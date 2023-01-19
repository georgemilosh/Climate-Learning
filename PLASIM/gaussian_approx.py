# '''
# Created in January 2023

# @author: Alessandro Lovo
# '''
import Learn2_new as ln
logger = ln.logger
early_stopping = ln.early_stopping
ut = ln.ut
np = ln.np
tf = ln.tf
keras = ln.keras
layers = keras.layers
pd = ln.pd

import scipy.special as ss

# log to stdout
import logging
import sys
import os
from pathlib import Path
logging.getLogger().level = logging.INFO
logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

def compute_weight_matrix(reshape_mask, lat):
    shape = reshape_mask.shape
    shape_r = (np.sum(reshape_mask),)
    if len(shape) != 3:
        raise ValueError(f'reshape_mask should be a 3d array! Instead {reshape_mask.shape = }')

    geosep = ut.Reshaper(reshape_mask)
    
    W = np.zeros(shape_r*2)
    #f -> field
    #i -> lat
    #j -> lon
    for f in range(shape[-1]):
        for i in range(shape[-3]):
            w = 1./np.cos(lat[i]*np.pi/180)
            for j in range(shape[-2]):
                # add latitude gradient
                try:
                    ind1 = geosep.reshape_index((i,j,f))
                    ind2 = geosep.reshape_index((i+1,j,f))
                except IndexError:
                    print(f'IndexError: {(i,j,f)}-{(i+1,j,f)}')
                else:
                    W[ind1,ind1] += 1
                    W[ind2,ind2] += 1
                    W[ind1,ind2] += -1
                    W[ind2,ind1] += -1
                
                # add longitude gradient
                try:
                    ind1 = geosep.reshape_index((i,j,f))
                    ind2 = geosep.reshape_index((i,j+1,f))
                except IndexError:
                    print(f'IndexError: {(i,j,f)}-{(i,j+1,f)}')
                else:
                    W[ind1,ind1] += w
                    W[ind2,ind2] += w
                    W[ind1,ind2] += -w
                    W[ind2,ind1] += -w
                
            # add periodic longitude point
            try:
                ind1 = geosep.reshape_index((i,shape[-2] - 1,f))
                ind2 = geosep.reshape_index((i,0,f))
            except IndexError:
                print(f'IndexError: {(i,shape[-2] - 1,f)}-{(i,0,f)}')
            else:
                W[ind1,ind1] += w
                W[ind2,ind2] += w
                W[ind1,ind2] += -w
                W[ind2,ind1] += -w
    return W

class GaussianCommittor(object):
    def __init__(self, regularization_matrix=None, threshold=0):
        self.regularization_matrix = regularization_matrix or 0
        self.threshold = threshold
        self.p = None
        self.f_tr = None
        self.f = None

    def fit(self, X, A):
        # compute the covariance matrix
        XAs = np.concatenate([X,A.reshape(-1,1)], axis=-1)
        logger.info(f'{XAs.shape = }')
        XAs_cov = np.cov(XAs.T)
        logger.info(f'{XAs_cov.shape = }')
        sigma_XX = XAs_cov[:-1,:-1]
        sigma_XA = XAs_cov[-1,:-1]

        # compute the (regularized) projection pattern
        self.p = np.linalg.inv(sigma_XX + self.regularization_matrix) @ sigma_XA
        self.p /= np.sqrt(np.sum(self.p**2))
        logger.info(f'{self.p.shape = }')

        # compute the projected coordinate and the rescaling
        self.f_tr = X @ self.p
        fA = np.stack([self.f_tr, A])
        fA_cov = np.cov(fA)
        logger.info(f'{fA_cov.shape = }')
        lam = np.linalg.inv(fA_cov)
        self.lam_AA = lam[-1,-1]
        self.lam_fA = lam[0,-1]

    def q(self,x):
        self.f = x @ self.p
        return 0.5*ss.erfc((self.lam_AA*self.threshold + self.lam_fA*self.f)/np.sqrt(2*self.lam_AA))

    def __call__(self,x):
        return self.q(x)


@ut.execution_time
@ut.indent_logger(logger)
def train_model(model, X_tr, A_tr, Y_tr, X_va, A_va, Y_va, folder, return_metric='val_CrossEntropyLoss'):
    '''
    Trains a given model

    Parameters
    ----------
    model : keras.models.Model
    X_tr : np.ndarray
        training data
    Y_tr : np.ndarray
        training labels
    X_va : np.ndarray
        validation data
    Y_va : np.ndarray
        validation labels
    folder : str or Path
        location where to save the checkpoints of the model
    return_metric : str, optional
        name of the metric of which the minimum value will be returned at the end of training

    Returns
    -------
    float
        value of `return_metric` during training
    '''
    folder = folder.rstrip('/')

    # log the amount af data that is entering the network
    logger.info(f'Training the network on {len(Y_tr)} datapoint and validating on {len(Y_va)}')

    # fit the model
    model.fit(X_tr, A_tr)

    # compute metrics
    q_tr = model(X_tr)
    r_tr = np.corrcoef(model.f, A_tr)[0,1]
    ce_tr = np.mean(ut.entropy(Y_tr, q_tr))

    q_va = model(X_va)
    r_va = np.corrcoef(model.f, A_va)[0,1]
    ce_va = np.mean(ut.entropy(Y_va, q_va))

    history = {'CrossEntropyLoss': [ce_tr], 'val_CrossEntropyLoss': [ce_va], 'r': [r_tr], 'val_r': [r_va]}

    ## save Y_va and Y_pred_unbiased
    np.save(f'{folder}/Y_va.npy', Y_va)
    np.save(f'{folder}/Y_pred_unbiased.npy', q_va)

    ## deal with history
    np.save(f'{folder}/history.npy', history)
    # log history
    df = pd.DataFrame(history)
    df.index.name = 'epoch-1'
    logger.log(25, str(df))
    df.to_csv(f'{folder}/history.csv', index=True)

    # return the best value of the return metric
    if return_metric not in history:
        logger.error(f'{return_metric = } is not one of the metrics monitored during training, returning NaN')
        score = np.NaN
    else:
        score = np.min(history[return_metric])
    logger.log(42, f'{score = }')
    return score