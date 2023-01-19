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
    def __init__(self):
        pass

@ut.execution_time
@ut.indent_logger(logger)
def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, return_metric='CrossEntropyLoss'):
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
    ### preliminary operations
    ##########################
    folder = folder.rstrip('/')


    ### training the model
    ######################

    # log the amount af data that is entering the network
    logger.info(f'Training the network on {len(Y_tr)} datapoint and validating on {len(Y_va)}')

    # prepare the data
    X0_tr = X_tr[Y_tr==0]
    X1_tr = X_tr[Y_tr==1]

    X0_va = X_va[Y_va==0]
    X1_va = X_va[Y_va==1]
    
    # prepare the model
    model.set_data(X0_tr,X1_tr)
    model.compute_rotation()

    # prepare the history
    history = {'invFisher': [], 'val_invFisher': []}

    # perform training for `num_epochs`
    best_epoch = 0
    best_value = np.inf
    non_improving_epochs = 0
    for epoch in range(1, num_epochs+1):
        model.compute_projection(n_directions=epoch)

        tr_score = model.inv_fisher
        va_score = 1./ms.score(model(X0_va), model(X1_va))

        history['invFisher'].append(tr_score)
        history['val_invFisher'].append(va_score)

        print(f'{epoch = }: {tr_score = }, {va_score = }')

        if checkpoint_every and epoch % checkpoint_every == 0:
            model.save_proj(f'{folder}/cp-{epoch:04d}.npy')

        if patience:
            if va_score < best_value:
                best_epoch = epoch
                non_improving_epochs = 0
                best_value = va_score
            else:
                non_improving_epochs += 1
                if non_improving_epochs > patience:
                    # checkpoint back and exit the loop
                    model.compute_projection(n_directions=best_epoch+1)
                    if not os.path.exists(f'{folder}/cp-{best_epoch:04d}.npy'):
                        model.save_proj(f'{folder}/cp-{best_epoch:04d}.npy')
                    break

    ## save Y_va and Y_pred_unbiased
    np.save(f'{folder}/Y_va.npy', Y_va)
    Y_pred = model(X_va)
    np.save(f'{folder}/Y_pred_unbiased.npy', Y_pred)

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