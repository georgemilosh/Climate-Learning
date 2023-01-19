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

        # compute the coefficients for the rescaling
        self.a = np.sqrt(self.lam_AA/2)*self.threshold
        self.b = self.lam_fA/np.sqrt(2*self.lam_AA)

    def q(self,x):
        self.f = x @ self.p
        # return 0.5*ss.erfc((self.lam_AA*self.threshold + self.lam_fA*self.f)/np.sqrt(2*self.lam_AA))
        return 0.5*ss.erfc(self.a + self.b*self.f)

    def __call__(self,x):
        return self.q(x)


class Trainer(ln.Trainer):
    def prepare_XY(self, fields, **prepare_XY_kwargs):
        if self._prepare_XY_kwargs != prepare_XY_kwargs:
            self._prepare_XY_kwargs = prepare_XY_kwargs
            X, self.Y, self.year_permutation, self.lat, self.lon, timeseries, threshold = ln.prepare_XY(fields, **prepare_XY_kwargs) # timeseries is not what we want!

            label_field = ut.extract_nested(prepare_XY_kwargs, 'label_field')
            if label_field not in fields:
                label_field = f'{label_field}_ghost'
            if label_field not in fields:
                logger.error(f'Unable to find label field {label_field} among the provided fields {list(self.fields.keys())}')
                raise KeyError
            k2i = {k:i for i,k in enumerate(fields)}
            A = timeseries[:,k2i[label_field]]

            assert self.Y.shape == A.shape
            # here we do something very ugly and bundle A, threshold and lat together with X to pass through ln.Trainer.run function
            self.X = (X,A,threshold,self.lat)
        return self.X, self.Y, self.year_permutation, self.lat, self.lon


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

    ## save A_va, Y_va and Y_pred_unbiased
    np.save(f'{folder}/A_va.npy', A_va)
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

@ut.execution_time
@ut.indent_logger(logger)
def k_fold_cross_val(folder, X, Y, train_model_kwargs=None, optimal_checkpoint_kwargs=None, load_from=None, nfolds=10, val_folds=1, u=1, normalization_mode='pointwise',
                    regularization='gradient', reg_c=0):
    '''
    Performs k fold cross validation on a model architecture.

    Parameters
    ----------
    folder : str
        folder in which to save data related to the folds
    X : np.ndarray
        all data (train + val)
    Y : np.ndarray
        all labels
    create_model_kwargs : dict
        dictionary with the parameters to create a model
    train_model_kwargs : dict
        dictionary with the parameters to train a model
        For most common use (command line) you can only specify arguments that have a default value and so appear in the config file.
        However when runing this function from a notebook you can use more advanced features like using another loss rather than the default cross entropy
        or an optimizer rather than Adam.
        This can be done specifying other parameters rather than the ones that appear in the config file, namely:
            num_epochs : int
                number of training epochs. `training_epochs` and `training_epochs_tl` are ignored
            optimizer : keras.optimizers.Optimizer
                optimizer object, `lr` is ignored
            loss : keras.metrics.Metric
                overrides the `loss`
            metrics : list of metrics objects
                overrides `fullmetrics`
    optimal_chekpoint_kwargs : dict
        dictionary with the parameters to find the optimal checkpoint
    load_from : None, int, str or 'last', optional
        from where to load weights for transfer learning. See the documentation of function `get_run`
        If not None it overrides `create_model_kwargs` (the model is loaded instead of created)
    nfolds : int, optional
        number of folds
    val_folds : int, optional
        number of folds to be used for the validation set for every split
    u : float, optional
        undersampling factor (>=1). If = 1 no undersampling is performed
    
    regularization : 'identity' or 'gradient'
        How to regularize the covariance matrix
    reg_c : float
        Amount of regularization

    Returns
    -------
    float
        average score of the run
    '''
    if train_model_kwargs is None:
        train_model_kwargs = {}
    if optimal_checkpoint_kwargs is None:
        optimal_checkpoint_kwargs = {}
    folder = folder.rstrip('/')

    if load_from is not None:
        raise NotImplementedError('Sorry: cannot do transfer learning with this code')
    if u != 1:
        raise NotImplementedError('Sorry, cannot use undersampling with this code')
    # get the folders from which to load the models
    load_from, info = ln.get_transfer_learning_folders(load_from, folder, nfolds, optimal_checkpoint_kwargs=optimal_checkpoint_kwargs)
    # here load_from is either None (no transfer learning) or a list of strings

    my_memory = []
    info['status'] = 'RUNNING'

    # unbundle X
    X, A, threshold, lat = X

    # reshape X to remove zero_variance features
    geosep = ut.Reshaper(np.std(X[:10], axis=0) != 0)
    logger.info(f'{geosep.reshape_mask.shape = }, {np.sum(geosep.reshape_mask) = }')

    logger.info(f'{X.shape = }, reshaping')
    X = geosep.reshape(X)
    logger.info(f'{X.shape = }')

    # compute regularization matrix
    reg_matrix = 0
    if reg_c:
        if regularization == 'identity':
            W = np.identity(geosep.surviving_coords)
        elif regularization == 'gradient':
            W = compute_weight_matrix(geosep.reshape_mask,lat)
            np.save(f'{folder}/W.npy', W)
        else:
            logger.error(f'Unrecognized regularization mode {regularization}')
            raise KeyError()
        reg_matrix = reg_c*W

    # create the model
    model = GaussianCommittor(reg_matrix,threshold=threshold)

    # k fold cross validation
    scores = []
    for i in range(nfolds):
        logger.info('=============')
        logger.log(35, f'fold {i} ({i+1}/{nfolds})')
        logger.info('=============')
        # create fold_folder
        fold_folder = f'{folder}/fold_{i}'
        os.mkdir(fold_folder)

        # split data
        X_tr, A_tr, Y_tr, X_va, A_va, Y_va = ln.k_fold_cross_val_split(i, X, A, Y, nfolds=nfolds, val_folds=val_folds)

        n_pos_tr = np.sum(Y_tr)
        n_neg_tr = len(Y_tr) - n_pos_tr
        logger.info(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')

        if normalization_mode: # normalize X_tr and X_va
            X_tr, X_mean, X_std = ln.normalize_X(X_tr, mode=normalization_mode)
            X_va = (X_va - X_mean)/X_std 

            # save X_mean and X_std
            np.save(f'{fold_folder}/X_mean.npy', X_mean) # GM: Why not include all of this in normalize_X? It may simplify the code -> AL: Because normalize_X doesn't know about fold_folder
            np.save(f'{fold_folder}/X_std.npy', X_std)
        
            
        logger.info(f'{X_tr.shape = }, {X_va.shape = }')

        # train the model
        score = train_model(model, X_tr, A_tr, Y_tr, X_va, A_va, Y_va, # arguments that are always computed inside this function
                            folder=fold_folder, # arguments that may come from train_model_kwargs for advanced uses but usually are computed here
                            **train_model_kwargs) # arguments which have a default value in the definition of `train_model` and thus appear in the config file

        # retrieve the projection pattern and save it
        np.save(f'{fold_folder}/proj.npy', geosep.inv_reshape(model.p))
        np.save(f'{fold_folder}/ab.npy', np.array([model.a, model.b])) # rescaling coefficients
        # with the knowledge of these two we can compute the committor
        

        scores.append(score)

        my_memory.append(ln.psutil.virtual_memory())
        logger.info(f'RAM memory: {my_memory[i][3]:.3e}') # Getting % usage of virtual_memory ( 3rd field)

        ln.gc.collect() # Garbage collector which removes some extra references to the objects. This is an attempt to micromanage the python handling of RAM
        
    np.save(f'{folder}/RAM_stats.npy', my_memory)        

    score_mean = np.mean(scores)
    score_std = np.std(scores)

    # log the scores
    info['scores'] = {}
    logger.info('\nFinal scores:')
    for i,s in enumerate(scores):
        logger.info(f'\tfold {i}: {s}')
        info['scores'][f'fold_{i}'] = s
    logger.log(45,f'Average score: {ln.ufloat(score_mean, score_std)}')
    info['scores']['mean'] = score_mean
    info['scores']['std'] = score_std

    info['scores'] = ln.ast.literal_eval(str(info['scores']))

    if info['status'] != 'PRUNED':
        info['status'] = 'COMPLETED'

    # return the average score
    return score_mean, info


#######################################################
# set the modified functions to override the old ones #
#######################################################
ln.k_fold_cross_val = k_fold_cross_val
ln.train_model = train_model
ln.Trainer = Trainer

ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary
ut.set_values_recursive(ln.CONFIG_DICT, {'return_timeseries': True, 'return_threshold': True}, inplace=True)

if __name__ == '__main__':
    ln.main()

    lock = ln.Path(__file__).resolve().parent / 'lock.txt'
    if os.path.exists(lock): # there is a lock
        # check for folder argument
        if len(sys.argv) == 2:
            folder = sys.argv[1]
            print(f'moving code to {folder = }')
            # copy this file
            path_to_here = ln.Path(__file__).resolve() # path to this file
            ln.shutil.copy(path_to_here, folder)
