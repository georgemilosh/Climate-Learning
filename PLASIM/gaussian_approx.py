# '''
# Created in January 2023

# @author: Alessandro Lovo
# '''
description = """Gaussian approximation"""
dependencies = None

import Learn2_new as ln
logger = ln.logger
ut = ln.ut
np = ln.np
tf = ln.tf
keras = ln.keras
layers = keras.layers
pd = ln.pd

import scipy.special as ss
from scipy import sparse

# log to stdout
import logging
import sys
import os

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

def compute_weight_matrix(reshape_mask, lat) -> sparse.dia_matrix:
    '''
    Compute the matrix W such that
    $$ H_2(p) = p^\top W p $$

    Parameters
    ----------
    reshape_mask : np.ndarray[bool]
        mask to flatten a snapshot `p` into a one dimensional array, eventually removing zero variance features
    lat : np.ndarray[float]
        latitude vector, used to compute the grid cell area and the proper longitudinal gradients

    Returns
    -------
    spares.dia_matrix[float]
        W
    '''
    shape = reshape_mask.shape
    shape_r = (np.sum(reshape_mask),)
    if len(shape) != 3:
        raise ValueError(f'reshape_mask should be a 3d array! Instead {reshape_mask.shape = }')

    geosep = ut.Reshaper(reshape_mask)

    W = sparse.lil_matrix(shape_r*2)
    #f -> field
    #i -> lat
    #j -> lon
    for f in range(shape[-1]):
        for i in range(shape[-3]):
            w = np.cos(lat[i]*np.pi/180)
            wi = 1./w
            for j in range(shape[-2]):
                # add latitude gradient
                try:
                    ind1 = geosep.reshape_index((i,j,f))
                    ind2 = geosep.reshape_index((i+1,j,f))
                except IndexError:
                    logger.debug(f'IndexError: {(i,j,f)}-{(i+1,j,f)}')
                else:
                    W[ind1,ind1] += w
                    W[ind2,ind2] += w
                    W[ind1,ind2] += -w
                    W[ind2,ind1] += -w

                # add longitude gradient
                try:
                    ind1 = geosep.reshape_index((i,j,f))
                    ind2 = geosep.reshape_index((i,j+1,f))
                except IndexError:
                    logger.debug(f'IndexError: {(i,j,f)}-{(i,j+1,f)}')
                else:
                    W[ind1,ind1] += wi
                    W[ind2,ind2] += wi
                    W[ind1,ind2] += -wi
                    W[ind2,ind1] += -wi

            # add periodic longitude point
            try:
                ind1 = geosep.reshape_index((i,shape[-2] - 1,f))
                ind2 = geosep.reshape_index((i,0,f))
            except IndexError:
                logger.debug(f'IndexError: {(i,shape[-2] - 1,f)}-{(i,0,f)}')
            else:
                W[ind1,ind1] += wi
                W[ind2,ind2] += wi
                W[ind1,ind2] += -wi
                W[ind2,ind1] += -wi
    return W.todia() # most of the entries of W are around the diagonal, so this is more efficient

class GaussianCommittor(object):
    def __init__(self, regularization_matrix=0, threshold=0, GPU=True, save_Sigmas=False):
        '''
        Object to compute a committor function under the gaussian assumption.

        Given a high dimensional input $x$ the probability (committor) $q$ of a target variable $a$ to be above a threshold $t$ is computed as

        $$ q = \frac{1}{2} \erfc(\alpha + \beta p^{\top} x) $$

        where $p$ is a norm 1 vector and $\alpha$ and $\beta$ are scalars. The three are fitted onto data thanks to the `self.fit` function.

        Parameters
        ----------
        regularization_matrix : np.ndarray or float, optional
            matrix to add to the covariance matrix before taking the inverse, by default 0
            If float it is multiplied by the identity matrix.
        threshold : float, optional
            `t`, by default 0
        '''
        self.regularization_matrix = regularization_matrix
        self.threshold = threshold
        self.p = None
        self.f_tr = None
        self.f = None
        self.save_Sigmas = save_Sigmas

        self.set_engine('GPU' if GPU else 'CPU')

    def set_engine(self, engine='GPU'):
        if engine == 'GPU':
            try:
                logger.info('Setting engine as GPU')
                import cupy as cp
                from gpuutils import GpuUtils
                df = GpuUtils.analyzeSystem().set_index('gpu_index')
                # get the value of 'gpu_index' of the GPU with the most free memory
                gpu_id = df['available_memories_in_mb'].idxmax()
                gpu_memory = df['available_memories_in_mb'].max()
                cp.cuda.runtime.setDevice(gpu_id)
                print(f'Using GPU Device {gpu_id}, with {gpu_memory/1024:.2f} GB free memory')
                self.engine = cp
                self.GPU = True
                self.precision = np.float32
                return
            except ModuleNotFoundError:
                logger.error('Please install cupy and gpuutils to use GPU')
            except:
                logger.error('Failed to use GPU, using CPU instead')

        logger.info('Setting engine as CPU')
        self.GPU = False
        self.engine = np
        self.precision = np.float64

    def fit(self,X,A):
        try:
            self._fit(X,A)
        except self.engine.cuda.memory.OutOfMemoryError:
            logger.error('Failed to allocate momory in GPU, retrying with CPU instead')
            self.set_engine('CPU')
            self._fit(X,A)

    def _fit(self, X, A):
        '''
        Fits the object on data

        Parameters
        ----------
        X : np.ndarray[float]
            Input data with shape (n_data_points, n_features)
        A : np.ndarray[float]
            Target variable with shape (n_data_points,)
        '''
        # check that the regularization matrix is indeed a matrix, if not make it a multiple of the identity matrix
        if not hasattr(self.regularization_matrix, 'shape') or self.regularization_matrix.shape == ():
            logger.info('multiplying scalar regularization matrix by the identity matrix')
            self.regularization_matrix = self.regularization_matrix * np.identity(X.shape[-1], dtype=float)

        # compute the covariance matrix
        XAs = np.concatenate([X,A.reshape(-1,1)], axis=-1)
        logger.info(f'{XAs.shape = }')

        if self.GPU:
            XAs = self.engine.asarray(XAs, dtype=self.precision) # convert to GPU array

        XAs_cov = self.engine.cov(XAs.T)
        # once we have the covariance matrix we don't need XAs anymore
        del XAs # this frees GPU memory

        logger.info(f'{XAs_cov.shape = }')
        sigma_XX = XAs_cov[:-1,:-1]
        sigma_XA = XAs_cov[-1,:-1]

        if self.save_Sigmas:
            if self.GPU:
                self._Sigma_XX = self.engine.asnumpy(sigma_XX)
                self._Sigma_XA = self.engine.asnumpy(sigma_XA)
            else:
                self._Sigma_XX = np.copy(sigma_XX)
                self._Sigma_XA = np.copy(sigma_XA)

        # now we don't need XAs_cov anymore
        del XAs_cov # this frees GPU memory

        assert self.regularization_matrix.shape == sigma_XX.shape

        # compute the (regularized) projection pattern
        self.p = self.engine.linalg.inv(sigma_XX + self.engine.asarray(self.regularization_matrix.toarray() if isinstance(self.regularization_matrix, sparse.spmatrix) else self.regularization_matrix, dtype=self.precision)) @ sigma_XA
        # now that we have the projection pattern we don't need sigma_XX and sigma_XA anymore
        del sigma_XX, sigma_XA # this frees GPU memory

        self.p /= self.engine.sqrt(self.engine.sum(self.p**2))
        logger.info(f'{self.p.shape = }')

        if self.GPU:
            # convert back to CPU
            self.p = np.copy(self.p)

        # compute the projected coordinate and the rescaling
        self.f_tr = X @ self.p
        fA = np.stack([self.f_tr, A]) # this is a 2 by 2 matrix: we don't need to use the GPU
        fA_cov = np.cov(fA)
        logger.info(f'{fA_cov.shape = }')
        lam = np.linalg.inv(fA_cov)

        self.lam_AA = lam[-1,-1]
        self.lam_fA = lam[0,-1]

        # compute the predicted standard deviation
        self.sigma = 1./np.sqrt(self.lam_AA)

        # compute the coefficient for the predicted mean (mu = m*f)
        self.m = -self.lam_fA/self.lam_AA

        # compute the coefficients for the rescaling
        # self.a = np.sqrt(self.lam_AA/2)*self.threshold
        # self.a = self.threshold / np.sqrt(2) / self.sigma
        # self.b = self.lam_fA/np.sqrt(2*self.lam_AA)
        # self.b = -self.m / np.sqrt(2) / self.sigma
        self.a, self.b = msigma2ab(self.m, self.sigma, self.threshold)


    def q(self,x=None):
        '''
        committor function

        Parameters
        ----------
        x : np.ndarray[float]
            observed input with shape (..., n_features)

        Returns
        -------
        np.ndarray[float]
            predicted committor with shape (...,)
        '''
        if x is not None:
            self.f = x @ self.p
        # return 0.5*ss.erfc((self.lam_AA*self.threshold + self.lam_fA*self.f)/np.sqrt(2*self.lam_AA))
        return 0.5*ss.erfc(self.a + self.b*self.f)

    def mu(self, x=None):
        if x is not None:
            self.f = x @ self.p
        return self.m * self.f

    def __call__(self,x):
        '''Alias for self.q'''
        return self.q(x)

def msigma2ab(m,sigma,threshold):
    a = threshold / np.sqrt(2) / sigma
    b = -m / np.sqrt(2) / sigma
    return a,b

def ab2msigma(a,b,threshold):
    sigma = threshold / np.sqrt(2) / a
    m = -b * np.sqrt(2) * sigma
    return m,sigma

orig_Trainer = ln.Trainer
# Here we redefine the `prepare_XY` function to save the heatwave amplitude A
class Trainer(ln.Trainer):
    def prepare_XY(self, fields, **prepare_XY_kwargs):
        if self._prepare_XY_kwargs != prepare_XY_kwargs:
            self._prepare_XY_kwargs = prepare_XY_kwargs
            X, self.Y, self.year_permutation, self.lat, self.lon, threshold = ln.prepare_XY(fields, **prepare_XY_kwargs) # timeseries is not what we want!

            label_field = ut.extract_nested(prepare_XY_kwargs, 'label_field')
            try:
                lf = fields[label_field]
            except KeyError:
                try:
                    lf = fields[f'{label_field}_ghost']
                except KeyError:
                    logger.error(f'Unable to find label field {label_field} among the provided fields {list(self.fields.keys())}')
                    raise KeyError

            A = lf.to_numpy(lf._time_average).reshape(lf.years, -1)[self.year_permutation].flatten()

            assert self.Y.shape == A.shape
            _Y = np.array(A >= threshold, dtype=int)
            diff = np.sum(np.abs(self.Y - _Y))
            assert diff == 0, f'{diff} datapoints do not match in labels'

            # here we do something very ugly and bundle A, threshold and lat together with X to pass through ln.Trainer.run function
            self.X = (X,A,threshold,self.lat)
        return self.X, self.Y, self.year_permutation, self.lat, self.lon

orig_train_model = ln.train_model
@ut.exec_time(logger)
@ut.indent_logger(logger)
def train_model(model, X_tr, A_tr, Y_tr, X_va, A_va, Y_va, folder, return_metric='val_CrossEntropyLoss', save_Sigmas=False):
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

    model.save_Sigmas = save_Sigmas
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

    ## save A_va, Y_va, f_va and Y_pred_unbiased (committor)
    np.save(f'{folder}/A_va.npy', A_va)
    np.save(f'{folder}/Y_va.npy', Y_va)
    np.save(f'{folder}/f_va.npy', model.f)
    np.save(f'{folder}/Y_pred_unbiased.npy', q_va)

    ## save Sigmas (takes a lot of space)
    if save_Sigmas:
        np.save(f'{folder}/Sigma_XX.npy', model._Sigma_XX)
        np.save(f'{folder}/Sigma_XA.npy', model._Sigma_XA)

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

orig_k_fold_cross_val = ln.k_fold_cross_val
@ut.exec_time(logger)
@ut.indent_logger(logger)
def k_fold_cross_val(folder, X, Y, train_model_kwargs=None, optimal_checkpoint_kwargs=None, load_from=None, nfolds=10, val_folds=1, u=1, normalization_mode='pointwise',
                    regularization='gradient', reg_c=0, use_GPU=True):
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

    #save threshold
    np.save(f'{folder}/threshold.npy', threshold)

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
            sparse.save_npz(f'{folder}/W.npz', W)
        else:
            logger.error(f'Unrecognized regularization mode {regularization}')
            raise KeyError()
        reg_matrix = reg_c*W

    # create the model
    model = GaussianCommittor(reg_matrix,threshold=threshold, GPU=use_GPU)

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
            X_tr, _, _ = ln.normalize_X(X_tr, fold_folder, mode=normalization_mode)
            #X_va = (X_va - X_mean)/X_std
            X_va, _, _ = ln.normalize_X(X_va, fold_folder) # we expect that the previous operation stores X_mean, X_std
            logger.info(f'after normalization: {X_tr.shape = }, {X_va.shape = }, {Y_tr.shape = }, {Y_va.shape = }')

        # train the model
        score = train_model(model, X_tr, A_tr, Y_tr, X_va, A_va, Y_va, # arguments that are always computed inside this function
                            folder=fold_folder, # arguments that may come from train_model_kwargs for advanced uses but usually are computed here
                            **train_model_kwargs) # arguments which have a default value in the definition of `train_model` and thus appear in the config file

        # retrieve the projection pattern and save it
        np.save(f'{fold_folder}/proj.npy', geosep.inv_reshape(model.p))
        np.save(f'{fold_folder}/ab.npy', np.array([model.a, model.b])) # rescaling coefficients
        # with the knowledge of these two we can compute the committor
        np.save(f'{fold_folder}/msigma.npy', np.array([model.m, model.sigma])) # m and sigma to easily compute predicted mean and std

        scores.append(score)

        my_memory.append(ln.psutil.virtual_memory())
        logger.info(f'RAM memory: {my_memory[i][3]:.3e}') # Getting % usage of virtual_memory (3rd field)

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
def enable():
    ln.add_mod(__file__, description, dependencies)
    ln.k_fold_cross_val = k_fold_cross_val
    ln.train_model = train_model
    ln.Trainer = Trainer
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary
    ut.set_values_recursive(ln.CONFIG_DICT, {'return_threshold': True}, inplace=True)

def disable():
    ln.remove_mod(__file__)
    ln.k_fold_cross_val = orig_k_fold_cross_val
    ln.train_model = orig_train_model
    ln.Trainer = orig_Trainer
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary

if __name__ == '__main__':
    enable()
    ln.main()
