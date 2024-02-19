# '''
# Created in February 2024

# @author: Alessandro Lovo
# '''
description = """Train a model as a perturbation on top of a gaussian model for probabilistic regression."""
dependencies = None

import Learn2_new as ln
logger = ln.logger
ut = ln.ut
np = ln.np
tf = ln.tf
keras = ln.keras
layers = keras.layers
pd = ln.pd
import os

from functools import wraps

# log to stdout
import logging
import sys
from pathlib import Path

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]


class GaPerturbation(object):
    def __init__(self, model, path_to_ga):
        self.model = model
        self.proj = tf.convert_to_tensor(np.load(f'{path_to_ga}/proj.npy'), dtype=tf.float32)
        self.m, self.sigma = np.load(f'{path_to_ga}/msigma.npy').astype(np.float32)

    def __call__(self, x):
        f = tf.tensordot(x, self.proj, axes=len(self.proj.shape))
        ga_output = tf.stack([f*self.m, tf.ones_like(f)*self.sigma], axis=-1)
        model_output = self.model(x)
        assert ga_output.shape == model_output.shape, (f'{ga_output.shape = } != {model_output.shape = }')
        return model_output + ga_output

    def __getattr__(self, name):
        if name in ['model', 'proj', 'm', 'sigma', '__call__']:
            return getattr(self, name)
        return getattr(self.model, name)


def create_perturbation_model(input_shape, path_to_ga, **create_model_kwargs):
    if create_model_kwargs is None:
        create_model_kwargs = {}
    model = ln.create_model(input_shape, **create_model_kwargs)
    return GaPerturbation(model, path_to_ga)


orig_k_fold_cross_val = ln.k_fold_cross_val

def k_fold_cross_val(folder, X, Y, path_to_ga=None, create_model_kwargs=None, train_model_kwargs=None, optimal_checkpoint_kwargs=None,
                     load_from='last', ignorable_keys=None, nfolds=10, val_folds=1, u=1, normalization_mode='pointwise',
                     fullmetrics=True, training_epochs=40, training_epochs_tl=10, loss='sparse_categorical_crossentropy', prune_threshold=None, min_folds_before_pruning=None,
                     Z_DIM=None, T=14, time_start=30, time_end=120, label_period_start=None, label_period_end=None):
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
    ignorable_keys : list of str, optional
        keys that are not important when comparing models for transfer learning, by default None
    nfolds : int, optional
        number of folds
    val_folds : int, optional
        number of folds to be used for the validation set for every split
    u : float, optional
        undersampling factor (>=1). If = 1 no undersampling is performed
    fullmetrics : bool, optional
        whether to use a set of evaluation metrics or just the loss
    training_epochs : int, optional
        number of training epochs when creating a model from scratch
    training_epochs_tl : int, optional
        numer of training epochs when using transfer learning
    loss : str, optional
        loss function to minimize, by default 'sparse_categorical_crossentropy'
        another possibility is 'unbiased_crossentropy',
        which will unbias the logits with the undersampling factor and then proceeds with the sparse_categorical_crossentropy

    prune_threshold : float, optional
        if the average score in the first `min_folds_before_pruning` is above `prune_threshold`, the run is pruned.
        This means that the run is considered not promising and hence we avoid losing time in computing the remaining folds.
        This is particularly useful when performing a hyperparameter optimization procedure.
        By default is None, which means that pruning is disabled
    min_folds_before_pruning : int, optional
        minimum number of folds to train before checking whether to prune the run
        By default None, which means that pruning is disabled
    Z_DIM: int, optional
        if Z_DIM is not None, pca decomposition is performed to Z_DIM components
    T : int, optional
        width of the window for the running average
    time_start : int, optional
        first day of the period of interest (copied from make_XY to be able to compute `timestamps`)
    time_end : int, optional
        first day after the end of the period of interst (copied from make_XY)
    label_period_start : int, optional
        if provided the first day of the period of interest for the label threshold determination (copied from make_XY)
        This variable is necessary if for some reason we need to also load data that lies outside the range of where
        the labels that we need for training/validation and testing directly
            leftmargin = label_period_start - time_start
            if positive will be treated as a
    label_period_end : int, optional
        if provided the first day after the end of the period of interst for the label threshold determination (copied from make_XY)
        This variable is necessary if for some reason we need to also load data that lies outside the range of where
        the labels that we need for training/validation and testing directly
    Returns
    -------
    float
        average score of the run
    '''

    if create_model_kwargs is None:
        create_model_kwargs = {}
    if train_model_kwargs is None:
        train_model_kwargs = {}
    if optimal_checkpoint_kwargs is None:
        optimal_checkpoint_kwargs = {}

    if path_to_ga is None or not os.path.exists(path_to_ga):
        raise ValueError(f'{path_to_ga = } does not exist')
    path_to_ga = path_to_ga.rstrip('/')
    folder = folder.rstrip('/')

    """leftmargin: (int), optional
        Specifies the number of timestamps that we use to make the prediction. By default is absent (one time stamp)
        The use of `leftmargin` is only recommended if PCA was performed since otherwise too much RAM is taken"""

    # X and Y are extracted from time_start to time_end, but we only care about the part inside label_period_start to label_period_end for training and testing
    if label_period_start is None: # label_period_start == time_start implied
        leftmargin = None # basically left margin
    else:
        leftmargin = label_period_start - time_start
        if leftmargin < 0:
            raise ValueError(f'leftmargin = label_period_start - time_start < 0 which is not allowed!')
        elif leftmargin == 0:
            leftmargin = None

    if label_period_end is None:
        rightmargin = None
    else:
        rightmargin = time_end - label_period_end - T + 1 # that's because when we perform running mean we have to avoid using last T days
        if rightmargin < 0:
            raise ValueError(f'leftmargin = time_end - label_period_end - T + 1 which is not allowed!')


    # get the folders from which to load the models
    load_from, info = ln.get_transfer_learning_folders(load_from, folder, nfolds, optimal_checkpoint_kwargs=optimal_checkpoint_kwargs, ignorable_keys=ignorable_keys)
    # here load_from is either None (no transfer learning) or a list of strings

    my_memory = []
    info['status'] = 'RUNNING'

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
        X_tr, Y_tr, X_va, Y_va = ln.k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)

        if normalization_mode: # normalize X_tr and X_va
            X_tr, _, _ = ln.normalize_X(X_tr, fold_folder, mode=normalization_mode)
            #X_va = (X_va - X_mean)/X_std
            X_va, _, _ = ln.normalize_X(X_va, fold_folder) # we expect that the previous operation stores X_mean, X_std
            logger.info(f'after normalization: {X_tr.shape = }, {X_va.shape = }, {Y_tr.shape = }, {Y_va.shape = }')

        if Z_DIM is not None:
            with ln.PCAer(Z_DIM=Z_DIM, folder=fold_folder) as pcaer:
                pcaer.fit_with_timeout(0,X_tr)
                X_tr = pcaer.encoder.predict(X_tr)
                X_va = pcaer.encoder.predict(X_va)
            logger.info(f'after PCA: {X_tr.shape = }, {X_va.shape = }')

        logger.info(f' {time_start = }, {time_end = }, {leftmargin = }, {rightmargin = }, {T = }')
        X_tr = ln.margin_removal_with_sliding_window(X_tr,time_start,leftmargin,rightmargin,time_end,T,sliding=True)
        X_va = ln.margin_removal_with_sliding_window(X_va,time_start,leftmargin,rightmargin,time_end,T,sliding=True)
        Y_tr = ln.margin_removal_with_sliding_window(Y_tr,time_start,leftmargin,rightmargin,time_end,T)
        Y_va = ln.margin_removal_with_sliding_window(Y_va,time_start,leftmargin,rightmargin,time_end,T)
        logger.info(f'After margin removal: {X_tr.shape = }, {X_va.shape = }, {Y_tr.shape = }, {Y_va.shape = }')

        # perform undersampling
        X_tr, Y_tr = ln.undersample(X_tr, Y_tr, u=u)

        n_pos_tr = np.sum(Y_tr)
        n_neg_tr = len(Y_tr) - n_pos_tr
        logger.info(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')
        # at this point data is ready to be fed to the networks


        # check for transfer learning
        model = None
        if load_from is None:
            fold_path_to_ga = f'{path_to_ga}/fold_{i}'
            model = ln.create_model(input_shape=X_tr.shape[1:], **create_model_kwargs)
            model = GaPerturbation(model, fold_path_to_ga)
        else:
            model = ln.load_model(load_from[i], compile=False)
        summary_buffer = ut.Buffer() # workaround necessary to log the structure of the network to the file, since `model.summary` uses `print`
        summary_buffer.append('\n')
        model.summary(print_fn = lambda x: summary_buffer.append(x + '\n'))
        logger.info(summary_buffer.msg)

        # number of training epochs
        num_epochs = train_model_kwargs.pop('num_epochs', None) # if num_epochs is not provided in train_model_kwargs, which is most of the time,
                                                                # we assign it according if we have to do transfer learning or not
        if num_epochs is None:
            if load_from is None:
                num_epochs = training_epochs
            else:
                num_epochs = training_epochs_tl

        # metrics
        metrics = train_model_kwargs.pop('metrics', None)
        if metrics is None:
            metrics = ln.get_default_metrics(fullmetrics, u=u)

        # optimizer
        optimizer = train_model_kwargs.pop('optimizer',keras.optimizers.Adam()) # if optimizer is not provided in train_model_kwargs use Adam
        # loss function
        loss_fn = train_model_kwargs.pop('loss',None)
        if loss_fn is None:
            loss_fn = ln.get_loss_function(loss, u=u)
        logger.info(f'Using {loss_fn.name} loss')


        # train the model
        score = ln.train_model(model, X_tr, Y_tr, X_va, Y_va, # arguments that are always computed inside this function
                            folder=fold_folder, num_epochs=num_epochs, optimizer=optimizer, loss=loss_fn, metrics=metrics, # arguments that may come from train_model_kwargs for advanced uses but usually are computed here
                            **train_model_kwargs) # arguments which have a default value in the definition of `train_model` and thus appear in the config file

        scores.append(score)

        my_memory.append(ln.psutil.virtual_memory())
        try:
            logger.info(f'RAM memory: {my_memory[i][3]:.3e}') # Getting % usage of virtual_memory ( 3rd field)
        except:
            logger.info(f'RAM memory: {my_memory[i]:.3e}') # Getting % usage of virtual_memory ( 3rd field)

        keras.backend.clear_session()
        ln.gc.collect() # Garbage collector which removes some extra references to the objects. This is an attempt to micromanage the python handling of RAM

        # check for pruning, i.e. if the run is not promising we don't compute all the folds to save time
        if min_folds_before_pruning is not None and prune_threshold is not None:
            if i >= min_folds_before_pruning - 1 and i < nfolds - 1:
                score_mean = np.mean(scores) # we compute the average score of the already computed folds
                if score_mean > prune_threshold: # score too high, we prune the run
                    info['status'] = 'PRUNED'
                    logger.log(41,f'Pruning after {i+1}/{nfolds} folds')
                    break

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
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary

def disable():
    ln.remove_mod(__file__)
    ln.k_fold_cross_val = orig_k_fold_cross_val
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])

if __name__ == '__main__':
    enable()
    ln.main()