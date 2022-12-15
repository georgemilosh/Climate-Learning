import Learn2_new as ln
logger = ln.logger
ut = ln.ut

# log to stdout
import logging
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
logging.getLogger().level = logging.INFO
logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

sys.path.append('../../MaxSeparation/')
import MaxSeparator as ms


@ut.execution_time
@ut.indent_logger(logger)
def k_fold_cross_val(folder, X, Y, train_model_kwargs=None, optimal_checkpoint_kwargs=None, nfolds=10, val_folds=1, u=1, normalization_mode='pointwise',
                    training_epochs=40, patience=0):
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
    lr : float, optional
        learning_rate for Adam optimizer

    prune_threshold : float, optional
        if the average score in the first `min_folds_before_pruning` is above `prune_threshold`, the run is pruned.
        This means that the run is considered not promising and hence we avoid losing time in computing the remaining folds.
        This is particularly useful when performing a hyperparameter optimization procedure.
        By default is None, which means that pruning is disabled
    min_folds_before_pruning : int, optional
        minimum number of folds to train before checking whether to prune the run
        By default None, which means that pruning is disabled

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
    folder = folder.rstrip('/')

    load_from = None
    # get the folders from which to load the models
    load_from, info = ln.get_transfer_learning_folders(load_from, folder, nfolds, optimal_checkpoint_kwargs=optimal_checkpoint_kwargs)
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

        # perform undersampling
        X_tr, Y_tr = ln.undersample(X_tr, Y_tr, u=u)

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

        # at this point data is ready to be fed to the networks


        # create the model
        model = ms.GeoSeparator()

        # number of training epochs
        num_epochs = train_model_kwargs.pop('num_epochs', None) # if num_epochs is not provided in train_model_kwargs, which is most of the time,
                                                                # we assign it according if we have to do transfer learning or not
        if num_epochs is None:
            num_epochs = training_epochs


        # train the model
        score = train_model(model, X_tr, Y_tr, X_va, Y_va, # arguments that are always computed inside this function
                            folder=fold_folder, num_epochs=num_epochs, # arguments that may come from train_model_kwargs for advanced uses but usually are computed here
                            **train_model_kwargs) # arguments which have a default value in the definition of `train_model` and thus appear in the config file

        scores.append(score)

        my_memory.append(ln.psutil.virtual_memory())
        logger.info(f'RAM memory: {my_memory[i][3]:.3e}') # Getting % usage of virtual_memory ( 3rd field)

        ln.gc.collect() # Garbage collector which removes some extra references to the objects. This is an attempt to micromanage the python handling of RAM
        
    np.save(f'{folder}/RAM_stats.npy', my_memory)

    # recompute the scores if collective=True
    # Here we want to use the `optimal_checkpoint` function to compute the best checkpoint for this network. 
    # Mind that before we used it to determine the optimal checkpoint from the network from which to perform transfer learning, so we need to change the parameters
    try:
        collective = optimal_checkpoint_kwargs['collective']
    except KeyError:
        collective = ln.get_default_params(ln.optimal_checkpoint)['collective']
    if collective:
        logger.log(35, 'recomputing scores and network predictions with the collective optimal checkpoint')
        try:
            return_metric = train_model_kwargs['return_metric']
        except KeyError:
            return_metric = ln.get_default_params(train_model)['return_metric']
        try:
            first_epoch = optimal_checkpoint_kwargs['first_epoch']
        except KeyError:
            first_epoch = ln.get_default_params(ln.optimal_checkpoint)['first_epoch']
            
        opt_checkpoint, fold_subfolder = ln.optimal_checkpoint(folder,nfolds, **optimal_checkpoint_kwargs)

        # recompute the scores
        for i in range(nfolds):
            scores[i] = np.load(f'{folder}/fold_{i}/{fold_subfolder}history.npy', allow_pickle=True).item()[return_metric][opt_checkpoint - first_epoch]

        # reload the models at their proper checkpoint and recompute Y_pred_unbiased
        batch_size = train_model_kwargs['batch_size']
        for i in range(nfolds):
            _, _, X_va, Y_va = ln.k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)
            fold_folder = f'{folder}/fold_{i}'
            model = ms.load_model(f'{fold_folder}/{fold_subfolder}cp-{opt_checkpoint:04d}.ckpt')

            Y_pred = []
            for b in range(Y_va.shape[0]//batch_size + 1):
                Y_pred.append(model(X_va[b*batch_size:(b+1)*batch_size]))
            Y_pred = np.concatenate(Y_pred)
            np.save(f'{fold_folder}/Y_pred_unbiased.npy', Y_pred)
        

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

ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary

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
