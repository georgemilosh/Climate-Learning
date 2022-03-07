# '''
# Created in February 2022

# @author: Alessandro Lovo
# '''
import Learn2_new as ln
logger = ln.logger
early_stopping = ln.early_stopping
ut = ln.ut
np = ln.np
keras = ln.keras
pd = ln.pd

# log to stdout
import logging
import sys
import os
logging.getLogger().level = logging.INFO
logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

def select(*arrays, amount=0.1, p=None):
    '''
    selects a given amount of data from a set of arrays according to a probability distribution
    Parameters
    ----------
    *arrays : M arrays with first dimension of size N  
        The arrays from which to select from. They are assumed to be synchronized, i.e. if, for example, the first element of an array is selected, it is selected in every array (same indices are selected)
    amount : int or float, optional
        Amount of data to select. If int: number of elements; if float fraction of elsements, by default 0.1, which means 10% of the elements
    p : 1D array-like of shape (N,), optional
        array of probabilities corresponding to each element in `arrays`. By default None, which implies a uniform distribution over the elements af the arrays

    Returns
    -------
    [
        (a_selected, a_remaining)
        for a in arrays
    ]

    Raises
    ------
    ValueError
        If `arrays` have different first dimension lenght or it is different from the lenght of `p` or `amount` is float but not in [0,1]

    Examples
    --------
    >>> np.random.seed(0)
    >>> a = np.arange(10)
    >>> select(a, amount=4)
    [(array([2, 8, 4, 9]), array([0, 1, 3, 5, 6, 7]))]

    >>> p = [0.1,0.1,0.5,0,0,0,0,0.1,0.1,0.1]
    >>> select(a, amount=4, p=p)
    [(array([2, 8, 9, 7]), array([0, 1, 3, 4, 5, 6]))]

    >>> b = 1 - np.arange(10)/10
    >>> select(a, b, amount=4, p=p)
    [(array([2, 9, 0, 1]), array([3, 4, 5, 6, 7, 8])), (array([0.8, 0.1, 1. , 0.9]), array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2]))]
    '''
    l = len(arrays[0])
    for a in arrays[1:]:
        if l != len(a):
            raise ValueError(f'Arrays with different lengths: {[len(a) for a in arrays]}')
    if isinstance(amount, float):
        if amount < 0 or amount > 1:
            raise ValueError('Amount must be either int or float between 0 and 1')
        amount = int(l*amount)

    if p is not None:
        s = np.sum(p)
        if np.abs(s - 1) > 1e-7:
            p = np.array(p, dtype=float)/s

    indexes = np.arange(l)
    selected_indexes = np.random.choice(indexes, size=amount, replace=False, p=p)
    remaining_indexes = np.delete(indexes, selected_indexes)

    output = []
    for a in arrays:
        output.append((a[selected_indexes], a[remaining_indexes]))

    return output


def compute_p_func(q, Y):
    # q0 = q[Y==0]
    # q1 = q[Y==1]
    # GM: It is not quite clear why it needs to be called with q and Y, especially since even q is not used below
    # AL: this is for future use where I compute the histogram of q and choose how to deal with it

    epsilon = 1e-7 # GM: I guess you are assuming float32 precision. Maybe 1e-15 could still work?

    @np.vectorize
    def p0_func(qs):
        return -np.log(np.maximum(epsilon, 1 - qs))

    @np.vectorize
    def p1_func(qs):
        return -np.log(np.maximum(epsilon, qs))

    return p0_func, p1_func


# we redefine optimal_checkpoint fonction to account for eons
def optimal_checkpoint(run_folder, nfolds, metric='val_CustomLoss', direction='minimize', first_epoch=1, collective=True, fold_subfolder='eon_last'):
    '''
    Computes the epoch that had the best score

    Parameters
    ----------
    folder : str
        folder where the model is located that contains sub folders with the n folds named 'fold_%i'
    nfolds : int
        number of folds,
    metric : str, optional
        metric with respect to which optimize, by default 'val_CustomLoss'
    direction : str, optional
        'maximize' or 'minimize', by default 'minimize'
    first_epoch : int, optional
        The number of the first epoch, by default 1
    collective : bool, optional
        Whether the optimal checkpoint should be the same for all folds (True) or the best for each fold
    fold_subfolder : str or 'eon_last', optional
        Name of the subfolder inside the fold folder in which to look for history and model checkpoints,
        By default 'eon_last', which means the subfolder will be the one of the last eon

    Returns
    -------
    opt_checkpoint
        if collective : int
            epoch number corresponding to the best checkpoint
        else : list
            of best epoch number for each fold
    fold_subfolder: str (collective = True) or list of str (collective = False)
        the fold subfolder where history and checkpoints are located

    Raises
    ------
    KeyError
        If `metric` is not present in the history
    ValueError
        If `direction` not in ['maximize', 'minimize']
    '''
    run_folder = run_folder.rstrip('/')
    
    if fold_subfolder == 'eon_last':
        fold_subfolder = []
        for i in range(nfolds):
            fold_folder = f'{run_folder}/fold_{i}'
            eon_dirs = [name for name in os.listdir(fold_folder) if (os.path.isdir(os.path.join(fold_folder, name)) and name.startswith('eon'))]
            if not len(eon_dirs):
                raise ValueError('No eon folder found')
            eons = [int(name.split('_',1)[-1]) for name in eon_dirs]
            fold_subfolder.append(f'eon_{max(eons)}/')
    else:
        fold_subfolder = (fold_subfolder.rstrip('/') + '/') if fold_subfolder else ''

    if isinstance(fold_subfolder, str):
        fold_subfolder = [fold_subfolder]*nfolds

    # Here we insert analysis of the previous training with the assessment of the ideal checkpoint
    history0 = np.load(f'{run_folder}/fold_0/{fold_subfolder[0]}history.npy', allow_pickle=True).item()
    if metric not in history0.keys():
        raise KeyError(f'{metric} not in history: cannot compute optimal checkpoint')
    historyCustom = [np.load(f'{run_folder}/fold_{i}/{fold_subfolder[i]}history.npy', allow_pickle=True).item()[metric] for i in range(nfolds)]

    if direction == 'minimize':
        opt_f = np.argmin
    elif direction == 'maximize':
        opt_f = np.argmax
    else:
        raise ValueError(f'Unrecognized {direction = }')

    if collective: # the optimal checkpoint is the same for all folds and it is based on the average performance over the folds
        if len(set(fold_subfolder)) > 1: # the set has more than one element if not all elements are the same
            logger.error('Cannot compute a collective checkpoint as folds have a different number of eons. Computing independent checkpoints instead')
            collective = False
        # check that the nfolds histories have the same length
        elif len(set([len(historyCustom[i] for i in range(nfolds))])) > 1:
            logger.error('Cannot compute a collective checkpoint from folds trained a different number of epochs. Computing independent checkpoints instead')
            collective = False
    
    if collective:
        fold_subfolder = fold_subfolder[0] # fold_subfolder is a list of elements that are all the same, so we take just the first
        historyCustom = np.mean(np.array(historyCustom),axis=0)
        opt_checkpoint = opt_f(historyCustom)
    else:
        opt_checkpoint = np.array([opt_f(h) for h in historyCustom]) # each fold independently
    
    opt_checkpoint += first_epoch

    if collective:
        opt_checkpoint = int(opt_checkpoint)
    else:
        opt_checkpoint = [int(oc) for oc in opt_checkpoint]

    return opt_checkpoint, fold_subfolder


# we redefine the train model function
@ut.execution_time
@ut.indent_logger(logger)
def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics, early_stopping_kwargs=None, # We always use early stopping
                batch_size=1024, checkpoint_every=1, additional_callbacks=['csv_logger'], return_metric='val_CustomLoss',
                num_eons=10, data_amount_per_eon=0.1):
    '''
    Trains a given model checkpointing its weights

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
    num_epochs : int
        number of maximum epochs for the training
    optimizer : keras.Optimizer object
    loss : keras.losses.Loss object
    metrics : list of keras.metrics.Metric or str
    early_stopping_kwargs : dict
        arguments to create the early stopping callback. Ignored if `enable_early_stopping` = False
    batch_size : int, optional
        by default 1024
    checkpoint_every : int or str, optional
        Examples:
        0: disabled
        5 or '5 epochs' or '5 e': every 5 epochs
        '100 batches' or '100 b': every 100 batches
        'best custom_loss': every time 'custom_loss' reaches a new minimum. 'custom_loss' must be in the list of metrics
    additional_callbacks : list of keras.callbacks.Callback objects or list of str, optional
        string items are interpreted, for example 'csv_logger' creates a CSVLogger callback
    return_metric : str, optional
        name of the metric of which the minimum value will be returned at the end of training

    Returns
    -------
    float
        minimum value of `return_metric` during training
    '''
    ### preliminary operations
    ##########################
    if early_stopping_kwargs is None:
        early_stopping_kwargs = {}
    folder = folder.rstrip('/')

    ## deal with callbacks
    callbacks = {}

    # additional callbacks
    if additional_callbacks is not None:
        for cb in additional_callbacks:
            if isinstance(cb, str):
                if cb.lower().startswith('csv'):
                    callbacks['csv_logger'] = keras.callbacks.CSVLogger(f'{folder}/history.csv', append=True)
                else:
                    raise ValueError(f'Unable to understand callback {cb}')
            else:
                callbacks[str(cb)] = cb

    # early stopping callback
    if 'patience' not in early_stopping_kwargs or early_stopping_kwargs['patience'] == 0:
        logger.warning('Skipping early stopping with patience = 0')
    else:
        callbacks['early_stopping'] = early_stopping(**early_stopping_kwargs)

    ### training the model
    ######################
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    np.save(f'{folder}/Y_va.npy', Y_va) # save validation labels
    
    # The data is split into positive and negative labels so that the same percentage enters
    i_tr = np.arange(Y_tr.shape[0]) # the subset of the data that will be used

    X0_remaining = X_tr[Y_tr == 0]
    Y0_remaining = Y_tr[Y_tr == 0]
    i0_remaining = i_tr[Y_tr == 0]
    X1_remaining = X_tr[Y_tr == 1]
    Y1_remaining = Y_tr[Y_tr == 1]
    i1_remaining = i_tr[Y_tr == 1]

    p0 = None
    p1 = None
    X_tr = X_tr[0:0] # this way we get the shape we need: (0, *X_tr.shape[1:]) that is the one we need for the first concatenation
    Y_tr = Y_tr[0:0]
    i_tr = i_tr[0:0]

    for eon in range(num_eons):
        logger.info(f'{eon = } ({eon+1}/{num_eons})')
        eon_folder = f'{folder}/eon_{eon}'
        ckpt_name = eon_folder + '/cp-{epoch:04d}.ckpt'

        model.save_weights(ckpt_name.format(epoch=0)) # save model before training

        # checkpointing callback
        ckpt_callback = ln.make_checkpoint_callback(ckpt_name, checkpoint_every=checkpoint_every)
        if ckpt_callback is not None:
            callbacks['model_checkpoint'] = ckpt_callback


        # augment training data
        (X0_selected, X0_remaining), (Y0_selected, Y0_remaining), (i0_selected, i0_remaining) = select(X0_remaining, Y0_remaining, i0_remaining, amount=data_amount_per_eon, p=p0)
        (X1_selected, X1_remaining), (Y1_selected, Y1_remaining), (i1_selected, i1_remaining) = select(X1_remaining, Y1_remaining, i1_remaining, amount=data_amount_per_eon, p=p1)

        X_tr = np.concatenate([X_tr, X0_selected, X1_selected], axis=0)
        Y_tr = np.concatenate([Y_tr, Y0_selected, Y1_selected], axis=0)
        i_tr =  np.concatenate([i_tr, i0_selected, i1_selected], axis=0)

        shuffle_permutation = np.random.permutation(Y_tr.shape[0]) # shuffle data
        X_tr = X_tr[shuffle_permutation]
        Y_tr = Y_tr[shuffle_permutation]
        i_tr = i_tr[shuffle_permutation]

        # save i_tr
        np.save(f'{eon_folder}/i_tr.npy', i_tr)

        # log the amount af data that is entering the network
        logger.info(f'Training the network on {len(Y_tr)} datapoint and validating on {len(Y_va)}')

        # perform training for `num_epochs`
        my_history=model.fit(X_tr, Y_tr, batch_size=batch_size, validation_data=(X_va,Y_va), shuffle=True,
                            callbacks=list(callbacks.values()), epochs=num_epochs, verbose=2, class_weight=None)


        ## deal with history
        history = my_history.history
        model.save(eon_folder)
        np.save(f'{eon_folder}/history.npy', history)
        # log history
        df = pd.DataFrame(history)
        df.index.name = 'epoch-1'
        logger.log(25, str(df))

        # compute best score of the eon

        # thanks to early stopping the model is reverted back to the best checkpoint
        
        # compute q on the training dataset (using batches so I am sure the data fits in memory)
        q_tr = []
        for b in range(Y_tr.shape[0]//batch_size + 1):
            q_tr.append(keras.layers.Softmax()(model(X_tr[b*batch_size:(b+1)*batch_size])).numpy()[:,1])
        q_tr = np.concatenate(q_tr)

        # compute Y_pred on the validation set
        q_va = []
        for b in range(Y_va.shape[0]//batch_size + 1):
            q_va.append(keras.layers.Softmax()(model(X_va[b*batch_size:(b+1)*batch_size])).numpy()[:,1])
        q_va = np.concatenate(q_va)

        # save predictions
        np.save(f'{eon_folder}/q_tr.npy', q_tr)
        np.save(f'{eon_folder}/Y_tr.npy', Y_tr)
        np.save(f'{eon_folder}/q_va.npy', q_va)

        # compute probability distribution for the next eon that would decide which data to add
        p0_func, p1_func = compute_p_func(q_tr, Y_tr)

        # compute q on the remaining dataset
        q0_remaining = []
        for b in range(Y0_remaining.shape[0]//batch_size + 1):
            q0_remaining.append(keras.layers.Softmax()(model(X0_remaining[b*batch_size:(b+1)*batch_size])).numpy()[:,1])
        q0_remaining = np.concatenate(q0_remaining)

        q1_remaining = []
        for b in range(Y1_remaining.shape[0]//batch_size + 1):
            q1_remaining.append(keras.layers.Softmax()(model(X1_remaining[b*batch_size:(b+1)*batch_size])).numpy()[:,1])
        q1_remaining = np.concatenate(q1_remaining)

        # np.save(f'{eon_folder}/q0_remaining.npy', q0_remaining)
        # np.save(f'{eon_folder}/q1_remaining.npy', q1_remaining)

        # attribute the probabilities based on the value of the predicted committor
        p0 = p0_func(q0_remaining)
        p1 = p1_func(q1_remaining)

    # return the best value of the return metric
    if return_metric not in history:
        logger.error(f'{return_metric = } is not one of the metrics monitored during training, returning NaN')
        return np.NaN
    return np.min(history[return_metric])

#####################################################
# set the modified function to override the old one #
#####################################################
ln.train_model = train_model
ln.optimal_checkpoint = optimal_checkpoint
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
