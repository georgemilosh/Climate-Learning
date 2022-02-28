import Learn2_new as ln
logger = ln.logger
early_stopping = ln.early_stopping
ut = ln.ut
np = ln.np
keras = ln.keras
pd = ln.pd

def select(*arrays, amount=0.1, p=None):
    '''
    selects a given amount of data from a set of arrays according to a probability distribution

    Parameters
    ----------
    *arrays : M arrays with first dimension of size N
        The arrays from which to select from. They are assumed synchronized, i.e. if, for example, the first element of an array is selected, it is selected in every array
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



# we redefine the train model function
@ut.execution_time
@ut.indent_logger(logger)
def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics, early_stopping_kwargs=None, # We always use early stopping
                batch_size=1024, checkpoint_every=1, additional_callbacks=['csv_logger'], return_metric='val_CustomLoss', num_eons=10, data_percent=10):
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
    ckpt_name = folder + '/cp-{epoch:04d}.ckpt'

    ## deal with callbacks
    callbacks = []

    # additional callbacks
    if additional_callbacks is not None:
        for cb in additional_callbacks:
            if isinstance(cb, str):
                if cb.lower().startswith('csv'):
                    callbacks.append(keras.callbacks.CSVLogger(f'{folder}/history.csv', append=True))
                else:
                    raise ValueError(f'Unable to understand callback {cb}')
            else:
                callbacks.append(cb)

    # checkpointing callback
    ckpt_callback = None
    if checkpoint_every == 0: # no checkpointing
        pass
    elif checkpoint_every == 1: # save every epoch
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1)
    elif isinstance(checkpoint_every, int): # save every `checkpoint_every` epochs 
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, period=checkpoint_every)
    elif isinstance(checkpoint_every, str): # parse string options
        if checkpoint_every[0].isnumeric():
            every, what = checkpoint_every.split(' ',1)
            every = int(every)
            if what.startswith('b'): # every batch
                ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, save_freq=every)
            elif what.startswith('e'): # every epoch
                ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, period=every)
            else:
                raise ValueError(f'Unrecognized value for {checkpoint_every = }')

        elif checkpoint_every.startswith('best'): # every best of something
            monitor = checkpoint_every.split(' ',1)[1]
            ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=1)
        else:
            raise ValueError(f'Unrecognized value for {checkpoint_every = }')
    else:
        raise ValueError(f'Unrecognized value for {checkpoint_every = }')

    if ckpt_callback is not None:
        callbacks.append(ckpt_callback)

    # early stopping callback
    if 'patience' not in early_stopping_kwargs or early_stopping_kwargs['patience'] == 0:
        logger.warning('Skipping early stopping with patience = 0')
    else:
        callbacks.append(early_stopping(**early_stopping_kwargs))

    ### training the model
    ######################
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.save_weights(ckpt_name.format(epoch=0)) # save model before training


    ############################################
    # Up to here is the same as ln.train_model #
    ############################################

    # perform training for `num_epochs`
    my_history=model.fit(X_tr, Y_tr, batch_size=batch_size, validation_data=(X_va,Y_va), shuffle=True,
                         callbacks=callbacks, epochs=num_epochs, verbose=2, class_weight=None)


    ## deal with history
    history = my_history.history
    model.save(folder)
    np.save(f'{folder}/history.npy', history)
    # log history
    df = pd.DataFrame(history)
    df.index.name = 'epoch-1'
    logger.log(25, str(df))

    # return the best value of the return metric
    if return_metric not in history:
        logger.error(f'{return_metric = } is not one of the metrics monitored during training, returning NaN')
        return np.NaN
    return np.min(history[return_metric])

#####################################################
# set the modified function to override the old one #
#####################################################
ln.train_model = train_model
ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary