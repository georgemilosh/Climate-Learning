# '''
# Created in February 2022

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

def q(mu, sig, thr):
    return 0.5*tf.math.erfc((thr - mu)/sig/np.sqrt(2))

def entropy(p, q, epsilon):
    q = tf.clip_by_value(q, epsilon, 1 - epsilon)
    return -p*tf.math.log(q) - (1-p)*tf.math.log(1 - q)

def phi(x):
    '''PDF of a standard Gaussian'''
    return 1/np.sqrt(2*np.pi)*tf.math.exp(-0.5*tf.math.square(x))

def Phi(x):
    '''CDF of a standard Gaussian'''
    return 0.5*(1 + tf.math.erf(x/np.sqrt(2)))

### custom losses/metrics
class PreTrainingLoss(keras.losses.Loss):
    def __init__(self, name=None):
        super().__init__(name=name or self.__class__.__name__)

    def call(self, y_true, y_pred):
        sig = y_pred[...,1:2]
        y_pred = y_pred[...,0:1]
        assert y_pred.shape == sig.shape == y_true.shape
        return tf.math.square(y_true - y_pred) + tf.math.square(sig - 1)

class CRPS(keras.losses.Loss):
    '''Continuous Ranked Probability Score'''
    def __init__(self,name=None, epsilon=None) -> None:
        super().__init__(name=name or self.__class__.__name__)
        self.epsilon = epsilon or keras.backend.epsilon()

    def call(self, y_true, y_pred):
        sig = tf.math.abs(y_pred[...,1:2]) + self.epsilon
        y_pred = y_pred[...,0:1]
        res = (y_true - y_pred)/sig
        assert y_pred.shape == sig.shape == y_true.shape == res.shape

        return sig*(res*tf.math.erf(res/np.sqrt(2)) + 2*phi(res) -1/np.sqrt(np.pi))

class ProbRegLoss(keras.losses.Loss):
    def __init__(self, name=None, epsilon=None, maxsig=None):
        super().__init__(name=name or self.__class__.__name__)
        self.epsilon = epsilon or keras.backend.epsilon()
        self.maxsig = maxsig

    def call(self, y_true, y_pred):
        sig2 = tf.math.square(y_pred[...,1:2]) + self.epsilon
        penalty = 0
        if self.maxsig:
            sig2 = tf.clip_by_value(sig2, 0, self.maxsig**2)
            penalty = tf.math.square(y_pred[...,1:2]) - sig2
        # mu = y_pred[...,0:1]
        y_pred = y_pred[...,0:1] # for memory efficiency
        assert y_pred.shape == sig2.shape == y_true.shape
        return tf.math.square(y_true - y_pred)/sig2 + tf.math.log(sig2) + penalty

class SeparateMRSOLinearModel(keras.Model):
    def __init__(self, m=1):
        self.m = m
        self.kernel = keras.layers.Dense(m, activation=None)
        self.conc = keras.layers.Concatenate()

    def __call__(self, x):
        x, mrso = x
        x = self.kernel(x)
        x = self.conc([x,mrso])
        return x
    
class ParametricCrossEntropyLoss(keras.losses.Loss):
    def __init__(self, threshold=0, epsilon=None):
        super().__init__(name=self.__class__.__name__)
        self.threshold = threshold
        self.epsilon = epsilon or keras.backend.epsilon()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true >= self.threshold, tf.float32)
        sig = tf.math.abs(y_pred[...,1:2])
        y_pred = y_pred[...,0:1]
        y_pred = q(y_pred,sig,self.threshold)
        assert y_pred.shape == y_true.shape, f'{y_pred.shape = }, {y_true.shape = }'
        return entropy(y_true, y_pred, self.epsilon)

def weighted(cls, function):
    class WeightedLoss(cls):
        def __init__(self, name=f'Weighted{cls.__name__}'):
            super().__init__(name=name or self.__class__.__name__)
            self.function = function

        def call(self, y_true, y_pred):
            weights = self.function(y_true)
            loss = super().call(y_true, y_pred)
            assert weights.shape == loss.shape
            return weights*loss
        
    return WeightedLoss

old_get_loss_function = ln.get_loss_function
def get_loss_function(loss_name: str, u=1):
    loss_name = loss_name.lower()
    args = []
    kwargs = {}
    func = None
    wk = 'weighted'
    if loss_name.startswith(wk):
        func = lambda x: tf.math.sigmoid(x - 2)
        loss_name = loss_name[len(wk):].strip('_')

    if loss_name.startswith('prob'):
        cls = ProbRegLoss
    elif loss_name.startswith('pretr'):
        cls = PreTrainingLoss
    elif loss_name.startswith('crps'):
        cls = CRPS
    # elif loss_name.startswith('weighted'):
    #     return WeightedProbRegLoss(a=2, b=1)
    else:
        return old_get_loss_function(loss_name, u=u)
    
    if func:
        return weighted(cls, func)(*args, **kwargs)
    return cls(*args, **kwargs)
    
def get_default_metrics(fullmetrics=False, u=1):
    if fullmetrics:
        metrics = [
            ParametricCrossEntropyLoss(ln._current_threshold),
        ]
    else:
        metrics=None
    return metrics

def postprocess(x):
    return x


## interpretable neural network architecture

class Dense2D(layers.Layer):
    def __init__(self, filters_per_field=[1,1,1], merge_to_one=True, regularizer=None, **kwargs):
        '''
        Layer for performing a linear projection of a color image treating the colors (fields) independently

        Parameters
        ----------
        filters_per_field : list[int], optional
            Number of patterns onto which to project for every color (field), by default [1,2,1]
        merge_to_one : bool, optional
            Whether to sum the outputs of the scalar products between filters and fields into a single neuron. This makes the reduced space one dimensional.
            With this setting there is no point in having more than 1 filter per field. Default is False
        regularizer : tf.keras.regularizers.Regularizer, optional
            Regularizer, by default None
        '''
        super().__init__(**kwargs)
        self.filters_per_field = filters_per_field
        self.nfields = len(self.filters_per_field)
        self.nfilters = np.sum(self.filters_per_field)
        self.merge_to_one = merge_to_one
        if self.nfilters == 0:
            raise ValueError(f'Layer with no filters is invalid: {filters_per_field = }')
        if self.merge_to_one:
            self.sum = keras.layers.Add()
        else:
            self.conc = keras.layers.Concatenate()

        self.m = 1 if self.merge_to_one else self.nfilters

        self.regularizer = regularizer

    def build(self, input_shape):
        if input_shape[-1] != self.nfields:
            raise ValueError(f'Expected {self.nfields} fields, received {input_shape[-1]}')
        kernel_shape = input_shape[-3:-1]

        self.kernels = []
        for i, fpf in enumerate(self.filters_per_field):
            if fpf:
                self.kernels.append(self.add_weight(
                    name=f"w_{i}",
                    shape=(*kernel_shape, fpf),
                    initializer="random_normal",
                    trainable=True,
                    regularizer=self.regularizer
                ))
            else:
                self.kernels.append(None) # no filters for this field
    
    def call(self, x):
        if x.shape[-1] != self.nfields:
            raise ValueError(f'Expected {self.nfields} fields, received {x.shape[-1]}')

        x = [tf.tensordot(x[...,i], k, axes=2) for i,k in enumerate(self.kernels) if k is not None]

        if self.nfilters == 1:
            x = x[0]
        elif self.merge_to_one:
            x = self.sum(x)
        else:
            x = self.conc(x)            

        return x


class GradientRegularizer(keras.regularizers.Regularizer):
    def __init__(self, mode='l2', c=1, weights=None, periodic_lon=True, normalize=True, lat=None):
        '''
        Makes a filter smooth by penalizing the difference between adjacent pixels

        Parameters
        ----------
        mode : 'l1' or 'l2, optional
            regularization mode, by default 'l2'
        c : float, optional
            regularization coefficient, by default 1
        weights : np.ndarray or str, optional
            weights to apply to the different pixels of the filter, special options are:
                - None : uniform weighting
                - 'sphere' : assumes a spherical topology (needs a latitude vector: `lat`)
                - 'auto' or 'compromise' : deprecated: it is a wrong version of the sphere mode
            By default None
        periodic_lon : bool, optional
            whether to consider periodicity on the longitude axis, by default True
        normalize : bool, optional
            whether to normalize the gradient so it is not sensitive to rescaling of the whole filter, by default True
        '''
        if mode in ['L1', 'l1', 'sparse']:
            self.mode = 'l1'
        else:
            if mode not in ['L2', 'l2', 'ridge']:
                logger.warning(f"Unrecognized regularization {mode = }: using 'l2'")
            self.mode = 'l2'
        self.c = c
        self.weights = weights
        self.periodic_lon = periodic_lon
        self.normalize = normalize
        self.lat = lat

        if self.weights is not None:
            if isinstance(self.weights, str):
                if self.weights == 'sphere':
                    if self.lat is None:
                        raise ValueError(f'{self.weights} regularization mode requires latitude vector')
                    self.coslat = np.cos(self.lat*np.pi/180)
                    self.broadcasted_coslat = None
                elif self.weights in ['auto', 'compromise']: # for backward compatibility
                    logger.warning(f"Deprecation warning: regularization weight in mode {self.weights} is deprecated. Please use None or 'sphere'")
                    apply_sqrt = self.weights == 'compromise'
                    if self.lat is None:
                        raise ValueError(f'{self.weights} regularization mode requires latitude vector')
                    self.weights = np.ones((22,128,2), dtype=np.float32)
                    # gradient in the lat (x) direction is uniform so we don't do anything
                    # gradient in the lon direction depends on latitude
                    self.weights[...,1] = (self.weights[...,1].T/np.cos(self.lat*np.pi/180)).T # these double transposition helps using numpy native operators
                    if apply_sqrt:
                        self.weights = np.sqrt(self.weights)
                else:
                    raise ValueError(f'Unrecognized string option for weights: {self.weights}')

            if not isinstance(self.weights, str): # the weights are numerical, so we normalize them
                if self.mode == 'l1':
                    self.weights = self.weights/tf.math.reduce_mean(tf.math.abs(self.weights)) # now the mean of the weights is 1
                else:
                    self.weights = self.weights/tf.math.sqrt(tf.math.reduce_mean(tf.math.square(self.weights)))

    def __call__(self, x):
        if self.c == 0:
            return 0
        nfilters = x.shape[-1]
        if self.weights is not None:
            if isinstance(self.weights, str):
                if self.broadcasted_coslat is None:
                    self.broadcasted_coslat = (np.ones(x.shape[:2]).T*self.coslat).T # these double transposition helps using numpy native operators
            elif self.weights.shape[:-1] != x.shape[:-1]:
                raise ValueError(f'weight shape {self.weights.shape} does not match received input shape {x.shape[:-1]}')
        
        if self.mode == 'l1':
            op = tf.math.abs
        else:
            op = tf.math.square

        s = 0
        for i in range(nfilters):
            if self.weights is not None:
                if isinstance(self.weights, str):
                    if self.weights == 'sphere':
                        # add gradient along x (lat)
                        _s = tf.math.reduce_sum(self.broadcasted_coslat[:-1,:]*op(x[1:,:,i] - x[:-1,:,i]))
                        # add gradient along y (lon)
                        _s = _s + tf.math.reduce_sum(self.broadcasted_coslat[:,:-1]*op((x[:,1:,i] - x[:,:-1,i])/self.broadcasted_coslat[:,:-1]))
                        # add periodic point
                        if self.periodic_lon:
                            _s = _s + tf.math.reduce_sum(self.broadcasted_coslat[:,-1]*op((x[:,0,i] - x[:,-1,i])/self.broadcasted_coslat[:,-1]))
                    else:
                        raise ValueError(f'Unrecognized string option for weights: {self.weights}')
                else:
                    # add gradient along x (lat)
                    _s = tf.math.reduce_sum(op((x[1:,:,i] - x[:-1,:,i])*self.weights[:-1,:,0]))
                    # add gradient along y (lon)
                    _s = _s + tf.math.reduce_sum(op((x[:,1:,i] - x[:,:-1,i])*self.weights[:,:-1,1]))
                    # add periodic point
                    if self.periodic_lon:
                        _s = _s + tf.math.reduce_sum(op((x[:,0,i] - x[:,-1,i])*self.weights[:,-1,1]))
            else:
                # add gradient along x (lat)
                _s = tf.math.reduce_sum(op(x[1:,:,i] - x[:-1,:,i]))
                # add gradient along y (lon)
                _s = _s + tf.math.reduce_sum(op(x[:,1:,i] - x[:,:-1,i]))
                # add periodic point
                if self.periodic_lon:
                    _s = _s + tf.math.reduce_sum(op(x[:,0,i] - x[:,-1,i]))

            if self.normalize:
                if isinstance(self.weights, str):
                    if self.weights == 'sphere':
                        _s = _s/tf.math.reduce_sum(self.broadcasted_coslat * op(x[...,i]))
                    else:
                        raise ValueError(f'Unrecognized string option for weights: {self.weights}')
                else:
                    _s = _s/tf.math.reduce_sum(op(x[...,i]))

            s = s + _s

        return self.c*s

    def get_config(self):
        return {'c': self.c, 'weights': self.weights, 'periodic_lon': self.periodic_lon, 'normalize': self.normalize}


def create_model(input_shape, filters_per_field=[1,1,1], merge_to_one=False, batch_normalization=False, reg_mode='l2', reg_c=1, reg_weights=None, reg_periodicity=True, reg_norm=True, dense_units=[8,2], dense_activations=['relu', None], dense_dropouts=False, dense_l2coef=None):
    '''
    Creates a neural network

    Parameters
    ----------
    input_shape : tuple
        shape of the data (without the batch axis)
    filters_per_field : list[int], optional
        Number of projection patterns for each of the fields ('ghost' fields should not be counted), by default [1,1,1]
    merge_to_one : bool, optional
        Whether to sum the outputs of the scalar products between filters and fields into a single neuron. This makes the reduced space one dimensional.
        With this setting there is no point in having more than 1 filter per field. Default is False
    batch_normalization : bool, optional
        whether to perform batch normalization after the projection. This helps if the input data is not normalized, by default False
    reg_mode : str, optional
        how to regularize the graident, either 'l1' or 'l2', by default 'l2'
    reg_c : float, optional
        coefficient for the gradient regularization penalty that is added to the loss, by default 1
    reg_weights : str, optional
        How to compute the gradient: either None (assuming euclidean distance between the gridpoints) or 'sphere' which accounts for the fact that the Earth is spherical, by default None
    reg_periodicity : bool, optional
        Whether to regularize the gradient over the Bering straight, by default True
    reg_norm : bool, optional
        Whether to normalize the gradient to the norm of the projection pattern. This avoids the tendency to simply push all the values in the pattern to zero, by default True
    dense_units : list[int], optional
        Number of neurons for each hidden layer of the classification network. The last layer must have 2 neurons, by default [8,2]
    dense_activations : str or list[str], optional
        Activation functions at the end of each layer. If not a string, must have the same length as `dense_units` and the last layer must have None activation. By default ['relu', None]
        If string, it will broadcasted to all layers except the last which will have activation=None
    dense_dropouts : list[float], optional
        Dropout rates for each layer. If False or None it is disabled, by default False

    Returns
    -------
    tf.keras.Model
        Neural network
    '''
    regularizer = None
    if reg_c:
        regularizer = GradientRegularizer(mode=reg_mode, c=reg_c, weights=reg_weights, periodic_lon=reg_periodicity, normalize=reg_norm, lat=ln.lat)
    
    model = keras.models.Sequential()

    model.add(Dense2D(filters_per_field=filters_per_field, merge_to_one=merge_to_one, regularizer=regularizer, input_shape=input_shape))

    if batch_normalization:
        model.add(keras.layers.BatchNormalization())

    # dense layers
    # adjust the shape of the arguments to be of the same length as `dense_units`
    args = [dense_activations, dense_dropouts, dense_l2coef]
    for j,arg in enumerate(args):
        if not isinstance(arg, list):
            args[j] = [arg]*len(dense_units)
            if j==0:
                args[j][-1] = None # the last layer cannot have activation
            elif j==1:
                args[j][-1] = False # the last layer cannot have dropout
        elif len(arg) != len(dense_units):
            raise ValueError(f'Invalid length for argument {arg}')
    logger.info(f'dense args = {args}')
    dense_activations, dense_dropouts, dense_l2coef = args
    # build the dense layers
    for i in range(len(dense_units)):
        model.add(layers.Dense(dense_units[i], activation=dense_activations[i], kernel_regularizer=keras.regularizers.l2(dense_l2coef[i]) if dense_l2coef[i] else None))
        if dense_dropouts[i]:
            model.add(layers.Dropout(dense_dropouts[i]))

    return model


orig_train_model = ln.train_model

def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics, early_stopping_kwargs=None, enable_early_stopping=False,
                u=1, batch_size=1024, checkpoint_every=1, additional_callbacks=['csv_logger'], return_metric='val_CustomLoss', load_kernels_from=None, learn_kernels=True):
    '''
    Extra arguments:

    load_kernels_from : None|str|list
        How to initialize the kernels
    learn_kernels : bool
        Whether to train the kernels or leave them as they are at the initialization. By default True
    '''
    if load_kernels_from is not None:
        if isinstance(load_kernels_from, str):
            if load_kernels_from.startswith('composite'):
                comp = np.mean(X_tr[Y_tr==1], axis=0)
                np.save(f'{folder}/composite.npy', comp)
            elif load_kernels_from.startswith('significance'):
                comp = np.mean(X_tr[Y_tr==1], axis=0)
                np.save(f'{folder}/composite.npy', comp)
                sig = np.std(X_tr[Y_tr==1], axis=0)
                sig[sig==0] = 1
                comp = comp/sig
                np.save(f'{folder}/significance.npy', comp)
            else:
                raise NotImplementedError(f'Unknown option {load_kernels_from}')

            load_kernels_from = []
            for i,fpf in enumerate(model.layers[0].filters_per_field):
                if fpf is None:
                    continue
                elif fpf == 1:
                    load_kernels_from.append(comp[...,i:i+1])
                else:
                    raise ValueError(f'It is dumb to set the composite as kernel {fpf} times')


        if not isinstance(load_kernels_from, list):
            raise TypeError(f'at this point load_kernels_from should be of type list, not {type(load_kernels_from)}')

        model.layers[0].set_weights(load_kernels_from)

    if not learn_kernels: # we can compute the result of the first layer on the data at once at the beginning. Also since we won't compute gradients through the projection layer, it is not trained.
        logger.info('Projection is not trainable: computing it at the beginning')

        # split the model
        proj = keras.models.Sequential(model.layers[:1])
        proj.save(f'{folder}/projection') # save the projection
        model = keras.models.Sequential(model.layers[1:]) # override model
        model.build(input_shape=proj.output_shape)

        # compute the output of the first layer
        _X_va = []
        for b in range(Y_va.shape[0]//batch_size + 1):
            _X_va.append(proj(X_va[b*batch_size:(b+1)*batch_size]).numpy())
        X_va = np.concatenate(_X_va) # override validation set

        _X_tr = []
        for b in range(Y_tr.shape[0]//batch_size + 1):
            _X_tr.append(proj(X_tr[b*batch_size:(b+1)*batch_size]).numpy())
        X_tr = np.concatenate(_X_tr) # override training set

        logger.info('New data shapes:')
        logger.info(f'{X_tr.shape = }, {X_va.shape = }, {Y_tr.shape = }, {Y_va.shape = }')
    
    return orig_train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics, early_stopping_kwargs=early_stopping_kwargs, enable_early_stopping=enable_early_stopping,
                            u=u, batch_size=batch_size, checkpoint_every=checkpoint_every, additional_callbacks=additional_callbacks, return_metric=return_metric)


def load_model(checkpoint, compile=False):
    '''
    Loads a neural network and its weights. Checkpoints with the weights are supposed to be in the same folder as where the model structure is

    Parameters
    ----------
    checkpoint : str
        path to the checkpoint is. For example with structure <folder>/cp-<epoch>.ckpt
    compile : bool, optional
        whether to compile the model, by default False

    Returns
    -------
    keras.models.Model
    '''
    model_folder = Path(checkpoint).parent
    model = keras.models.load_model(model_folder, compile=compile)
    model.load_weights(checkpoint)

    proj_folder = model_folder / 'projection'
    if proj_folder.exists():
        logger.info('Detected separate projection: loading and concatenating')
        proj = keras.models.load_model(proj_folder, compile=compile)

        model = keras.models.Sequential([proj, model])
    return model


## Modifications to the trainer object

ln._current_threshold = None
class Trainer(ln.Trainer):
    def prepare_XY(self, fields, **prepare_XY_kwargs):
        if self._prepare_XY_kwargs != prepare_XY_kwargs:
            self._prepare_XY_kwargs = prepare_XY_kwargs
            self.X, self.Y, self.year_permutation, self.lat, self.lon, threshold = ln.prepare_XY(fields, **prepare_XY_kwargs)

            logger.info('Saving latitude as module level variable')
            ln.lat = self.lat

            # retrieve A
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

            # overwrite self.Y with A
            self.Y = A
            ln._current_threshold = threshold # save the threshold in a module level variable
        return self.X, self.Y, self.year_permutation, self.lat, self.lon

#######################################################
# set the modified functions to override the old ones #
#######################################################
ln.get_default_metrics = get_default_metrics
ln.get_loss_function = get_loss_function
ln.postprocess = postprocess

ln.create_model = create_model
ln.train_model = train_model
ln.load_model = load_model
ln.Trainer = Trainer
ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary

ut.set_values_recursive(ln.CONFIG_DICT,
                        {
                            'return_threshold': True,
                            'loss': 'prob_reg_loss',
                            'return_metric': 'val_ParametricCrossEntropyLoss',
                            'monitor' : 'val_ParametricCrossEntropyLoss',
                            'metric' : 'val_ParametricCrossEntropyLoss',
                        },
                        inplace=True) 

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
