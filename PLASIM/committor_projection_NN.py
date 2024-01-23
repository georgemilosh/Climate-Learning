# '''
# Created in February 2022

# @author: Alessandro Lovo
# '''
description = """Intrinsically interpretable neural network architecture. Can be used to perform committor projection."""
dependencies = None

import Learn2_new as ln
logger = ln.logger
# early_stopping = ln.early_stopping # this seems useless
ut = ln.ut
np = ln.np
tf = ln.tf
keras = ln.keras
layers = keras.layers
pd = ln.pd

from functools import wraps

# log to stdout
import logging
import sys
from pathlib import Path

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

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

orig_prepare_XY = ln.prepare_XY
@wraps(orig_prepare_XY)
def prepare_XY(fields, **kwargs):
    res = orig_prepare_XY(fields, **kwargs)
    # res = X, Y, year_permutation, lat, lon, [threshold]
    logger.info('Saving latitude as module level variable')
    ln.lat = res[3]
    logger.info(f'{ln.lat = }')

    return res

orig_create_model = ln.create_model
def create_model(input_shape, filters_per_field=[1,1,1], merge_to_one=False, batch_normalization=False, regularization='gradient', reg_mode='l2', reg_c=1, reg_weights=None, reg_periodicity=True, reg_norm=True, dense_units=[8,2], dense_activations=['relu', None], dense_dropouts=False, dense_l2coef=None):
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
    regularization : str, optional
        whether to regularize the projection by penalizing the spatial gradient ('gradient') or rather simply apply a `reg_mode` penalty to the weights, by default 'gradient'
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
        if regularization == 'gradient':
            regularizer = GradientRegularizer(mode=reg_mode, c=reg_c, weights=reg_weights, periodic_lon=reg_periodicity, normalize=reg_norm, lat=ln.lat)
        else:
            if reg_mode == 'l2':
                regularizer = keras.regularizers.l2(reg_c)
            elif reg_mode == 'l1':
                regularizer = keras.regularizers.l1(reg_c)
            else:
                raise ValueError(f'Unrecognized value for reg_mode: {reg_mode}')

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

def split_model(model, maxiter=5):
    proj = model.layers[0]
    rest = model.layers[1:]
    for i in range(maxiter):
        if hasattr(proj, 'filters_per_field'):
            break
        proj = proj.layers[0]
        rest = proj.layers[1:] + rest
    return proj, keras.models.Sequential(rest)


orig_train_model = ln.train_model
def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics, load_kernels_from=None, learn_kernels=True, orig_train_model_kwargs=None):
    '''
    Wrapper of the original train_model function. The extra arguments are:

    load_kernels_from : None|str|list
        How to initialize the kernels
    learn_kernels : bool
        Whether to train the kernels or leave them as they are at the initialization. By default True
    '''
    if orig_train_model_kwargs is None:
        orig_train_model_kwargs = {}

    if load_kernels_from is not None:
        if isinstance(load_kernels_from, str):
            if load_kernels_from.startswith('composite'):
                comp = np.mean(X_tr[Y_tr > 0.5], axis=0)
                np.save(f'{folder}/composite.npy', comp)
            elif load_kernels_from.startswith('significance'):
                comp = np.mean(X_tr[Y_tr > 0.5], axis=0)
                np.save(f'{folder}/composite.npy', comp)
                sig = np.std(X_tr[Y_tr > 0.5], axis=0)
                sig[sig==0] = 1
                comp = comp/sig
                np.save(f'{folder}/significance.npy', comp)
            else:
                raise NotImplementedError(f'Unknown option {load_kernels_from}')

            # split the model
            proj, _ = split_model(model)
            FPF = proj.filters_per_field

            load_kernels_from = []
            for i,fpf in enumerate(FPF):
                if fpf is None:
                    continue
                elif fpf == 1:
                    load_kernels_from.append(comp[...,i:i+1])
                else:
                    raise ValueError(f'It is dumb to set the composite as kernel {fpf} times')


        if not isinstance(load_kernels_from, list):
            raise TypeError(f'at this point load_kernels_from should be of type list, not {type(load_kernels_from)}')

        proj.set_weights(load_kernels_from)

    if not learn_kernels: # we can compute the result of the first layer on the data at once at the beginning. Also since we won't compute gradients through the projection layer, it is not trained.
        logger.info('Projection is not trainable: computing it at the beginning')

        # split the model
        proj, rest = split_model(model)

        proj = keras.models.Sequential([proj])
        proj.save(f'{folder}/projection') # save the projection
        model = rest # override model
        model.build(input_shape=proj.output_shape)

        # compute the output of the first layer
        batch_size = orig_train_model_kwargs.get('batch_size', 1024)
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

    return orig_train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics, **orig_train_model_kwargs)

orig_load_model = ln.load_model
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


#######################################################
# set the modified functions to override the old ones #
#######################################################
def enable():
    ln.add_mod(__file__, description, dependencies)
    ln.orig_train_model = orig_train_model
    ln.train_model = train_model
    ln.create_model = create_model
    ln.load_model = load_model
    ln.prepare_XY = prepare_XY
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary

def disable():
    ln.remove_mod(__file__)
    del ln.orig_train_model
    ln.train_model = orig_train_model
    ln.create_model = orig_create_model
    ln.load_model = orig_load_model
    ln.prepare_XY = orig_prepare_XY
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])

if __name__ == '__main__':
    enable()
    ln.main()
