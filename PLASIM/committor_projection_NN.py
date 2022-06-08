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

    def __init__(self, filters_per_field=[1,2,1], regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.filters_per_field = filters_per_field
        self.nfields = len(self.filters_per_field)
        self.m = np.sum(self.filters_per_field)

        self.regularizer = regularizer

        self.conc = keras.layers.Concatenate()

    def build(self, input_shape):
        if input_shape[-1] != self.nfields:
            raise ValueError(f'Expected {self.nfields} fields, received {input_shape[-1]}')
        kernel_shape = input_shape[-3:-1]

        self.kernels = []
        for i, fpf in enumerate(self.filters_per_field):
            self.kernels.append(self.add_weight(
                name=f"w_{i}",
                shape=(*kernel_shape, fpf),
                initializer="random_normal",
                trainable=True,
                regularizer=self.regularizer
            ))
    
    def call(self, x):
        if x.shape[-1] != self.nfields:
            raise ValueError(f'Expected {self.nfields} fields, received {x.shape[-1]}')

        x = [tf.tensordot(x[...,i], self.kernels[i], axes=2) for i in range(self.nfields)]
        x = self.conc(x)

        return x


class GradientRegularizer(keras.regularizers.Regularizer):
    def __init__(self, mode='l2', c=1, weights=None, periodic_lon=True, normalize=True):
        if mode in ['L1', 'l1', 'sparse']:
            self.mode = 'l1'
        else:
            if mode not in ['L2', 'l2']:
                logger.warning(f"Unrecognized regularization {mode = }: using 'l2'")
            self.mode = 'l2'
        self.c = c
        self.weights = weights
        self.periodic_lon = periodic_lon
        self.normalize = normalize

        if self.weights is not None:
            self.weights = self.weights/tf.math.reduce_mean(tf.math.abs(self.weights)) # now the mean of the weights is 1

    def __call__(self, x):
        if self.c == 0:
            return 0
        nfilters = x.shape[-1]
        if self.weights is not None and self.weights.shape[:-1] != x.shape[:-1]:
            raise ValueError(f'weight shape {self.weights.shape} does not match received input shape {x.shape[:-1]}')
        s = 0
        for i in range(nfilters):
            if self.mode == 'l1':
                op = tf.math.abs
            else:
                op = tf.math.square
            
            if self.weights is not None:
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
                _s = _s/tf.math.reduce_sum(op(x))

            s = s + _s

        return self.c*s

    def get_config(self):
        return {'c': self.c, 'weights': self.weights, 'periodic_lon': self.periodic_lon, 'normalize': self.normalize}


def create_model(input_shape, filters_per_field=[1,1,1], reg_mode='l2', reg_c=1, reg_weights=None, reg_periodicity=True, reg_norm=True, dense_units=[8,2], dense_activations=['relu', None], dense_dropouts=False):
    if not reg_c:
        regularizer = None
    else:
        regularizer = GradientRegularizer(mode=reg_mode, c=reg_c, weights=reg_weights, periodic_lon=reg_periodicity, normalize=reg_norm)
    
    model = keras.models.Sequential()

    model.add(Dense2D(filters_per_field=filters_per_field, regularizer=regularizer, input_shape=input_shape))

    # dense layers
    # adjust the shape of the arguments to be of the same length as conv_channels
    args = [dense_activations, dense_dropouts]
    for j,arg in enumerate(args):
        if not isinstance(arg, list):
            args[j] = [arg]*len(dense_units)
        elif len(arg) != len(dense_units):
            raise ValueError(f'Invalid length for argument {arg}')
    logger.info(f'dense args = {args}')
    dense_activations, dense_dropouts = args
    # build the dense layers
    for i in range(len(dense_units)):
        model.add(layers.Dense(dense_units[i], activation=dense_activations[i]))
        if dense_dropouts[i]:
            model.add(layers.Dropout(dense_dropouts[i]))

    return model



#####################################################
# set the modified function to override the old one #
#####################################################
ln.create_model = create_model
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
