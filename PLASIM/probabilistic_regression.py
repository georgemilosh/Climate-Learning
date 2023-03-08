import Learn2_new as ln
ut = ln.ut
logger = ln.logger
# log to stdout
import logging
import sys
import os
logging.getLogger().level = logging.INFO
logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

tf = ln.tf
keras = ln.keras

import numpy as np
from scipy import special as ss

def q(mu, sig, thr):
    return 0.5*ss.erfc((thr - mu)/sig/np.sqrt(2))

### custom losses/metrics
class ProbRegLoss(keras.losses.Loss):
    def __init__(self, name='ProbRegLoss'):
        self.name = name
        self.epsilon = keras.backend.epsilon()

    def call(self, y_true, y_pred):
        mu = y_pred[...,0:1]
        sig = tf.math.square(y_pred[...,1:2]) + self.epsilon
        assert mu.shape == sig.shape == y_true.shape
        return tf.math.square(y_true - mu)/sig + tf.math.log(sig)

class ParametricCrossEntropyLoss(keras.losses.Loss):
    def __init__(self, threshold=0, name='ParametricCrossEntropyLoss'):
        self.name = name
        self.threshold = threshold

    def call(self, y_true, y_pred):
        labels = tf.cast(y_true >= self.threshold, tf.float32)
        mu = y_pred[...,0:1]
        sig = tf.math.square(y_pred[...,1:2])
        prob = q(mu,sig,self.threshold)

        return ut.entropy(labels, prob)


# create a module level variable to store the threshold
ln._current_threshold = None

# redefine prepare_XY to use A instead of Y
class Trainer(ln.Trainer):
    def prepare_XY(self, fields, **prepare_XY_kwargs):
        if self._prepare_XY_kwargs != prepare_XY_kwargs:
            self._prepare_XY_kwargs = prepare_XY_kwargs
            self.X, self.Y, self.year_permutation, self.lat, self.lon, threshold = ln.prepare_XY(fields, **prepare_XY_kwargs)

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
    
def get_loss_function(loss_name: str, u=1):
    loss_name = loss_name.lower()
    if loss_name.startswith('prob'):
        return ProbRegLoss()
    else:
        return ln.get_loss_function(loss_name, u=u)
    
def get_default_metrics(fullmetrics=False, u=1):
    if fullmetrics:
        metrics = [
            'loss',
            ParametricCrossEntropyLoss(ln._current_threshold),
        ]
    else:
        metrics=['loss']
    return metrics

#######################################################
# set the modified functions to override the old ones #
#######################################################
ln.Trainer = Trainer
ln.get_default_metrics = get_default_metrics
ln.get_loss_function = get_loss_function

# uptade module level config dictionary
ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])

# change default values without modifying functions, below an example
ut.set_values_recursive(ln.CONFIG_DICT, {'return_threshold': True, 'loss': 'prob_reg_loss', 'return_metric': 'val_ParametricCrossEntropyLoss'}, inplace=True) 

# override the main function as well
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

