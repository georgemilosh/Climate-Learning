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
    def __init__(self):
        self.epsilon = keras.backend.epsilon()

    def call(self, y_true, y_pred):
        mu = y_pred[...,0:1]
        sig = tf.math.square(y_pred[...,1:2]) + self.epsilon
        assert mu.shape == sig.shape == y_true.shape
        return tf.math.square(y_true - mu)/sig + tf.math.log(sig)

class ParametricCrossEntropy(keras.losses.Loss):
    def __init__(self, threshold=0):
        self.threshold = threshold

    def call(self, y_true, y_pred):
        labels = tf.cast(y_true >= self.threshold, tf.float32)
        mu = y_pred[...,0:1]
        sig = tf.math.square(y_pred[...,1:2])
        prob = q(mu,sig,self.threshold)

        return ut.entropy(labels, prob)




class Trainer(ln.Trainer):
    def extra_feature(self):
        pass

#######################################################
# set the modified functions to override the old ones #
#######################################################
ln.Trainer = Trainer

# uptade module level config dictionary
ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])

# change default values without modifying functions, below an example
ut.set_values_recursive(ln.CONFIG_DICT, {'return_threshold': True}, inplace=True) 

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

