description = """Probabilistic regression."""
dependencies = None

import Learn2_new as ln
ut = ln.ut
logger = ln.logger
# log to stdout
import logging
import sys

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

tf = ln.tf
keras = ln.keras

import numpy as np

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
    '''Continuous Ranked Probability Score. Assumes predicted sigma is positive'''
    def __init__(self,name=None, epsilon=None) -> None:
        super().__init__(name=name or self.__class__.__name__)
        self.epsilon = epsilon or keras.backend.epsilon()

    def call(self, y_true, y_pred):
        sig = y_pred[...,1:2]
        tf.debugging.assert_non_negative(sig, 'Model predicted negative variance, please fix it!')
        sig = sig + self.epsilon
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
        assert y_pred.shape == sig2.shape == y_true.shape, f'{y_pred.shape = }, {sig2.shape = }, {y_true.shape = }'
        return tf.math.square(y_true - y_pred)/sig2 + tf.math.log(sig2) + penalty

class NLLLoss(ProbRegLoss):
    def __init__(self, name=None, epsilon=None):
        super().__init__(name=name or self.__class__.__name__, epsilon=epsilon)

class ParametricCrossEntropyLoss(keras.losses.Loss):
    def __init__(self, threshold=0, epsilon=None):
        super().__init__(name=self.__class__.__name__)
        self.threshold = threshold
        self.epsilon = epsilon or keras.backend.epsilon()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true >= self.threshold, tf.float32)
        sig = y_pred[...,1:2]
        tf.debugging.assert_non_negative(sig, 'Model predicted negative variance, please fix it!')
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


# custom activation for sigma
class Sigma_Activation(tf.keras.layers.Layer):
    def __init__(self, activation='relu', min_sigma=None, name=None,):
        super().__init__(trainable=False, name=name or self.__class__.__name__)
        if activation == 'abs':
            self.activation = tf.math.abs
        elif activation == 'exp':
            self.activation = tf.math.exp
        else:
            self.activation = tf.keras.activations.get(activation)
        self.min_sigma = min_sigma

    def call(self, x):
        sigma = self.activation(x[...,-1:])
        if self.min_sigma is not None:
            sigma = tf.maximum(sigma, self.min_sigma)
        return tf.concat([x[...,:-1], sigma], axis=-1)

create_core_model = ln.create_model

def create_model(input_shape, sigma_activation='relu', min_sigma=None, create_core_model_kwargs=None):
    if create_core_model_kwargs is None:
        create_core_model_kwargs = {}
    model = create_core_model(input_shape, **create_core_model_kwargs)
    model = keras.models.Sequential([model, Sigma_Activation(activation=sigma_activation, min_sigma=min_sigma)])
    return model

# create a module level variable to store the threshold
ln._current_threshold = None

# redefine prepare_XY to use A instead of Y
orig_Trainer = ln.Trainer
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

# we redefine check_config_dict to ensure return_threshold is True
orig_check_config_dict = ln.check_config_dict
def check_config_dict(config_dict, correct_mistakes=True):
    config_dict_flat = orig_check_config_dict(config_dict, correct_mistakes=correct_mistakes)
    if not config_dict_flat['return_threshold']:
        if correct_mistakes:
            config_dict_flat['return_threshold'] = True
        else:
            raise ValueError('return_threshold must be True')
    return config_dict_flat

orig_get_loss_function = ln.get_loss_function
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
        return orig_get_loss_function(loss_name, u=u)

    if func:
        return weighted(cls, func)(*args, **kwargs)
    return cls(*args, **kwargs)

orig_get_default_metrics = ln.get_default_metrics
def get_default_metrics(fullmetrics=False, u=1):
    if fullmetrics:
        metrics = [
            ParametricCrossEntropyLoss(ln._current_threshold),
            ProbRegLoss(),
            CRPS(),
        ]
    else:
        metrics=None
    return metrics

orig_postprocess = ln.postprocess
def postprocess(x):
    return x

#######################################################
# set the modified functions to override the old ones #
#######################################################
def enable():
    ln.add_mod(__file__, description, dependencies)
    ln.Trainer = Trainer
    ln.check_config_dict = check_config_dict
    ln.get_default_metrics = get_default_metrics
    ln.get_loss_function = get_loss_function
    ln.postprocess = postprocess
    ln.create_model = create_model
    ln.create_core_model = create_core_model

    # uptade module level config dictionary
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])

    # change default values without modifying functions, below an example
    ut.set_values_recursive(ln.CONFIG_DICT,
                            {
                                'return_threshold': True,
                                'loss': 'crps',
                                'return_metric': 'val_ParametricCrossEntropyLoss',
                                'monitor' : 'val_ParametricCrossEntropyLoss',
                                'metric' : 'val_ParametricCrossEntropyLoss',
                            },
                            inplace=True)

def disable():
    ln.remove_mod(__file__)
    ln.Trainer = orig_Trainer
    ln.check_config_dict = orig_check_config_dict
    ln.get_default_metrics = orig_get_default_metrics
    ln.get_loss_function = orig_get_loss_function
    ln.postprocess = orig_postprocess
    ln.create_model = create_core_model
    del ln.create_core_model
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])

# override the main function as well
if __name__ == '__main__':
    enable()
    ln.main()
