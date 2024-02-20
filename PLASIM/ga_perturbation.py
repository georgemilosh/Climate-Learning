# '''
# Created in February 2024

# @author: Alessandro Lovo
# '''
description = """Train a model as a perturbation on top of a gaussian model for probabilistic regression."""
dependencies = None

import probabilistic_regression as pr
pr.enable()

ln = pr.ln
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


class GaussianLayer(layers.Layer):
    def __init__(self, path_to_ga, **kwargs):
        super().__init__(**kwargs)
        ln.logger.log(45, f'Loading GA Perturbation from {path_to_ga}...')
        m, self._sigma = np.load(f'{path_to_ga}/msigma.npy').astype(np.float32)
        self._proj = m*tf.convert_to_tensor(np.load(f'{path_to_ga}/proj.npy'), dtype=tf.float32)
        self.naxes = len(self._proj.shape)

    def build(self, input_shape):
        assert input_shape[-self.naxes:] == self._proj.shape
        self.proj = self.add_weight(name='proj', shape=self._proj.shape, initializer='zeros', trainable=False)
        # set weight values from self._proj
        self.proj.assign(self._proj)
        self.sigma = self.add_weight(name='sigma', shape=(), initializer='zeros', trainable=False)
        self.sigma.assign(self._sigma)

    def call(self, x):
        mu = tf.tensordot(x, self.proj, axes=len(self.proj.shape))
        return tf.stack([mu, tf.ones_like(mu)*self.sigma], axis=-1)


orig_k_fold_cross_val_split = ln.k_fold_cross_val_split

@wraps(orig_k_fold_cross_val_split)
def k_fold_cross_val_split(*args, **kwargs):
    ln.current_fold = args[0]
    ln.logger.info(f'{ln.current_fold = }')
    return orig_k_fold_cross_val_split(*args, **kwargs)

create_inner_model = ln.create_model

def create_model(input_shape, path_to_ga=None, create_inner_model_kwargs=None):
    if path_to_ga is None:
        raise ValueError('path_to_ga must be provided')
    if create_inner_model_kwargs is None:
        create_inner_model_kwargs = {}
    model = create_inner_model(input_shape, **create_inner_model_kwargs)

    # split the model to put the sigma_activation at the end
    assert len(model.layers) == 2, model.layers
    model, act = model.layers

    # create the gaussian model
    ga_model = GaussianLayer(path_to_ga)

    # create the full perturbative model
    inputs = tf.keras.Input(shape=input_shape, name='input')
    ga_output = ga_model(inputs)
    model_output = model(inputs)
    merged_output = layers.Add()([ga_output, model_output])
    outputs = act(merged_output)

    model = keras.Model(inputs=inputs, outputs=outputs, name='perturbative_model')
    return model

pr.disable()
#######################################################
# set the modified functions to override the old ones #
#######################################################
def enable():
    pr.enable()
    ln.add_mod(__file__, description, dependencies)
    ln.k_fold_cross_val_split = k_fold_cross_val_split
    ln.create_inner_model = create_inner_model
    ln.create_model = create_model
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary

def disable():
    ln.remove_mod(__file__)
    ln.k_fold_cross_val_split = orig_k_fold_cross_val_split
    ln.create_model = create_inner_model
    del ln.create_inner_model
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])
    pr.disable()

if __name__ == '__main__':
    enable()
    ln.main()