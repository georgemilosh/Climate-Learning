# This file is an adaptation of tutorial.ipynb

import Learn2_new as ln
ut = ln.ut # utilities
ef = ln.ef # ERA_Fields_New

# log to stdout
import logging
import sys
import os
logging.getLogger().level = logging.INFO
logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

# set spacing of the indentation
ut.indentation_sep = '  '

# create a work directory
work_dir = './test'

t = ln.Trainer(work_dir)

print(ut.dict2str(ut.extract_nested(t.config_dict, 'create_model_kwargs')))

d = {
    'conv_channels': [64, 24], # set number of convolutional kernels per layer
    'kernel_sizes': [7, 3],
    'strides': [3, 2],
    'batch_normalization': False, # disable batch normalization
    'max_pool_sizes': False, # disable max pool
    'conv_dropout': False, # disable dropout in the convolutional layer

    'dense_units': [128, 64, 16, 2], # number of neurons per fully connected layer
    'dense_dropouts': [0.5, 0.5, 0.1, 0]
}

ut.set_values_recursive(t.config_dict, d, inplace=True)
print(ut.dict2str(t.config_dict))

t.schedule(percent=[5,1], tau=[0, -5, -10, -15, -20, -25, -30], load_from='last--percent__same')

t.run_multiple()