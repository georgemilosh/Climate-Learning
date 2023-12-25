# This file is an adaptation of tutorial.ipynb

import Learn2_new as ln
ut = ln.ut # utilities
ef = ln.ef # ERA_Fields_New

# log to stdout
import logging
import sys
import os

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

# set spacing of the indentation
ut.indentation_sep = '  '

# create a work directory
work_dir = './__test__/t1'

t = ln.Trainer(work_dir)

print(ut.dict2str(ut.extract_nested(t.config_dict, 'create_model_kwargs')))

d = {
    #'dataset_years': 1000, # choose the dataset with less years to avoid 
    'year_list' : 'range(100)', # choose the first 100 years for training and testing (everything)
    'conv_channels': [64, 24], # set number of convolutional kernels per layer
    'kernel_sizes': [7, 3],
    'strides': [3, 2],
    'batch_normalization': False, # disable batch normalization
    'max_pool_sizes': False, # disable max pool
    'conv_dropout': False, # disable dropout in the convolutional layer

    'dense_units': [128, 64, 16, 2], # number of neurons per fully connected layer
    'dense_dropouts': [0.5, 0.5, 0.1, 0],
    'dense_activations': ['relu', 'relu', 'relu', None],
    'training_epochs': 5
}


ut.set_values_recursive(t.config_dict, d, inplace=True)
print(ut.dict2str(t.config_dict))

t.schedule(tau=[-10, -15], percent=[5,1], load_from='last--percent__same')
# t.schedule(percent=5, tau=5, load_from='last_percent__same--tau__-5')

t.run_multiple()
