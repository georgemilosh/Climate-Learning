# George Miloshevich 2021
# Train a neural network

# Import librairies
import os as os
from pathlib import Path
import sys
import warnings
import time
import shutil
import gc
from numpy.lib.arraysetops import isin
from numpy.random.mtrand import permutation
import psutil
import numpy as np
import inspect
import json

this_module = sys.modules[__name__]
path_to_here = Path(__file__).resolve().parent
path_to_ERA = path_to_here / 'ERA' # when absolute path, so you can run the script from another folder (outside plasim)
if not os.path.exists(path_to_ERA):
    path_to_ERA = path_to_here.parent / 'ERA'
    if not os.path.exists(path_to_ERA):
        raise FileNotFoundError('Could not find ERA folder')
print(path_to_ERA)
sys.path.insert(1, str(path_to_ERA))
# sys.path.insert(1, '../ERA/')
import ERA_Fields as ef # general routines
import TF_Fields as tff # tensorflow routines 

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from operator import mul
from functools import reduce

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.python.types.core import Value

########## USAGE ###############################
def usage():
    s = '''
    How to use this script:
    '''
    return s


########## ARGUMENT PARSING ####################

def run_smart(func, default_kwargs, **kwargs): # this is not as powerful as it looks like
    evaluate = True
    for k,v in kwargs.items():
        if k not in default_kwargs:
            raise KeyError(f'Unknown argument {k}')
        iterate = False
        if isinstance(v, list): # possible need to iterate over the argument
            if isinstance(default_kwargs[k], list):
                if isinstance(v[0], list):
                    iterate = True
            else:
                iterate = True
        if iterate:
            evaluate = False
            for _v in v:
                kwargs[k] = _v
                run_smart(func, default_kwargs, **kwargs)
            break
    if evaluate:
        f_kwargs = default_kwargs
        for k,v in kwargs.items():
            f_kwargs[k] = v
        func(**f_kwargs)

#### CONFIG FILE #####

def get_default_params(func, recursive=False):
    '''
    Given a function returns a dictionary with the default values of its parameters
    '''
    s = inspect.signature(func)
    default_params = {
        k:v.default for k,v in s.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    if recursive: # look for parameters ending in '_kwagrs' and extract further default arguments
        possible_other_params = [k for k,v in s.parameters.items() if (v.default is inspect.Parameter.empty and k.endswith('_kwargs'))]
        for k in possible_other_params:
            func_name = k.rsplit('_',1)[0] # remove '_kwargs'
            try:
                default_params[k] = get_default_params(getattr(this_module, func_name), recursive=True)
            except:
                print(f'Could not find function {func_name}')
    return default_params

def read_json(filename):
    '''
    Reads a json file `filename` as a dictionary
    '''
    with open(filename, 'r') as j:
        d = json.load(j)
    return d

def write_to_json(d, filename):
    '''
    Saves a dictionary `d` to a json file `filename`
    '''
    with open(filename, 'w') as j:
        json.dump(d, j, indent=4)

def build_config_dict(functions):
    '''
    Creates a config file with the default arguments of the functions in the list `functions`

    Parameters:
    -----------
        functions: list of functions or string with the function name
    
    Returns:
    --------
        d: dictionary
    '''
    d = {}
    for f in functions:
        if isinstance(f, str):
            f_name = f
            f = getattr(this_module, f_name)
        else:
            f_name = f.__name__
        d[f_name] = get_default_params(f, recursive=True)
    return d

def collapse_dict(d_nested, d_flat=None):
    '''
    Flattens a nested dictionary `d_nested` into a flat one `d_flat`. 
    `d_nested` can contain dictionaries and other types. If a key is present more times the associated values must be the same, otherwise an error will be raised
    '''
    if d_flat is None:
        d_flat = {}

    for k,v in d_nested.items():
        if isinstance(v, dict):
            d_flat = collapse_dict(v,d_flat)
        else:
            if k in d_flat and v != d_flat[k]:
                raise ValueError(f'Multiple definitions for argument {k}')
            d_flat[k] = v
    return d_flat

def parse_run_name(run_name):
    d = {}
    args = run_name.split('__')
    for arg in args:
        if '_' not in arg:
            continue
        key, value = arg.rsplit('_',1)
        d[key] = value
    return d

def get_run(load_from, current_run_name=None):
    '''
    Parameters:
    -----------
        load_from: int, str or 'last'. If int it is the number of the run. If 'last' it is the last completed run. Otherwise it can be a piece of the run name.
            If the choice is ambiguous an error will be raised.
        current_run_name: optional, used to check for compatibility issues when loading a model
    '''
    if load_from is None:
        return None
    # get run_folder name
    with open('runs.txt', 'r') as runs_file:
        runs = runs_file.readlines()
    if len(runs) == 0:
        print('No runs to load from')
        return None
    try:
        l = int(load_from)
    except ValueError: # cannot convert load_from to int, so it must be a string that doesn't contain only numbers
        if load_from == 'last':
            l = -1
        else:
            found = False
            for i,r in enumerate(runs):
                if load_from in r:
                    if not found:
                        found = True
                        l = r
                    else:
                        raise KeyError(f'Multiple runs contain {load_from}, at least {l} and {i}')
    run_name = runs[l].rstrip('\n')
    
    if current_run_name is not None: # check for compatibility issues when loading
        # parse run_name for arguments
        run_dict = parse_run_name(run_name)
        current_run_dict = parse_run_name(current_run_name)

        # NOTE: continue here

    return run_name

########## COPY SOURCE FILES #########

def move_to_folder(folder):
    '''
    Copies this file and its dependencies to a given folder.
    '''
    folder = Path(folder).resolve()
    ERA_folder = folder / 'ERA'

    if os.path.exists(ERA_folder):
        raise FileExistsError(f'Cannot copy scripts to {folder}: you already have some there')
    ERA_folder.mkdir(parents=True,exist_ok=True)

    # copy this file
    path_to_here = Path(__file__).resolve() # path to this file
    shutil.copy(path_to_here, folder)

    # copy other files in the same directory as this one
    path_to_here = path_to_here.parent
    # shutil.copy(path_to_here / 'config', folder)

    # copy files in ../ERA/
    path_to_here = path_to_here.parent / 'ERA'
    shutil.copy(path_to_here / 'cartopy_plots.py', ERA_folder)
    shutil.copy(path_to_here / 'ERA_Fields.py', ERA_folder)
    shutil.copy(path_to_here / 'TF_Fields.py', ERA_folder)

    # copy additional files
    # History.py
    # Metrics.py
    # Recalc_Tau_Metrics.py
    # Recalc_History.py
    
    print(f'Now you can go to {folder} and run the learning from there:\n')
    print(f'cd \"{folder}\"\n')
    
    

########## DATA PREPROCESSING ##############

fields_infos = {
    't2m': { # temperature
        'name': 'tas',
        'filename_suffix': 'tas',
        'label': 'Temperature',
    },
    'mrso': { # soil moisture
        'name': 'mrso',
        'filename_suffix': 'mrso',
        'label': 'Soil Moisture',
    },
}

for h in [200,300,500,850]: # geopotential heights
    fields_infos[f'zg{h}'] = {
        'name': 'zg',
        'filename_suffix': f'zg{h}',
        'label': f'{h} mbar Geopotential',
    }


def load_data(dataset_years=8000, year_list=None, sampling='', Model='Plasim', area='France', filter_area='France',
              lon_start=0, lon_end=128, lat_start=0, lat_end=22, mylocal='/local/gmiloshe/PLASIM/',fields=['t2m','zg500','mrso_filtered']):
    '''
    Loads the data.

    Parameters:
    -----------
        dataset_years: number of years of the dataset, 8000 or 1000
        year_list: list of years to load from the dataset
        sampling: '' (dayly) or '3hrs'
        Model: 'Plasim', 'CESM', ...
        area: region of interest, e.g. 'France'
        filter_area: area over which to keep filtered fields
        lon_start, lon_end, lat_start, lat_end: longitude and latitude extremes of the data expressed in indices (model specific)
        mylocal: path the the data storage. For speed it is better if it is a local path.
        fields: list of field names to be loaded. Add '_filtered' to the name to have the velues of the field outside `area` set to zero.

    Returns:
    --------
        _fields: dictionary of ERA_Fields.Plasim_Field objects
    '''

    if area != filter_area:
        warnings.warn(f'Fields will be filtered on a different area ({filter_area}) than the region of interest ({area})')

    if dataset_years == 1000:
        dataset_suffix = ''
    elif dataset_years == 8000:
        dataset_suffix = '_LONG'
    else:
        raise ValueError('Invalid number of dataset years')
   

    mask, cell_area, lsm = ef.ExtractAreaWithMask(mylocal,Model,area) # extract land-sea mask and multiply it by cell area

    if sampling == '3hrs': 
        prefix = ''
        file_suffix = f'../Climate/Data_Plasim{dataset_suffix}/'
    else:
        prefix = f'ANO{dataset_suffix}_'
        file_suffix = f'Data_Plasim{dataset_suffix}/'

    # load the fields
    _fields = {}
    for field_name in fields:
        do_filter = False
        if field_name.endswith('_filtered'): # TO IMPROVE: if you have to filter the data load just the interesting part
            field_name = field_name.rsplit('_', 1)[0] # remove '_filtered'
            do_filter = True
        if field_name not in fields_infos:
            raise KeyError(f'Unknown field {field_name}')
        f_infos = fields_infos[field_name]
        # create the field object
        field = ef.Plasim_Field(f_infos['name'], prefix+f_infos['filename_suffix'], f_infos['label'],
                                Model=Model, lat_start=lat_start, lat_end=lat_end, lon_start=lon_start, lon_end=lon_end,
                                myprecision='single', mysampling=sampling, years=dataset_years)
        # load the data
        field.load_data(mylocal+file_suffix, year_list=year_list)
        # Set area integral
        field.abs_area_int, field.ano_area_int = field.Set_area_integral(area,mask,'Postproc')
        # filter
        if do_filter: # set to zero all values outside `filter_area`
            filter_mask = ef.create_mask(Model, filter_area, field.var, axes='last 2', return_full_mask=True)
            field.var *= filter_mask

        _fields[field_name] = field  
    
    return _fields


def assign_labels(field, time_start=30, time_end=120, T=14, percent=5, threshold=None):
    '''
    Given a field of anomalies it computes the `T` days forward convolution of the integrated anomaly and assigns label 1 to anomalies above a given `threshold`.
    If `threshold` is not provided, then it is computed from `percent`, namely to identify the `percent` most extreme anomalies.

    Returns:
    --------
        labels: 2D array with shape (years, days) and values 0 or 1
    '''
    A, A_flattened, threshold =  field.ComputeTimeAverage(time_start, time_end, T=T, percent=percent, threshold=threshold)[:3]
    return np.array(A >= threshold, dtype=int)

def make_X(fields, time_start=30, time_end=120, T=14, tau=0):
    '''
    Cuts the fields in time and stacks them. The original fields are not modified

    Returns:
    --------
        X: array with shape (years, days, lat, lon, field)
    '''
    # stack the fields
    X = np.array([field.var[:, time_start+tau:time_end+tau-T+1, ...] for field in fields.values()])
    # now transpose the array so the field index becomes the last
    X = X.transpose(*range(1,len(X.shape)), 0)
    return X

def make_XY(fields, label_field='t2m', time_start=30, time_end=120, T=14, tau=0, percent=5, threshold=None):
    '''
    Combines make_X and assign labels

    Returns:
    --------
        X: array with shape (years, days, lat, lon, field)
        Y: array with shape (years, days)
    '''
    X = make_X(fields, time_start=time_start, time_end=time_end, T=T, tau=tau)
    Y = assign_labels(fields[label_field], time_start=time_start, time_end=time_end, T=T, percent=percent, threshold=threshold)
    return X,Y

def roll_X(X, roll_axis='lon', roll_steps=64):
    '''
    Rolls `X` along a given axis. useful for example for moving France away from the Greenwich meridian

    Parameters:
    -----------
        X: array with shape (years, days, lat, lon, field)
        axis: 'year' (or 'y'), 'day' (or 'd'), 'lat', 'lon', 'field' (or 'f')
        steps: number of gridsteps to roll
            a positive value for 'steps' means that the elements of the array are moved forward in it, e.g. `steps` = 1 means that the old first element is now in the second place
            This means that for every axis a positive value of `steps` yields a shift of the array
            'year', 'day' : forward in time
            'lat' : southward
            'lon' : eastward
            'field' : forward in the numbering of the fields
    '''
    if roll_steps == 0:
        return X
    if roll_axis.startswith('y'):
        roll_axis = 0
    elif roll_axis.startswith('d'):
        roll_axis = 1
    elif roll_axis == 'lat':
        roll_axis = 2
    elif roll_axis == 'lon':
        roll_axis = 3
    elif roll_axis.startswith('f'):
        roll_axis = 4
    else:
        raise ValueError(f'Unknown valur for axis: {roll_axis}')
    return np.roll(X,roll_steps,axis=roll_axis)

####### MIXING ########

def invert_permutation(permutation):
    '''
    Inverts a permutation.
    e.g.:
        a = np.array([3,4,2,5])
        p = np.random.permutation(np.arange(4))
        a_permuted = a[p]
        p_inverse = invert_permutation(p)

        `a` and `a_permuted[p_inverse]` will be equal

    Parameters:
    -----------
        permutation: 1D array that must be a permutation of an array of the kind `np.arange(n)` with `n` integer
    '''
    return np.argsort(permutation)

def compose_permutations(permutations):
    '''
    Composes a series of permutations
    e.g.:
        a = np.array([3,4,2,5])
        p1 = np.random.permutation(np.arange(4))
        p2 = np.random.permutation(np.arange(4))
        p_composed = compose_permutations([p1,p2])
        a_permuted1 = a[p1]
        a_permuted2 = a_permuted1[p2]
        a_permuted_c = a[p_composed]

        `a_permuted_c` and `a_permuted2` will be equal

    Parameters:
    -----------
        permutations: list of 1D arrays that must be a permutation of an array of the kind `np.arange(n)` with `n` integer and the same for every permutation
    '''
    l = len(permutations[0])
    for p in permutations[1:]:
        if len(p) != l:
            raise ValueError('All permutations must have the same length')
    ps = permutations[::-1]
    p = ps[0]
    for _p in ps[1:]:
        p = _p[p]
    return p
    

def shuffle_years(X, permutation=None, seed=0, apply=False):
    '''
    Permutes `X` along the first axis

    Parameters:
    -----------
        X: array with the data to permute
        permutation: None or 1D array that must be a permutation of an array of `np.arange(X.shape[0])`
        seed: int, if `permutation` is None, then it is computed using the provided seed.
        apply: bool, if True the function returns the permuted data, otherwise the permutation is returned
    '''
    if permutation is None:
        if seed is not None:
            np.random.seed(seed)
            permutation = np.random.permutation(X.shape[0])
    if len(permutation) != X.shape[0]:
        raise ValueError(f'Shape mismatch between X ({X.shape[0] = }) and permutation ({len(permutation) = })')
    if apply:
        return X[permutation,...]
    return permutation

def balance_folds(weights, nfolds=10, verbose=False):
    '''
    Returns a permutation that, once applied to `weights` would make the consecutive `nfolds` pieces of equal length have their sum the most similar to each other.

    Parameters:
    -----------
        weights: 1D array
        nfolds: int, must be a divisor of `len(weights)`

    Returns:
    --------
        permutation: permutation of `np.arange(len(weights))`
    '''
    class Fold():
        def __init__(self, target_length, target_sum, name=None):
            self.indexs = []
            self.length = target_length
            self.target_sum = target_sum
            self.sum = 0
            self.hunger = np.infty # how much a fold 'wants' the next datapoint
            self.name = name
        
        def add(self, a):
            self.indexs.append(a[1])
            self.sum += a[0]
            if self.length == len(self.indexs):
                print(f'fold {self.name} done!')
                return True
            self.hunger = (self.target_sum - self.sum)/(self.length - len(self.indexs))
            return False

    fold_length = len(weights)//nfolds
    if len(weights) != fold_length*nfolds:
        raise ValueError(f'Cannot make {nfolds} folds of equal lenght out of {len(weights)} years of data')
    target_sum = np.sum(weights)/nfolds

    # create nfolds Folds that will redistribute the data
    folds = [Fold(fold_length, target_sum, name=i) for i in range(nfolds)]
    permutation = []

    # sort the weights keeping track of the original order
    ws = [(a,i) for i,a in enumerate(weights)]
    ws.sort()
    ws = ws[::-1]
    
    sums = []
    # run over the weights and distribute data to the folds
    for a in ws:
        # determine the hungriest fold, i.e. the one that has its sum the furthest from its target
        j = np.argmax([f.hunger for f in folds])
        if folds[j].add(a): # add datapoint and check if the fold is full
            f = folds.pop(j)
            sums.append(f.sum)
            permutation += f.indexs

    if len(permutation) != len(weights):
        raise ValueError('balance_folds: Something went wrong during balancing: either missing or duplicated data')

    if verbose:
        sums = np.array(sums)
        print(f'Sums of the balanced {nfolds} folds:\n{sums}\nstd/avg = {np.std(sums)/target_sum}\nmax relative deviation = {np.max(np.abs(sums - target_sum))/target_sum*100}\%')

    return permutation


########## NEURAL NETWORK DEFINITION ###########

def create_model(input_shape, conv_channels=[32,64,64], kernel_sizes=3, strides=1,
                 batch_normalizations=True, conv_activations='relu', conv_dropouts=0.2, max_pool_sizes=[2,2,False],
                 dense_units=[64,2], dense_activations=['relu', None], dense_dropouts=[0.2,False]):
    '''
    Creates a model consisting of a series of convolutional layers followed by fully connected ones
    '''
    model = models.Sequential()

    # convolutional layers
    # adjust the shape of the arguments to be of the same length as conv_channels
    args = kernel_sizes, strides, batch_normalizations, conv_activations, conv_dropouts, max_pool_sizes
    for j,arg in enumerate(range(len(args))):
        if not isinstance(arg, list):
            args[j] = [arg]*len(conv_channels)
        elif len(arg) != len(conv_channels):
            raise ValueError(f'Invalid length for argument {arg}')
    kernel_sizes, strides, batch_normalizations, conv_activations, conv_dropouts, max_pool_sizes = args
    # build the convolutional layers
    for i in range(len(conv_channels)):
        if i == 0:
            model.add(layers.Conv2D(conv_channels[i], kernel_sizes[i],
                      strides=strides[i], input_shape=input_shape))
        else:
            model.add(layers.Conv2D(conv_channels[i], kernel_sizes[i],
                      strides=strides[i]))
        if batch_normalizations[i]:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(conv_activations[i]))
        if conv_dropouts[i]:
            model.add(layers.SpatialDropout2D(conv_dropouts[i]))
        if max_pool_sizes[i]:
            model.add(layers.MaxPooling2D(max_pool_sizes[i]))

    # flatten
    model.add(layers.Flatten())

    # dense layers
    # adjust the shape of the arguments to be of the same length as conv_channels
    args = dense_activations, dense_dropouts
    for j,arg in enumerate(range(len(args))):
        if not isinstance(arg, list):
            args[j] = [arg]*len(dense_units)
        elif len(arg) != len(dense_units):
            raise ValueError(f'Invalid length for argument {arg}')
    dense_activations, dense_dropouts = args
    # build the dense layers
    for i in range(len(dense_units)):
        model.add(layers.Dense(dense_units[i], activation=dense_activations[i]))
        if dense_dropouts[i]:
            model.add(layers.Dropout(dense_dropouts[i]))
    
    print(model.summary())

    return model


###### TRAINING THE NETWORK ############

def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics,
                batch_size=1024, checkpoint_every=1, additional_callbacks=None):
    '''
    Trains a given model checkpointing its weights

    Parameters:
    -----------
        model: keras.models.Model object
        X_tr: training data
        Y_tr: training labels
        X_va: validation data
        Y_va: validation labels
        folder: loaction where to save the checkpoints of the model
        num_epochs: int, number of maximum epochs for the training
        optimizer: keras.Optimizer object
        loss: keras.losses.Loss object
        metrics: list of keras.metrics.Metric or str
        batch_size: int, default 1024
        checkpoint_every: int or str. Examples:
            0: disabled
            5 or '5 epochs' or '5 e': every 5 epochs
            '100 batches' or '100 b': every 100 batches
            'best custom_loss': every time 'custom_loss' reaches a new minimum. 'custom_loss' must be in the list of metrics
        additional_callbacks: list of keras.callbacks.Callback objects, for example EarlyStopping
    '''
    folder = folder.rstrip('/')
    ckpt_name = folder + '/cp-{epoch:04d}.ckpt'
    if additional_callbacks is None:
        additional_callbacks = []

    ckpt_callback = None
    if checkpoint_every == 0:
        pass
    elif checkpoint_every == 1: # save every epoch
        ckpt_callpback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1)
    elif isinstance(checkpoint_every, int): # save every checkpoint_every epochs 
        ckpt_callpback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, period=checkpoint_every)
    elif isinstance(checkpoint_every, str): # parse string options
        if checkpoint_every[0].isnumeric():
            every, what = checkpoint_every.split(' ',1)
            every = int(every)
            if what.startswith('b'): # every batch
                ckpt_callpback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, save_freq=every)
            elif what.startswith('e'): # every epoch
                ckpt_callpback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, period=every)
            else:
                raise ValueError(f'Unrecognized value for {checkpoint_every = }')

        elif checkpoint_every.startswith('best'): # every best of something
            monitor = checkpoint_every.split(' ',1)[1]
            ckpt_callpback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=1)
        else:
            raise ValueError(f'Unrecognized value for {checkpoint_every = }')
    else:
        raise ValueError(f'Unrecognized value for {checkpoint_every = }')

    if ckpt_callpback is not None:
        additional_callbacks.append(ckpt_callpback)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.save_weights(ckpt_name.format(epoch=0))

    my_history=model.fit(X_tr, Y_tr, batch_size=batch_size, validation_data=(X_va,Y_va), shuffle=True,
                         callbacks=additional_callbacks, epochs=num_epochs, verbose=2, class_weight=None)

    model.save(folder)
    np.save(f'{folder}_history.npy', my_history.history)

def k_fold_cross_val_split(i, X, Y, nfolds=10, val_folds=1):
    '''
    Splits X and Y in a training and validation set according to k fold cross validation algorithm

    Parameters:
    -----------
        i: int, fold number from 0 to `nfolds`-1
        X: data
        Y: labels
        nfolds: number of folds
        val_folds: number of consecutive folds for the validation set (between 1 and `nfolds`-1). Default 1.

    Returns:
    --------
        X_tr, Y_tr, X_va, Y_va
    '''
    if i < 0 or i >= nfolds:
        raise ValueError(f'fold number i is out of the range [0, {nfolds - 1}]')
    if val_folds >= nfolds or val_folds <= 0:
        raise ValueError(f'val_folds out of the range [1, {nfolds - 1}]')
    fold_len = X.shape[0]//nfolds
    lower = i*fold_len % X.shape[0]
    upper = (i+val_folds) % X.shape[0]
    if lower < upper:
        X_va = X[lower:upper]
        Y_va = Y[lower:upper]
        X_tr = np.concatenate([X[upper:], X[:lower]], axis=0)
        Y_tr = np.concatenate([Y[upper:], Y[:lower]], axis=0)
    else: # upper overshoots
        X_va = np.concatenate([X[lower:], X[:upper]], axis=0)
        Y_va = np.concatenate([Y[lower:], Y[:upper]], axis=0)
        X_tr = X[upper:lower]
        Y_tr = Y[upper:lower]
    return X_tr, Y_tr, X_va, Y_va

def k_fold_cross_val(folder, X, Y, create_model_kwargs, load_from=None, nfolds=10, val_folds=1, u=1,
                     fullmetrics=True, training_epochs=40, training_epochs_tl=10, lr=1e-4, **kwargs):
    '''
    Performs k fold cross validation on a model architecture.

    Parameters:
    -----------
    folder: folder in which to save data related to the folds
    X: all data (train + val)
    Y: all labels
    create_model_kwargs: dictionary with the parameters to create a model
    load_from: from where to load weights for transfer learning. If not None it overrides `create_model_kwargs` (the model is loaded instead of created)
    nfolds: int, number of folds
    val_folds: number of folds to be used for the validation set for every split
    u: float, undersampling factor (>=1)
    fullmetrics: bool, whether to use a set of evaluation metrics or just the loss
    training_epochs: number of training epochs when creating a model from scratch
    training_epochs_tl: numer of training epochs when using transfer learning
    lr: learning_rate for Adam optimizer

    **kwargs: additional arguments to pass to `train_model` (see its docstring), in particular
        num_epochs: overrides `training_epochs` and `training_epochs_tl`
        optimizer: overrides `lr`
        loss: overrides the default SparseCrossEntropyLoss
        metrics: overrides `fullmetrics`
    '''
    folder = folder.rstrip('/')

    load_from = get_run(load_from, current_run_name=folder.rstrip('/').rsplit('/',1)[1])

    my_memory = []

    # find the optimal checkpoint
    opt_checkpoint = None
    if load_from is not None:
        load_from = load_from.rstrip('/')
        # Here we insert analysis of the previous training with the assessment of the ideal checkpoint
        history0 = np.load(f'{load_from}/fold_0_history.npy', allow_pickle=True).item()
        if 'val_CustomLoss' not in history0.keys():
            raise KeyError('val_CustomLoss not in history: cannot compute optimal checkpoint')
        historyCustom = [np.load(f'{load_from}/fold_{i}_history.npy', allow_pickle=True).item()['val_CustomLoss'] for i in range(nfolds)]
        historyCustom = np.mean(np.array(historyCustom),axis=0)
        opt_checkpoint = np.argmin(historyCustom) # We will use optimal checkpoint in this case!
        print(f'{opt_checkpoint = }')

    # k fold cross validation
    for i in range(nfolds):
        # split data
        X_tr, Y_tr, X_va, Y_va = k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)

        n_pos_tr = np.sum(Y_tr)
        n_neg_tr = len(Y_tr) - n_pos_tr
        print(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')

        # perform undersampling
        if u > 1:
            undersampling_strategy = n_pos_tr/(n_neg_tr/u)
            pipeline = Pipeline(steps=[('u', RandomUnderSampler(random_state=42, sampling_strategy=undersampling_strategy))])
            # reshape data to feed it to the pipeline
            X_tr_shape = X_tr.shape
            X_tr = X_tr.reshape(X_tr_shape[0], np.product(X_tr_shape[1:]))
            X_tr, Y_tr = pipeline.fit_resample(X_tr, Y_tr)
            X_tr = X_tr.reshape(X_tr.shape[0], *X_tr_shape[1:])
            n_pos_tr = np.sum(Y_tr)
            n_neg_tr = len(Y_tr) - n_pos_tr
            print(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')

        # renormalize data with pointwise mean and std
        X_mean = np.mean(X_tr, axis=0)
        X_std = np.std(X_tr, axis=0)
        print(f'{np.sum(X_std < 1e-5)/np.product(X_std.shape)*100}\% of the data have std below 1e-5')
        X_std[X_std==0] = 1 # If there is no variance we shouldn't divide by zero ### hmmm: this may create discontinuities

        # save X_mean and X_std
        np.save(f'{folder}/fold_{i}_X_mean.npy', X_mean)
        np.save(f'{folder}/fold_{i}_X_std.npy', X_std)

        X_tr = (X_tr - X_mean)/X_std
        X_va = (X_va - X_mean)/X_std

        print(f'{X_tr.shape = }, {X_va.shape = }')


        # check for transfer learning
        model = None        
        if load_from is None:
            model = create_model(input_shape=X_tr.shape[1:], **create_model_kwargs)
        else:
            model = keras.models.load_model(f'{load_from}/fold_{i}', compile=False)
            model.load_weights(f'{load_from}/fold_{i}/cp-{opt_checkpoint:04d}.ckpt')            
        print(model.summary())

        num_epochs = kwargs.pop('num_epochs', None)
        if num_epochs is None:
            if load_from is None:
                num_epochs = training_epochs
            else:
                num_epochs = training_epochs_tl

        tf_sampling = tf.cast([0.5*np.log(u), -0.5*np.log(u)], tf.float32)
        metrics = kwargs.pop('metrics', None)
        if metrics is None:
            if fullmetrics:
                metrics=['accuracy',tff.MCCMetric(2),tff.ConfusionMatrixMetric(2),tff.CustomLoss(tf_sampling)]#keras.metrics.SparseCategoricalCrossentropy(from_logits=True)]#CustomLoss()]   # the last two make the code run longer but give precise discrete prediction benchmarks
            else:
                metrics=['loss']
        fold_folder = f'{folder}/fold_{i}'
        optimizer = kwargs.pop('optimizer',keras.optimizers.Adam(learning_rate=lr))
        loss = kwargs.pop('loss',keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        train_model(model, X_tr, Y_tr, X_va, Y_va,
                    folder=fold_folder, num_epochs=num_epochs, optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

        my_memory.append(psutil.virtual_memory())
        print('RAM memory:', my_memory[i][3]) # Getting % usage of virtual_memory ( 3rd field)

        keras.backend.clear_session()
        gc.collect() # Garbage collector which removes some extra references to the objects
        
    np.save(f'{folder}/RAM_stats.npy', my_memory)


########## PUTTING THE PIECES TOGETHER ###########

def prepare_data(load_data_kwargs, make_XY_kwargs, roll_X_kwargs, premix_seed=0, nfolds=10):
    # load data
    found = False
    for field_name in load_data_kwargs['fields']:
        if field_name.startswith(make_XY_kwargs['label_field']):
            found = True
            break
    if not found:
        raise KeyError(f"field {make_XY_kwargs['label_field']} is not a loaded field")

    fields = load_data(**load_data_kwargs)

    X,Y = make_XY(fields, **make_XY_kwargs)
    
    # move greenwich_meridian
    X = roll_X(X, **roll_X_kwargs)

    # mixing
    premix_permutation = shuffle_years(X, seed=premix_seed, apply=False)
    Y = Y[premix_permutation]
    # balance folds:
    weights = np.sum(Y, axis=1) # get the number of heatwave events per year
    balance_permutation = balance_folds(weights,nfolds=nfolds)
    Y = Y[balance_permutation]
    tot_permutation = compose_permutations([premix_permutation, balance_permutation])
    X = X[tot_permutation]

    return X, Y, tot_permutation


def run(folder, prepare_data_kwargs, k_fold_cross_val_kwargs):
    folder = folder.rstrip('/')
    # prepare the data
    X,Y, permutation = prepare_data(**prepare_data_kwargs)
    np.save(f'{folder}/year_permutation.npy',permutation)

    # run kfold
    k_fold_cross_val(folder, X, Y, **k_fold_cross_val_kwargs)
    


if __name__ == '__main__':
    # check if there is a lock:
    lock = Path(__file__).resolve().parent / 'lock.txt'
    if os.path.exists(lock): # there is a lock
        # check for folder argument
        if len(sys.argv) < 2:
            print(usage())
            exit(0)
        if len(sys.argv) == 2:
            folder = sys.argv[1]
            print(f'moving code to {folder = }')
            move_to_folder(folder)

            d = build_config_dict([run])
            write_to_json(d,f'{folder}/config.json')

            exit(0)
        else:
            with open(lock) as l:
                raise ValueError(l.read())
    
    # if there is a lock, the previous block of code would have ended the run, so the code below is executed only if there is no lock
    
    # load config file
    config_dict = read_json('config.json')
    config_dict_flat = collapse_dict(config_dict)
    print(config_dict_flat)

    # parse command line arguments
    cl_args = sys.argv[1:]
    i = 0
    arg_dict = {}
    while(i < len(cl_args)):
        key = cl_args[i]
        if '=' in key:
            key, value = key.split('=')
            i += 1
        else:
            value = cl_args[i+1]
            i += 2
        if key not in config_dict_flat:
            raise KeyError(f'Unknown argument {key}')
        arg_dict[key] = value
    print(arg_dict)

    folder = ''
    for k in sorted(arg_dict):
        folder += f'{k}_{arg_dict[k]}__'
    folder = folder[:-2] # remove the last '__'
    print(folder)

    # add folder name to the list of runs
    # NOTE: enable transfer learning from previous run

    # run()

    

    