# '''
# Created in December 2021

# @author: Alessandro Lovo
# '''

'''
Module for training a Convolutional Neural Network on climate data.

Usage
-----
First you need to move the code to a desired folder by running
    python Learn2_new.py <folder>

This will copy this code and its dependencies to your desired location and will create a config file from the default values in the functions specified in this module.

`cd` into your folder and have a look at the config file, modify all the parameters you want BEFORE the first run, but AFTER the first successful run the config file becomes read-only. There is a reason for it, so don't try to modify it anyways!

The config file will store the default values for the arguments of the functions.

When running the code you can specify some parameters to deviate from their default value, for example running
    python Learn2_new.py tau=5

will run the code with all parameters at their default values but `tau` which will now be 5

Beware that arguments are parsed with spaces, so valid syntaxes are
    python Learn2_new.py tau=5 lr=1e-4
    python Learn2_new.py tau 5 lr 1e-4
Invalid syntaxes are:
    python Learn2_new.py tau=5 lr = 1e-4
    python Learn2_new.py tau=5, lr=1e-4

You can also provide arguments as lists, in that case the program will iterate over them. For example:
    python Learn2_new.py tau='[0,1,2]'

will perform three runs with tau=1, tau=2, tau=3.


Beware that you need to enclose the list inside a string or the terminal will complain. If you are passing a list of strings, use double apices, i.e.
    python Learn2.py area="['France', 'Scandinavia']"


If by default an argument is already a list, the provided list is not interpreted as something to be iterated over, for example the argument `fields` has default value ['t2m','zg500','mrso_filtered']. So running
    python Learn2.py fields="['t2m', 'zg500']"

will result in a single run performed with fields=['t2m', 'zg500']

If you provide more than one argument to iterate over, all combinations will be performed, e.g.:
    python Learn2.py fields="[['t2m'], ['t2m', 'zg500']]" tau='[1,2]'

will result in 4 runs:
    fields=['t2m'], tau=1
    fields=['t2m'], tau=2
    fields=['t2m', 'zg500'], tau=1
    fields=['t2m', 'zg500'], tau=2


Logging levels:
level   name                events

0       logging.NOTSET

10      logging.DEBUG

20      logging.INFO

30      logging.WARNING

35                          Which fold is running

40      logging.ERROR

41                          From where the models are loaded or created

42                          Folder name of the run
                            Single run completes

45                          Added and removed telegram logger
                            Tell number of scheduled runs
                            Skipping already performed run

49                          All runs completed

50      logging.CRITICAL    The program stops due to an error
'''

### IMPORT LIBRARIES #####

## general purpose
import os as os
from pathlib import Path
from stat import S_IREAD
import sys
import traceback
import warnings
import time
import shutil
import gc
import psutil
import numpy as np
import inspect
import ast
import logging

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO


## machine learning
# from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

## user defined modules
this_module = sys.modules[__name__]
path_to_here = Path(__file__).resolve().parent
path_to_ERA = path_to_here / 'ERA' # when absolute path, so you can run the script from another folder (outside plasim)
if not os.path.exists(path_to_ERA):
    path_to_ERA = path_to_here.parent / 'ERA'
    if not os.path.exists(path_to_ERA):
        raise FileNotFoundError('Could not find ERA folder')
        
# go to the parent so vscode is happy with code completion :)
path_to_ERA = path_to_ERA.parent
path_to_ERA = str(path_to_ERA)
logger.info(f'{path_to_ERA = }/ERA/')
if not path_to_ERA in sys.path:
    sys.path.insert(1, path_to_ERA)

import ERA.ERA_Fields_New as ef # general routines
import ERA.TF_Fields as tff # tensorflow routines
import ERA.utilities as ut


arg_sep = '--'
value_sep = '__'

########## USAGE ###############################
def usage(): 
    '''
    Returns the documentation of this module that explains how to use it.
    '''
    return this_module.__doc__

#### CONFIG FILE #####

def get_default_params(func, recursive=False):
    '''
    Given a function returns a dictionary with the default values of its parameters

    Parameters
    ----------
    func : callable
        to be able to use the recursive feature, the function must be defined inside this module
    recursive : bool, optional
        if True arguments of the type '*_kwargs' will be interpreted as kwargs to pass to function * and `get_default_params` will be applied to * as well to retrieve the default values
    
    Returns
    -------
    default_params : dict
        dictionary of the default parameter values

    Examples
    --------
    >>> get_default_params(balance_folds) # see function balance_folds
    {'nfolds': 10, 'verbose': False}

    >>> def pnroll(X, roll_X_kwargs, greeting='Hello there!'):
    ...     print(greeting)
    ...     return roll_X(X, **roll_X_kwargs)
    >>> get_default_params(pnroll, recursive=True)
    {'greeting': 'Hello there!', 'roll_X_kwargs': {'roll_axis': 'lon', 'roll_steps': 64}}
    '''
    s = inspect.signature(func)
    default_params = {
        k:v.default for k,v in s.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    if recursive: # look for parameters ending in '_kwargs' and extract further default arguments
        possible_other_params = [k for k,v in s.parameters.items() if (v.default is inspect.Parameter.empty and k.endswith('_kwargs'))]
        for k in possible_other_params:
            func_name = k.rsplit('_',1)[0] # remove '_kwargs'
            try:
                default_params[k] = get_default_params(getattr(this_module, func_name), recursive=True)
            except:
                logger.warning(f'From get_default_params:  Could not find function {func_name}')
    return default_params

def build_config_dict(functions):
    '''
    Creates a config file with the default arguments of the functions in the list `functions`. See also function `get_default_params`

    Parameters:
    -----------
    functions : list
        list of functions or string with the function name
    
    Returns
    -------
    d : dict

    Examples
    --------
    If calling on one function only it is the same as to use get_default_params recursively
    >>> d1 = build_config_dict([k_fold_cross_val])
    >>> d2 = get_default_params(k_fold_cross_val, recursive=True)
    >>> d1 == {'k_fold_cross_val': d2}
    True
    '''
    d = {}
    for f in functions:
        if isinstance(f, str):
            f_name = f
            f = getattr(this_module, f_name)
        else:
            f_name = f.__name__
        d[f'{f_name}_kwargs'] = get_default_params(f, recursive=True)
    return d

def check_config_dict(config_dict):
    '''
    Checks that the confic dictionary is consistent

    Returns
    -------
    config_dict_flat : dict
        flattened config_dict

    Raises
    ------
    KeyError
        if the config dictionary is inconsistent.
    '''
    try:
        config_dict_flat = ut.collapse_dict(config_dict) # in itself this checks for multiple values for the same key
        found = False
        for field_name in config_dict_flat['fields']:
            if field_name.startswith(config_dict_flat['label_field']):
                found = True
                break
        if not found:
            raise KeyError(f"field {config_dict_flat['label_field']} is not a loaded field")
    except Exception as e:
        raise KeyError('Invalid config dictionary') from e
    return config_dict_flat

def parse_run_name(run_name):
    '''
    Parses a string into a dictionary

    Parameters
    ----------
    run_name: str
        run name formatted as *<param_name>_<param_value>__*
    
    Returns
    -------
    d: dict
        Values of the dictionary are strings
    
    Examples
    --------
    >>> parse_run_name('a_5__b_7')
    {'a': '5', 'b': '7'}
    >>> parse_run_name('test_arg_bla__b_7')
    {'test_arg': 'bla', 'b': '7'}
    '''
    d = {}
    args = run_name.split(arg_sep)
    for arg in args:
        if '_' not in arg:
            continue
        key, value = arg.rsplit(value_sep,1)
        d[key] = value
    return d

def get_run(load_from, current_run_name=None):
    '''
    Parameters
    ----------
    load_from : dict, int, str, 'last' or None
        If dict it is a dictionary with arguments of the run. If int it is the number of the run.
        If 'last' it is the last completed run. Otherwise it can be a piece of the run name. If None the function returns None
        If the choice is ambiguous an error will be raised.
    current_run_name : str, optional
        used to check for compatibility issues when loading a model
    
    Returns
    -------
    run_name : str
        name of the run from which to load
        If there are no runs to load from or `load_from` = None, None is returned instead of raising an error

    Raises
    ------
    KeyError
        if run_name is not found or the choice is ambiguous
    '''
    if load_from is None:
        return None

    create_model_keys = list(get_default_params(create_model).keys()) + ['nfolds', 'fields'] # arguments relevant for model architecture

    def check(run_name, current_run_name=None):
        if current_run_name is None:
            return True
        # parse run_name for arguments
        run_dict = parse_run_name(run_name)
        current_run_dict = parse_run_name(current_run_name)
        # keep only arguments that are model kwargs
        run_dict = {k:v for k,v in run_dict.items() if k in create_model_keys}
        current_run_dict = {k:v for k,v in current_run_dict.items() if k in create_model_keys}
        return run_dict == current_run_dict
    
    runs = ut.json2dict('runs.json')

    # select only completed runs
    runs = {k: v for k,v in runs.items() if v['status'] == 'COMPLETED'}

    # select only compatible runs
    runs = {k: v for k,v in runs.items() if check(v['name'], current_run_name)}

    if len(runs) == 0:
        logger.warning('No valid runs to load from')
        return None
    if isinstance(load_from, dict):
        found = False
        for i,r in runs.items():
            if load_from.items() <= r['args']: # check if the provided arguments are a subset of the run argument
                if not found:
                    found = True
                    l = int(i)
                else: # ambiguity
                    raise KeyError(f"Multiple runs contain {load_from}, at least {l} and {i}")
        if not found:
            raise KeyError(f'No previous compatible run has {load_from}')
    elif isinstance(load_from, int):
        l = load_from
    elif isinstance(load_from, str):
        try:
            l = int(load_from) # run number
        except ValueError: # cannot convert load_from to int, so it must be a string that doesn't contain only numbers
            if load_from == 'last':
                l = -1
            else:
                load_from_dict = parse_run_name(load_from)
                found = False
                for i,r in runs.items():
                    r_dict = parse_run_name(r['name']) # cannot use directly r['args'] because of types (we need argument values in string format)
                    if load_from_dict.items() <= r_dict.items(): # check if the provided arguments are a subset of the run argument
                        if not found:
                            found = True
                            l = int(i)
                        else: # ambiguity
                            raise KeyError(f'Multiple runs contain {load_from_dict}, at least {l} and {i}')
                if not found:
                    raise KeyError(f'No previous compatible run has {load_from_dict}')
    else:
        raise TypeError(f'Unsupported type {type(load_from)} for load_from')
    # now l is an int
    if l < 0:
        r = list(runs.values())[l]
    else:
        r = runs[str(l)]
    run_name = r['name']
    
    return run_name

########## COPY SOURCE FILES #########

def move_to_folder(folder):
    '''
    Copies this file and its dependencies to a given folder.
    '''
    folder = Path(folder).resolve()
    ERA_folder = folder / 'ERA' # GM: container of the useful routines in a subfolder "folder/ERA". The abbreviation comes from the original routines deveoped for ERA5 reanalysis

    if os.path.exists(ERA_folder):
        raise FileExistsError(f'Cannot copy scripts to {folder}: you already have some there')
    ERA_folder.mkdir(parents=True,exist_ok=True)

    # copy this file
    path_to_here = Path(__file__).resolve() # path to this file
    shutil.copy(path_to_here, folder)

    # copy other files in the same directory as this one
    path_to_here = path_to_here.parent

    # copy useful files from ../ERA/ to folder/ERA/
    path_to_here = path_to_here.parent / 'ERA'
    shutil.copy(path_to_here / 'cartopy_plots.py', ERA_folder)
    shutil.copy(path_to_here / 'ERA_Fields.py', ERA_folder)
    shutil.copy(path_to_here / 'TF_Fields.py', ERA_folder)
    shutil.copy(path_to_here / 'utilities.py', ERA_folder)

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

@ut.execution_time  # GM: I guess the point is to measure elapsed time but the way this works is not transparent to me yet
@ut.indent_logger(logger)   # GM: same, I guess the idea is to ensure something about print statements
def load_data(dataset_years=1000, year_list=None, sampling='', Model='Plasim', area='France', filter_area='France',
              lon_start=0, lon_end=128, lat_start=0, lat_end=22, mylocal='/local/gmiloshe/PLASIM/',fields=['t2m','zg500','mrso_filtered']):
    '''
    Loads the data into Plasim_Fields objects

    Parameters
    ----------
    dataset_years : int, optional
        number of years of the dataset, for now 8000 or 1000.
    year_list : list or None, optional
        list of years to load from the dataset. If None all years are loaded
    sampling : str, optional
        '' (dayly) or '3hrs'
    Model : str, optional
        'Plasim', 'CESM', ... For now only Plasim is implemented
    area : str, optional
        region of interest, e.g. 'France'
    filter_area : str, optionla
        area over which to keep filtered fields, ususlly the same of `area`
    lon_start, lon_end, lat_start, lat_end : int
        longitude and latitude extremes of the data expressed in indices (model specific)
    mylocal : str or Path, optional
        path the the data storage. For speed it is better if it is a local path.
    fields : list, optional
        list of field names to be loaded. Add '_filtered' to the name to have the values of the field outside `filter_area` set to zero.

    Returns
    -------
    _fields: dict
        dictionary of ERA_Fields.Plasim_Field objects
    '''

    if area != filter_area:
        warnings.warn(f'Fields will be filtered on a different area ({filter_area}) than the region of interest ({area})')

    if dataset_years == 1000:
        dataset_suffix = ''
    elif dataset_years == 8000:
        dataset_suffix = '_LONG'
    else:
        raise ValueError(f'Invalid number of {dataset_years = }')
   

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
        field.load_field(mylocal+file_suffix, year_list=year_list)
        # Set area integral
        field.abs_area_int, field.ano_area_int = field.Set_area_integral(area,mask,containing_folder=None) # don't save area integrals in order to avoid conflicts between different runs
        # filter
        if do_filter: # set to zero all values outside `filter_area`
            filter_mask = ef.create_mask(Model, filter_area, field.var, axes='last 2', return_full_mask=True)
            field.var *= filter_mask

        _fields[field_name] = field  
    
    return _fields

@ut.execution_time
@ut.indent_logger(logger)
def assign_labels(field, time_start=30, time_end=120, T=14, percent=5, threshold=None):
    '''
    Given a field of anomalies it computes the `T` days forward convolution of the integrated anomaly and assigns label 1 to anomalies above a given `threshold`.
    If `threshold` is not provided, then it is computed from `percent`, namely to identify the `percent` most extreme anomalies.

    Parameters
    ----------
    field : Plasim_Field object
    time_start : int, optional
        first day of the period of interest
    time_end : int, optional
        first day after the end of the period of interst
    T : int, optional
        width of the window for the running average
    percent : float, optional
        percentage of the most extreme heatwaves
    threshold : float, optional
        if provided overrides `percent`.

    Returns:
    --------
    labels : np.ndarray
        2D array with shape (years, days) and values 0 or 1
    '''
    A, A_flattened, threshold =  field.ComputeTimeAverage(time_start, time_end, T=T, percent=percent, threshold=threshold)[:3]
    return np.array(A >= threshold, dtype=int)

@ut.execution_time
@ut.indent_logger(logger)
def make_X(fields, time_start=30, time_end=120, T=14, tau=0):
    '''
    Cuts the fields in time and stacks them. The original fields are not modified

    Parameters
    ----------
    fields : dict of Plasim_Field objects
    time_start : int, optional
        first day of the period of interest
    time_end : int, optional
        first day after the end of the period of interst
    T : int, optional
        width of the window for the running average
    tau : int, optional
        delay between observation and prediction

    Returns
    -------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    '''
    # stack the fields
    X = np.array([field.var[:, time_start+tau:time_end+tau-T+1, ...] for field in fields.values()]) # NOTE: maybe chenge to -tau
    # now transpose the array so the field index becomes the last
    X = X.transpose(*range(1,len(X.shape)), 0)
    return X

@ut.execution_time
@ut.indent_logger(logger)
def make_XY(fields, label_field='t2m', time_start=30, time_end=120, T=14, tau=0, percent=5, threshold=None):
    '''
    Combines `make_X` and `assign_labels`

    Parameters:
    -----------
    fields : dict of Plasim_Field objects
    label_field : str, optional
        key for the field used for computing labels
    time_start : int, optional
        first day of the period of interest
    time_end : int, optional
        first day after the end of the period of interst
    T : int, optional
        width of the window for the running average
    tau : int, optional
        delay between observation and prediction
    percent : float, optional
        percentage of the most extreme heatwaves
    threshold : float, optional
        if provided overrides `percent`

    Returns:
    --------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    Y : np.ndarray
        with shape (years, days)
    '''
    X = make_X(fields, time_start=time_start, time_end=time_end, T=T, tau=tau)
    Y = assign_labels(fields[label_field], time_start=time_start, time_end=time_end, T=T, percent=percent, threshold=threshold)
    return X,Y

@ut.execution_time
@ut.indent_logger(logger)
def roll_X(X, roll_axis='lon', roll_steps=64):
    '''
    Rolls `X` along a given axis. useful for example for moving France away from the Greenwich meridian.
    In other words this allows one, for example, to shift the grid so that desired areas are not found at the boundary.
    In principle this function allows us to roll along arbitrary axis, including days or years.

    Parameters
    ----------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    axis : str, optional
        'year' (or 'y'), 'day' (or 'd'), 'lat', 'lon', 'field' (or 'f')
    steps : int, optional
        number of gridsteps to roll: a positive value for 'steps' means that the elements of the array are moved forward in it,
        e.g. `steps` = 1 means that the old first element is now in the second place
        This means that for every axis a positive value of `steps` yields a shift of the array
        'year', 'day' : forward in time
        'lat' : southward
        'lon' : eastward
        'field' : forward in the numbering of the fields
    
    Returns
    -------
    X_rolled : np.ndarray
        of the same shape of `X`
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

def shuffle_years(X, permutation=None, seed=0, apply=False):
    '''
    Permutes `X` along the first axis

    Parameters
    ----------
    X : np.ndarray
        array with the data to permute
    permutation : None or 1D np.ndarray, optional,
        must be a permutation of an array of `np.arange(X.shape[0])`
    seed : int, optional
        if `permutation` is None, then it is computed using the provided seed.
    apply : bool, optional
        if True the function returns the permuted data, otherwise the permutation is returned
    
    Returns
    -------
    if apply:
        np.ndarray of the same shape of X
    else:
        np.ndarray of shape (X.shape[0],)
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

@ut.execution_time
@ut.indent_logger(logger)
def balance_folds(weights, nfolds=10, verbose=False):
    '''
    Returns a permutation that, once applied to `weights` would make the consecutive `nfolds` pieces of equal length have their sum the most similar to each other.

    Parameters
    ----------
    weights : 1D array-like
    nfolds : int, optional
        must be a divisor of `len(weights)`
    verbose : bool, optional

    Returns
    -------
    permutation : permutation of `np.arange(len(weights))`
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
                if verbose:
                    logger.info(f'fold {self.name} done!')
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
    if verbose:
        logger.info('Balancing folds')
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
        logger.info(f'Sums of the balanced {nfolds} folds:\n{sums}\nstd/avg = {np.std(sums)/target_sum}\nmax relative deviation = {np.max(np.abs(sums - target_sum))/target_sum*100}\%')

    return permutation


########## NEURAL NETWORK DEFINITION ###########

def create_model(input_shape, conv_channels=[32,64,64], kernel_sizes=3, strides=1,
                 batch_normalizations=True, conv_activations='relu', conv_dropouts=0.2, max_pool_sizes=[2,2,False],
                 dense_units=[64,2], dense_activations=['relu', None], dense_dropouts=[0.2,False]):
    '''
    Creates a model consisting of a series of convolutional layers followed by fully connected ones

    Parameters
    ----------
    input_shape : tuple
        shape of input data excluding the data_ID axis
    conv_channels : list of int, optional
        number of channels corresponding to the convolutional layers
    kernel_sizes : int, 2-tuple or list of ints or 2-tuples, optional
        If list must be of the same size of `conv_channels`
    strides : int, 2-tuple or list of ints or 2-tuples, optional
        same as kernel_sizes
    batch_normalizations : bool or list of bools, optional
        whether to add a BatchNormalization layer after each Conv2D layer
    conv_activations : str or list of str, optional
        activation functions after each convolutional layer
    conv_dropouts : float in [0,1] or list of floats in [0,1], optional
        dropout to be applied after the BatchNormalization layer. If 0 no dropout is applied
    max_pool_sizes : int or list of int, optional
        size of max pooling layer to be applied after dropout. If 0 no max pool is applied

    dense_units : list of int, optional
        number of neurons for each fully connected layer
    dense_activations : str or list of str, optional
        activation functions after each fully connected layer
    dense_dropouts : float in [0,1] or list of floats in [0,1], optional

    Returns
    -------
    model : keras.models.Model
    '''
    model = models.Sequential()

    # convolutional layers
    # adjust the shape of the arguments to be of the same length as conv_channels
    args = [kernel_sizes, strides, batch_normalizations, conv_activations, conv_dropouts, max_pool_sizes]
    for j,arg in enumerate(args):
        if not isinstance(arg, list):
            args[j] = [arg]*len(conv_channels)
        elif len(arg) != len(conv_channels):
            raise ValueError(f'Invalid length for argument {arg}')
    logger.info(f'convolutional args = {args}')
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


###### TRAINING THE NETWORK ############
@ut.execution_time
@ut.indent_logger(logger)
def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics,
                batch_size=1024, checkpoint_every=1, additional_callbacks=['csv_logger']):
    '''
    Trains a given model checkpointing its weights

    Parameters
    ----------
    model : keras.models.Model
    X_tr : np.ndarray
        training data
    Y_tr : np.ndarray
        training labels
    X_va : np.ndarray
        validation data
    Y_va : np.ndarray
        validation labels
    folder : str or Path
        loaction where to save the checkpoints of the model
    num_epochs : int
        number of maximum epochs for the training
    optimizer : keras.Optimizer object
    loss : keras.losses.Loss object
    metrics : list of keras.metrics.Metric or str
    batch_size : int, optional
    checkpoint_every : int or str, optional
        Examples:
        0: disabled
        5 or '5 epochs' or '5 e': every 5 epochs
        '100 batches' or '100 b': every 100 batches
        'best custom_loss': every time 'custom_loss' reaches a new minimum. 'custom_loss' must be in the list of metrics
    additional_callbacks : list of keras.callbacks.Callback objects or list of str, optional
        for example EarlyStopping
        string items are interpreted, for example 'csv_logger' creates a CSVLogger callback
    '''
    folder = folder.rstrip('/')
    ckpt_name = folder + '/cp-{epoch:04d}.ckpt'
    
    callbacks = []
    if additional_callbacks is not None:
        for cb in additional_callbacks:
            if isinstance(cb, str):
                if cb.lower().startswith('csv'):
                    callbacks.append(keras.callbacks.CSVLogger(f'{folder}/history.csv', append=True))
                else:
                    raise ValueError(f'Unable to understand callback {cb}')
            else:
                callbacks.append(cb)

    ckpt_callback = None
    if checkpoint_every == 0:
        pass
    elif checkpoint_every == 1: # save every epoch
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1)
    elif isinstance(checkpoint_every, int): # save every checkpoint_every epochs 
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, period=checkpoint_every)
    elif isinstance(checkpoint_every, str): # parse string options
        if checkpoint_every[0].isnumeric():
            every, what = checkpoint_every.split(' ',1)
            every = int(every)
            if what.startswith('b'): # every batch
                ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, save_freq=every)
            elif what.startswith('e'): # every epoch
                ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1, period=every)
            else:
                raise ValueError(f'Unrecognized value for {checkpoint_every = }')

        elif checkpoint_every.startswith('best'): # every best of something
            monitor = checkpoint_every.split(' ',1)[1]
            ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=1)
        else:
            raise ValueError(f'Unrecognized value for {checkpoint_every = }')
    else:
        raise ValueError(f'Unrecognized value for {checkpoint_every = }')

    if ckpt_callback is not None:
        callbacks.append(ckpt_callback)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.save_weights(ckpt_name.format(epoch=0))

    my_history=model.fit(X_tr, Y_tr, batch_size=batch_size, validation_data=(X_va,Y_va), shuffle=True,
                         callbacks=callbacks, epochs=num_epochs, verbose=2, class_weight=None)

    model.save(folder)
    np.save(f'{folder}/history.npy', my_history.history)

@ut.execution_time
@ut.indent_logger(logger)
def k_fold_cross_val_split(i, X, Y, nfolds=10, val_folds=1):
    '''
    Splits X and Y in a training and validation set according to k fold cross validation algorithm

    Parameters
    ----------
    i : int
        fold number from 0 to `nfolds`-1
    X : np.ndarray
        data
    Y : np.ndarray
        labels
    nfolds : int, optional
        number of folds
    val_folds : int, optional
        number of consecutive folds for the validation set (between 1 and `nfolds`-1). Default 1.

    Returns
    -------
    X_tr : np.ndarray
        training data
    Y_tr : np.ndarray
        training labels
    X_va : np.ndarray
        validation data
    Y_va : np.ndarray
        validation labels
    '''
    if i < 0 or i >= nfolds:
        raise ValueError(f'fold number i is out of the range [0, {nfolds - 1}]')
    if val_folds >= nfolds or val_folds <= 0:
        raise ValueError(f'val_folds out of the range [1, {nfolds - 1}]')
    fold_len = X.shape[0]//nfolds
    lower = i*fold_len % X.shape[0]
    upper = (i+val_folds)*fold_len % X.shape[0]
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

@ut.execution_time
@ut.indent_logger(logger)
def k_fold_cross_val(folder, X, Y, create_model_kwargs, train_model_kwargs, load_from='last', nfolds=10, val_folds=1, u=1,
                     fullmetrics=True, training_epochs=40, training_epochs_tl=10, lr=1e-4):
    '''
    Performs k fold cross validation on a model architecture.

    Parameters
    ----------
    folder : str
        folder in which to save data related to the folds
    X : np.ndarray
        all data (train + val)
    Y : np.ndarray
        all labels
    create_model_kwargs : dict
        dictionary with the parameters to create a model
    train_model_kwargs : dict
        dictionary with the parameters to train a model
        The follwing special arguments will override other parameters of this function:
            num_epochs: overrides `training_epochs` and `training_epochs_tl`
            optimizer: overrides `lr`
            loss: overrides the default SparseCrossEntropyLoss
            metrics: overrides `fullmetrics`
    load_from : None, int, str or 'last', optional
        from where to load weights for transfer learning. See the documentation of function `get_run`
        If not None it overrides `create_model_kwargs` (the model is loaded instead of created)
    nfolds : int, optional
        number of folds
    val_folds : int, optional
        number of folds to be used for the validation set for every split
    u : float, optional
        undersampling factor (>=1). If = 1 no undersampling is performed
    fullmetrics : bool, optional
        whether to use a set of evaluation metrics or just the loss
    training_epochs : int, optional
        number of training epochs when creating a model from scratch
    training_epochs_tl : int, optional 
        numer of training epochs when using transfer learning
    lr : float, optional
        learning_rate for Adam optimizer       
    '''
    folder = folder.rstrip('/')

    load_from = get_run(load_from, current_run_name=folder.rsplit('/',1)[-1])
    if load_from is None:
        logger.log(41, 'Models will be trained from scratch')
    else:
        logger.log(41, f'Models will be loaded from {load_from}')

    my_memory = []

    # find the optimal checkpoint
    opt_checkpoint = None
    if load_from is not None:
        load_from = load_from.rstrip('/')
        # Here we insert analysis of the previous training with the assessment of the ideal checkpoint
        history0 = np.load(f'{load_from}/fold_0/history.npy', allow_pickle=True).item()
        if 'val_CustomLoss' not in history0.keys():
            raise KeyError('val_CustomLoss not in history: cannot compute optimal checkpoint')
        historyCustom = [np.load(f'{load_from}/fold_{i}/history.npy', allow_pickle=True).item()['val_CustomLoss'] for i in range(nfolds)]
        historyCustom = np.mean(np.array(historyCustom),axis=0)
        opt_checkpoint = np.argmin(historyCustom) + 1 # We will use optimal checkpoint in this case! Add 1 because epochs start from 1
        logger.info(f'{opt_checkpoint = }')

    # k fold cross validation
    for i in range(nfolds):
        logger.info('=============')
        logger.log(35, f'fold {i} ({i+1}/{nfolds})')
        logger.info('=============')
        # create fold_folder
        fold_folder = f'{folder}/fold_{i}'
        os.mkdir(fold_folder)

        # split data
        X_tr, Y_tr, X_va, Y_va = k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)

        n_pos_tr = np.sum(Y_tr)
        n_neg_tr = len(Y_tr) - n_pos_tr
        logger.info(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')

        # perform undersampling
        if u > 1:
            undersampling_strategy = n_pos_tr/(n_neg_tr/u)
            if undersampling_strategy > 1:
                # print(f'Too high undersmapling factor, maximum for this dataset is u={n_neg_tr/n_pos_tr}')
                # print(f'using the maximum undersampling instead')
                # undersampling_strategy = 1
                # u = n_neg_tr/n_pos_tr
                raise ValueError(f'Too high undersmapling factor, maximum for this dataset is u={n_neg_tr/n_pos_tr}')
            pipeline = Pipeline(steps=[('u', RandomUnderSampler(random_state=42, sampling_strategy=undersampling_strategy))])
            # reshape data to feed it to the pipeline
            X_tr_shape = X_tr.shape
            X_tr = X_tr.reshape((X_tr_shape[0], np.product(X_tr_shape[1:])))
            X_tr, Y_tr = pipeline.fit_resample(X_tr, Y_tr)
            X_tr = X_tr.reshape((X_tr.shape[0], *X_tr_shape[1:]))
            n_pos_tr = np.sum(Y_tr)
            n_neg_tr = len(Y_tr) - n_pos_tr
            logger.info(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')

        # renormalize data with pointwise mean and std
        X_mean = np.mean(X_tr, axis=0)
        X_std = np.std(X_tr, axis=0)
        logger.info(f'{np.sum(X_std < 1e-5)/np.product(X_std.shape)*100}\% of the data have std below 1e-5')
        X_std[X_std==0] = 1 # If there is no variance we shouldn't divide by zero ### hmmm: this may create discontinuities

        # save X_mean and X_std
        np.save(f'{fold_folder}/X_mean.npy', X_mean)
        np.save(f'{fold_folder}/X_std.npy', X_std)

        X_tr = (X_tr - X_mean)/X_std
        X_va = (X_va - X_mean)/X_std

        logger.info(f'{X_tr.shape = }, {X_va.shape = }')


        # check for transfer learning
        model = None        
        if load_from is None:
            model = create_model(input_shape=X_tr.shape[1:], **create_model_kwargs)
        else:
            model = keras.models.load_model(f'{load_from}/fold_{i}', compile=False)
            model.load_weights(f'{load_from}/fold_{i}/cp-{opt_checkpoint:04d}.ckpt')
        summary_buffer = ut.Buffer()
        summary_buffer.append('\n')
        model.summary(print_fn = lambda x: summary_buffer.append(x + '\n'))
        logger.info(summary_buffer.msg)

        num_epochs = train_model_kwargs.pop('num_epochs', None)
        if num_epochs is None:
            if load_from is None:
                num_epochs = training_epochs
            else:
                num_epochs = training_epochs_tl

        tf_sampling = tf.cast([0.5*np.log(u), -0.5*np.log(u)], tf.float32)
        metrics = train_model_kwargs.pop('metrics', None)
        if metrics is None:
            if fullmetrics:
                metrics=['accuracy',tff.MCCMetric(2),tff.ConfusionMatrixMetric(2),tff.CustomLoss(tf_sampling)]#keras.metrics.SparseCategoricalCrossentropy(from_logits=True)]#CustomLoss()]   # the last two make the code run longer but give precise discrete prediction benchmarks
            else:
                metrics=['loss']
        optimizer = train_model_kwargs.pop('optimizer',keras.optimizers.Adam(learning_rate=lr))
        loss = train_model_kwargs.pop('loss',keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        train_model(model, X_tr, Y_tr, X_va, Y_va,
                    folder=fold_folder, num_epochs=num_epochs, optimizer=optimizer, loss=loss, metrics=metrics, **train_model_kwargs)

        my_memory.append(psutil.virtual_memory())
        logger.info(f'RAM memory: {my_memory[i][3]:.3e}') # Getting % usage of virtual_memory ( 3rd field)

        keras.backend.clear_session()
        gc.collect() # Garbage collector which removes some extra references to the objects
        
    np.save(f'{folder}/RAM_stats.npy', my_memory)


########## PUTTING THE PIECES TOGETHER ###########
@ut.execution_time
@ut.indent_logger(logger)
def prepare_XY(fields, make_XY_kwargs, roll_X_kwargs, do_premix=False, premix_seed=0, do_balance_folds=True, nfolds=10, flatten_time_axis=True):
    '''
    Performs all operations to extract from the fields X and Y ready to be fed to the neural network.

    Parameters
    ----------
    fields : dict of ef.Plasim_Field objects
    make_XY_kwargs : dict
        arguments to pass to the function `make_XY`
    roll_X_kwargs : dict
        arguments to pass to the function `roll_X`
    premix_seed : int, optional
        seed for premixing, by default 0
    nfolds : int, optional
        necessary for balancing folds
    flatten_time_axis : bool, optional
        whether to flatten the time axis consisting of years and days

    Returns
    -------
    X : np.ndarray
        data. If flatten_time_axis with shape (days, lat, lon, fields), else (years, days, lat, lon, fields)
    Y : np.ndarray 
        labels. If flatten_time_axis with shape (days,), else (years, days)
    tot_permutation : np.ndarray
        with shape (years,), final permutaion of the years that reproduces X and Y once applied to the just loaded data
    '''
    X,Y = make_XY(fields, **make_XY_kwargs)
    
    # move greenwich_meridian
    X = roll_X(X, **roll_X_kwargs)

    # mixing
    logger.info('Mixing')
    start_time = time.time()
    tot_permutation = None

    # premixing
    if do_premix:
        premix_permutation = shuffle_years(X, seed=premix_seed, apply=False)
        Y = Y[premix_permutation]
        tot_permutation = premix_permutation

    # balance folds:
    if do_balance_folds:
        weights = np.sum(Y, axis=1) # get the number of heatwave events per year
        balance_permutation = balance_folds(weights,nfolds=nfolds, verbose=True)
        Y = Y[balance_permutation]
        if tot_permutation is None:
            tot_permutation = balance_permutation
        else:
            tot_permutation = ut.compose_permutations([tot_permutation, balance_permutation])

    # apply permutation to X
    if tot_permutation is not None:    
        X = X[tot_permutation]
    logger.info(f'Mixing completed in {ut.pretty_time(time.time() - start_time)}\n')
    logger.info(f'{X.shape = }, {Y.shape = }')

    if flatten_time_axis:
        X = X.reshape((X.shape[0]*X.shape[1],*X.shape[2:]))
        Y = Y.reshape((Y.shape[0]*Y.shape[1]))
        logger.info(f'Flattened time: {X.shape = }, {Y.shape = }')

    return X, Y, tot_permutation


@ut.execution_time
@ut.indent_logger(logger)
def prepare_data(load_data_kwargs, prepare_XY_kwargs):
    '''
    Combines all the steps from loading the data to the creation of X and Y

    Parameters
    ----------
    load_data_kwargs: dict
        arguments to pass to the function `load_data`
    prepare_XY_kwargs: dict
        arguments to pass to the function `prepare_XY`
        
    Returns
    -------
    X : np.ndarray
        data. If flatten_time_axis with shape (days, lat, lon, fields), else (years, days, lat, lon, fields)
    Y : np.ndarray 
        labels. If flatten_time_axis with shape (days,), else (years, days)
    tot_permutation : np.ndarray
        with shape (years,), final permutaion of the years that reproduces X and Y once applied to the just loaded data
    '''
    # load data
    fields = load_data(**load_data_kwargs)

    return prepare_XY(fields, **prepare_XY_kwargs)


def run(folder, prepare_data_kwargs, k_fold_cross_val_kwargs, log_level=logging.INFO):
    '''
    Perfroms a single full run

    Parameters:
    -----------
    folder : str
        folder where to perform the run
    prepare_data_kwargs : dict
        arguments to pass to the `prepare_data` function
    k_fold_cross_val_kwargs : dict
        arguments to pass to the `k_fold_cross_val` function
    '''
    label_field = ut.extract_nested(prepare_data_kwargs, 'label_field')
    for field_name in prepare_data_kwargs['load_data_kwargs']['fields']:
        if field_name.startswith(label_field):
            found = True
            break
    if not found:
        raise KeyError(f"field {label_field} is not a loaded field")
    start_time = time.time()
    folder = folder.rstrip('/')
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        raise FileExistsError(f'{folder} already exists')
    # setup logger
    fh = logging.FileHandler(f'{folder}/log.log')
    fh.setLevel(log_level)
    logger.handlers.append(fh)

    # check tf version and GPUs
    logger.info(f"{tf.__version__ = }")
    if int(tf.__version__[0]) < 2:
        logger.info(f"{tf.test.is_gpu_available() = }")
        GPU = tf.test.is_gpu_available()
    else:
        logger.info(f"{tf.config.list_physical_devices('GPU') = }")
        GPU = len(tf.config.list_physical_devices('GPU'))
    if not GPU:
        warnings.warn('\n\nThis machine does not have a GPU: training may be very slow\n\n')

    # prepare the data
    X,Y, permutation = prepare_data(**prepare_data_kwargs)
    if permutation is not None:
        np.save(f'{folder}/year_permutation.npy', permutation)

    logger.info(f'{X.shape = }, {Y.shape = }')
    # flatten the time axis
    X = X.reshape((X.shape[0]*X.shape[1],*X.shape[2:]))
    Y = Y.reshape((Y.shape[0]*Y.shape[1]))
    logger.info(f'Flattened time: {X.shape = }, {Y.shape = }')

    # run kfold
    k_fold_cross_val(folder, X, Y, **k_fold_cross_val_kwargs)

    logger.info(f'\ntotal run time: {ut.pretty_time(time.time() - start_time)}')

    # remove logger
    logger.handlers.remove(fh)


###### EFFICIENT MANAGEMENT OF MULTIPLE RUNS #######

class Trainer():
    def __init__(self):
        # load config file and parse arguments
        self.config_dict = ut.json2dict('config.json')
        self.config_dict_flat = check_config_dict(self.config_dict)
        
        self.fields = None
        self.X = None
        self.Y = None
        self.year_permutation = None

        # extract default arguments for each function
        self.default_run_kwargs = ut.extract_nested(self.config_dict, 'run_kwargs')
        self.telegram_kwargs = ut.extract_nested(self.config_dict, 'telegram_kwargs')

        # setup last evaluation arguments
        self._load_data_kwargs = None
        self._prepare_XY_kwargs = None

        self.scheduled_kwargs = None

        # check tf version and GPUs
        print(f"{tf.__version__ = }")
        if int(tf.__version__[0]) < 2:
            print(f"{tf.test.is_gpu_available() = }")
            GPU = tf.test.is_gpu_available()
        else:
            print(f"{tf.config.list_physical_devices('GPU') = }")
            GPU = len(tf.config.list_physical_devices('GPU'))
        if not GPU:
            warnings.warn('\nThis machine does not have a GPU: training may be very slow\n')

    def schedule(self, **kwargs):
        '''
        Here kwargs can be iterables. This function schedules several runs and calls on each of them `self._run`
        You can also set telegram kwargs with this function.
        '''

        # detect variables over which to iterate
        iterate_over = []
        non_iterative_kwargs = {}
        for k,v in kwargs.items():
            if k not in self.config_dict_flat:
                raise KeyError(f'Invalid argument {k}')
            if k in self.telegram_kwargs:
                self.telegram_kwargs[k] = v
                continue
            iterate = False
            if isinstance(v, list): # possible need to iterate over the argument
                if isinstance(self.config_dict_flat[k], list):
                    if isinstance(v[0], list):
                        iterate = True
                else:
                    iterate = True
            if iterate:
                iterate_over.append(k)
            elif v != self.config_dict_flat[k]: # skip parameters already at their default value
                non_iterative_kwargs[k] = v

        new_iterate_over = []
        # arguments for loading fields
        to_add = []
        for k in iterate_over:
            if k in self.default_run_kwargs['load_data_kwargs']:
                to_add.append(k)
        new_iterate_over += to_add
        for k in to_add:
            iterate_over.remove(k)
        # arguments for preparing XY
        to_add = []
        for k in iterate_over:
            if k in self.default_run_kwargs['prepare_XY_kwargs']:
                to_add.append(k)
        new_iterate_over += to_add
        for k in to_add:
            iterate_over.remove(k)
        # remaining arguments
        new_iterate_over += iterate_over
        
        iterate_over = new_iterate_over

        iteration_values = [kwargs[k] for k in iterate_over]
        # expand the iterations into a list
        iteration_values = list(zip(*[m.flatten() for m in np.meshgrid(*iteration_values, indexing='ij')]))
        # ensure json serialazability by converting to string and back
        iteration_values = ast.literal_eval(str(iteration_values))

        self.scheduled_kwargs = [{**non_iterative_kwargs, **{k: l[i] for i,k in enumerate(iterate_over)}} for l in iteration_values]

        if len(self.scheduled_kwargs) == 0:
            self.scheduled_kwargs = [non_iterative_kwargs]
            if len(non_iterative_kwargs) == 0:
                logger.info('Scheduling 1 run at default values')
            else:
                logger.info(f'Scheduling 1 run at values {non_iterative_kwargs}')
        else:
            logger.info(f'Scheduled the following {len(self.scheduled_kwargs)} runs:')
            for i,kw in enumerate(self.scheduled_kwargs):
                logger.info(f'{i}: {kw}')
    
    def telegram(self, telegram_bot_token='~/ENSMLbot.txt', chat_ID=None, telegram_logging_level=31, telegram_logging_format=None):
        th = None
        if telegram_bot_token is not None and chat_ID is not None:
            th = ut.new_telegram_handler(chat_ID=chat_ID, token=telegram_bot_token, level=telegram_logging_level, formatter=telegram_logging_format)
            logger.handlers.append(th)
            logger.log(45, 'Added telegram logger: you should receive this message on telegram.')
        return th

    def run_multiple(self):
        th = self.telegram(**self.telegram_kwargs)
        logger.log(45, f'Starting {len(self.scheduled_kwargs)} runs')
        try:
            for kwargs in self.scheduled_kwargs:
                self._run(**kwargs)
            logger.log(49, '\n\n\n\n\n\nALL RUNS COMPLETED\n\n')
        finally:
            if th is not None:
                logger.handlers.remove(th)
                logger.log(45, 'Removed telegram logger')

    def run(self, folder, load_data_kwargs, prepare_XY_kwargs, k_fold_cross_val_kwargs, log_level=logging.INFO):
        os.mkdir(folder)

        # setup logger
        fh = logging.FileHandler(f'{folder}/log.log')
        fh.setLevel(log_level)
        logger.handlers.append(fh)

        try:
            # load the fields
            if self._load_data_kwargs != load_data_kwargs:
                self._load_data_kwargs = load_data_kwargs
                self._prepare_XY_kwargs = None # force the computation of prepare_XY
                self.fields = load_data(**load_data_kwargs)

            # prepare XY
            if self._prepare_XY_kwargs != prepare_XY_kwargs:
                self._prepare_XY_kwargs = prepare_XY_kwargs
                self.X, self.Y, self.year_permutation = prepare_XY(self.fields, **prepare_XY_kwargs)
            if self.year_permutation is not None:
                np.save(f'{folder}/year_permutation.npy',self.year_permutation)

            # do kfold
            k_fold_cross_val(folder, self.X, self.Y, **k_fold_cross_val_kwargs)

            # make the config file read-only after the first successful run
            if os.access('config.json', os.W_OK): # the file is writeable
                os.chmod('config.json', S_IREAD)
        
        except Exception as e:
            logger.critical(f'Run on {folder = } failed due to {repr(e)}')
            tb = traceback.format_exc()
            logger.error(tb)
            raise RuntimeError('Run failed') from e

        finally:
            logger.handlers.remove(fh)


    def _run(self, **kwargs):
        '''
        Parses kwargs and performs a single run, kwargs are not interpreted as iterables
        '''
        runs = ut.json2dict('runs.json')

        # check if the run has already been performed
        for r in runs.values():
            if r['status'] == 'COMPLETED' and r['args'] == kwargs:
                logger.log(45, f"Skipping already performed run {r['name']}")
                return None

        # get run number
        run_id = str(len(runs))

        folder = f'{run_id}{arg_sep}'
        for k in sorted(kwargs):
            folder += f'{k}{value_sep}{kwargs[k]}{arg_sep}'
        folder = folder[:-len(arg_sep)] # remove the last arg_sep
        folder = ut.make_safe(folder)
        logger.log(42, f'{folder = }\n')
        
        runs[run_id] = {'name': folder, 'args': kwargs, 'status': 'RUNNING', 'start_time': ut.now()}
        ut.dict2json(runs, 'runs.json')

        try:
            run_kwargs = ut.set_values_recursive(self.default_run_kwargs, kwargs)
            
            self.run(folder, **run_kwargs)

            runs = ut.json2dict('runs.json')
            runs[run_id]['status'] = 'COMPLETED'
            logger.log(42, 'run completed!!!\n\n')

        except Exception as e:
            runs = ut.json2dict('runs.json')
            runs[run_id]['status'] = 'FAILED'
            runs[run_id]['name'] = f'F{folder}'
            shutil.move(folder, f'F{folder}')
            raise e

        finally:
            runs[run_id]['end_time'] = ut.now()
            ut.dict2json(runs,'runs.json')

        

        

        


    


if __name__ == '__main__':
    # check if there is a lock:
    lock = Path(__file__).resolve().parent / 'lock.txt'
    if os.path.exists(lock): # there is a lock
        # check for folder argument
        if len(sys.argv) < 2: 
            print(usage())
            sys.exit(0)
        if len(sys.argv) == 2:
            folder = sys.argv[1]
            print(f'moving code to {folder = }')
            move_to_folder(folder)
            
            # config file will be built from the default parameters of the functions given here
            # GM: build_config_dict will recursively find the keyword parameters of function run (including the functions it calls) and build a corresponding dictionary tree in config file
            # GM: Can some of these functions be moved to ../ERA/utilities.py later at some point?
            d = build_config_dict([Trainer.run, Trainer.telegram]) 
            print(f"{d = }") # GM: Doing some tests
            ut.dict2json(d,f'{folder}/config.json')

            # runs file (which will keep track of various runs performed in newly created folder)
            ut.dict2json({},f'{folder}/runs.json')

            sys.exit(0)
        else:
            with open(lock) as l:
                raise ValueError(l.read())
    
    # if there is a lock, the previous block of code would have ended the run, so the code below is executed only if there is no lock
    
    #parse command line arguments
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
        # `value` is a string. Here we try to cast it to the correct type
        try:
            value = ast.literal_eval(value)
        except:
            logger.warning(f'Could not evaluate {value}. Keeping string type')
        arg_dict[key] = value

    logger.info(f'{arg_dict = }')

    # create trainer
    trainer = Trainer()

    # schedule runs
    trainer.schedule(**arg_dict)

    # o = input('Start training? (Y/[n]) ')
    # if o != 'Y':
    #     logger.error('Aborting')
    #     sys.exit(0)
    
    trainer.run_multiple()




    ### THIS OLD CODE USES THE FUNCTION run instead of the Trainer object. Hence it is suited only for 1 run

    # # load config file
    # config_dict = ut.json2dict('config.json')
    # config_dict_flat = ut.collapse_dict(config_dict) # GM: flatten the dictionary
    # print(f'{config_dict = }')

    
    # # parse command line arguments
    # cl_args = sys.argv[1:]
    # i = 0
    # arg_dict = {}
    # while(i < len(cl_args)):
    #     key = cl_args[i]
    #     if '=' in key:
    #         key, value = key.split('=')
    #         i += 1
    #     else:
    #         value = cl_args[i+1]
    #         i += 2
    #     if key not in config_dict_flat:
    #         raise KeyError(f'Unknown argument {key}')
    #     # `value` is a string. Here we try to cast it to the correct type
    #     if config_dict_flat[key] is not None:
    #         try:
    #             value = ast.literal_eval(value)
    #         except:
    #             print(f'Could not evaluate {value}. Keeping string type')
    #     # now check if the provided value is equal to the default one
    #     if value == config_dict_flat[key]:
    #         print(f'Skipping given argument {key} as it is at its default value {value}')
    #     else:
    #         arg_dict[key] = value

    # print("arg_dict = ", arg_dict)
    # # get run number
    # runs = ut.json2dict('runs.json')
    # run_id = len(runs)
    # print("run_id = ", run_id)
    

    # folder = f'{run_id}__'
    # for k in sorted(arg_dict):
    #     folder += f'{k}_{arg_dict[k]}__'
    # folder = folder[:-2] # remove the last '__'
    # print(f'{folder = }')
    # runs[run_id] = {'name': folder, 'args': arg_dict}
    
    # print(f'{runs = }')
    # # set the arguments provided into the nested dictionaries
    # run_kwargs = ut.set_values_recursive(config_dict['run'], arg_dict) # GM: set the values of arg_dict to config_dict['run']
    # print(f'{run_kwargs = }')

    # runs[run_id]['status'] = 'RUNNING'
    # runs[run_id]['start_time'] = ut.now()
    # ut.dict2json(runs, 'runs.json')

    # try:
    #     run(folder, **run_kwargs)
    # except Exception as e:
    #     runs = ut.dict2json('runs.json')
    #     runs[run_id]['status'] = 'FAILED'
    #     runs[run_id]['end_time'] = ut.now()
    #     ut.dict2json(runs,'runs.json')
    #     raise RuntimeError('Run failed') from e
    
    # runs = ut.dict2json('runs.json')
    # runs[run_id]['status'] = 'COMPLETED'
    # runs[run_id]['end_time'] = ut.now()
    # ut.dict2json(runs,'runs.json')

    # print('\n\nrun completed!!!\n\n')
