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
    python Learn2_new.py area="['France', 'Scandinavia']"


If by default an argument is already a list, the provided list is not interpreted as something to be iterated over, for example the argument `fields` has default value ['t2m','zg500','mrso_filtered']. So running
    python Learn2_new.py fields="['t2m', 'zg500']"

will result in a single run performed with fields=['t2m', 'zg500']

If you provide more than one argument to iterate over, all combinations will be performed, e.g.:
    python Learn2_new.py fields="[['t2m'], ['t2m', 'zg500']]" tau='[1,2]'

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
# GM: specify the sign of tau and what it means

# GM: Why is the default to look for Data_Plasim rather than Data_Plasim_LONG?
#        FileNotFoundError: [Errno 2] No such file or directory: b'/local/gmiloshe/PLASIM/Data_Plasim/ANO_tas.nc'
# GM: What if I want to work with 1000 years that are a subset of 8000 years of Plasim_LONG?

### IMPORT LIBRARIES #####

## general purpose
from copy import deepcopy
import os as os
from pathlib import Path
from stat import S_IREAD
import sys
from tkinter.messagebox import NO
import traceback
from unittest import skip
import warnings
import time
import shutil
import gc
from matplotlib import path
from matplotlib.pyplot import hist, loglog
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

# separators to create the run name from the run arguments
arg_sep = '--' # separator between arguments
value_sep = '__' # separator between an argument and its value

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
    >>> d1 == {'k_fold_cross_val_kwargs': d2}
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

####################################
### OPERATIONS WITH RUN METADATA ###
####################################

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
    >>> parse_run_name('a__5--b__7')
    {'a': '5', 'b': '7'}
    >>> parse_run_name('test_arg__bla--b__7')
    {'test_arg': 'bla', 'b': '7'}
    '''
    d = {}
    args = run_name.split(arg_sep)
    for arg in args:
        if value_sep not in arg:
            continue
        key, value = arg.rsplit(value_sep,1)
        d[key] = value
    return d

def check_compatibility(run_name, current_run_name=None, relevant_keys=None):
    '''
    Check if `run_name` is compatible with `current_run_name` for transfer learning, namely if the values corresponding to `relevant_keys` coincide
    '''
    if current_run_name is None or relevant_keys is None:
        return True
    # parse run_name for arguments
    run_dict = parse_run_name(run_name)
    current_run_dict = parse_run_name(current_run_name)
    # keep only arguments that are relevant for model architecture
    run_dict = {k:v for k,v in run_dict.items() if k in relevant_keys}
    current_run_dict = {k:v for k,v in current_run_dict.items() if k in relevant_keys}
    return run_dict == current_run_dict

def select_compatible(run_args, conditions, require_unique=True, path_to_config=None):
    '''
    Selects which runs are compatible with given conditions

    Parameters
    ----------
    run_args : dict
        dictionary where each item is a dictionary of the arguments of the run
    conditions : dict
        dictionary of run arguments that has to be contained in the arguments of a compatible run
    require_unique : bool, optional
        whether you want a single run or a subset of compatible runs, by default True
    path_to_config : str, optional
        path to where the config file is located (without 'config.json').
        If provided allows to beter check when a candition is at its default level, since it won't appear in the list of arguments of the run

    Returns
    -------
    if require_unique:
        the key in run_args corresponding to the only run satisfying the compatibility requirements
    else:
        the list of keys of compatible runs

    Raises
    ------
    KeyError
        If require_unique and either none or more than one run are found compatible.

    Examples   
    --------
    >>> run_args = {'1': {'tau': -5}, '2': {'percent': 1, 'tau': -10}, '3': {'percent': 1, 'tau': -5}}
    >>> select_compatible(run_args, {'tau': 10})
    '2'
    >>> select_compatible(run_args, {'tau': -10}, require_unique=False)
    ['2']
    >>> select_compatible(run_args, {'percent': 1}, require_unique=False)
    ['2', '3']
    '''
    _run_args = deepcopy(run_args)
    if path_to_config is not None:
        path_to_config.rstrip('/')
        config_dict_flat = ut.collapse_dict(ut.json2dict(f'{path_to_config}/config.json'))
        conditions_at_default = {k:v for k,v in conditions.items() if v == config_dict_flat[k]}
        for args in _run_args.values():
            for k,v in conditions_at_default.items():
                if k not in args:
                    args[k] = v

    compatible_keys = [k for k,v in _run_args.items() if conditions.items() <= v.items()]
    if not require_unique:
        return compatible_keys

    if len(compatible_keys) == 0:
        raise KeyError(f'No previous compatible satisfies {conditions = }')
    elif len(compatible_keys) > 1:
        raise KeyError(f"Multiple runs contain satisfy {conditions = } ({compatible_keys}) and your are requiring just one")
    return compatible_keys[0]

def remove_args_at_default(run_args, config_dict_flat):
    '''
    Removes from a dictionary of parameters the values that are at their default one.

    Parameters
    ----------
    run_args : dict
        dictionary where each item is a dictionary of the arguments of the run
    config_dict_flat : dict
        flattened config dictionary with the default values

    Returns
    -------
    dict
        epurated run_args
    '''
    _run_args = deepcopy(run_args)
    for k,args in _run_args.items():
        new_args = {}
        for arg,value in args:
            if value != config_dict_flat[arg]:
                new_args[arg] = value
        _run_args[k] = new_args
    return _run_args

def group_by_varying(run_args, variable='tau', config_dict_flat=None):
    '''
    Groups a series of runs into sets where only `variable` varies and other parameters are shared inside the set

    Parameters
    ----------
    run_args : dict
        dictionary where each item is a dictionary of the arguments of the run
    variable : str, optional
        argument that varies inside each group, by default 'tau'
    config_dict_flat : dict, optional
        flattened config dictionary with the default values, by default None

    Returns
    -------
    list
        list of dictionaries, one for each group. See examples for its structure

    Examples
    --------
    >>> run_args = {'1': {'tau': 0, 'percent':5}, '2': {'percent': 1, 'tau': 0}, '3': {'percent': 1, 'tau': -5}}
    >>> group_by_varying(run_args)
    [{'args': {'percent': 5}, 'runs': ['1'], 'tau': [0]}, {'args': {'percent': 1}, 'runs': ['2', '3'], 'tau': [0, -5]}]
    >>> group_by_varying(run_args, 'percent')
    [{'args': {'tau': 0}, 'runs': ['1', '2'], 'percent': [5, 1]}, {'args': {'tau': -5}, 'runs': ['3'], 'percent': [1]}]
    '''
    _run_args = deepcopy(run_args)
    # add default values for the varaible of interest
    if config_dict_flat is not None:
        for args in _run_args.values():
            if variable not in args:
                args[variable] = config_dict_flat[variable]
    
    # find the groups
    variable_dict = {k:v.pop(variable) for k,v in _run_args.items()} # move the variable to a separate dictionary removing it from the arguments in run_args

    group_args = []
    group_runs = []
    for k,v in _run_args.items():
        try:
            i = group_args.index(v)
            group_runs[i].append(k)
        except ValueError:
            group_args.append(v)
            group_runs.append([k])

    groups = []
    for i in range(len(group_args)):
        groups.append({'args': group_args[i], 'runs': group_runs[i], variable:[variable_dict[k] for k in group_runs[i]]})
    return groups

def get_run(load_from, current_run_name=None):
    '''
    Parameters
    ----------
    load_from : dict, int, str, 'last' or None
        If dict it is a dictionary with arguments of the run. If int it is the number of the run.
        If 'last' it is the last completed run. Otherwise it can be a piece of the run name or the full run name. If None, the function returns None
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

    # arguments relevant for model architecture
    relevant_keys = list(get_default_params(create_model).keys()) + list(get_default_params(load_data).keys()) + ['nfolds']
    
    runs = ut.json2dict('runs.json')

    # select only completed runs
    runs = {k: v for k,v in runs.items() if v['status'] == 'COMPLETED'}

    # select only compatible runs
    runs = {k: v for k,v in runs.items() if check_compatibility(v['name'], current_run_name, relevant_keys=relevant_keys)}

    if len(runs) == 0:
        logger.warning('No valid runs to load from')
        return None

    if isinstance(load_from, int):
        l = load_from
    elif isinstance(load_from, dict):
        l = int(select_compatible({k:v['args'] for k,v in runs.items()}, load_from, require_unique=True))
    elif isinstance(load_from, str):
        try:
            l = int(load_from) # run number
        except ValueError: # cannot convert load_from to int, so it must be a string that doesn't contain only numbers
            if load_from == 'last':
                l = -1
            else:
                run_names = {k:v['name'] for k,v in runs.items()}
                if load_from in run_names.values(): # full run name provided
                    return load_from
                load_from_dict = parse_run_name(load_from)
                l = int(select_compatible({k:parse_run_name(v) for k,v in run_names.items()}, load_from_dict, require_unique=True))
    else:
        raise TypeError(f'Unsupported type {type(load_from)} for load_from')

    # now l is an int
    if l < 0:
        r = list(runs.values())[l]
    else:
        r = runs[str(l)]
    run_name = r['name']
    
    return run_name

######################################
########## COPY SOURCE FILES #########
######################################

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
    shutil.copy(path_to_here / 'import_config.py', folder)
    # copy additional files
    # History.py
    # Metrics.py
    # Recalc_Tau_Metrics.py
    # Recalc_History.py

    # copy useful files from ../ERA/ to folder/ERA/
    path_to_here = path_to_here.parent / 'ERA'
    shutil.copy(path_to_here / 'cartopy_plots.py', ERA_folder)
    shutil.copy(path_to_here / 'ERA_Fields_New.py', ERA_folder)
    shutil.copy(path_to_here / 'TF_Fields.py', ERA_folder)
    shutil.copy(path_to_here / 'utilities.py', ERA_folder)

    print(f'Now you can go to {folder} and run the learning from there:\n')
    print(f'cd \"{folder}\"\n')
    
    
############################################
########## DATA PREPROCESSING ##############
############################################

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

@ut.execution_time  # prints the time it takes for the function t run
@ut.indent_logger(logger)   # indents the log messages produced by this function
# GM: perhaps 'mask' is a better title, rather than filter, but given many functions already carry this name it is too late
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
    filter_area : str, optional
        area over which to keep filtered fields, usually the same of `area`. `filter` implies a mask
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
def early_stopping(monitor='val_CustomLoss', min_delta=0, patience=0, mode='auto'):
    '''
    Creates an early stopping callback

    Parameters
    ----------
    monitor : str, optional
        metric to monitor, by default 'val_CustomLoss'
    min_delta : int, optional
        minimum change in the monitored metric to qualify as improvement, by default 0
    patience : int, optional
        maximum number of epochs to wait for an improvement of the monitored metric, by default 0, which means early stopping is disabled
    mode : 'min' or 'max' or 'auto', optional
        whether the monitored metric needs to be maximized or minimized, by default 'auto', which means it is inferrend by the name of the monitored quantity

    Returns
    -------
    keras.callbacks.EarlyStopping
        Early stopping callback
    '''
    if mode == 'auto':
        mode = 'min'
        for v in ['acc', 'MCC', 'skill']:
            if v in monitor:
                mode = 'max'
                break
    return keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, mode=mode)

@ut.execution_time
@ut.indent_logger(logger)
def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics, early_stopping_kwargs, enable_early_stopping=False,
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
        location where to save the checkpoints of the model
    num_epochs : int
        number of maximum epochs for the training
    optimizer : keras.Optimizer object
    loss : keras.losses.Loss object
    metrics : list of keras.metrics.Metric or str
    early_stopping_kwargs : dict
        arguments to create the early stopping callback. Ignored if `enable_early_stopping` = False
    enable_early_stopping : bool, optional
        whether to perform early stopping or not, by default False
    batch_size : int, optional
        by default 1024
    checkpoint_every : int or str, optional
        Examples:
        0: disabled
        5 or '5 epochs' or '5 e': every 5 epochs
        '100 batches' or '100 b': every 100 batches
        'best custom_loss': every time 'custom_loss' reaches a new minimum. 'custom_loss' must be in the list of metrics
    additional_callbacks : list of keras.callbacks.Callback objects or list of str, optional
        string items are interpreted, for example 'csv_logger' creates a CSVLogger callback
    '''
    folder = folder.rstrip('/')
    ckpt_name = folder + '/cp-{epoch:04d}.ckpt'

    ## deal with callbacks
    callbacks = []

    # additional callbacks
    if additional_callbacks is not None:
        for cb in additional_callbacks:
            if isinstance(cb, str):
                if cb.lower().startswith('csv'):
                    callbacks.append(keras.callbacks.CSVLogger(f'{folder}/history.csv', append=True))
                else:
                    raise ValueError(f'Unable to understand callback {cb}')
            else:
                callbacks.append(cb)

    # checkpointing callback
    ckpt_callback = None
    if checkpoint_every == 0: # no checkpointing
        pass
    elif checkpoint_every == 1: # save every epoch
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=ckpt_name, save_weights_only=True, verbose=1)
    elif isinstance(checkpoint_every, int): # save every `checkpoint_every` epochs 
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

    # early stopping callback
    if enable_early_stopping:
        if 'patience' not in early_stopping_kwargs or early_stopping_kwargs['patience'] == 0:
            logger.warning('Skipping early stopping with patience = 0')
        else:
            callbacks.append(early_stopping(**early_stopping_kwargs))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.save_weights(ckpt_name.format(epoch=0)) # save model before training

    # perform training for `num_epochs`
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
    else: # `upper` overshoots
        X_va = np.concatenate([X[lower:], X[:upper]], axis=0)
        Y_va = np.concatenate([Y[lower:], Y[:upper]], axis=0)
        X_tr = X[upper:lower]
        Y_tr = Y[upper:lower]
    return X_tr, Y_tr, X_va, Y_va


def optimal_checkpoint(run_folder, nfolds, metric='val_CustomLoss', direction='minimize', first_epoch=1, collective=True, bypass=None):
    '''
    Computes the epoch that had the best score

    Parameters
    ----------
    folder : str
        folder where the model is located that contains sub folders with the n folds named 'fold_%i'
    nfolds : int, optional
        number of folds,
    metric : str, optional
        metric with respect to which optimize, by default 'val_CustomLoss'
    direction : str, optional
        'maximize' or 'minimize', by default 'minimize'
    first_epoch : int, optional
        The number of the first epoch, by default 1
    collective : bool, optional
        Whether the optimal checkpoint should be the same for all folds (True) or the best for each fold
    bypass : np.ndarray, optional
        If provided the function immediately returns `bypass`
        (See Trainer._run for practical use)
    Returns
    -------
    if collective:
        int
            epoch number corresponding to the best checkpoint
    else:
        list
            of best epoch number for each fold

    Raises
    ------
    KeyError
        If `metric` is not present in the history
    ValueError
        If `direction` not in ['maximize', 'minimize']
    '''
    if bypass is not None:
        return bypass
    
    run_folder = run_folder.rstrip('/')
    # Here we insert analysis of the previous training with the assessment of the ideal checkpoint
    history0 = np.load(f'{run_folder}/fold_0/history.npy', allow_pickle=True).item()
    if metric not in history0.keys():
        raise KeyError(f'{metric} not in history: cannot compute optimal checkpoint')
    historyCustom = [np.load(f'{run_folder}/fold_{i}/history.npy', allow_pickle=True).item()[metric] for i in range(nfolds)]

    if direction == 'minimize':
        opt_f = np.argmin
    elif direction == 'maximize':
        opt_f = np.argmax
    else:
        raise ValueError(f'Unrecognized {direction = }')

    if collective: # the optimal checkpoint is the same for all folds and it is based on the average performance over the folds
        # check that the nfolds histories have the same length
        l0 = len(historyCustom[0])
        for h in historyCustom[1:]:
            if len(h) != l0:
                logger.error('Cannot compute a collective checkpoint from folds trained a different number of epochs. Computing independent checkpoints instead')
                collective = False
                break
    if collective:
        historyCustom = np.mean(np.array(historyCustom),axis=0)
        opt_checkpoint = opt_f(historyCustom)
    else:
        opt_checkpoint = np.array([opt_f(h) for h in historyCustom]) # each fold independently
    
    opt_checkpoint += first_epoch

    if collective:
        opt_checkpoint = int(opt_checkpoint)
    else:
        opt_checkpoint = [int(oc) for oc in opt_checkpoint]
    return opt_checkpoint

@ut.execution_time
@ut.indent_logger(logger)
def k_fold_cross_val(folder, X, Y, create_model_kwargs, train_model_kwargs, optimal_checkpoint_kwargs, load_from='last', nfolds=10, val_folds=1, u=1,
                     fullmetrics=True, training_epochs=40, training_epochs_tl=10, loss='sparse_categorical_crossentropy', lr=1e-4):
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
        For most common use (command line) you can only specify arguments that have a default value and so appear in the config file.
        However if run this function from a notebook you can use more advanced features like using another loss rather than the default cross entropy
        or an optimizer rather than Adam.
        This can be done specifying other parameters rather than the ones that appear in the config file, namely:
            num_epochs : int
                number of training epochs. `training_epochs` and `training_epochs_tl` are ignored
            optimizer : keras.optimizers.Optimizer
                optimizer object, `lr` is ignored
            loss : keras.metrics.Metric
                overrides the `loss`
            metrics : list of metrics objects
                overrides `fullmetrics`
    optimal_chekpoint_kwargs : dict
        dictionary with the parameters to find the optimal checkpoint
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
    loss : str, optional
        loss function to minimize, by default 'sparse_categorical_crossentropy'
        another possibility is 'unbiased_crossentropy',
        which will unbias the logits with the undersampling factor and then proceeds with the sparse_categorical_crossentropy
    lr : float, optional
        learning_rate for Adam optimizer       
    '''
    folder = folder.rstrip('/')

    # get the actual run name from where to load
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
        opt_checkpoint = optimal_checkpoint(load_from, nfolds, **optimal_checkpoint_kwargs)
        if isinstance(opt_checkpoint,int):
            # this happens if the optimal checkpoint is computed with `collective` = True
            #  so we simply broadcast the single optimal checkpoint to all the folds
            opt_checkpoint = [opt_checkpoint]*nfolds

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
            if undersampling_strategy > 1: # you cannot undersample so much that the majority class becomes the minority one
                raise ValueError(f'Too high undersmapling factor, maximum for this dataset is u={n_neg_tr/n_pos_tr}')
            pipeline = Pipeline(steps=[('u', RandomUnderSampler(random_state=42, sampling_strategy=undersampling_strategy))])
            # reshape data to feed it to the pipeline
            X_tr_shape = X_tr.shape
            X_tr = X_tr.reshape((X_tr_shape[0], np.product(X_tr_shape[1:])))
            X_tr, Y_tr = pipeline.fit_resample(X_tr, Y_tr) # apply pipeline
            X_tr = X_tr.reshape((X_tr.shape[0], *X_tr_shape[1:])) # reshape back
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

        # at this point data is ready to be fed to the networks


        # check for transfer learning
        model = None        
        if load_from is None:
            model = create_model(input_shape=X_tr.shape[1:], **create_model_kwargs)
        else:
            model = keras.models.load_model(f'{load_from}/fold_{i}', compile=False)
            model.load_weights(f'{load_from}/fold_{i}/cp-{opt_checkpoint[i]:04d}.ckpt')
        summary_buffer = ut.Buffer() # workaround necessary to log the structure of the network to the file, since `model.summary` uses `print`
        summary_buffer.append('\n')
        model.summary(print_fn = lambda x: summary_buffer.append(x + '\n'))
        logger.info(summary_buffer.msg)

        # number of training epochs
        num_epochs = train_model_kwargs.pop('num_epochs', None) # if num_epochs is not provided in train_model_kwargs, whihc is most of the time,
                                                                # we assign it according if we have to du transfer learning or not
        if num_epochs is None:
            if load_from is None:
                num_epochs = training_epochs
            else:
                num_epochs = training_epochs_tl

        # metrics
        tf_sampling = tf.cast([0.5*np.log(u), -0.5*np.log(u)], tf.float32)
        metrics = train_model_kwargs.pop('metrics', None)
        if metrics is None:
            if fullmetrics:
                metrics=[
                    'accuracy',
                    tff.MCCMetric(undersampling_factor=1),  # GM: Freddy says 1, try both but if it is too slow not worth it
                    tff.MCCMetric(undersampling_factor=u, name='UnbiasedMCC'),
                    tff.ConfusionMatrixMetric(2, undersampling_factor=u), # GM: Freddy says 1
                    tff.BrierScoreMetric(undersampling_factor=u),
                    tff.CustomLoss(tf_sampling)
                ]# the last two make the code run longer but give precise discrete prediction benchmarks
            else:
                metrics=['loss']

        # optimizer
        optimizer = train_model_kwargs.pop('optimizer',keras.optimizers.Adam(learning_rate=lr)) # if optimizer is not provided in train_model_kwargs use Adam
        # loss function
        loss_fn = train_model_kwargs.pop('loss',None)
        if loss_fn is None:
            if loss.startswith('unbiased'):
                loss_fn = tff.UnbiasedCrossentropyLoss(undersampling_factor=u)
            else:
                loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        logger.info(f'Using {loss_fn.name} loss')


        # train the model
        train_model(model, X_tr, Y_tr, X_va, Y_va, # arguments that are always computed inside this function
                    folder=fold_folder, num_epochs=num_epochs, optimizer=optimizer, loss=loss_fn, metrics=metrics, # arguments that may come from train_model_kwargs for advanced uses but usually are computed here
                    **train_model_kwargs) # arguments which have a default value in the definition of `train_model` and thus appear in the config file

        # TODO: compute metrics here where it we have easy access to model and data

        my_memory.append(psutil.virtual_memory())
        logger.info(f'RAM memory: {my_memory[i][3]:.3e}') # Getting % usage of virtual_memory ( 3rd field)

        keras.backend.clear_session()
        gc.collect() # Garbage collector which removes some extra references to the objects
        
    np.save(f'{folder}/RAM_stats.npy', my_memory)


########## PUTTING THE PIECES TOGETHER ###########
@ut.execution_time
@ut.indent_logger(logger)
def prepare_XY(fields, make_XY_kwargs, roll_X_kwargs,
               do_premix=False, premix_seed=0, do_balance_folds=True, nfolds=10, year_permutation=None, flatten_time_axis=True):
    '''
    Performs all operations to extract from the fields X and Y ready to be fed to the neural network.

    Parameters
    ----------
    fields : dict of ef.Plasim_Field objects
    make_XY_kwargs : dict
        arguments to pass to the function `make_XY`
    roll_X_kwargs : dict
        arguments to pass to the function `roll_X`
    do_premix : bool, optional
        whether to perform premixing, by default False
    premix_seed : int, optional
        seed for premixing, by default 0
    do_balance_folds : bool, optional
        whether to balance folds
    nfolds : int, optional
        necessary for balancing folds
    year_permutation : np.ndarray, optional
        if provided overrides both premixing and fold balancing, useful for transfer learning as avoids contaminating test sets. By default None
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
    
    if year_permutation is None:
        # premixing
        if do_premix:
            premix_permutation = shuffle_years(X, seed=premix_seed, apply=False)
            Y = Y[premix_permutation]
            year_permutation = premix_permutation

        # balance folds:
        if do_balance_folds:
            weights = np.sum(Y, axis=1) # get the number of heatwave events per year
            balance_permutation = balance_folds(weights,nfolds=nfolds, verbose=True)
            Y = Y[balance_permutation]
            if year_permutation is None:
                year_permutation = balance_permutation
            else:
                year_permutation = ut.compose_permutations([year_permutation, balance_permutation])
    else:
        year_permutation = np.array(year_permutation)
        Y = Y[year_permutation]
        logger.warning('Mixing overriden by provided permutation')

    # apply permutation to X
    if year_permutation is not None:    
        X = X[year_permutation]
    logger.info(f'Mixing completed in {ut.pretty_time(time.time() - start_time)}\n')
    logger.info(f'{X.shape = }, {Y.shape = }')

    # flatten the time axis dropping the organizatin in years
    if flatten_time_axis:
        X = X.reshape((X.shape[0]*X.shape[1],*X.shape[2:]))
        Y = Y.reshape((Y.shape[0]*Y.shape[1]))
        logger.info(f'Flattened time: {X.shape = }, {Y.shape = }')

    return X, Y, year_permutation


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
    year_permutation : np.ndarray
        with shape (years,), final permutaion of the years that reproduces X and Y once applied to the just loaded data
    '''
    # load data
    fields = load_data(**load_data_kwargs)

    return prepare_XY(fields, **prepare_XY_kwargs)

@ut.execution_time
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
    load_data_kwargs = prepare_data_kwargs['load_data_kwargs']
    prepare_XY_kwargs = prepare_data_kwargs['prepare_XY_kwargs']
    label_field = ut.extract_nested(prepare_data_kwargs, 'label_field')
    for field_name in load_data_kwargs['fields']:
        if field_name.startswith(label_field):
            found = True
            break
    if not found:
        raise KeyError(f"field {label_field} is not a loaded field")

    trainer = Trainer()
    trainer.run(folder,load_data_kwargs, prepare_XY_kwargs, k_fold_cross_val_kwargs, log_level=log_level)


###### EFFICIENT MANAGEMENT OF MULTIPLE RUNS #######

class Trainer():
    '''
    Class for performing training of neural networks over multiple runs with different paramters in an efficient way
    '''
    def __init__(self, skip_existing_run=True):
        '''
        Constructor

        Parameters
        ----------
        skip_existing_run : bool, optional
            Whether to skip runs that have already been performed in the same folder, by default True
            If False the existing run is not overwritten but a new one is performed
        '''
        self.skip_existing_run = skip_existing_run

        # load config file and parse arguments
        self.config_dict = ut.json2dict('config.json')
        self.config_dict_flat = check_config_dict(self.config_dict)
        
        # cached (heavy) variables
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

        Special arguments:
            first_from_scratch : bool, optional
                Whether the first run should be created from scratch or from transfer learning, by default False (transfer learning)
        '''
        first_from_scratch = kwargs.pop('first_from_scratch', False)  # this argument is removed from the kwargs because it affects only the first run
        
        # detect variables over which to iterate
        iterate_over = [] # list of names of arguments that are lists and so need to be iterated over
        non_iterative_kwargs = {} # dictionary of provided arguments that have a single value
        for k,v in kwargs.items():
            if k not in self.config_dict_flat:
                raise KeyError(f'Invalid argument {k}')
            if k in self.telegram_kwargs: # deal with telegram arguments separately
                self.telegram_kwargs[k] = v
                continue
            iterate = False
            if isinstance(v, list): # the argument is a list: possible need to iterate over the argument
                if isinstance(self.config_dict_flat[k], list): # the default value is a list as well, so maybe we don't need to iterate over v
                    if isinstance(v[0], list): # v is a list of lists: we need to iterate over it
                        iterate = True
                else:
                    iterate = True
            if iterate:
                iterate_over.append(k)
            elif v != self.config_dict_flat[k]: # skip parameters already at their default value
                non_iterative_kwargs[k] = v


        # rearrange the order of the arguments over which we need to iterate such that the runs are performed in the most efficient way
        # namely we want arguments for loading data to tick like hours, arguments for preparing X,Y to tick like minutes and arguments for k_fold_cross_val like seconds
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

        # retrieve values of the arguments
        iteration_values = [kwargs[k] for k in iterate_over]
        # expand the iterations into a list performing the meshgrid
        iteration_values = list(zip(*[m.flatten() for m in np.meshgrid(*iteration_values, indexing='ij')]))
        # ensure json serialazability by converting to string and back
        iteration_values = ast.literal_eval(str(iteration_values))

        # add the non iterative kwargs
        self.scheduled_kwargs = [{**non_iterative_kwargs, **{k: l[i] for i,k in enumerate(iterate_over)}} for l in iteration_values]

        if len(self.scheduled_kwargs) == 0: # GM: this is fix to avoid empty scheduled_kwargs if it happens there are no iterative kwargs
            self.scheduled_kwargs = [non_iterative_kwargs]
            if len(non_iterative_kwargs) == 0:
                logger.info('Scheduling 1 run at default values')
            else:
                logger.info(f'Scheduling 1 run at values {non_iterative_kwargs}')
        else:
            logger.info(f'Scheduled the following {len(self.scheduled_kwargs)} runs:')
            for i,kw in enumerate(self.scheduled_kwargs):
                logger.info(f'{i}: {kw}')

        if first_from_scratch: 
            self.scheduled_kwargs[0]['load_from'] = None # disable transfer learning for the first run
            logger.warning('Forcing the first run to be loaded from scratch')
    
    def telegram(self, telegram_bot_token='~/ENSMLbot.txt', chat_ID=None, telegram_logging_level=31, telegram_logging_format=None):
        '''
        Adds a telegram handler to the logger of this module, if `telegram_bot_token` and `chat_ID` are both not None
        To be able to receive messages on telegram from this bot go on telegram and start a conversation with `ENSMLbot`

        Parameters
        ----------
        telegram_bot_token : str, optional
            token for the telegram bot or path to the file where it is stored, by default '~/ENSMLbot.txt'
        chat_ID : int, optional
            chat id of the telegram user/group to whom send the log messages, by default None
            To find your chat id go in telegram and type /start in a chat with `userinfobot`
        telegram_logging_level : int, optional
            logging level for this handler, by default 31
        telegram_logging_format : srt, optional
            format of the logging messages, by default None

        Returns
        -------
        telegram_handler.handlers.TelegramHandler
            telegram handler object
        '''
        th = None
        if telegram_bot_token is not None and chat_ID is not None:
            th = ut.new_telegram_handler(chat_ID=chat_ID, token=telegram_bot_token, level=telegram_logging_level, formatter=telegram_logging_format)
            logger.handlers.append(th)
            logger.log(45, 'Added telegram logger: you should receive this message on telegram.')
        return th

    def run_multiple(self):
        '''
        Performs all the scheduled runs
        '''
        # add telegram logger
        th = self.telegram(**self.telegram_kwargs)
        logger.log(45, f'Starting {len(self.scheduled_kwargs)} runs')
        try:
            for kwargs in self.scheduled_kwargs:
                self._run(**kwargs)
            logger.log(49, '\n\n\n\n\n\nALL RUNS COMPLETED\n\n')
        finally:
            # remove telegram logger
            if th is not None:
                logger.handlers.remove(th)
                logger.log(45, 'Removed telegram logger')

    def run(self, folder, load_data_kwargs, prepare_XY_kwargs, k_fold_cross_val_kwargs, log_level=logging.INFO):
        '''
        Performs a single full run

        Parameters
        ----------
        folder : str
            folder where to perform the run
        load_data_kwargs : dict
            arguments for the function load_data
        prepare_XY_kwargs : dict
            arguments for the function prepare_XY
        k_fold_cross_val_kwargs : dict
            arguments for the function k_fold_cross_val
        log_level : int, optional
            logging level for the log file, by default logging.INFO

        Raises
        ------
        RuntimeError
            If an exception is raised during the run
        '''
        os.mkdir(folder)

        # setup logger to file
        fh = logging.FileHandler(f'{folder}/log.log')
        fh.setLevel(log_level)
        logger.handlers.append(fh)

        try:
            # load the fields only if the arguments have changed, otherwise self.fields is already at the correct value
            if self._load_data_kwargs != load_data_kwargs:
                self._load_data_kwargs = load_data_kwargs
                self._prepare_XY_kwargs = None # force the computation of prepare_XY
                self.fields = load_data(**load_data_kwargs)

            # prepare XY only if the arguments have changed, as above
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
            tb = traceback.format_exc() # log the traceback to the log file
            logger.error(tb)
            raise RuntimeError('Run failed') from e

        finally:
            logger.handlers.remove(fh) # stop writing to the log file


    def _run(self, **kwargs):
        '''
        Parses kwargs and performs a single run, kwargs are not interpreted as iterables.
        It checks for transfer learning and if the run has already been performed, in which case, if `self.skip_existing_run` is True, it is skipped
        Basically it is a wrapper of the `self.run` function.
        '''
        runs = ut.json2dict('runs.json') # get runs dictionary

        # check if the run has already been performed
        for r in runs.values():
            if r['status'] == 'COMPLETED' and r['args'] == kwargs:
                if self.skip_existing_run:
                    logger.log(45, f"Skipping already performed run {r['name']}")
                    return None
                else:
                    logger.log(45, f"Rerunning {r['name']}")

        # get run number
        run_id = str(len(runs))

        # create run name from kwargs
        folder = f'{run_id}{arg_sep}'
        for k in sorted(kwargs):
            folder += f'{k}{value_sep}{kwargs[k]}{arg_sep}'
        folder = folder[:-len(arg_sep)] # remove the last arg_sep
        folder = ut.make_safe(folder) 

        # correct the default kwargs with the ones provided
        run_kwargs = ut.set_values_recursive(self.default_run_kwargs, kwargs)

        check_config_dict(run_kwargs) # check if the arguments are consistent with each other

        # check for transfer learning
        load_from = ut.extract_nested(run_kwargs, 'load_from')
        load_from = get_run(load_from,current_run_name=folder)
        tl_from = None
        if load_from is not None: # we actually do transfer learning
            nfolds = ut.extract_nested(run_kwargs, 'nfolds')
            optimal_checkpoint_kwargs = ut.extract_nested(run_kwargs, 'optimal_checkpoint_kwargs')
            opt_checkpoint = optimal_checkpoint(load_from,nfolds, **optimal_checkpoint_kwargs) # get the optimal checkpoint

            tl_from = {'run': load_from, 'optimal_checkpoint': opt_checkpoint}

            # avoid computing the optimal checkpoint again inside k_fold_cross_val by setting up a bypass for when `optimal_checkpoint` is called inside k_fold_cross_val
            run_kwargs = ut.set_values_recursive(run_kwargs, {'load_from': load_from, 'bypass': opt_checkpoint})

            # force the dataset to the same year permutation
            year_permutation = list(np.load(f'{load_from}/year_permutation.npy', allow_pickle=True))
            run_kwargs = ut.set_values_recursive(run_kwargs, {'year_permutation': year_permutation})

            # these arguments are ignored due to transfer learning, so warn the user if they had been provided
            overridden_kwargs = ['do_premix', 'premix_seed', 'do_balance_folds']
            ignored_kwargs = [k for k in kwargs if k in overridden_kwargs]
            if len(ignored_kwargs) > 0:
                # remove ignored kwargs
                for k in ignored_kwargs:
                    kwargs.pop(k)
                    logger.log(45, f'Ignoring provided argument {k} due to transfer learning compatibility')

                # check again if the run has already been done since now kwargs is potentially changed
                for r in runs.values():
                    if self.skip_existing_run:
                        logger.log(45, f"Skipping already performed run {r['name']}")
                        return None
                    else:
                        logger.log(45, f"Rerunning {r['name']}")
                        
                # update the folder name
                folder = f'{run_id}{arg_sep}'
                for k in sorted(kwargs):
                    folder += f'{k}{value_sep}{kwargs[k]}{arg_sep}'
                folder = folder[:-len(arg_sep)] # remove the last arg_sep
                folder = ut.make_safe(folder)

        logger.log(42, f'{folder = }\n')
        
        runs[run_id] = {'name': folder, 'args': kwargs, 'transfer_learning_from': tl_from, 'status': 'RUNNING', 'start_time': ut.now()}
        ut.dict2json(runs, 'runs.json') # save runs.json

        # run
        try:            
            self.run(folder, **run_kwargs)

            runs = ut.json2dict('runs.json')
            runs[run_id]['status'] = 'COMPLETED'
            logger.log(42, 'run completed!!!\n\n')

        except Exception as e: # run failed
            runs = ut.json2dict('runs.json')
            runs[run_id]['status'] = 'FAILED'
            runs[run_id]['name'] = f'F{folder}'
            shutil.move(folder, f'F{folder}')
            raise e

        finally: # in any case we need to save the end time and save runs to json
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
            # GM: build_config_dict will recursively find the keyword parameters of function run 
            # (including the functions it calls) and build a corresponding dictionary tree in config file
            # GM: Can some of these functions be moved to ../ERA/utilities.py later at some point?
            d = build_config_dict([Trainer.run, Trainer.telegram]) 
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

    # o = input('Start training? (Y/[n]) ') # ask for confirmation
    # if o != 'Y':
    #     logger.error('Aborting')
    #     sys.exit(0)
    
    trainer.run_multiple()