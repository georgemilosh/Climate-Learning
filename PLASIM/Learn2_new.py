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

If you want to import config parameters from another config file (for example another folder with a previous version of this code), you can do it by running
    python import_config.py <path_to_config_from_which_to_import>

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

You can also import parameters from a previous run by using the argument 'import_params_from'.
    Let's say you have performed run 0 with
        percent=5
        tau=0
        T=12
    Now if you run
        python Learn2_new.py import_params_from=0 percent=1 fields="['t2m']"
    You will perform a run with the same parameters as run 0, except for those explicitly provided. In particular for this case
        percent=1
        tau=0
        T=12
        fields="['t2m']"
        
To facilitate debugging and development it is highly advised to work with small dataset. We have orginazed our files so that 
    if you choose: datafolder=Data_CESM_short the code will load for a different source (with fewer number of years) 

Somewhat less obvious is the treatment of skip connections that are used in `create_model` method. The point is that we convert the input
to the dictionary inside the function but we couldn't include it as a kwarg for Learn2_new.py because of conflicts with file names when
saving model checkpoints. Thus we provide an input which is subsequently processed, e.g.:

    python Learn2_new.py conv_skip="[[[0,2]],[[0,2],[1,2]]]"=
will result in in two runs, one with a skip connections between layers 0 and 2, and the second run with two skip connections, one between
0 and 2 layers and one between 1 and 2 layers. 

    While originally we developed the code to work with a single time, we have subsequently modified it to allow inputs which
    use sliding windows with time. To engage such a sliding window it is important to specify the input parameters precisely:
        if `label_period_start` is None -> single time will be used as usual
        if `label_period_start` is specified and it is larger than 'time_start' then the code expects that the user wishes
        to use temporal information for the inference and thus expands X to an extra dimension which carries slided windows
        in time. The difference that is computed `leftmargin` = `label_period_start` - `time_start` tells us how long the extra
        time information is (going back in time). The window will be slided on a daily basis, so the extra dimension length
        will become `leftmargin`+1. It should be noted that currently we do not support tensorflow Datasets and thus one should
        be careful and not apply big time windows, or possibly risk to overrun RAM. having leftmargin nonzero is safer if 
        dimensionality reduction was performed beforehand.


FAQ
---
Q: what is tau?
    It is the delay between the prediction and the observation, and is usually supposed to be negative (prediction before observation)

Q: What if I want to work with 1000 years that are a subset of 8000 years of Plasim_LONG? 
    You can do it by running with `dataset_years = 8000` and `year_list = 'range(1000)'`, which will use the first 1000 years of the 8000 year dataset
    you can also provide `year_list = 'range(1000,3000,2)'` which will take the even years between 1000 and 3000

Q: what if I want to have a smaller training set and a bigger validation one
    If you want to have kfold validation where the validation dataset is bigger than the training one you can do it as well by providing the argument val_folds.
    For example `nfolds = 10, val_folds = 9` will use 90% of the data for testing and 10% for training


Logging levels:
level   name                events

0       logging.NOTSET

10      logging.DEBUG

20      logging.INFO

25                          History of training

30      logging.WARNING

35                          Which fold is running
                            No runs from which to perform transfer learning
                            Recomputing scores with optimal checkpoint

40      logging.ERROR

41                          From where the models are loaded or created
                            Final score of k_fold_cross_val
                            Run pruned

42                          Folder name of the run
                            Single run completes

44                          Non default arguments of the run

45                          Added and removed telegram logger
                            Tell number of scheduled runs
                            Skipping/rerunning already performed run
                            Average score of the run

48                          Progressive number among the scheduled runs

49                          All runs completed

50      logging.CRITICAL    The program stops due to an error
'''

### IMPORT LIBRARIES #####

## general purpose
from copy import deepcopy
import os as os
from pathlib import Path
from stat import S_IREAD, S_IROTH, S_IRGRP
import sys
import traceback
import time
import shutil
import gc
import psutil
import numpy as np
import pandas as pd
import inspect
import ast
import pickle # in case we need to open labels from outside
import logging
from uncertainties import ufloat
from functools import wraps
import socket
from functools import partial # this one we use for the scheduler
from sklearn.decomposition import PCA
import signal


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO

HOSTNAME = socket.gethostname()


## machine learning
# from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models

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
try:
    import general_purpose.utilities as ut
except ImportError:
    import ERA.utilities as ut

try:
    from np.lib.stride_tricks import sliding_window_view
except ImportError: # custom copy for numpy<1.20
    logger.warning('Could not import sliding_window_view from np.lib.stride_tricks. Using custom copy for numpy<1.20')
    try:
        sliding_window_view = ef.sliding_window_view
    except AttributeError:
        logger.warning('This version of ERA_Fields_New does not have the sliding_window_view function. Using custom copy for numpy<1.20')
        # define a function which will raise error when called
        def sliding_window_view(*args, **kwargs):
            raise NotImplementedError("you need to update ERA_Fields_New.py to have the sliding_window_view function, or update numpy to 1.20 or higher")
    

# separators to create the run name from the run arguments
arg_sep = '--' # separator between arguments
value_sep = '__' # separator between an argument and its value

########## USAGE ###############################
def usage(): 
    '''
    Returns the documentation of this module that explains how to use it.
    '''
    return this_module.__doc__

#######################################
#### CONFIG FILE RELATED ROUTINES #####
#######################################

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
    {'greeting': 'Hello there!', 'roll_X_kwargs': {'roll_axis': 'lon', 'roll_steps': 0}}
    '''
    s = inspect.signature(func)
    default_params = {
        k:v.default for k,v in s.parameters.items()
        if (v.default is not inspect.Parameter.empty and not k.endswith('_kwargs'))
    }
    if recursive: # look for parameters ending in '_kwargs' and extract further default arguments
        possible_other_params = [k for k,v in s.parameters.items() if k.endswith('_kwargs')]
        for k in possible_other_params:
            func_name = k.rsplit('_',1)[0] # remove '_kwargs'
            try:
                default_params[k] = get_default_params(getattr(this_module, func_name), recursive=True)
            except:
                logger.warning(f'From get_default_params:  Could not find function {func_name}')
    return default_params

def build_config_dict(functions):
    '''
    Creates a config dictionary with the default arguments of the functions in the list `functions`. See also function `get_default_params`
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

def check_config_dict(config_dict, correct_mistakes=True):
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
        label_field = config_dict_flat['label_field']
        for field_name in config_dict_flat['fields']:
            if field_name.startswith(label_field):
                found = True
                break
        if not found:
            if correct_mistakes:
                logger.warning(f"field {label_field} is not a loaded field: adding ghost field")
                config_dict_flat['fields'].append(f'{label_field}_ghost')
                ut.set_values_recursive(config_dict, {'fields': config_dict_flat['fields']}, inplace=True)
            else:
                raise ValueError(f"{label_field = } is not one of the loaded fields: please add a ghost field as {label_field+'_ghost'}")

        if 'enable_early_stopping' in config_dict_flat and config_dict_flat['enable_early_stopping']:
            if config_dict_flat['patience'] == 0:
                logger.warning('Setting `patience` to 0 disables early stopping')
            elif config_dict_flat['collective']:
                raise ValueError('Using collective checkpoint together with early stopping is highly deprecated')
    except Exception as e:
        raise KeyError('Invalid config dictionary') from e
    return config_dict_flat

####################################
### OPERATIONS WITH RUN METADATA ###
####################################

def make_run_name(run_id, **kwargs):
    folder = f'{run_id}{arg_sep}'
    for k in sorted(kwargs):
        if k == 'load_from': # skip putting load_from in the name as it messes it
            continue
        folder += f'{k}{value_sep}{kwargs[k]}{arg_sep}'
    folder = folder[:-len(arg_sep)] # remove the last arg_sep
    folder = ut.make_safe(folder)
    return folder

def parse_run_name(run_name, evaluate=False):
    '''
    Parses a string into a dictionary

    Parameters
    ----------
    run_name: str
        run name formatted as *<param_name>_<param_value>__*
    evaluate : bool, optional
        whether to try to evaluate the string expressions (True), or leave them as strings (False).
        If unable to evalate an expression, it is left as is
    
    Returns
    -------
    dict
    
    Examples
    --------
    >>> parse_run_name('a__5--b__7')
    {'a': '5', 'b': '7'}
    >>> parse_run_name('test_arg__bla--b__7')
    {'test_arg': 'bla', 'b': '7'}
    >>> parse_run_name('test_arg__bla--b__7', evaluate=True)
    {'test_arg': 'bla', 'b': 7}
    '''
    d = {}
    args = run_name.split(arg_sep)
    for arg in args:
        if value_sep not in arg:
            continue
        key, value = arg.rsplit(value_sep,1)
        if evaluate:
            try:
                value = ast.literal_eval(value)
            except:
                pass
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

def select_compatible(run_args, conditions, require_unique=True, config=None):
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
    config : dict or str, optional
        if dict: config file
        if str: path to the config file
        If provided allows to beter check when a condition is at its default level, since it won't appear in the list of arguments of the run

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
    >>> select_compatible(run_args, {'tau': -10})
    '2'
    >>> select_compatible(run_args, {'tau': -10}, require_unique=False)
    ['2']
    >>> select_compatible(run_args, {'percent': 1}, require_unique=False)
    ['2', '3']
    >>> select_compatible(run_args, {'percent': 10}, require_unique=False)
    []
    '''
    _run_args = deepcopy(run_args)
    if config is not None:
        if isinstance(config, dict):
            config_dict = config
        elif isinstance(config, str):
            config_dict = ut.json2dict(config)
        else:
            raise TypeError(f'Invalid type {type(config)} for config')
        config_dict_flat = ut.collapse_dict(config_dict)
        conditions_at_default = {k:v for k,v in conditions.items() if v == config_dict_flat[k]}
        for args in _run_args.values():
            for k,v in conditions_at_default.items():
                if k not in args:
                    args[k] = v

    compatible_keys = [k for k,v in _run_args.items() if conditions.items() <= v.items()]
    if not require_unique:
        return compatible_keys

    if len(compatible_keys) == 0:
        raise KeyError(f'No previous compatible run satisfies {conditions = }')
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

def group_by_varying(run_args, variable='tau', config_dict_flat=None, sort=False, ignore=None):
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
    sort: True, False or 'descending', optional
        wether and how to sort the runs according to the variable, default False, i.e. no sorting, which means the runs will be in chronological order
    ignore : str or list[str], optional
        kwarg or list of kwargs to ignore when grouping rins together

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
    >>> group_by_varying(run_args, 'percent', sort=True)
    [{'args': {'tau': 0}, 'runs': ['2', '1'], 'percent': [1, 5]}, {'args': {'tau': -5}, 'runs': ['3'], 'percent': [1]}]
    >>> group_by_varying(run_args, 'percent', ignore='tau')
    [{'args': {}, 'runs': ['1', '2', '3'], 'percent': [5, 1, 1]}]
    '''
    _run_args = deepcopy(run_args)
    
    # remove arguments to ignore
    if ignore is not None:
        if isinstance(ignore, str):
            ignore = [ignore]
        if variable in ignore:
            raise ValueError('Cannot ignore the variable!')
        for args in _run_args.values():
            for ign in ignore:
                args.pop(ign, None)
    
    # find the groups
    try:
        # move the variable to a separate dictionary removing it from the arguments in _run_args
        variable_dict = {k:v.pop(variable) if variable in v else config_dict_flat[variable] for k,v in _run_args.items()}
    except TypeError as e:
        raise TypeError(f'{variable} is at default value in some runs, please provide config_dict_flat') from e

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

    if sort:
        for g in groups:
            isort = np.argsort(g[variable])
            if sort == 'descending':
                isort = isort[::-1]

            g['runs'] = [g['runs'][i] for i in isort]
            g[variable] = [g[variable][i] for i in isort]

    return groups

def make_groups(runs, variable='tau', config_dict_flat=None, sort=False, ignore=None):
    '''
    A wrapper of `group by varying` that allows to use directly the runs dictionary rather then needing to extract the run arguments

    Parameters
    ----------
    runs : dict
        dictionary with the runs
    variable : str, optional
        argument that varies inside each group, by default 'tau'
    config_dict_flat : dict, optional
        flattened config dictionary with the default values, by default None
    sort : True, False or 'descending', optional
        wether and how to sort the runs according to the variable, default False, i.e. no sorting, which means the runs will be in chronological order
    ignore : list[str], optional
        list of kwargs to ignore when grouping rins together
    
    Returns
    -------
    list
        list of groups. 
        Each group is a dictionary with the same structure as the output of `group_by_varying` but the argument 'runs' contains the full run dictionary instead of just the run numbers
    '''
    run_args = {k:v['args'] for k,v in runs.items()}
    groups = group_by_varying(run_args, variable=variable, config_dict_flat=config_dict_flat, sort=sort, ignore=ignore)
    for g in groups:
        g['runs'] = [runs[k] for k in g['runs']]
    return groups

def get_subset(runs, conditions, config_dict=None):
    '''
    Wrapper of `select_compatible` that allows to extract a subset of runs that satisfy certain conditions

    Parameters
    ----------
    runs : dict
        dictionary with the runs
    conditions : dict
        dictionary of run arguments that has to be contained in the arguments of a compatible run
    config : dict or str, optional
        if dict: config file
        if str: path to the config file
        If provided allows to beter check when a condition is at its default level, since it won't appear in the list of arguments of the run

    Returns
    -------
    dict
        subset of `runs`
    '''
    run_args = {k:v['args'] for k,v in runs.items()}
    subset = select_compatible(run_args, conditions, require_unique=False, config=config_dict)
    subset = {k:runs[k] for k in subset}
    return subset

def get_run(load_from, current_run_name=None, runs_path='./runs.json'):
    '''
    Parameters
    ----------
    load_from : dict, int, str, or None
        If dict:
            it is a dictionary with arguments of the run, plus the optional argument 'if_ambiguous_choose'.
            The latter can have value either 'last' or 'first', and tells the function which run to choose if multiple satisfy the compatibility conditions.
            If it is not provided, the function will require to have only one compatible run and will raise an error if the choice is ambiguous.
            The other items in the dictionary can be set to the special value 'same', which will set them to the value assumed by that argument in the current run.
        If str:
            it is parsed into a dictionary using `parse_run_name` function. `if_ambiguous_choose` is inferred from the beginning of `load_from`.
            For example providing 'last', will look for the most recent run without further conditions than normal compatibility
            Providing 'first--percent__1' will return the first compatible performed run with `percent` = 1
            Providing 'last--percent__1--tau__same' will return the last compatible run with `percent` = 1 and `tau` at the same value of the current run
            To use the 'same' keyword you must provide `current_run_name`
            `load_from` can also be the full name of a run, in which case compatibility checks are skipped. 
        If int:
            it is the number of the run (>0) or if negative is the n-th last run
        If None: 
            the function returns None
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
    # GM: examples could help to understand this function better.
    '''
    if load_from is None:
        return None

    runs = ut.json2dict(runs_path)

    # select only completed runs
    runs = {k: v for k,v in runs.items() if v['status'] == 'COMPLETED'}

    if isinstance(load_from, str) and load_from in [r['name'] for r in runs.values()]: # if load_from is precisely the name of one of the runs, we don't need to do anything more
        return load_from

    if_ambiguous_choose = None # either None, 'last' or 'first'

    # get if_ambiguous_choose and deal with the string type
    if isinstance(load_from, str):
        if load_from.startswith('last'):
            if_ambiguous_choose = 'last'
        elif load_from.startswith('first'):
            if_ambiguous_choose = 'first'
        
        try:
            load_from = int(load_from)
        except ValueError: # cannot convert load_from to int, so it must be a string that doesn't contain only numbers:
            load_from = parse_run_name(load_from, evaluate=True)
    elif isinstance(load_from, dict):
        if_ambiguous_choose = load_from.pop('if_ambiguous_choose', None)
    
    # now load_from is either int or dict

    # handle 'same' options
    additional_relevant_keys = []
    if isinstance(load_from, dict):
        for k,v in load_from.items():
            if v == 'same':
                if current_run_name is None:
                    raise ValueError("Cannot use 'same' special value without providing current_run_name")
                additional_relevant_keys.append(k)
        for k in additional_relevant_keys:
            load_from.pop(k)

    # arguments relevant for model architecture
    relevant_keys = list(get_default_params(create_model).keys()) + list(get_default_params(load_data).keys()) + ['nfolds'] + additional_relevant_keys

    # select only compatible runs
    runs = {k: v for k,v in runs.items() if check_compatibility(v['name'], current_run_name, relevant_keys=relevant_keys)}

    if len(runs) == 0:
        logger.warning('None of the previous runs is compatible with this one for performing transfer learning')
        # GM: It would be nice if the warning specifies the function that reports them.
        # AL: This can be achieved in formatting the logger
        return None

    if isinstance(load_from, int):
        l = load_from
    elif isinstance(load_from, dict):
        require_unique = if_ambiguous_choose is None
        l = select_compatible({k:v['args'] for k,v in runs.items()}, load_from, require_unique=require_unique)
        if require_unique: # l is a string
            l = int(l)
        else: # l is a list of str
            if len(l) == 0:
                logger.log(35, f'None of the previous runs satisfy the conditions {load_from = }')
                return None
            if if_ambiguous_choose == 'first':
                l = int(l[0])
            elif if_ambiguous_choose == 'last':
                l = int(l[-1])
            else:
                raise NotImplementedError(f'Unrecognized option {if_ambiguous_choose = }')
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

def move_to_folder(folder, additional_files=None):
    '''
    Copies this file and its dependencies to a given folder.

    Parameters
    ----------
    folder : str or Path
        destination folder
    additional_files : list[Path], optional
        list of additional files to copy in `folder`, by default None

    Raises
    ------
    FileExistsError
        If there is already code in `folder`
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
    if additional_files:
        for file in additional_files:
            shutil.copy(file, folder)

    # copy useful files from other directories to 'folder/ERA/'
    shutil.copy(path_to_here.parent / 'ERA/ERA_Fields_New.py', ERA_folder)
    shutil.copy(path_to_here.parent / 'ERA/TF_Fields.py', ERA_folder)
    shutil.copy(path_to_here.parent / 'general_purpose/cartopy_plots.py', ERA_folder)
    shutil.copy(path_to_here.parent / 'general_purpose/utilities.py', ERA_folder)

    print(f'Now you can go to {folder} and run the learning from there:\n')
    print(f'\n\ncd \"{folder}\"\n')
    # print(f'cd \"{folder}\"\n has been copied to your clipboard :)')
    # pyperclip.copy(f'cd \"{folder}\"')
    
    
############################################
########## DATA PREPROCESSING ##############
############################################

try:
    fields_infos = ut.json2dict('fields_infos.json')
    logger.info('Loaded field_infos from file')
except FileNotFoundError:
    logger.warning("Could not load field_infos: using the hardcoded version")
    fields_infos_Plasim = {
        't2m': { # how we label the field
            'name': 'tas', # how the variable is called in the *.nc files
            'filename_suffix': 'tas', # the ending of the filename
            'label': 'Temperature',
        },
        'mrso': { # how we label the field
            'name': 'mrso', # how the variable is called in the *.nc files
            'filename_suffix': 'mrso', # the ending of the filename
            'label': 'Soil Moisture',
        },
        't2m_inter': { # how we label the field
            'name': 'tas', # how the variable is called in the *.nc files
            'filename_suffix': 'tas_inter', # the ending of the filename
            'label': '3 day Temperature', # interpolated data
        },
        'mrso_inter': { # how we label the field
            'name': 'mrso', # how the variable is called in the *.nc files
            'filename_suffix': 'mrso_inter', # the ending of the filename
            'label': '3 day Soil Moisture', # interpolated data
        },
    }

    for h in [200,300,500,850]: # geopotential heights
        fields_infos_Plasim[f'zg{h}'] = {
            'name': 'zg',
            'filename_suffix': f'zg{h}',
            'label': f'{h} hPa Geopotential Height',
        }
    for h in [200,300,500,850]: # geopotential heights
        fields_infos_Plasim[f'zg{h}_inter'] = {
            'name': 'zg',
            'filename_suffix': f'zg{h}_inter',
            'label': f'3 day {h} hPa Geopotential Height',
        }

    field_infos_CESM = {
        "t2m": {
            "name": "TSA",
            "filename_suffix": "TSA",
            "label": "Temperature"
        },
        "mrso": {
            "name": "H2OSOI",
            "filename_suffix": "H2OSOI",
            "label": "Soil Moisture"},
        "zg500": {
            "name": "Z3",
            "filename_suffix": "Z3.500hPa",
            "label": "500 hPa Geopotential Height"
        }
    }

    field_infos_ERA5 = {
        't2m': {
            'name': 't2m',
            'filename_suffix': 't2m',
            'label': '2 meter temperature',
        },
        'zg500': {
            'name': 'z',
            'filename_suffix': 'zg',
            'label': '500 hPa Geopotential Height'
        },
        'mrso': {
            'name': 'swvl1',
            'filename_suffix': 'mrso',
            'label': 'Surface Soil Moisture',
        }
    }
    
    fields_infos = {
        'Plasim' : fields_infos_Plasim,
        'CESM'   : field_infos_CESM,
        'ERA5'   : field_infos_ERA5,
    }
    

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
def load_data(dataset_years=8000, year_list=None, sampling='', Model='Plasim', area='France', filter_area='France',
              lon_start=-64, lon_end=64, lat_start=0, lat_end=22, fillna=None, mylocal='/local/gmiloshe/PLASIM/',
              fields=['t2m','zg500','mrso_filtered'], preprefix='ANO_', datafolder=None):
    # AL: can't you use the `Model` argument to reconstruct datafolder?
    '''
    Loads the data into Plasim_Fields objects

    Parameters
    ----------
    dataset_years : int, optional
        number of years of the dataset, for now 8000 or 1000.
    year_list : array-like or str or int or tuple or None, optional
        list of years to load from the dataset. If None all years are loaded
        if str must be in the format 'range([<start>],<end>,[<step>])', where square brackets mean the argument is optional. It will be interpreted as np.range([<start>],<end>,[<step>])
        if tuple must be in format ([<start>],<end>,[<step>])
        if int is just like providing only <end>
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
        If `lon_start` >= `lon_end` the selection will start from `lon_start`, go over the end of the array and then continue from the beginning up to `lon_end`.
        Providing `lon_start` = `lon_end` will result in the longitude being rolled by `lon_start` steps
    fillna : float, optional
            value to fill the missing values with, by default None
    mylocal : list[str or Path], optional
        paths to the data storage. The program will look for each data file in the first path, if not found it will look in the next one and so on.
        For speed it is better if they are local paths.
    fields : list, optional
        list of field names to be loaded. Add '_filtered' to the name to have the values of the field outside `filter_area` set to zero.
        Add '_ghost' to the name of the field to load it but not use it for learning.
        This happens when you need to compute the labels on a field that won't be fed to the network.
    preprefix: str, optional
        The name of the input file starts with preprefix. In practice it is either null or 'ANO' which indicates precomputed anomalies
    datafolder: str, optional
        The name of the folder which lies inside `mylocal`, it defaults to Data_Plasim
    
        
    Returns
    -------
    _fields: dict
        dictionary of ERA_Fields.Plasim_Field objects
    '''
    
    if area != filter_area:
        logger.warning(f'Fields will be filtered on a different area ({filter_area}) than the region of interest ({area}). If {area} is not a subset of {filter_area} the area integral will be different with and without filtering.')
    if datafolder is None:
        datafolder = f'Data_{Model}'
    elif Model.lower() not in datafolder.lower():
        logger.warning(f'{datafolder = } does not contain the name of the model ({Model})')
    
    dataset_suffix = ''
    if Model.lower() == 'plasim':
        if dataset_years == 1000:
            dataset_suffix = ''
        elif dataset_years == 8000:
            dataset_suffix = 'LONG'
        else:
            raise ValueError(f'Invalid number of {dataset_years = }')

    if isinstance(year_list, str):
        if '(' not in year_list or ')' not in year_list:
            raise ValueError(f'Unable to parse {year_list = }')
        year_list = f"({year_list.split('(',1)[1].split(')',1)[0]})" # get just the arguments
        year_list = ast.literal_eval(year_list) # now year_list is int or tuple

    if isinstance(year_list,int):
        year_list = np.arange(year_list)
    elif isinstance(year_list, tuple):
        year_list = np.arange(*year_list) # unpack the arguments of the tuple

    if sampling == '3hrs': 
        prefix = ''
        if dataset_suffix == '':
            file_suffix = f'../Climate/{datafolder}/'
        else:
            file_suffix = f'../Climate/{datafolder}_{dataset_suffix}/'
    else:
        if dataset_suffix == '':
            prefix = f'{preprefix}'
            file_suffix = f'{datafolder}/'
        else:
            prefix = f'{preprefix}{dataset_suffix}_'
            file_suffix = f'{datafolder}_{dataset_suffix}/'

    # load the fields
    _fields = {}
    for field_name in fields:
        ghost = False
        if field_name.endswith('_ghost'):
            field_name = field_name.rsplit('_', 1)[0] # remove '_ghost'
            ghost = True

        do_filter = False
        if field_name.endswith('_filtered'): # TO IMPROVE: if you have to filter the data load just the interesting part
            field_name = field_name.rsplit('_', 1)[0] # remove '_filtered'
            do_filter = True
            
        if field_name not in fields_infos[Model]:
            raise KeyError(f'Unknown field {field_name}')
        f_infos = fields_infos[Model][field_name]
        # create the field object
        field = ef.Plasim_Field(f_infos['name'], f"{file_suffix}{prefix}{f_infos['filename_suffix']}.nc", f_infos['label'], Model,
                                years=dataset_years, mylocal=mylocal)
        # select years
        field.select_years(year_list)

        # Sort the latitudes
        field.sort_lat()

        # select longitude and latitude
        field.select_lonlat(lat_start,lat_end,lon_start,lon_end,fillna) # this loads the field into memory

        # filter
        if do_filter: # set to zero all values outside `filter_area
            logger.info(f'Filtering field {field_name} over {filter_area}')
            field.set_mask(filter_area)
            field.filter()


        # prepare to compute area integral when needed
        field.set_mask(area)

        if ghost:
            field_name += '_ghost'

        _fields[field_name] = field
    
    return _fields

@ut.execution_time
@ut.indent_logger(logger)
def assign_labels(field, time_start=30, time_end=120, T=14, percent=5, threshold=None, label_period_start=None, label_period_end=None, A_weights=None, return_threshold=False):
    '''
    Given a field of anomalies it computes the `T` days forward convolution of the integrated anomaly and assigns label 1 to anomalies above a given `threshold`.
    If `threshold` is not provided, then it is computed from `percent`, namely to identify the `percent` most extreme anomalies.

    Parameters
    ----------
    field : Plasim_Field object
    time_start : int, optional
        first day of the period of interest (0 is the means the first datapoint of each year, so time_start is an index, not the day of the year)
    time_end : int, optional
        first day after the end of the period of interst
    T : int, optional
        width of the window for the running average
    percent : float, optional
        percentage of the most extreme heatwaves
    threshold : float, optional
        if provided overrides `percent`.
    label_period_start : int, optional
        if provided the first day of the period of interest for the label threshold determination
    label_period_end : int, optional
        if provided the first day after the end of the period of interst for the label threshold determination
    A_weights: list, optional
        if provided will influence how running mean is computed
    return_threshold: bool, optional
        if provided as True the output also involves threshold_new

    Returns:
    --------
    labels : np.ndarray
        2D array with shape (years, days) and values 0 or 1
    (threshold) if return_threshold : float
        the computed threshold
    '''
    day0 = field.field.time.dt.dayofyear[0]
    logger.info(f"{A_weights = }")
    if threshold is None:
        if (label_period_start is not None) and (label_period_end is None):
            A = field.compute_time_average(day_start=day0+label_period_start, day_end=day0+time_end, T=T, weights=A_weights)
            _, threshold = ef.is_over_threshold(field.to_numpy(A), threshold=None, percent=percent)
        elif (label_period_start is None) and (label_period_end is not None):
            A = field.compute_time_average(day_start=day0+time_start, day_end=day0+label_period_end, T=T, weights=A_weights)
            _, threshold = ef.is_over_threshold(field.to_numpy(A), threshold=None, percent=percent)
        elif (label_period_start is not  None) and (label_period_end is not None):
            A = field.compute_time_average(day_start=day0+label_period_start, day_end=day0+label_period_end, T=T, weights=A_weights)
            _, threshold = ef.is_over_threshold(field.to_numpy(A), threshold=None, percent=percent)
    
    A = field.compute_time_average(day_start=day0+time_start, day_end=day0+time_end, T=T, weights=A_weights)
    if hasattr(field, 'A'): 
        field.A = A # This is a placeholder variable that can be used as a running mean save. Problem is that our routines do not output it and rewritting could cause issues with backward compatibility
    labels, threshold = ef.is_over_threshold(field.to_numpy(A), threshold=threshold, percent=percent)
    logger.info(f"{threshold = }")
    if return_threshold:
        return np.array(labels, dtype=int), threshold
    return np.array(labels, dtype=int)

@ut.execution_time
@ut.indent_logger(logger)
def make_X(fields, time_start=30, time_end=120, T=14, tau=0):
    '''
    Cuts the fields in time and stacks them. The original fields are not modified

    Parameters
    ----------
    fields : dict of Plasim_Field objects
    time_start : int or None, optional
        first day of the period of interest (counting from the start of the dataset, so 0 means the first day in the dataset).
        If None, time_start will be max(-tau,0), i.e. the minimum value possible without causing problems with the shift of the data
    time_end : int or None, optional
        first day after the end of the period of interst (counting from the start of the dataset)
        If None time_end will be the maximum value possible without causing problems with the shift of the data
    T : int, optional
        width of the window for the running average
    tau : int, optional
        delay between observation and prediction (meaningful when negative)

    Returns
    -------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    '''
    if time_start is None:
        time_start = max(-tau,0)
        logger.info(f'Setting {time_start = } (maximum range)')
    if time_end is None:
        time_end = list(fields.values())[0].var.shape[1] - max(tau-T+1,0)
        logger.info(f'Setting {time_end = } (maximum range)')
    if time_start + tau < 0:
        raise IndexError(f'Too large delay {tau = }, the maximum delay is {-time_start = }')
    # stack the fields
    X = np.array([field.var[:, time_start+tau:time_end+tau-T+1, ...] for field_name,field in fields.items() if not field_name.endswith('_ghost')])
    # now transpose the array so the field index becomes the last
    X = X.transpose(*range(1,len(X.shape)), 0)
    return X

@ut.execution_time
@ut.indent_logger(logger)
def make_XY(fields, label_field='t2m', time_start=30, time_end=120, T=14, tau=0, percent=5, threshold=None, label_period_start=None, label_period_end=None, A_weights=None, return_threshold=False):
    '''
    Combines `make_X` and `assign_labels`

    Parameters:
    -----------
    fields : dict of Plasim_Field objects
    label_field : str, optional
        key for the field used for computing labels
    time_start : int or None, optional
        first day of the period of interest (counting from the start of the dataset)
        If None defaults to the maximum range available
    time_end : int or None, optional
        first day after the end of the period of interst (counting from the start of the dataset)
        If None defaults to the maximum range available
    T : int, optional
        width of the window for the running average
    tau : int, optional
        delay between observation and prediction
    percent : float, optional
        percentage of the most extreme heatwaves
    threshold : float, optional
        if provided overrides `percent`
    label_period_start : int, optional
        if provided the first day of the period of interest for the label threshold determination
    label_period_end : int, optional
        if provided the first day after the end of the period of interst for the label threshold determination
    A_weights: list, optional
        if provided will influence how running mean is computed
    return_threshold: bool, optional
        if provided as True the output also involves threshold_new
        
    Returns:
    --------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    Y : np.ndarray
        with shape (years, days)
    (threshold) if return_threshold : float
        the computed threshold
    '''
    #if label_period_start is not None:
    #    if label_period_start < time_start:
    #       raise ValueError(f'Bad parameters specified: {label_period_start = } is less than {time_start = }')
    #if label_period_end is not None:
    #    if label_period_end > time_end:
    #       raise ValueError(f'Bad parameters specified: {label_period_end = } is more than {time_end = }')
    logger.info(f'{time_start = }, {time_end = }, {label_period_start = }, {label_period_end = }, {T = }')
    if time_start is None:
        time_start = max(-tau,0)
        logger.info(f'Setting {time_start = } (maximum range)')
    if time_end is None:
        time_end = list(fields.values())[0].var.shape[1] - max(tau-T+1,0)
        logger.info(f'Setting {time_end = } (maximum range)')
    X = make_X(fields, time_start=time_start, time_end=time_end, T=T, tau=tau)
    try:
        lf = fields[label_field]
    except KeyError:
        try:
            lf = fields[f'{label_field}_ghost']
        except KeyError:
            raise KeyError(f'Unable to find label field {label_field} among the provided fields {list(fields.keys())}')

    Y = assign_labels(lf, time_start=time_start, time_end=time_end, T=T, percent=percent, threshold=threshold, label_period_start=label_period_start, label_period_end=label_period_end, A_weights=A_weights, return_threshold=return_threshold)
    if return_threshold: # Y is actually a tuple
        Y, threshold_new = Y
        return X,Y,threshold_new
    return X,Y

@ut.execution_time
@ut.indent_logger(logger)
def roll_X(X, roll_axis='lon', roll_steps=0):
    '''
    Rolls `X` along a given axis. useful for example for moving France away from the Greenwich meridian.
    In other words this allows one, for example, to shift the grid so that desired areas are not found at the boundary.
    In principle this function allows us to roll along arbitrary axis, including days or years.

    Parameters
    ----------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    roll_axis : int or str, optional
        'year' (or 'y'), 'day' (or 'd'), 'lat', 'lon', 'field' (or 'f')
    roll_steps : int, optional
        number of gridsteps to roll: a positive value for `roll_steps` means that the elements of the array are moved forward in it,
        e.g. `roll_steps` = 1 means that the old first element is now in the second place
        This means that for every axis a positive value of `roll_steps` yields a shift of the array
        'year', 'day' : forward in time
        'lat' : southward
        'lon' : eastward
        'field' : forward in the numbering of the fields
    
    Returns
    -------
    np.ndarray
        of the same shape of `X`
    '''
    if roll_steps == 0:
        return X
    logger.warn('DeprecationWarning: Roll the data when loading it using the arguments `lon_start`, `lon_end` of function `load_data`')
    if isinstance(roll_axis, str):
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
    elif not isinstance(roll_axis, int):
        raise TypeError(f'roll_axis can be int or str, not {type(roll_axis)}')
    # at this point roll_axis is an int
    return np.roll(X,roll_steps,axis=roll_axis)

@ut.execution_time
@ut.indent_logger(logger)
def normalize_X(X, fold_folder, mode='pointwise', recompute=False):
    '''
    Performs data normalization x_norm = (x - x_mean)/x_std and saves it into fold_folder
    if the normalization already exists the x_mean and x_std are reloaded and mode is ignored

    Parameters
    ----------
    X : np.ndarray of shape (N, lat, lon, fields)
        data
    fold_folder:
        where the normalization and other fold specific parameters are stored
    mode : str, optional
        how to perform the normalization, possibilities are:
            'pointwise': every gridpoint of every field is treated independenly (default)
            'global': mean and std are computed globally on each field
            'mean': mean and std are computed pointwise and then averaged over each field
    recompute: bool
        If True the normalization will be computed again based on the inputs, even if the 
        normalization files already exists. Notice, this action will overwrite the files

    Returns
    -------
    X_n : np.ndarray of same shape as X
        normalized data
    X_mean : np.ndarray 
        mean of X along the axes given by the normalization mode
    X_std : np.ndarray
        std of X along the axes given by the normalization mode

    Raises
    ------
    NotImplementedError
        if mode != 'pointwise'
    '''
    if os.path.exists(f'{fold_folder}/X_mean.npy') and os.path.exists(f'{fold_folder}/X_std.npy') and not recompute:
        logger.info(f'loading from: {fold_folder}/X_mean.npy and {fold_folder}/X_std.npy')
        X_mean = np.load(f'{fold_folder}/X_mean.npy')
        X_std = np.load(f'{fold_folder}/X_std.npy')
    else:
        if mode == 'pointwise':
            # renormalize data with pointwise mean and std
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
        elif mode == 'global':
            # mean over all axes except the last one (field axis). This does not work very well with filtered fields
            axis = tuple(range(len(X.shape) - 1))
            X_mean = np.mean(X, axis=axis)
            X_std = np.std(X, axis=axis)
        elif mode == 'mean':
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            nfields = X.shape[-1]
            for i in range(nfields):
                mask = X_std[...,i] != 0 # work only on the points that display some fluctuations
                non_zero_non_fluctuating_points = np.sum(X_mean[...,i][~mask] != 0) # number of gridpoints that have a non zero value that is always the same
                if non_zero_non_fluctuating_points:
                    logger.warning(f'Field {i} has {non_zero_non_fluctuating_points} non zero non fluctuating gridpoints')
                X_mean[...,i][mask] = np.mean(X_mean[...,i][mask])
                X_std[...,i][mask] = np.mean(X_std[...,i][mask])
        else:
            raise NotImplementedError(f'Unknown normalization {mode = }')

        logger.info(f'{np.sum((X_std < 1e-4)*(X_std > 0))/np.product(X_std.shape)*100 :.4f}\% of the data have non zero std below 1e-4')
        X_std[X_std==0] = 1 # If there is no variance we shouldn't divide by zero ### hmmm: this may create discontinuities
                            # This is necessary because we will be masking (filtering) certain parts of the map 
                            #     for certain fields, e.g. mrso, setting them to zero. Also there are some fields that don't
                            #     vary over the ocean. I've tried normalizing by field, rather than by grid point, in which case
                            #     the problem X_std==0 does not arise, but then the results came up slightly worse.
        # save X_mean and X_std
        logger.info(f'saving to: {fold_folder}/X_mean.npy and {fold_folder}/X_std.npy')
        np.save(f'{fold_folder}/X_mean.npy', X_mean) 
        np.save(f'{fold_folder}/X_std.npy', X_std)
    
    return  (X - X_mean)/X_std, X_mean, X_std


#####################################################
########### PCA Autoencoder  ################        
##################################################### 

class PCAencoder(PCA):
    """_summary_

    Args:
        PCA (_type_): _description_
    """
    def predict(self,*args,**kwargs):
        _X = self.transform(args[0].reshape(args[0].shape[0],-1))
        return _X
    def summary(self):
        print(f'We are computing PCA')

class PCAer:
    """
        Essentially decorator class that keeps the inputs and outputs maximally similar to autoencoder so that we could use the same routines
        Z_DIM: int
            dimensionality of the latent space
        folder: string
            where PCAer stores its encoder
    """
    def __init__(self, *args, Z_DIM=2, folder='./', **kwargs):
        self.Z_DIM = Z_DIM
        if os.path.exists(f'{folder}/encoder.pkl'):
            logger.info(f"laoding existing file from {folder}/encoder.pkl")
            with open(f'{folder}/encoder.pkl', 'rb') as file_pi:
                self.encoder = pickle.load(file_pi)
        else:
            logger.info(f"The file {folder}/encoder.pkl does not exist, creating new PCAer")
            self.encoder = PCAencoder(n_components=Z_DIM, svd_solver="randomized", whiten=True)
        self.shape = None # it gets a value when calling method fit()
        self.folder = folder # where the Autoencoder is saved
    # To be able to create this class using `with` statement
    def __enter__(self):
        # Return the instance of the class
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        # Clean up any resources used by the class
        pass

    def fit(self,*args, **kwargs):
        if os.path.exists(f'{self.folder}/encoder.pkl'):
            logger.warning("Fit will not be performed because the file exists")
        else:
            logger.info(f'The relevant file is absent so performing fit to the shape {args[0].shape = }')
            result_fit = self.encoder.fit(args[0].reshape(args[0].shape[0],-1)) # PCA expects the input of type fit(X) such that X is 2 dimensional
            self.shape = args[0].shape[0]
            logger.info(f'{np.sum(self.encoder.explained_variance_ratio_) = }')
            logger.info(f'saving in {self.folder}/encoder.pkl')
            with open(f'{self.folder}/encoder.pkl', 'wb') as file_pi:
                pickle.dump(self.encoder, file_pi)
        return result_fit 

    def fit_with_timeout(self,counter,*args,timeout=300,maxIter=3, **kwargs):
        '''
            This is a wrapper which runs fit() for maxIter times until
            the fit() produces result (doesn't hang) in `timeout` seconds.
            Basically this method addresses the problem that the PCA.fit often hangs
        '''
        def handler(signum, frame):
            raise TimeoutError("Computation timed out")
        logger.info(f'{counter = }')
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        counter=counter+1
        if counter<maxIter:
            try:
                logger.info("calling self.fit(*args, **kwargs)")
                result = self.fit(*args, **kwargs)
            except TimeoutError:
                logger.warning("Computation timed out. Restarting...")
                result = self.fit_with_timeout(counter,*args, timeout=timeout,maxIter=maxIter, **kwargs)
            finally:
                signal.alarm(0)
            return result
        else:
            logger.error("reached too many iterations")
            return None
    
    def score(self,*args,**kwargs):
        return self.encoder.score(args[0].reshape(args[0].shape[0],-1))
    
    def decoder(self,X):
        return self.encoder.inverse_transform(X).reshape(self.shape)
    
    def summary(self):
        logger.info(f'PCA with {self.Z_DIM} components')

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
        np.ndarray of the same shape of X (permuted data)
    else:
        np.ndarray of shape (X.shape[0],) (permutation)
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
    When applied in this context it shuffles the years in such a way that the `nfolds` folds have a number of heatwave events which is as equal as possible

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
        logger.info(f'Sums of the balanced {nfolds} folds:\n{sums}\nstd/avg = {np.std(sums)/target_sum :.3f}\nmax relative deviation = {np.max(np.abs(sums - target_sum))/target_sum*100 :.3f}\%')

    return permutation

def undersample(X, Y, u=1, random_state=42):
    '''
    Performs undersampling of the majority class. Warning: modifies the provided X,Y

    Parameters
    ----------
    X : np.ndarray of shape (N, ...)
        data
    Y : np.ndarray of shape (N,)
        labels
        N = n0 + n1 with n0 > n1 corresponds to the majority class
    u : float >= 1, optional
        undersampling factor , by default 1
    random_state : int, optional
        seed for the undersampler, by default 42

    Returns
    -------
    X : np.ndarray of shape (M, ...)
        undersampled data
    Y : np.ndarray of shape (M,)
        undersampled labels
        M = n0/u + n1

    Raises
    ------
    NotImplementedError
        If u < 1
    ValueError
        If u > n0/n1
    '''
    if u < 1:
        raise NotImplementedError(f'{u = } < 1')
    elif u == 1:
        return X, Y

    n_pos_tr = np.sum(Y)
    n_neg_tr = len(Y) - n_pos_tr
    logger.info(f'number of training data before undersampling: {len(Y)} of which {n_neg_tr} negative and {n_pos_tr} positive')
    
    undersampling_strategy = n_pos_tr/(n_neg_tr/u)
    # TODO #52 seems that this condition is restrictive. I don't see a reason why we need to limit the majority class to never be smaller@georgemilosh
    # as indicated by Alessandro there could be problems if we tried to remove this valueerror
    if undersampling_strategy > 1: # you cannot undersample so much that the majority class becomes the minority one 
        raise ValueError(f'Too high undersmapling factor, maximum for this dataset is u={n_neg_tr/n_pos_tr}')
    pipeline = Pipeline(steps=[('u', RandomUnderSampler(random_state=random_state, sampling_strategy=undersampling_strategy))])
    # reshape data to feed it to the pipeline
    X_shape = X.shape
    X = X.reshape((X_shape[0], np.product(X_shape[1:])))
    X, Y = pipeline.fit_resample(X, Y) # apply pipeline
    X = X.reshape((X.shape[0], *X_shape[1:])) # reshape back

    return X, Y
    

################################################
########## NEURAL NETWORK DEFINITION ###########
################################################

def create_model(input_shape, conv_channels=[32,64,64], kernel_sizes=3, strides=1, padding='valid',
                 batch_normalizations=True, conv_activations='relu', conv_dropouts=0.2, max_pool_sizes=[2,2,False], conv_l2coef=None, conv_skip=None,
                 rnn_units=None, rnn_type='LSTM', rnn_activations=None, rnn_dropouts=False, rnn_l2coef=None, rnn_batch_norm=None, rnn_return_sequences=False,
                 dense_units=[64,2], dense_activations=['relu', None], dense_dropouts=[0.2,False], dense_l2coef=None, dense_batch_norm=None):
    '''
    Creates a model consisting of a series of convolutional layers followed by fully connected ones

    Parameters
    ----------
    input_shape : tuple
        shape of a single input datapoint, i.e. not counting the axis corresponding to iteration through the datapoints (batch axis)
    conv_channels : list of int, optional
        number of channels corresponding to the convolutional layers
    kernel_sizes : int, 2-tuple or list of ints or 2-tuples, optional
        If list must be of the same size of `conv_channels`
    strides : int, 2-tuple or list of ints or 2-tuples, optional
        same as kernel_sizes
    padding : string, optional defaults to 'valid'
        one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to 
        the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input
    batch_normalizations : bool or list of bools, optional
        whether to add a BatchNormalization layer after each Conv2D layer
    conv_activations : str or list of str, optional
        activation functions after each convolutional layer
    conv_dropouts : float in [0,1] or list of floats in [0,1], optional
        dropout to be applied after the BatchNormalization layer. If 0 no dropout is applied
    max_pool_sizes : int or list of int, optional
        size of max pooling layer to be applied after dropout. If 0 no max pool is applied
    conv_l2coef : list of floats which encodes the values of L2 regularizers in convolutional layers, optional, defaults to None
    encoder_conv_skip: a list of lists that gets converted to a dictionary
        creates a skip connection between two layers given by key and value entries in the dictionary. 
        If empty no skip connections are included. The skip connection will not work if 
        the dimensions of layers mismatch. For this convolutional architecture should be implemented in future

    rnn_units : list of int, optional
        number of cells for each RNN layer
    rnn_type :  list of string, optional
        Either 'LSTM' or 'GRU'
    rnn_activations : str or list of str, optional
        activation functions after each RNN layer
    rnn_dropouts : float in [0,1] or list of floats in [0,1], optional 
    rnn_l2coef :list of floats, optional
        hich encodes the values of L2 regularizers in RNN layers, optional, defaults to None
    rnn_batch_norm :  bool or list of bools, optional
        whether to add a BatchNormalization layer after each dense layer
    rnn_return_sequences : bool or list of bools, optional
        whether to return internal states of RNNs
        

    dense_units : list of int, optional
        number of neurons for each fully connected layer
    dense_activations : str or list of str, optional
        activation functions after each fully connected layer
    dense_dropouts : float in [0,1] or list of floats in [0,1], optional
    dense_l2coef :list of floats, optional
        which encodes the values of L2 regularizers in dense layers, optional, defaults to None
    dense_batch_norm:  bool or list of bools, optional
        whether to add a BatchNormalization layer after each dense layer

    Returns
    -------
    model : keras.models.Model
    '''
    if conv_skip is not None:
        conv_skip = dict(tuple(map(tuple, conv_skip)))
    else:
        conv_skip = dict({})
    
    # convolutional layers
    # adjust the shape of the arguments to be of the same length as conv_channels
    args = [kernel_sizes, strides, batch_normalizations, conv_activations, conv_dropouts, max_pool_sizes, conv_l2coef, padding]
    logger.info(f'{args = }')
    x = []
    inputs = tf.keras.Input(shape=input_shape, name='input')
    x.append(inputs)
    if conv_channels is not None:
        for j,arg in enumerate(args):
            if not isinstance(arg, list):
                args[j] = [arg]*len(conv_channels)
            elif len(arg) != len(conv_channels):
                raise ValueError(f'Invalid length for argument {arg = } when compared with {conv_channels = }')
        logger.info(f'convolutional args = {args}')
        kernel_sizes, strides, batch_normalizations, conv_activations, conv_dropouts, max_pool_sizes, conv_l2coef, padding = args
        
        n_layers = len(conv_channels)
        
        # Add convolutional layers
        for i in range(n_layers):
            if conv_l2coef[i] is not None:
                kernel_regularizer=tf.keras.regularizers.l2(conv_l2coef[i])
            else:
                kernel_regularizer=None
            # print(i, f"Conv2D, filters = {encoder_conv_filters[i]}, kernel_size = {encoder_conv_kernel_size[i]}, strides = {encoder_conv_strides[i]}, padding = {encoder_conv_padding[i]}")
            conv = layers.Conv2D(filters = conv_channels[i], 
                    kernel_size = kernel_sizes[i],
                    strides = strides[i], 
                    padding = padding[i],
                    kernel_regularizer=kernel_regularizer,
                    name = f'conv_layer_{i}')(x[-1])

            if batch_normalizations[i]:
                conv = layers.BatchNormalization(name=f'batch_norm_{i}')(conv)
                # print("conv = BatchNormalization()(conv)")
                
            if conv_activations[i] == 'LeakyRelu':
                actv = layers.LeakyReLU(name=f'conv_activation_{i}')(conv)
                # print("actv = LeakyReLU()(conv)")
            else:
                actv = layers.Activation(conv_activations[i], name=f'conv_activation_{i}')(conv)
                # print("actv = Activation(conv_activation[i])(conv)")

            if conv_dropouts[i]:
                actv = layers.SpatialDropout2D(rate=conv_dropouts[i], name=f'spatial_dropout_{i}')(actv)
                # print("actv = Dropout(rate=0.25)(actv)")
            if max_pool_sizes[i]: # otherwise I get an error if max_pool_sizes[i] is None  because it cannot compare NoneType and int
                if max_pool_sizes[i] > 1:
                    actv = layers.MaxPooling2D(max_pool_sizes[i], name=f'max_pool_{i}')(actv)
            
            if conv_skip is not None:
                #logger.info(f'{i = },{conv_skip = }')
                if i in conv_skip.keys(): # The arrow of the skip connection starts here
                    # print('arrow_start = actv')
                    arrow_start = actv
                    logger.info(f'{arrow_start = }')
                    
                if i in conv_skip.values(): # The arrow of the skip connection end here
                    # print('conv = keras.layers.add([conv, arrow_start])')
                    actv = keras.layers.Add()([actv, arrow_start])
                    logger.info(f'{actv = }')
                    #if encoder_use_batch_norm:
                    #    actv = BatchNormalization()(actv)
                    #    # print("actv = BatchNormalization()(actv)")
            
            x.append(actv)
    
    if rnn_units is not None:
        feature_shape = tff.K.int_shape(x[-1])[1:] # The idea is to keep the sequence (axis=1) shape assuming it exists
        logger.info(f'{feature_shape = }')
        feature_shape = (feature_shape[0],np.prod(np.array(feature_shape[1:]))) # flatten from axis=1 on
        logger.info(f'{feature_shape = }')
        x.append(layers.Reshape(feature_shape)(x[-1]))
        args = [rnn_activations, rnn_dropouts, rnn_l2coef, rnn_batch_norm, rnn_type, rnn_return_sequences]
        for j,arg in enumerate(args):
            if not isinstance(arg, list):
                args[j] = [arg]*len(rnn_units)
            elif len(arg) != len(rnn_units):
                raise ValueError(f'Invalid length for argument {arg = } when compared with {rnn_units = }')
        logger.info(f'rnn args = {args}')
        rnn_activations, rnn_dropouts, rnn_l2coef, rnn_batch_norm, rnn_type, rnn_return_sequences = args
        # build the dense layers
        for i in range(len(rnn_units)):
            if rnn_l2coef[i] is not None:
                kernel_regularizer=tf.keras.regularizers.l2(rnn_l2coef[i])
            else:
                kernel_regularizer=None
            
            if rnn_type[i] == 'LSTM':
                rnn = layers.LSTM(rnn_units[i], kernel_regularizer=kernel_regularizer, return_sequences=rnn_return_sequences[i], name=f"rnn_layer_{i}")(x[-1])
            elif rnn_type[i] == 'GRU':
                rnn = layers.GRU(rnn_units[i], kernel_regularizer=kernel_regularizer, return_sequences=rnn_return_sequences[i], name=f"rnn_layer_{i}")(x[-1])
            else:
                raise ValueError(f'{rnn_type[i]} not implemented')
            if rnn_batch_norm[i]:
                rnn = layers.BatchNormalization(name=f"rnn_batch_{i}")(rnn)
            if rnn_activations[i] == 'LeakyRelu':
                actv = layers.LeakyReLU(name=f"rnn_activation_{i}")(rnn)
            else:
                actv = layers.Activation(rnn_activations[i],name=f"rnn_activation_{i}")(rnn)
            if rnn_dropouts[i]:
                actv = layers.Dropout(rate=rnn_dropouts[i],name=f"rnn_dropout_{i}")(actv)
            x.append(actv)
        
    # dense layers
    # adjust the shape of the arguments to be of the same length as conv_channels
    

    if dense_units is not None:
        x.append(layers.Flatten()(x[-1]))
        args = [dense_activations, dense_dropouts, dense_l2coef, dense_batch_norm]
        for j,arg in enumerate(args):
            if not isinstance(arg, list):
                args[j] = [arg]*len(dense_units)
            elif len(arg) != len(dense_units):
                raise ValueError(f'Invalid length for argument {arg = } when compared with {dense_units = }')
        logger.info(f'dense args = {args}')
        dense_activations, dense_dropouts, dense_l2coef, dense_batch_norm = args
        # build the dense layers
        for i in range(len(dense_units)):
            if dense_l2coef[i] is not None:
                kernel_regularizer=tf.keras.regularizers.l2(dense_l2coef[i])
            else:
                kernel_regularizer=None
            
            dense = layers.Dense(dense_units[i], kernel_regularizer=kernel_regularizer, name=f"dense_layer_{i}")(x[-1])
            if dense_batch_norm[i]:
                dense = layers.BatchNormalization(name=f"dense_batch_{i}")(dense)
            if dense_activations[i] == 'LeakyRelu':
                actv = layers.LeakyReLU(name=f"dense_activation_{i}")(dense)
            else:
                actv = layers.Activation(dense_activations[i],name=f"dense_activation_{i}")(dense)
            if dense_dropouts[i]:
                actv = layers.Dropout(rate=dense_dropouts[i],name=f"dense_dropout_{i}")(actv)
            x.append(actv)
        
        #model.add(layers.Dense(dense_units[i], activation=dense_activations[i],kernel_regularizer=kernel_regularizer))
        
        #if dense_dropouts[i]:
        #    model.add(layers.Dropout(dense_dropouts[i]))
    model = tf.keras.Model(x[0], x[-1], name="model")
    return model

########################################
###### TRAINING THE NETWORK ############
########################################


def scheduler(epoch, lr=5e-4, epoch_tol=None, warmup=False, lr_min=5e-4, decay=0.1):
    '''
    If `warmup`=False this function keeps the initial learning rate for the first `epoch_tol` epochs
      and decreases it exponentially after that.
    If `warmup`=True starts with 0 learning rate and increases it until we reach epoch_tol
    Parameters
    ----------
    epoch_tol: int
        epoch until which we apply flat lr learning rate, if None learning rate will be fixed
    lr: float
        base learning rate
    lr_min: float
        minimal learning rate we are supposed to reach asymptotically
    decay: float
        the parameter which defines how quickly our learning rate decays
  '''
    if epoch_tol is None:
        return lr
    elif epoch < epoch_tol:
        if warmup: # we assume linearly increasing learning rate
            return lr*epoch/epoch_tol
        else:
            return lr
    else:
        new_lr = lr*tf.math.exp(-decay*(epoch-epoch_tol+1))
        if new_lr < lr_min:
            new_lr = lr_min
        return new_lr


class PrintLR(tf.keras.callbacks.Callback):
    '''
        Prints learning rate given the input model
    '''
    def __init__(self, model):
        self.model = model
    def on_epoch_end(self, epoch, logs=None):
        logger.info('\nLearning rate for epoch {} is {}'.format(        epoch + 1, self.model.optimizer.lr.numpy()))

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
    return keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, mode=mode, restore_best_weights=True)

def make_checkpoint_callback(file_path, checkpoint_every=1):
    '''
    Creates a ModelCheckpoint callback

    Parameters
    ----------
    file_path : str
        path to the folder where the checkpoints will be stored. Can also have a format, for example <folder>/cp-{epoch:04d}.ckpt
    checkpoint_every : int or str, optional
        Examples:
        0: disabled
        5 or '5 epochs' or '5 e': every 5 epochs
        '100 batches' or '100 b': every 100 batches
        'best custom_loss': every time 'custom_loss' reaches a new minimum. 'custom_loss' must be in the list of metrics

    Returns
    -------
    keras.callbacks.ModelCheckpoint

    Raises
    ------
    ValueError
        If the system doesn't manage to interpret `checkpoint_every`
    '''
    ckpt_callback = None
    if checkpoint_every == 0: # no checkpointing
        pass
    elif checkpoint_every == 1: # save every epoch
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=file_path, save_weights_only=True, verbose=1)
    elif isinstance(checkpoint_every, int): # save every `checkpoint_every` epochs 
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=file_path, save_weights_only=True, verbose=1, period=checkpoint_every)
    elif isinstance(checkpoint_every, str): # parse string options
        if checkpoint_every[0].isnumeric():
            every, what = checkpoint_every.split(' ',1)
            every = int(every)
            if what.startswith('b'): # every batch
                ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=file_path, save_weights_only=True, verbose=1, save_freq=every)
            elif what.startswith('e'): # every epoch
                ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=file_path, save_weights_only=True, verbose=1, period=every)
            else:
                raise ValueError(f'Unrecognized value for {checkpoint_every = }')

        elif checkpoint_every.startswith('best'): # every best of something
            monitor = checkpoint_every.split(' ',1)[1]
            ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=1)
        else:
            raise ValueError(f'Unrecognized value for {checkpoint_every = }')
    else:
        raise ValueError(f'Unrecognized value for {checkpoint_every = }')

    return ckpt_callback

def postprocess(x):
    return keras.layers.Softmax()(x)

@ut.execution_time
@ut.indent_logger(logger)
def train_model(model, X_tr, Y_tr, X_va, Y_va, folder, num_epochs, optimizer, loss, metrics, early_stopping_kwargs=None, enable_early_stopping=False, scheduler_kwargs=None,
                u=1, batch_size=1024, checkpoint_every=1, additional_callbacks=['csv_logger'], return_metric='val_CustomLoss'):
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
    scheduler_kwargs: dict
        arguments which define the behavior of the learning rate schedule.
    early_stopping_kwargs : dict
        arguments to create the early stopping callback. Ignored if `enable_early_stopping` = False
    enable_early_stopping : bool, optional
        whether to perform early stopping or not, by default False
    u : float, optional
        undersampling factor (>=1). Used for unbiasing and saving the committor
    batch_size : int, optional
        by default 1024
    checkpoint_every : int or str, optional
        Examples:
        0: disabled
        5 or '5 epochs' or '5 e': every 5 epochs
        '100 batches' or '100 b': every 100 batches
        'best custom_loss': every time 'custom_loss' reaches a new minimum. 'custom_loss' must be in the list of metrics
    additional_callbacks : list of keras.callbacks.Callback objects or list of str, optional
        string items are interpreted, for example 'csv_logger' creates a CSVLogger callback that saves the history to a csv file
    return_metric : str, optional
        name of the metric of which the minimum value will be returned at the end of training

    Returns
    -------
    float
        minimum value of `return_metric` during training
    '''
    ### preliminary operations
    ##########################
    if early_stopping_kwargs is None:
        early_stopping_kwargs = {}
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
    ckpt_callback = make_checkpoint_callback(ckpt_name, checkpoint_every=checkpoint_every)

    if ckpt_callback is not None:
        callbacks.append(ckpt_callback)

    # early stopping callback
    if enable_early_stopping:
        if 'patience' not in early_stopping_kwargs or early_stopping_kwargs['patience'] == 0:
            logger.warning('Skipping early stopping with patience = 0')
            enable_early_stopping = False
        else:
            callbacks.append(early_stopping(**early_stopping_kwargs))
            
     
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    logger.info(f"{scheduler_kwargs = }")
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_callback,save_weights_only=True,verbose=1)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(partial(scheduler, **scheduler_kwargs)) 
    callbacks.append(scheduler_callback)
    callbacks.append(PrintLR(**dict(model=model))) # print learning rate in the terminal
    #callbacks.append(TerminateOnNaN()) # fail during training on NaN loss

    ### training the model
    ######################
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.save_weights(ckpt_name.format(epoch=0)) # save model before training

    # save Y_va
    np.save(f'{folder}/Y_va.npy', Y_va)
    np.save(f'{folder}/Y_tr.npy', Y_tr)

    with tf.device('CPU'): # convert data to tensors to fix a bugfix in tf > 2.6 that was otherwise throwing OutOfMemory errors
        logger.info('Converting training data to tensors')
        X_tr = tf.convert_to_tensor(X_tr)
        Y_tr = tf.convert_to_tensor(Y_tr)

        logger.info('Converting validation data to tensors')
        X_va = tf.convert_to_tensor(X_va)
        Y_va = tf.convert_to_tensor(Y_va)

    # log the amount af data that is entering the network
    logger.info(f'Training the network on {len(Y_tr)} datapoint and validating on {len(Y_va)}')

    # perform training for `num_epochs`
    my_history=model.fit(X_tr, Y_tr, batch_size=batch_size, validation_data=(X_va,Y_va), shuffle=True,
                         callbacks=callbacks, epochs=num_epochs, verbose=2, class_weight=None)

    ## compute and save Y_pred_unbiased
    Y_pred = []
    for b in range(Y_va.shape[0]//batch_size + 1):
        Y_pred.append(postprocess(model(X_va[b*batch_size:(b+1)*batch_size])).numpy())
    Y_pred = np.concatenate(Y_pred)
    Y_pred_unbiased = ut.unbias_probabilities(Y_pred, u=u)
    np.save(f'{folder}/Y_pred_unbiased.npy', Y_pred_unbiased)

    ## deal with history
    history = my_history.history
    model.save(folder)
    np.save(f'{folder}/history.npy', history)
    # log history
    df = pd.DataFrame(history)
    df.index.name = 'epoch-1'
    logger.log(25, str(df))

    # return the best value of the return metric
    if return_metric not in history:
        logger.error(f'{return_metric = } is not one of the metrics monitored during training, returning NaN')
        score = np.NaN
    else:
        score = np.min(history[return_metric])
    logger.log(42, f'{score = }')
    return score

@ut.execution_time
@ut.indent_logger(logger)
def k_fold_cross_val_split(i, *arr, nfolds=10, val_folds=1):
    '''
    Splits a series of arrays in a training and validation set according to k fold cross validation algorithm

    Parameters
    ----------
    i : int
        fold number from 0 to `nfolds`-1
    *arr : np.ndarray
        Series of arrays to split
    nfolds : int, optional
        number of folds
    val_folds : int, optional
        number of consecutive folds for the validation set (between 1 and `nfolds`-1). Default 1.

    Returns
    -------
    a_tr for a in arr : series of np.arrays
        training data
    a_va for a in arr : series of np.arrays
        validation data

    Examples
    --------
    >>> X = np.arange(5)
    >>> Y = np.arange(5)*2
    >>> k_fold_cross_val_split(0, X, Y, nfolds=5)
    (array([1, 2, 3, 4]), array([2, 4, 6, 8]), array([0]), array([0]))
    '''
    if i < 0 or i >= nfolds:
        raise ValueError(f'fold number i is out of the range [0, {nfolds - 1}]')
    if val_folds >= nfolds or val_folds <= 0:
        raise ValueError(f'val_folds out of the range [1, {nfolds - 1}]')
    arr_tr = []
    arr_va = []
    fold_len = arr[0].shape[0]//nfolds
    lower = i*fold_len % arr[0].shape[0]
    upper = (i+val_folds)*fold_len % arr[0].shape[0]
    if lower < upper:
        for a in arr:
            arr_va.append(a[lower:upper])
            arr_tr.append(np.concatenate([a[upper:], a[:lower]], axis=0))
    else: # `upper` overshoots
        for a in arr:
            arr_va.append(np.concatenate([a[lower:], a[:upper]], axis=0))
            arr_tr.append(a[upper:lower])
    return *arr_tr, *arr_va


def optimal_checkpoint(run_folder, nfolds, metric='val_CustomLoss', direction='minimize', first_epoch=1, collective=True, fold_subfolder=None):
    '''
    Computes the epoch that had the best score

    Parameters
    ----------
    folder : str
        folder where the model is located that contains sub folders with the n folds named 'fold_%i'
    nfolds : int
        number of folds,
    metric : str, optional
        metric with respect to which optimize, by default 'val_CustomLoss'
    direction : str, optional
        'maximize' or 'minimize', by default 'minimize'
    first_epoch : int, optional
        The number of the first epoch, by default 1
    collective : bool, optional
        Whether the optimal checkpoint should be the same for all folds (True) or the best for each fold
    fold_subfolder : str, optional
        Name of the subfolder inside the fold folder in which to look for history and model checkpoints,
        useful for more advanced usage. By default None
    Returns
    -------
    opt_checkpoint
        if collective : int
            epoch number corresponding to the best checkpoint
        else : list
            of best epoch number for each fold
    fold_subfolder: str or list of str
        the fold subfolder where history and checkpoints are located

    Raises
    ------
    KeyError
        If `metric` is not present in the history
    ValueError
        If `direction` not in ['maximize', 'minimize']
    '''
    run_folder = run_folder.rstrip('/')
    
    fold_subfolder = (fold_subfolder.rstrip('/') + '/') if fold_subfolder else ''

    # Here we insert analysis of the previous training with the assessment of the ideal checkpoint
    history0 = np.load(f'{run_folder}/fold_0/{fold_subfolder}history.npy', allow_pickle=True).item()
    if metric not in history0.keys():
        raise KeyError(f'{metric} not in history: cannot compute optimal checkpoint')
    historyCustom = [np.load(f'{run_folder}/fold_{i}/{fold_subfolder}history.npy', allow_pickle=True).item()[metric] for i in range(nfolds)]

    if direction == 'minimize':
        opt_f = np.argmin
    elif direction == 'maximize':
        opt_f = np.argmax
    else:
        raise ValueError(f'Unrecognized {direction = }')

    if collective: # the optimal checkpoint is the same for all folds and it is based on the average performance over the folds
        # check that the nfolds histories have the same length
        lm = np.min([len(historyCustom[i]) for i in range(nfolds)])
        lM = np.max([len(historyCustom[i]) for i in range(nfolds)])
        if lm < lM: # folds have different history length
            logger.warning('Using collective checkpoint on histories of different length is deprecated! Longer histories will be clipped to the shortest one')
            historyCustom = [historyCustom[i][:lm] for i in range(nfolds)]

        historyCustom = np.mean(np.array(historyCustom),axis=0)
        opt_checkpoint = opt_f(historyCustom)
    else:
        opt_checkpoint = np.array([opt_f(h) for h in historyCustom]) # each fold independently
    
    opt_checkpoint += first_epoch

    if collective:
        opt_checkpoint = int(opt_checkpoint)
    else:
        opt_checkpoint = [int(oc) for oc in opt_checkpoint]

    return opt_checkpoint, fold_subfolder


def get_transfer_learning_folders(load_from, current_run_folder, nfolds, optimal_checkpoint_kwargs=None):
    '''
    Creates the names of the checkpoints from which to load for every fold

    Parameters
    ----------
    load_from : list, dict, int, str, or None
        From where to load. If list `load_from` is returned, skipping other computations, otherwise see function `get_run`
    current_run_folder : str
        folder where the current run is happening
    nfolds : int
        number of folds
    optimal_checkpoint_kwargs : dict, optional
        arguments for the function `optimal_checkpoint`, by default None

    Returns
    -------
    load_from : None or list of str
        list of the checkpoint names
    info : dict
    '''
    if optimal_checkpoint_kwargs is None:
        optimal_checkpoint_kwargs = {}
    info = {}

    if isinstance(load_from, list):
        return load_from, info

    # get the actual run name from where to load
    spl = current_run_folder.rsplit('/',1) # it is either [root_folder, run_name] or [run_name]. The latter if there was no '/' in `folder`
    if len(spl) == 2:
        root_folder, current_run_name = spl
    else:
        root_folder = './'
        current_run_name = spl[-1]
    
    # Find the model which has the weights we can use for transfer learning, if it is possible
    load_from = get_run(load_from, current_run_name=current_run_name, runs_path=f'{root_folder}/runs.json')
    if load_from is None:
        logger.log(41, 'Models will be trained from scratch')
    else:
        logger.log(41, f'Models will be loaded from {load_from}')

    # find the optimal checkpoint
    if load_from is not None:
        load_from = load_from.rstrip('/')
        opt_checkpoint, fold_subfolder = optimal_checkpoint(f'{root_folder}/{load_from}', nfolds, **optimal_checkpoint_kwargs)
        info['tl_from'] = {'run': load_from, 'optimal_checkpoint': opt_checkpoint}
        if isinstance(opt_checkpoint,int):
            # this happens if the optimal checkpoint is computed with `collective` = True, so we simply broadcast the single optimal checkpoint to all the folds
            opt_checkpoint = [opt_checkpoint]*nfolds
        if isinstance(fold_subfolder, str):
            fold_subfolder = [fold_subfolder]*nfolds

        # make the folders name of the checkpoint
        load_from = [f'{root_folder}/{load_from}/fold_{i}/{fold_subfolder[i]}cp-{opt_checkpoint[i]:04d}.ckpt' for i in range(nfolds)]

    return load_from, info

def load_model(checkpoint, compile=False):
    '''
    Loads a neural network and its weights. Checkpoints with the weights are supposed to be in the same folder as where the model structure is

    Parameters
    ----------
    checkpoint : str
        path to the checkpoint is. For example with structure <folder>/cp-<epoch>.ckpt
    compile : bool, optional
        whether to compile the model, by default False

    Returns
    -------
    keras.models.Model
    '''
    model_folder = Path(checkpoint).parent
    model = keras.models.load_model(model_folder, compile=compile)
    model.load_weights(checkpoint)
    return model

def get_loss_function(loss_name: str, u=1):
    loss_name = loss_name.lower()
    if loss_name.startswith('unbiased'):
        return tff.UnbiasedCrossentropyLoss(undersampling_factor=u)
    elif 'crossentropy' in loss_name:
        return keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise ValueError(f'Could not parse {loss_name = }')
    
def get_default_metrics(fullmetrics=False, u=1):
    if fullmetrics:
        tf_sampling = tf.cast([0.5*np.log(u), -0.5*np.log(u)], tf.float32) # Debiasor of logits (this translates into debiasing the probabilities)
        metrics=[
            'accuracy',
            tff.MCCMetric(undersampling_factor=1),
            tff.MCCMetric(undersampling_factor=u, name='UnbiasedMCC'),
            tff.ConfusionMatrixMetric(2, undersampling_factor=u),
            tff.BrierScoreMetric(undersampling_factor=u),
            tff.CustomLoss(tf_sampling)
        ]# the last two make the code run longer but give precise discrete prediction benchmarks
    else:
        metrics=None
    return metrics

@ut.execution_time
@ut.indent_logger(logger)
def margin_removal_with_sliding_window(X,time_start,leftmargin,rightmargin,time_end,T,sliding=False):
    '''
    the procedure is to stride the time axis with the window size "leftmargin+1" 
       <- there is a +1 because this argument of `sliding_window_view` does nothing when input is 1
       `sliding_window_view` normally just puts the new axis at the end, but for using LSTMs, for instance, 
       this is not well adapted so we move the axis right after the natural time (0-th axis)

    Input
        X: ndarray
            The input array that will have its margins removed and possibly slided
        time_start: int
        time_end: int
        leftmargin: int
        rightmargin: int,
        T: int
        sliding: bool
    '''
    if (leftmargin is not None) or (rightmargin is not None):
        X = X.reshape(-1,time_end-time_start-T+1,*X.shape[1:])
        logger.info(f'preparing for margin removal: {X.shape = }')
            
    if (leftmargin is not None) and (sliding is True): # adding a dimension to X_tr and X_va with a time moving window
        X = np.moveaxis(sliding_window_view(X, leftmargin+1, axis=1),-1,2)
        logger.info(f'after sliding window: {X.shape = }')

    if (leftmargin is not None):
        if not sliding: # if sliding this operation should be automatic
            X = X[:,leftmargin:None,...]
        X = X.reshape(-1,*X.shape[2:]) # skip the unnecessary margins
        logger.info(f'after removing margins and shaping back: {X.shape = }')

    if (rightmargin is not None): # right margin affects whether or not we have sliding
        X = X[:,None:rightmargin,...].reshape(-1,*X.shape[2:]) # skip the unnecessary margins
        logger.info(f'after removing margins and shaping back: {X.shape = }')

    return X


@ut.execution_time
@ut.indent_logger(logger)
def k_fold_cross_val(folder, X, Y, create_model_kwargs=None, train_model_kwargs=None, optimal_checkpoint_kwargs=None, load_from='last', nfolds=10, val_folds=1, u=1, normalization_mode='pointwise',
                     fullmetrics=True, training_epochs=40, training_epochs_tl=10, loss='sparse_categorical_crossentropy', prune_threshold=None, min_folds_before_pruning=None,
                     Z_DIM=None, T=14, time_start=30, time_end=120, label_period_start=None, label_period_end=None):
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
        However when runing this function from a notebook you can use more advanced features like using another loss rather than the default cross entropy
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

    prune_threshold : float, optional
        if the average score in the first `min_folds_before_pruning` is above `prune_threshold`, the run is pruned.
        This means that the run is considered not promising and hence we avoid losing time in computing the remaining folds.
        This is particularly useful when performing a hyperparameter optimization procedure.
        By default is None, which means that pruning is disabled
    min_folds_before_pruning : int, optional
        minimum number of folds to train before checking whether to prune the run
        By default None, which means that pruning is disabled
    Z_DIM: int, optional
        if Z_DIM is not None, pca decomposition is performed to Z_DIM components
    T : int, optional
        width of the window for the running average  
    time_start : int, optional 
        first day of the period of interest (copied from make_XY to be able to compute `timestamps`)
    time_end : int, optional
        first day after the end of the period of interst (copied from make_XY)
    label_period_start : int, optional
        if provided the first day of the period of interest for the label threshold determination (copied from make_XY)
        This variable is necessary if for some reason we need to also load data that lies outside the range of where 
        the labels that we need for training/validation and testing directly
            leftmargin = label_period_start - time_start
            if positive will be treated as a 
    label_period_end : int, optional
        if provided the first day after the end of the period of interst for the label threshold determination (copied from make_XY)
        This variable is necessary if for some reason we need to also load data that lies outside the range of where 
        the labels that we need for training/validation and testing directly
    Returns
    -------
    float
        average score of the run
    '''
    
    if create_model_kwargs is None:
        create_model_kwargs = {}
    if train_model_kwargs is None:
        train_model_kwargs = {}
    if optimal_checkpoint_kwargs is None:
        optimal_checkpoint_kwargs = {}
    folder = folder.rstrip('/')
    
    """leftmargin: (int), optional
        Specifies the number of timestamps that we use to make the prediction. By default is absent (one time stamp)
        The use of `leftmargin` is only recommended if PCA was performed since otherwise too much RAM is taken"""
    
    # X and Y are extracted from time_start to time_end, but we only care about the part inside label_period_start to label_period_end for training and testing
    if label_period_start is None: # label_period_start == time_start implied
        leftmargin = None # basically left margin
    else:
        leftmargin = label_period_start - time_start
        if leftmargin < 0:
            raise ValueError(f'leftmargin = label_period_start - time_start < 0 which is not allowed!')
        elif leftmargin == 0:
            leftmargin = None
    
    if label_period_end is None:
        rightmargin = None
    else:
        rightmargin = time_end - label_period_end - T + 1 # that's because when we perform running mean we have to avoid using last T days
        if rightmargin < 0:
            raise ValueError(f'leftmargin = time_end - label_period_end - T + 1 which is not allowed!')
    # get the folders from which to load the models
    load_from, info = get_transfer_learning_folders(load_from, folder, nfolds, optimal_checkpoint_kwargs=optimal_checkpoint_kwargs)
    # here load_from is either None (no transfer learning) or a list of strings

    my_memory = []
    info['status'] = 'RUNNING'

    # k fold cross validation
    scores = []
    for i in range(nfolds):
        logger.info('=============')
        logger.log(35, f'fold {i} ({i+1}/{nfolds})')
        logger.info('=============')
        # create fold_folder
        fold_folder = f'{folder}/fold_{i}'
        os.mkdir(fold_folder)

        # split data
        X_tr, Y_tr, X_va, Y_va = k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)
            
        

        if normalization_mode: # normalize X_tr and X_va
            X_tr, _, _ = normalize_X(X_tr, fold_folder, mode=normalization_mode)
            #X_va = (X_va - X_mean)/X_std 
            X_va, _, _ = normalize_X(X_va, fold_folder) # we expect that the previous operation stores X_mean, X_std
            logger.info(f'after normalization: {X_tr.shape = }, {X_va.shape = }, {Y_tr.shape = }, {Y_va.shape = }')
        
        if Z_DIM is not None:
            with PCAer(Z_DIM=Z_DIM, folder=fold_folder) as pcaer:
                pcaer.fit_with_timeout(0,X_tr)
                X_tr = pcaer.encoder.predict(X_tr)
                X_va = pcaer.encoder.predict(X_va)
                logger.info(f'after PCA: {X_tr.shape = }, {X_va.shape = }')
        logger.info(f' {time_start = }, {time_end = }, {leftmargin = }, {rightmargin = }, {T = }')
        #logger.info(f'{Y_va[1*82:2*82]}')
        #logger.info(f'{np.where(Y_va == 1)}')
        #logger.info(f'{X_va[5*82:6*82,35,30,0]}')
        X_tr = margin_removal_with_sliding_window(X_tr,time_start,leftmargin,rightmargin,time_end,T,sliding=True)
        X_va = margin_removal_with_sliding_window(X_va,time_start,leftmargin,rightmargin,time_end,T,sliding=True)
        Y_tr = margin_removal_with_sliding_window(Y_tr,time_start,leftmargin,rightmargin,time_end,T)
        Y_va = margin_removal_with_sliding_window(Y_va,time_start,leftmargin,rightmargin,time_end,T)
        #logger.info(f'{Y_va[1*79:2*79]}')
        #logger.info(f'{np.where(Y_va == 1)}')
        #for i in range(X_va.shape[1]):   
        #    logger.info(f'{X_va[5*79:6*79,i,35,30,0]}')
        logger.info(f'After margin removal: {X_tr.shape = }, {X_va.shape = }, {Y_tr.shape = }, {Y_va.shape = }')

        # perform undersampling
        X_tr, Y_tr = undersample(X_tr, Y_tr, u=u)

        n_pos_tr = np.sum(Y_tr)
        n_neg_tr = len(Y_tr) - n_pos_tr
        logger.info(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')
        # at this point data is ready to be fed to the networks


        # check for transfer learning
        model = None        
        if load_from is None:
            model = create_model(input_shape=X_tr.shape[1:], **create_model_kwargs)
        else:
            model = load_model(load_from[i], compile=False)
        summary_buffer = ut.Buffer() # workaround necessary to log the structure of the network to the file, since `model.summary` uses `print`
        summary_buffer.append('\n')
        model.summary(print_fn = lambda x: summary_buffer.append(x + '\n'))
        logger.info(summary_buffer.msg)

        # number of training epochs
        num_epochs = train_model_kwargs.pop('num_epochs', None) # if num_epochs is not provided in train_model_kwargs, which is most of the time,
                                                                # we assign it according if we have to do transfer learning or not
        if num_epochs is None:
            if load_from is None:
                num_epochs = training_epochs
            else:
                num_epochs = training_epochs_tl

        # metrics
        metrics = train_model_kwargs.pop('metrics', None)
        if metrics is None:
            metrics = get_default_metrics(fullmetrics, u=u)

        # optimizer
        optimizer = train_model_kwargs.pop('optimizer',keras.optimizers.Adam()) # if optimizer is not provided in train_model_kwargs use Adam
        # loss function
        loss_fn = train_model_kwargs.pop('loss',None)
        if loss_fn is None:
            loss_fn = get_loss_function(loss, u=u)
        logger.info(f'Using {loss_fn.name} loss')


        # train the model
        score = train_model(model, X_tr, Y_tr, X_va, Y_va, # arguments that are always computed inside this function
                            folder=fold_folder, num_epochs=num_epochs, optimizer=optimizer, loss=loss_fn, metrics=metrics, # arguments that may come from train_model_kwargs for advanced uses but usually are computed here
                            **train_model_kwargs) # arguments which have a default value in the definition of `train_model` and thus appear in the config file

        scores.append(score)

        my_memory.append(psutil.virtual_memory())
        try:
            logger.info(f'RAM memory: {my_memory[i][3]:.3e}') # Getting % usage of virtual_memory ( 3rd field)
        except:
            logger.info(f'RAM memory: {my_memory[i]:.3e}') # Getting % usage of virtual_memory ( 3rd field)

        keras.backend.clear_session()
        gc.collect() # Garbage collector which removes some extra references to the objects. This is an attempt to micromanage the python handling of RAM

        # check for pruning, i.e. if the run is not promising we don't compute all the folds to save time
        if min_folds_before_pruning is not None and prune_threshold is not None:
            if i >= min_folds_before_pruning - 1 and i < nfolds - 1:
                score_mean = np.mean(scores) # we compute the average score of the already computed folds
                if score_mean > prune_threshold: # score too high, we prune the run
                    info['status'] = 'PRUNED'
                    logger.log(41,f'Pruning after {i+1}/{nfolds} folds')
                    break
        
    np.save(f'{folder}/RAM_stats.npy', my_memory)

    # recompute the scores if collective=True
    # Here we want to use the `optimal_checkpoint` function to compute the best checkpoint for this network. 
    # Mind that before we used it to determine the optimal checkpoint from the network from which to perform transfer learning, so we need to change the parameters
    try:
        collective = optimal_checkpoint_kwargs['collective']
    except KeyError:
        collective = get_default_params(optimal_checkpoint)['collective']
    if collective:
        logger.log(35, 'recomputing scores and network predictions with the collective optimal checkpoint')
        try:
            return_metric = train_model_kwargs['return_metric']
        except KeyError:
            return_metric = get_default_params(train_model)['return_metric']
        try:
            first_epoch = optimal_checkpoint_kwargs['first_epoch']
        except KeyError:
            first_epoch = get_default_params(optimal_checkpoint)['first_epoch']
            
        opt_checkpoint, fold_subfolder = optimal_checkpoint(folder,nfolds, **optimal_checkpoint_kwargs)

        # recompute the scores
        for i in range(nfolds):
            scores[i] = np.load(f'{folder}/fold_{i}/{fold_subfolder}history.npy', allow_pickle=True).item()[return_metric][opt_checkpoint - first_epoch]

        # reload the models at their proper checkpoint and recompute Y_pred_unbiased
        batch_size = train_model_kwargs['batch_size']
        for i in range(nfolds):
            _, _, X_va, Y_va = k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)
            logger.info(f'{X_tr.shape = }, {X_va.shape = }')
            fold_folder = f'{folder}/fold_{i}'
            if normalization_mode: # normalize X_tr and X_va
                #X_va = (X_va - X_mean)/X_std 
                X_va, _, _ = normalize_X(X_va, fold_folder) # we expect that the previous operation stores X_mean, X_std
                logger.info(f'after normalization: {X_va.shape = }, {Y_va.shape = }')

            if Z_DIM is not None:
                with PCAer(Z_DIM=Z_DIM, folder=fold_folder) as pcaer: # the fit is expected to have already been performed thus we must merely load
                    X_va = pcaer.encoder.predict(X_va)
                    logger.info(f'after PCA: {X_va.shape = }')
            
            X_va = margin_removal_with_sliding_window(X_va,time_start,leftmargin,rightmargin,time_end,T,sliding=True)
            Y_va = margin_removal_with_sliding_window(Y_va,time_start,leftmargin,rightmargin,time_end,T)
            
            
            model = load_model(f'{fold_folder}/{fold_subfolder}cp-{opt_checkpoint:04d}.ckpt')

            Y_pred = []
            for b in range(Y_va.shape[0]//batch_size + 1):
                Y_pred.append(postprocess(model(X_va[b*batch_size:(b+1)*batch_size])).numpy())
            Y_pred = np.concatenate(Y_pred)
            Y_pred_unbiased = ut.unbias_probabilities(Y_pred, u=u)
            np.save(f'{fold_folder}/Y_pred_unbiased.npy', Y_pred_unbiased)
        

    score_mean = np.mean(scores)
    score_std = np.std(scores)

    # log the scores
    info['scores'] = {}
    logger.info('\nFinal scores:')
    for i,s in enumerate(scores):
        logger.info(f'\tfold {i}: {s}')
        info['scores'][f'fold_{i}'] = s
    logger.log(45,f'Average score: {ufloat(score_mean, score_std)}')
    info['scores']['mean'] = score_mean
    info['scores']['std'] = score_std

    info['scores'] = ast.literal_eval(str(info['scores']))

    if info['status'] != 'PRUNED':
        info['status'] = 'COMPLETED'

    # return the average score
    return score_mean, info

##################################################
########## PUTTING THE PIECES TOGETHER ###########
##################################################

@ut.execution_time
@ut.indent_logger(logger)
def prepare_XY(fields, make_XY_kwargs=None, roll_X_kwargs=None,
               do_premix=False, premix_seed=0, do_balance_folds=True, nfolds=10, year_permutation=None, flatten_time_axis=True, return_time_series=False):
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
    return_time_series : bool, optional
        If True it appends to the return statement the time series integrated over the area

    Returns
    -------
    X : np.ndarray
        data. If flatten_time_axis with shape (days, lat, lon, fields), else (years, days, lat, lon, fields)
    Y : np.ndarray 
        labels. If flatten_time_axis with shape (days,), else (years, days)
    tot_permutation : np.ndarray
        with shape (years,), final permutaion of the years that reproduces X and Y once applied to the just loaded data
    lat : np.ndarray
        latitude, with shape (lat,) (rolled if necessary)
    lon : np.ndarray
        longitude, with shape (lon,) (rolled if necessary)
    (time_series) if return_time_series : np.ndarray
        output which is conditional on the input return_time_series
    (threshold) if return_threshold in make_XY_kwargs : float
        the threshold used for the labels
    '''
    if make_XY_kwargs is None:
        make_XY_kwargs = {}
    if roll_X_kwargs is None:
        roll_X_kwargs = {}
    roll_X_kwargs = ut.set_values_recursive(get_default_params(roll_X), roll_X_kwargs) # get the default values not provided

    # get lat and lon
    f = list(fields.values())[0] # take the first field
    lat = np.copy(f.field.lat.data) # 1d array
    lon = np.copy(f.field.lon.data) # 1d array

    
    return_threshold = ut.extract_nested(make_XY_kwargs, 'return_threshold', False)
    print(f"{return_threshold = }")

    if return_threshold:
        X,Y, threshold = make_XY(fields, **make_XY_kwargs)
    else:
        X,Y = make_XY(fields, **make_XY_kwargs)
    
    time_start = ut.extract_nested(make_XY_kwargs, 'time_start', None) # We need to extract these values to limit the season of Y which matters for balancing folds (see below)
    time_end = ut.extract_nested(make_XY_kwargs, 'time_end', None)
    label_period_start = ut.extract_nested(make_XY_kwargs, 'label_period_start', None)
    label_period_end = ut.extract_nested(make_XY_kwargs, 'label_period_end', None)
    tau = ut.extract_nested(make_XY_kwargs, 'tau', None)
    T = ut.extract_nested(make_XY_kwargs, 'T', None)

    if label_period_start is None:
        label_period_start = time_start
    if label_period_end is None:
        label_period_end = time_end
    # move greenwich_meridian
    X = roll_X(X, **roll_X_kwargs)
    # roll also lat and lon
    roll_axis = roll_X_kwargs['roll_axis']
    roll_steps = roll_X_kwargs['roll_steps']
    if roll_axis == 'lon':
        lon = np.roll(lon, roll_steps)

        # make the longitude monotonically increasing
        lon = lon % 360 # move all values between 0 and 360
        i = np.argmin(lon) # index of the minimum longitude (most close to Greenwich)
        if i+1 == len(lon):
            increases = lon[i-1] < lon[i] # whether lon is monotonically increasing or not 
        else:
            increases = lon[i] < lon[i+1] # whether lon is monotonically increasing or not 
        if increases:
            lon[:i] -= 360 # make negative the values before i
        else:
            if i+1 < len(lon):
                lon[i+1:] -= 360 # make negative the values after i
    if roll_axis == 'lat':
        lat = np.roll(lat, roll_steps)

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
            # GM: now the weights will be computed solely based on the time of interest, so the balancing will only care about heatwaves inside this time
            if (label_period_start and time_start and label_period_end and time_start and T) is None: # i.e. if any of the variables between parentheses is None
                weights = np.sum(Y, axis=1) # get the number of heatwave events per year
            else:
                logger.info(f" {label_period_start = } ;{time_start = } ;{time_end = } ;{label_period_end = } ")
                logger.info(f"{Y.shape = }, from {label_period_start-time_start} to {label_period_end-time_start-T+1} ")
                weights = np.sum(Y[:,(label_period_start-time_start):(label_period_end-time_start-T+1)], axis=1) # get the number of heatwave events per year
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
    else:
        logger.info(f'Time not flattened: {X.shape = }, {Y.shape = }')

    if return_time_series:
        time_series = []
        # logger.info(f"{fields.values() = }")
        for field in fields.values():
            #logger.info(f"{field.area_integral =}")=
            temp = (field.area_integral.to_numpy().reshape(field.years,-1))[year_permutation]
            # flatten the time axis dropping the organizatin in years
            if flatten_time_axis:
                if (time_start and time_start and T and tau) is None:
                    time_series.append(temp.flatten()) 
                else:
                    time_series.append((temp[:,time_start+tau:time_end+tau-T+1]).flatten()) 
            else:
                if (time_start and time_start and T and tau) is None:
                    time_series.append(temp)
                else:
                    time_series.append((temp[:,time_start+tau:time_end+tau-T+1])) 
        
        logger.info(f"{time_series = }")
        time_series = np.array(time_series).T
        logger.info(f"{time_series.shape = }")
        if return_threshold:
            return X, Y, year_permutation, lat, lon, time_series, threshold
        else:
            return X, Y, year_permutation, lat, lon, time_series
    
    if return_threshold:
        return X, Y, year_permutation, lat, lon, threshold
    else:
        return X, Y, year_permutation, lat, lon


@ut.execution_time
@ut.indent_logger(logger)
def prepare_data_and_mask(load_data_kwargs=None, prepare_XY_kwargs=None):
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
    (
        X : np.ndarray
            data. If flatten_time_axis with shape (days, lat, lon, fields), else (years, days, lat, lon, fields)
        Y : np.ndarray 
            labels. If flatten_time_axis with shape (days,), else (years, days)
        year_permutation : np.ndarray
            with shape (years,), final permutaion of the years that reproduces X and Y once applied to the just loaded data
        mask : np.ndarray
            a mask obtained from the first element of fields.
        lat
        lon
    )
    mask
    '''
    if load_data_kwargs is None:
        load_data_kwargs = {}
    if prepare_XY_kwargs is None:
        prepare_XY_kwargs = {}
    # load data
    fields = load_data(**load_data_kwargs)

    return prepare_XY(fields, **prepare_XY_kwargs), fields[next(iter(fields))].mask

@ut.execution_time
@ut.indent_logger(logger)
def prepare_data(load_data_kwargs=None, prepare_XY_kwargs=None):
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
    lat
    lon
    '''
    if load_data_kwargs is None:
        load_data_kwargs = {}
    if prepare_XY_kwargs is None:
        prepare_XY_kwargs = {}
    # load data
    fields = load_data(**load_data_kwargs)

    return prepare_XY(fields, **prepare_XY_kwargs)  

@ut.execution_time
def run(folder, load_data_kwargs=None, prepare_XY_kwargs=None, k_fold_cross_val_kwargs=None, log_level=logging.INFO):
    '''
    Perfroms a single full run

    Parameters
    ----------
    folder : str
        folder where to perform the run
    prepare_data_kwargs : dict
        arguments to pass to the `prepare_data` function
    k_fold_cross_val_kwargs : dict
        arguments to pass to the `k_fold_cross_val` function

    Returns
    -------
    float
        average score of the run
    '''
    if load_data_kwargs is None:
        load_data_kwargs = get_default_params(load_data, recursive=True)
    if prepare_XY_kwargs is None:
        prepare_XY_kwargs = get_default_params(prepare_XY, recursive=True)
    if k_fold_cross_val_kwargs is None:
        k_fold_cross_val_kwargs = get_default_params(k_fold_cross_val, recursive=True)

    # check that we are not asking to label the events with a field that was not loaded
    label_field = ut.extract_nested(prepare_XY_kwargs, 'label_field')
    if not any([field_name.startswith(label_field) for field_name in load_data_kwargs['fields']]):
        raise KeyError(f"field {label_field} is not a loaded field")

    trainer = Trainer()
    return trainer.run(folder,load_data_kwargs, prepare_XY_kwargs, k_fold_cross_val_kwargs, log_level=log_level)

####################################################
###### EFFICIENT MANAGEMENT OF MULTIPLE RUNS #######
####################################################

class Trainer():
    '''
    Class for performing training of neural networks over multiple runs with different paramters in an efficient way
    '''
    def __init__(self, root_folder='./', config='detect', skip_existing_run=True, upon_failed_run='raise'):
        '''
        Constructor

        Parameters
        ----------
        root_folder : str, optional
            path to the folder where to perform the runs, default corrent directory
        config : dict or str or None or 'detect', optional
            if dict: config dictionary
            if str: path to config file
            if None: the default values specified in this file will be used
            if 'detect': if in the current folder there is a config file, that one is used, otherwise the default config file will be used
        skip_existing_run : bool, optional
            Whether to skip runs that have already been performed in the same folder, by default True
            If False the existing run is not overwritten but a new one is performed
        upon_filed_run : 'raise' or 'continue', optional
            What to do if a run fails. If 'raise' an exception will be raised and all the program stops.
            Otherwise the run will be treated as a pruned run, namely, the Trainer proceeds with the following runs.
            By default 'raise'
        '''
        self.skip_existing_run = skip_existing_run
        self.upon_failed_run = upon_failed_run

        self.root_folder = root_folder.rstrip('/')
        if not os.path.exists(self.root_folder):
            rf = Path(self.root_folder).resolve()
            rf.mkdir(parents=True, exist_ok=True)
        self.config_file = f'{self.root_folder}/config.json'
        self.fields_infos_file = f'{self.root_folder}/fields_infos.json'
        self.runs_file = f'{self.root_folder}/runs.json'
        self.allow_run = None

        # load config file and parse arguments
        if config == 'detect':
            config = self.config_file if os.path.exists(self.config_file) else None

        if config is None:
            self.config_dict = CONFIG_DICT
            logger.info('Initializing config dictionary from default values')
        elif isinstance(config, dict):
            self.config_dict = config
        elif isinstance(config, str):
            self.config_dict = ut.json2dict(config)
            logger.info(f"Loading config file from folder {config.rsplit('/',1)[0]}")
        else:
            raise TypeError(f'Invalid type {type(config)} for config')
        
        self.config_dict_flat = check_config_dict(self.config_dict, correct_mistakes=False)
        
        # cached (heavy) variables
        self.fields = None
        self.X = None
        self.Y = None
        self.year_permutation = None
        self.lon = None
        self.lat = None

        self._old_lat_lon = None # cache of self.lat, self.lon. See the workings of self.LAT or self.LON
        self._LONLAT = None # meshgrid of self.lat, self.lon

        # extract default arguments for each function
        self.default_run_kwargs = ut.extract_nested(self.config_dict, 'run_kwargs').copy()
        self.telegram_kwargs = ut.extract_nested(self.config_dict, 'telegram_kwargs').copy()

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
            logger.warn('\nThis machine does not have a GPU: training may be very slow\n')

    @property
    def LON(self):
        '''
        Meshgridded longitude
        '''
        if (self.lat, self.lon) != self._old_lat_lon:
            self._old_lat_lon = (self.lat, self.lon)
            self._LONLAT = np.meshgrid(self.lon, self.lat)
        return self._LONLAT[0]

    @property
    def LAT(self):
        '''
        Meshgridded latitude
        '''
        if (self.lat, self.lon) != self._old_lat_lon:
            self._old_lat_lon = (self.lat, self.lon)
            self._LONLAT = np.meshgrid(self.lon, self.lat)
        return self._LONLAT[1]

    def schedule(self, **kwargs):
        '''
        Here kwargs can be iterables. This function schedules several runs and calls on each of them `self._run`
        You can also set telegram kwargs with this function.

        Special arguments:
            first_from_scratch : bool, optional
                Whether the first run should be created from scratch or from transfer learning, by default False (i.e. by default transfer learning)
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
        iteration_values = ut.zipped_meshgrid(*iteration_values)
        # ensure json serializability by converting to string and back
        iteration_values = ast.literal_eval(str(iteration_values))

        # add the non iterative kwargs
        self.scheduled_kwargs = [{**non_iterative_kwargs, **{k: l[i] for i,k in enumerate(iterate_over) if l[i] != self.config_dict_flat[k]}} for l in iteration_values]

        ## this block of code does exactly the same of the previous line but possibly in a clearer way
        # self.scheduled_kwargs = []
        # for l in iteration_values:
        #     self.scheduled_kwargs.append(non_iterative_kwargs) # add non iterative kwargs
        #     # add the iterative kwargs one by one checking if they are at their default value
        #     for i,k in enumerate(iterate_over):
        #         v = l[i]
        #         if v != self.config_dict_flat[k]: # skip parameters at their default value
        #             self.scheduled_kwargs[-1][k] = v

        if len(self.scheduled_kwargs) == 0: # this is fix to avoid empty scheduled_kwargs if it happens there are no iterative kwargs
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
        nruns = len(self.scheduled_kwargs)
        logger.log(45, f"Starting {nruns} run{'' if nruns == 1 else 's'}")
        try:
            for i,kwargs in enumerate(self.scheduled_kwargs):
                logger.log(48, f'{HOSTNAME}: Run {i+1}/{nruns}')
                self._run(**kwargs)
            logger.log(49, f'{HOSTNAME}: \n\n\n\n\n\nALL RUNS COMPLETED\n\n')
        finally:
            # remove telegram logger
            if th is not None:
                logger.handlers.remove(th)
                logger.log(45, 'Removed telegram logger')

    @wraps(load_data) # it transfers the docstring, signature and default values from the module level function `load_data`
    def load_data(self, **load_data_kwargs):
        # load the fields only if the arguments have changed, otherwise self.fields is already at the correct value
        if self._load_data_kwargs != load_data_kwargs:
            self._load_data_kwargs = load_data_kwargs
            self._prepare_XY_kwargs = None # force the computation of prepare_XY
            self.fields = load_data(**load_data_kwargs)
        return self.fields

    @wraps(prepare_XY)
    def prepare_XY(self, fields, **prepare_XY_kwargs):
        # prepare XY only if the arguments have changed, as above
        if self._prepare_XY_kwargs != prepare_XY_kwargs:
            self._prepare_XY_kwargs = prepare_XY_kwargs
            self.X, self.Y, self.year_permutation, self.lat, self.lon = prepare_XY(fields, **prepare_XY_kwargs)
        return self.X, self.Y, self.year_permutation, self.lat, self.lon

    @wraps(prepare_data)
    def prepare_data(self, load_data_kwargs=None, prepare_XY_kwargs=None):
        if load_data_kwargs is None:
            load_data_kwargs = ut.extract_nested(self.default_run_kwargs, 'load_data_kwargs')
        if prepare_XY_kwargs is None:
            prepare_XY_kwargs = ut.extract_nested(self.default_run_kwargs, 'prepare_XY_kwargs')

        self.load_data(**load_data_kwargs)
        return self.prepare_XY(self.fields, **prepare_XY_kwargs)

    def run(self, folder, load_data_kwargs=None, prepare_XY_kwargs=None, k_fold_cross_val_kwargs=None, log_level=logging.INFO):
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

        Returns
        -------
        float
            average score of the run
        '''
        if load_data_kwargs is None:
            load_data_kwargs = {}
        if prepare_XY_kwargs is None:
            prepare_XY_kwargs = {}
        if k_fold_cross_val_kwargs is None:
            k_fold_cross_val_kwargs = {}

        if not os.path.exists(folder):
            os.mkdir(folder)
        elif os.path.exists(f'{folder}/fold_0'):
            raise FileExistsError(f'A run has already been performed in {folder = }')

        # setup logger to file
        fh = logging.FileHandler(f'{folder}/log.log')
        fh.setLevel(log_level)
        logger.handlers.append(fh)

        try:
            self.load_data(**load_data_kwargs) # compute self.fields

            self.prepare_XY(self.fields, **prepare_XY_kwargs) # compute self.X, self.Y, self.year_permutation, self.lat, self.lon
            if self.year_permutation is not None:
                np.save(f'{folder}/year_permutation.npy',self.year_permutation)

            # save area integral and A
            label_field = ut.extract_nested(prepare_XY_kwargs, 'label_field')
            try:
                lf = self.fields[label_field]
            except KeyError:
                try:
                    lf = self.fields[f'{label_field}_ghost']
                except KeyError:
                    logger.error(f'Unable to find label field {label_field} among the provided fields {list(self.fields.keys())}')
                    raise KeyError
            
            np.save(f'{folder}/area_integral.npy', lf.to_numpy(lf.area_integral))
            ta = lf.to_numpy(lf._time_average)
            np.save(f'{folder}/time_average.npy', ta)
            np.save(f'{folder}/time_average_permuted.npy', ta[self.year_permutation])

            # save labels
            np.save(f'{folder}/labels_permuted.npy', self.Y)
            

            # do kfold
            score, info = k_fold_cross_val(folder, self.X, self.Y, **k_fold_cross_val_kwargs)

            # make the config file and fields_infos file read-only after the first successful run
            if os.access(self.config_file, os.W_OK): # the file is writeable
                os.chmod(self.config_file, S_IREAD|S_IRGRP|S_IROTH) # we make it readable for all users
            if os.access(self.fields_infos_file, os.W_OK): # the file is writeable
                os.chmod(self.fields_infos_file, S_IREAD|S_IRGRP|S_IROTH) # we make it readable for all users
        
        except Exception as e:
            logger.critical(f'Run on {folder = } failed due to {repr(e)}')
            tb = traceback.format_exc() # log the traceback to the log file
            logger.error(tb)
            if isinstance(e, KeyboardInterrupt):
                raise e
            raise RuntimeError('Run failed') from e

        finally:
            logger.handlers.remove(fh) # stop writing to the log file

        return score, info


    def _run(self, **kwargs):
        '''
        Parses kwargs and performs a single run, kwargs are not interpreted as iterables.
        It checks for transfer learning and if the run has already been performed, in which case, if `self.skip_existing_run` is True, it is skipped.
        It also deals with the runs.json file.
        Basically it is a wrapper of the `self.run` function that performs all the extra steps besides a simply training the network.
        '''
        # check if we can run
        if self.allow_run is None: # compute allow_run
            if os.path.exists(f'{self.root_folder}/lock.txt'): # check if there is a lock
                self.allow_run = False
                logger.error('Lock detected: cannot run')
            elif os.path.exists(self.config_file): # if there is a config file we check it is compatible with self.config_dict
                config_in_folder = ut.json2dict(self.config_file)
                if config_in_folder == self.config_dict:
                    self.allow_run = True
                else:
                    self.allow_run = False
            else: # if there is no config file we create it
                ut.dict2json(self.config_dict, self.config_file)
                self.allow_run = True

        if not self.allow_run:
            raise FileExistsError('You cannot run in this folder with the provided config file. Other runs have already been performed with a different config file')

        if not os.path.exists(self.runs_file): # create run dictionary if not found
            ut.dict2json({},self.runs_file)
        
        runs = ut.json2dict(self.runs_file) # get runs dictionary

        # check if the run has already been performed
        for r in runs.values():
            if r['status'] == 'COMPLETED' and r['args'] == kwargs:
                if self.skip_existing_run:
                    logger.log(45, f"Skipping already performed run {r['name']}")
                    return None
                else:
                    logger.log(45, f"Rerunning {r['name']}")


        ############################
        ## preliminary operations ##
        ############################

        # get run number
        run_id = str(len(runs))
        # TODO #55 it would be convenient to call a function here that tests if the provided architecture would result in a valid neural network, thus avoiding waiting for the data to be loaded for no reason
        # create run name from kwargs
        folder = make_run_name(run_id, **kwargs)

        # correct the default kwargs with the ones provided
        run_kwargs = ut.set_values_recursive(self.default_run_kwargs, kwargs)

        check_config_dict(run_kwargs) # check if the arguments are consistent with each other

        # check for transfer learning
        load_from = ut.extract_nested(run_kwargs, 'load_from')
        nfolds = ut.extract_nested(run_kwargs, 'nfolds')
        optimal_checkpoint_kwargs = ut.extract_nested(run_kwargs, 'optimal_checkpoint_kwargs')
        load_from, tl_info = get_transfer_learning_folders(load_from, f'{self.root_folder}/{folder}', nfolds,
                                                           optimal_checkpoint_kwargs=optimal_checkpoint_kwargs)
        if tl_info:
            tl_info = tl_info['tl_from']

        if load_from is not None: # we actually do transfer learning
            # avoid computing the transfer learning folders by setting up a bypass for when `get_transfer_learning_folders` is called inside `k_fold_cross_val`
            run_kwargs = ut.set_values_recursive(run_kwargs, {'load_from': load_from})

            # force the dataset to the same year permutation
            year_permutation = list(np.load(f"{self.root_folder}/{tl_info['run']}/year_permutation.npy", allow_pickle=True))
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
                folder = make_run_name(run_id, **kwargs)

        logger.log(42, f'{folder = }\n')

        
        ###################
        ## start running ##
        ###################

        folder = f'R{folder}'

        start_time = time.time()
        
        runs[run_id] = {
            'name': folder, 
            'args': kwargs, 
            'transfer_learning_from': tl_info if tl_info else None,
            'status': 'RUNNING',
            'start_time': ut.now()
        }
        ut.dict2json(runs, self.runs_file) # save runs.json

        # write kwargs to logfile
        os.mkdir(f'{self.root_folder}/{folder}')
        with open(f'{self.root_folder}/{folder}/log.log', 'a') as logfile:
            logfile.write(f'{run_id = }\n\n')
            logfile.write(f'Running on machine: {HOSTNAME}\n\n')
            logfile.write('Non default parameters:\n')
            logfile.write(ut.dict2str(kwargs))
            logfile.write('\n')
            if tl_info:
                logfile.write('Transfer learning from:\n')
                logfile.write(ut.dict2str(tl_info))
                logfile.write('\n')
            else:
                logfile.write('No transfer learning\n\n')
            logfile.write('\n\n\n')

        # log kwargs
        logger.log(44, ut.dict2str(kwargs))

        # run
        score, info = None, {}
        try:            
            score, info = self.run(f'{self.root_folder}/{folder}', **run_kwargs)
            
            runs = ut.json2dict(self.runs_file)
            runs[run_id]['status'] = info['status'] # either COMPLETED or PRUNED
            if info['status'] == 'PRUNED':
                runs[run_id]['name'] = f'P{folder[1:]}'
                shutil.move(f'{self.root_folder}/{folder}', f'{self.root_folder}/P{folder[1:]}')
            elif info['status'] == 'COMPLETED': # remove the leading R
                runs[run_id]['name'] = f'{folder[1:]}'
                shutil.move(f'{self.root_folder}/{folder}', f'{self.root_folder}/{folder[1:]}')
            
            runs[run_id]['score'] = ast.literal_eval(str(score)) # ensure json serializability
            runs[run_id]['scores'] = info['scores']
            logger.log(42, 'run completed!!!\n\n')

        except Exception as e: # run failed
            runs = ut.json2dict(self.runs_file)
            runs[run_id]['status'] = 'FAILED'
            runs[run_id]['name'] = f'F{folder[1:]}'
            shutil.move(f'{self.root_folder}/{folder}', f'{self.root_folder}/F{folder[1:]}')

            if self.upon_failed_run == 'raise' or isinstance(e, KeyboardInterrupt):
                raise e
            info['status'] = 'FAILED'

        finally: # in any case we need to save the end time and save runs to json
            if runs[run_id]['status'] == 'RUNNING': # the run has not completed but the above except block has not been executed (e.g. due to KeybordInterruptError)
                runs[run_id]['status'] = 'FAILED'
                runs[run_id]['name'] = f'F{folder[1:]}'
                shutil.move(f'{self.root_folder}/{folder}', f'{self.root_folder}/F{folder[1:]}')
            runs[run_id]['end_time'] = ut.now()
            run_time = time.time() - start_time
            run_time_min = int(run_time/0.6)/100 # 2 decimal places of run time in minutes
            runs[run_id]['run_time'] = ut.pretty_time(run_time)
            runs[run_id]['run_time_min'] = run_time_min

            ut.dict2json(runs,self.runs_file)

        return score, info



CONFIG_DICT = build_config_dict([Trainer.run, Trainer.telegram]) # module level config dictionary
# config file will be built from the default parameters of the functions given here and of the functions they call in a recursive manner
        

def deal_with_lock(**kwargs):
    '''
    Checks if there is a lock and moves the code to the folder parsed from the input as well as 
    control json dictionaries such as `config.json`, `runs.json` and `fields_infos.json`.
    **kwargs are passed to `move_to_folder`

    Returns
    -------
    b : bool
        True if there is a lock, False otherwise

    Raises
    ------
    ValueError
        If invalid command line
    '''
    # check if there is a lock:
    lock = Path(__file__).resolve().parent / 'lock.txt'
    if os.path.exists(lock): # there is a lock
        # check for folder argument
        if len(sys.argv) < 2: 
            print(usage())
            return True
        if len(sys.argv) == 2:
            folder = sys.argv[1]
            print(f'moving code to {folder = }')
            move_to_folder(folder, **kwargs)
            
            ut.dict2json(CONFIG_DICT,f'{folder}/config.json')

            # runs file (which will keep track of various runs performed in newly created folder)
            ut.dict2json({},f'{folder}/runs.json')
            ut.dict2json(fields_infos, f'{folder}/fields_infos.json')

            return True
        else:
            with open(lock) as l:
                raise ValueError(l.read())
    
    return False

def parse_command_line():
    '''Parses command line arguments into a dictionary'''
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

    return arg_dict

        
def main():
    if deal_with_lock():
        return
    
    # the code below is executed only if there is no lock
    
    arg_dict = parse_command_line()

    logger.info(f'{arg_dict = }')

    trainer_kwargs = get_default_params(Trainer)
    trainer_kwargs.pop('config')
    trainer_kwargs.pop('root_folder') # these two parameters cannot be changed
    for k in trainer_kwargs:
        if k in arg_dict:
            trainer_kwargs[k] = arg_dict.pop(k)

    # check if we want to import the parameters from another run (see usage description in the beginnig of this file)
    import_params_from = arg_dict.pop('import_params_from', None)
    if import_params_from is not None:
        runs = ut.json2dict('./runs.json')
        try:
            rargs = runs[str(import_params_from)]['args']
        except KeyError:
            raise KeyError(f'{import_params_from} is not a valid run')
        logger.info(f'Importing parameters from run {import_params_from}')
        logger.info(ut.dict2str(rargs))
        
        for k,v in rargs.items(): # add the imported parameters to arg_dict, but not the ones explicitly provided in the command line
            if k not in arg_dict:
                arg_dict[k] = v

        logger.info(f'\n\nEquivalent command line arguments:\n{ut.dict2str(arg_dict)}')

    # create trainer
    trainer = Trainer(config='./config.json', **trainer_kwargs)

    # schedule runs
    trainer.schedule(**arg_dict)

    # o = input('Start training? (Y/[n]) ') # ask for confirmation
    # if o != 'Y':
    #     logger.error('Aborting')
    #     sys.exit(0)
    
    trainer.run_multiple()


if __name__ == '__main__':
    main()
