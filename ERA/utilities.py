# '''
# Created on 2022-01-13

# @author: Alessandro Lovo
# '''
'''
Set of general purpose functions
'''

# import libraries
import numpy as np
import sys
from functools import wraps
import time
from datetime import datetime
import json
import logging

######## time formatting ##########
def now():
    '''
    Returns the current time as string formatted as year-month-day hour:minute:second
    '''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def pretty_time(t):
    '''
    Takes a time in seconds and returns it in a string with the format <hours> h <minutes> min <seconds> s

    Examples:
    ---------
    >>> pretty_time(124)
    '2 min 4.0 s'
    >>> pretty_time(3601.4)
    '1 h 1.4 s'
    '''
    h = t//3600
    t = t - h*3600
    m = t//60
    s = t - m*60
    pt = ''
    if h > 0:
        pt += f'{h:.0f} h '
    if m > 0:
        pt += f'{m:.0f} min '
    pt += f'{s:.1f} s'
    return pt

default_formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

###### function decorators for logging ###
class Indenter():
    '''
    Indents the output of `print` statements.

    Usage:
    ------
    old_stdout = sys.stdout
    sys.stdout = Indenter()
    # do your stuff
    sys.stdout = old_stdout # restore old output system
    '''
    def __init__(self, terminal=sys.stdout):
        self.terminal = terminal
    def write(self, message):
        if message == '\n'*len(message):
            self.terminal.write(message)
        else:
            self.terminal.write('\t'+'\n\t'.join(message.split('\n')))
    def flush(self):
        pass

def indent_stdout(func):
    '''
    Indents the output produced by a function
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_std_out = sys.stdout
        sys.stdout = Indenter()
        try:
            r = func(*args, **kwargs)
        except Exception as e:
            sys.stdout = old_std_out
            raise e
        sys.stdout = old_std_out
        return r
    return wrapper

def execution_time(func):
    '''
    Prints the execution time of a function
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f'{func.__name__}:')
        r = func(*args, **kwargs)
        print(f'{func.__name__}: completed in {pretty_time(time.time() - start_time)}')
        return r
    return wrapper

#### TELEGRAM LOGGER ####

def new_telegram_handler(chat_ID=None, token=None, level=logging.WARNING, formatter=default_formatter, **kwargs):
    '''
    Creates a telegram handler object

    Parameters:
    -----------
        chat_ID : int or None, optional
            chat ID of the telegram user or group to whom send the logs. If None it is the last used.
            To find your chat ID go to telegram and search for 'userinfobot' and type '/start'. The bot will provide you with your chat ID.
            You can do the same with a telegram group, and, in this case, you will need to invite 'autoJASCObot' to the group.
            The default is None.
        token: str, token for the telegram bot or path to a text file where the first line is the token
        level : logging level: int or logging.(NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL), optional
            The default is logging.WARNING.
        formatter : logging.Formatter, optional
            The formatter used to log the messages. The default is default_formatter.
        **kwargs: additional arguments for telegram_handler.handlers.TelegramHandler

    Returns:
    --------
        th: TelegramHandler object
    '''
    import telegram_handler # NOTE: to install this package run pip install python-telegram-handler
    try:
        with open(token, 'r') as token_file:
            token = token_file.readline().rstrip('\n')
    except FileNotFoundError:
        pass
    th = telegram_handler.handlers.TelegramHandler(token=token, chat_id=chat_ID, **kwargs)
    th.setFormatter(formatter)
    th.setLevel(level)
    return th


########## ARGUMENT PARSING ####################

def run_smart(func, default_kwargs, **kwargs): # this is not as powerful as it looks like
    '''
    Runs a function in a vectorized manner:

    Parameters:
    -----------
        func: function with signature func(**kwargs) -> None
        default_kwargs: dict: default values for the keyword arguments of func
        **kwargs: non default values of the keyword arguments. If a list is provided, the function is run iterating over the list

    Examples:
    ---------
    >>> def add(x, y=0):
    ...     print(x + y)
    >>> run_smart(add, {'x': 0, 'y': 0}, x=1)
    1
    >>> run_smart(add, {'x': 0, 'y': 0}, x=1, y=[1,2,3]) # iterates over y
    2
    3
    4
    >>> run_smart(add, {'x': 0, 'y': 0}, x=[0, 10], y=[1,2]) # iterates over x and y
    1
    2
    11
    12
    >>> run_smart(add, {'x': [0], 'y': [0]}, x=[1,2], y=[1]) # correctly interprets lists when not supposed to iterate over them
    [1, 2, 1]
    >>> run_smart(add, {'x': [0], 'y': [0]}, x=[1,2], y=[[1], [0]]) # to iterate over list arguments, nest the lists
    [1, 2, 1]
    [1, 2, 0]
    '''
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

#### JSON IO #########

def json2dict(filename):
    '''
    Reads a json file `filename` as a dictionary

    Returns:
    --------
        d: dict
    '''
    with open(filename, 'r') as j:
        d = json.load(j)
    return d

def dict2json(d, filename):
    '''
    Saves a dictionary `d` to a json file `filename`
    '''
    with open(filename, 'w') as j:
        json.dump(d, j, indent=4)

#### MANAGE NESTED DICTIONARIES #####

def collapse_dict(d_nested, d_flat=None):
    '''
    Flattens a nested dictionary `d_nested` into a flat one `d_flat`.

    Parameters:
    -----------
        d_nested: dict, can contain dictionaries and other types.
            If a key is present more times the associated values must be the same, otherwise an error will be raised
        d_flat: dict (optional), flat dictionary into which to store the items of `d_nested`
    
    Returns:
    --------
        d_flat: dict

    Raises:
    -------
        ValueError: if a key appears more than once with different values

    Examples:
    ---------
    >>> collapse_dict({'a': 10, 'b': {'a': 10, 'c': 4}})
    {'a': 10, 'c': 4}
    >>> collapse_dict({'a': 10, 'b': {'a': 10, 'c': 4}}, d_flat={'a': 10, 'z': 7})
    {'a': 10, 'z': 7, 'c': 4}
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

def set_values_recursive(d_nested, d_flat):
    '''
    Given a nested dictionary `d_nested` replaces its values at any level of indentation according according to the ones in `d_flat`.

    Example:
    --------
    >>> set_values_recursive({'a': 10, 'b': {'a': 10, 'c': 8}}, {'a': 'hello'})
    {'a': 'hello', 'b': {'a': 'hello', 'c': 8}}
    '''
    if len(d_flat) == 0:
        return d_nested

    for k,v in d_nested.items():
        if isinstance(v, dict):
            d_nested[k] = set_values_recursive(v, d_flat)
        elif k in d_flat:
            d_nested[k] = d_flat[k]
    return d_nested

#### PERMUTATIONS ####

def invert_permutation(permutation):
    '''
    Inverts a permutation.

    Parameters:
    -----------
        permutation: 1D array that must be a permutation of an array of the kind `np.arange(n)` with `n` integer

    Examples:
    ---------
    >>> a = np.array([3,4,2,5])
    >>> p = np.random.permutation(np.arange(4))
    >>> a_permuted = a[p]
    >>> p_inverse = invert_permutation(p)
    >>> all(a == a_permuted[p_inverse])
    True
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
    
    Examples:
    ---------
    >>> a = np.array([3,4,2,5])
    >>> p1 = np.random.permutation(np.arange(4))
    >>> p2 = np.random.permutation(np.arange(4))
    >>> p_composed = compose_permutations([p1,p2])
    >>> a_permuted1 = a[p1]
    >>> a_permuted2 = a_permuted1[p2]
    >>> a_permuted_c = a[p_composed]
    >>> all(a_permuted_c == a_permuted2)
    True
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