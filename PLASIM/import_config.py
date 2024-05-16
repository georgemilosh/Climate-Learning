'''
Imports the argument from a given config file to the one in the current directory.

Usage:
    python import_config.py <path to config file from which to import arguments>

Examples:
    python import_config.py ../t1/config.json  # imports arguments from ../t1/config.json
    python import_config.py ../t1              # imports arguments from ../t1/config.json
    python import_config.py ../t1/43           # imports arguments from the parameters of run 43 in ../t1, i.e. both the config file ../t1/config.json and the non-default paraamters specified by run 43

If the two config files have different arguments it is not a problem since the importation will affect only the arguments in common between the two.
'''
# GM: perhaps it would be nice to explain reasons for why this might be necessary
import sys
import os
from pathlib import Path

from Learn2_new import ut

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(0)

if not os.path.exists('./config.json'):
    raise FileNotFoundError('There is no config file in this directory')

if not os.access('./config.json', os.W_OK):
    raise ValueError('config is not writeable!')

config = ut.json2dict('./config.json')

run_id=None
path_to_import_config = Path(sys.argv[1])
if not path_to_import_config.stem.startswith('config'):
    # maybe is just missing a 'config.json'
    if (path_to_import_config / 'config.json').exists():
        path_to_import_config = path_to_import_config / 'config.json'
    elif (path_to_import_config.parent / 'config.json').exists():
        try:
            run_id = f'{int(path_to_import_config.stem)}'
        except ValueError:
            raise ValueError('run_id must be the string representation of an integer')
        path_to_import_config = path_to_import_config.parent / 'config.json'
    else:
        raise FileNotFoundError('There is no config file in this directory')

config_to_import = ut.json2dict(path_to_import_config)
config_to_import_flat = ut.collapse_dict(config_to_import)

ut.set_values_recursive(config, config_to_import_flat, inplace=True)

if run_id is not None:
    runs = ut.json2dict(path_to_import_config.parent / 'runs.json')
    run = runs[run_id]
    ut.set_values_recursive(config, run['args'], inplace=True)

ut.dict2json(config, './config.json')