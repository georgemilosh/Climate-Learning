'''
Imports the argument from a given config file to the one in the current directory.

Usage:
    python import_config.py <path to config file from which to import arguments>

If the two config files have different arguments it is not a problem since the importation will affect only the arguments in common between the two.
'''

import sys
import os
from pathlib import Path

from Learn2_new import ut

if len(sys.argv) < 2:
    print(__doc__)

if not os.path.exists('./config.json'):
    raise FileNotFoundError('There is no config file in this directory')

if not os.acces('./config.json', os.W_OK):
    raise ValueError('config is not writeable!')

config = ut.json2dict('./config.json')

path_to_import_config = sys.argv[1]
if not path_to_import_config.endswith('config.json'):
    path_to_import_config = path_to_import_config.rstrip('/')
    path_to_import_config += '/config.json'

config_to_import = ut.json2dict(path_to_import_config)
config_to_import_flat = ut.collapse_dict(config_to_import)

config = ut.set_values_recursive(config, config_to_import_flat)

ut.dict2json(config, './config.json')