#######################################################################################################
# In this file we showcase an example of inheritance from Learn2_new.py                               #
# For an example of multiple inheritance, see intrinsically_interpretable_probabilistic_regression.py #
#######################################################################################################
description = """Inheritance template"""
dependencies = None

import Learn2_new as ln
ut = ln.ut
logger = ln.logger
# log to stdout
import logging
import sys
from pathlib import Path

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

#########################
# import other packages #
#########################


################################################################################
# define your custom functions                                                 #
# below is an example redefining the Trainer class and a module level function #
################################################################################
orig_Trainer = ln.Trainer
class Trainer(ln.Trainer):
    def extra_feature(self):
        pass

orig_normalize_X = ln.normalize_X
def normalize_X(X, fold_folder, mode='mycustommode', recompute=False):
    if mode == 'mycustommode':
        # do custom stuff
        return X
    # else use the normal function
    return ln.normalize_X(X, fold_folder, mode=mode, recompute=recompute)

#######################################################
# set the modified functions to override the old ones #
#######################################################
def enable():
    ln.add_mod(__file__, description, dependencies) # add this mod
    # override functions
    ln.Trainer = Trainer
    ln.normalize_X = normalize_X

    # uptade module level config dictionary
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])

    # change default values without modifying functions, below an example
    ut.set_values_recursive(ln.CONFIG_DICT, {'return_threshold': True}, inplace=True)

def disable():
    ln.remove_mod(__file__)
    # restore original functions
    ln.Trainer = orig_Trainer
    ln.normalize_X = orig_normalize_X
    ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # rebuild config dict

if __name__ == '__main__':
    enable() # enable this mod
    ln.main()
