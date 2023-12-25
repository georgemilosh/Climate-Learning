#########################################################################
# In this file we showcase an example of inheritance from Learn2_new.py #
######################################################################### 

import Learn2_new as ln
ut = ln.ut
logger = ln.logger
# log to stdout
import logging
import sys
import os

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

class Trainer(ln.Trainer):
    def extra_feature(self):
        pass

def normalize_X(X, fold_folder, mode='mycustommode', recompute=False):
    if mode == 'mycustommode':
        # do custom stuff
        return X
    # else use the normal function
    return ln.normalize_X(X, fold_folder, mode=mode, recompute=recompute)

#######################################################
# set the modified functions to override the old ones #
#######################################################
ln.Trainer = Trainer
ln.normalize_X = normalize_X

# uptade module level config dictionary
ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram])

# change default values without modifying functions, below an example
ut.set_values_recursive(ln.CONFIG_DICT, {'return_threshold': True}, inplace=True) 

# override the main function as well (you don't need to edit the following code)
if __name__ == '__main__':
    ln.main()

    lock = ln.Path(__file__).resolve().parent / 'lock.txt'
    if os.path.exists(lock): # there is a lock
        # check for folder argument
        if len(sys.argv) == 2:
            folder = sys.argv[1]
            print(f'moving code to {folder = }')
            # copy this file
            path_to_here = ln.Path(__file__).resolve() # path to this file
            ln.shutil.copy(path_to_here, folder)

