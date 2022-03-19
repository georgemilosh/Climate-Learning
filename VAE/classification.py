# George Miloshevich 2022
# This routine is written for two parameters: input folder for VAE weights and the given epoch. It shows us how good the classification of the VAE works
import os, sys
import shutil
from pathlib import Path
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set


folder = Path(sys.argv[1])  # The name of the folder where the weights have been stored
checkpoint = int(sys.argv[2])       # The checkpoint at which the weights have been stored

import logging
from colorama import Fore # support colored output in terminal
from colorama import Style
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO

import importlib.util
def module_from_file(module_name, file_path): #The code that imports the file which originated the training with all the instructions
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
logger.info(f"{Fore.BLUE}") #  indicates we are inside the routine  
print("fold_folder = ", folder)
print(f"loading module from  {folder}/Funs.py")
from importlib import import_module
#foo = import_module(fold_folder+'/Funs.py', package=None)
foo = module_from_file("foo", f'{folder}/Funs.py')
ef = foo.ef # Inherit ERA_Fields_New from the file we are calling

print("==Importing tensorflow packages===")
import random as rd  
from scipy.stats import norm
import numpy as np
from sklearn.linear_model import LogisticRegression

tff = foo.tff # tensorflow routines 
ut = foo.ut # utilities
ln = foo.ln #Learn2_new.py
print("==Checking GPU==")
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

print("==Checking CUDA==")
tf.test.is_built_with_cuda()

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.insert(1, '../ERA')

print("==Reading data==")

year_permutation = np.load(f'{folder}/year_permutation.npy')

def classify(X_tr, z_tr, Y_tr, X_va, z_va, Y_va, u=1):
    '''
    insert description
    '''
    u=1 #10
    
    logger.info(f"{Fore.BLUE}")
    logger.info(f"==classify of classification.py==")
    logger.info(f"{X_va[23,15,65,0] = }, {z_va[23,14] = }, {Y_va[23] = }") # Just testing if data is processed properly (potentially remove this line)
    logger.info(f"{X_tr[23,15,65,0] = }, {z_tr[23,14] = }, {Y_tr[23] = }") # Just testing if data is processed properly (potentially remove this line)
    logger.info(f"Before undersampling: {len(Y_tr) = }, {len(Y_va) = }, {np.sum(Y_tr==1) = }, {np.sum(Y_va==1) = }")    
    z_tr, Y_tr = ln.undersample(z_tr, Y_tr, u=u)  
    logger.info(f"After undersampling: {len(Y_tr) = }, {len(Y_va) = }, {np.sum(Y_tr==1) = }, {np.sum(Y_va==1) = }")    
    logreg = LogisticRegression(solver='liblinear',C=1e5)
    logreg.fit(z_tr, Y_tr)
    Y_pr = logreg.predict(z_va) 

    TP, TN, FP, FN, MCC = ef.ComputeMCC(Y_va, Y_pr, 'True')
    logger.info(f"{Y_pr[23] = }")
    logger.info(f"{Style.RESET_ALL}")
    return TP, TN, FP, FN, MCC 
#z_tr[23,24], Y_tr[23], z_va[23,24], Y_va[23]#

foo.classify = classify
logger.info(f"{Style.RESET_ALL}")
history, history_loss, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, LAT, LON, Y, vae, X_va, Y_va, X_tr, Y_tr, score = foo.run_vae(folder, myinput='N', evaluate_epoch=checkpoint)
logger.info(f"{Fore.BLUE}") #  indicates we are inside the routine 
# the rest of the code goes here
logger.info(f"{Style.RESET_ALL}")
# Construct 2D array for lon-lat:
print(score)
