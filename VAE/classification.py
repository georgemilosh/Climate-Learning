# George Miloshevich 2022
# This routine is written for two parameters: input folder for VAE weights and the given epoch. It shows us how good the classification of the VAE works
import os, sys
import shutil
from pathlib import Path
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set


folder = Path(sys.argv[1])  # The name of the folder where the weights have been stored
checkpoint = sys.argv[2]       # The checkpoint at which the weights have been stored

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

def classify(z_tr, Y_tr, z_va, Y_va):
    '''
    At the moment is void
    '''

    logger.info(f"{Fore.YELLOW}==classify of classification.py=={Style.RESET_ALL}")
    return None

foo.classify = classify

history, history_loss, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, LAT, LON, Y, vae, X_va, Y_va, X_tr, Y_tr, score = foo.run_vae(folder, myinput='N')
# Construct 2D array for lon-lat:

"""
print(f"{X_tr.shape = }, {np.max(X_tr) = }, {np.min(X_tr) = }, {np.mean(X_tr[:,5,5,0]) = }, {np.std(X_tr[:,5,5,0]) = }")
print(f"==loading the model: {fold_folder}")
vae = tf.keras.models.load_model(fold_folder, compile=False)

nb_zeros_c = 4-len(str(checkpoint))
checkpoint_i = '/cp_vae-'+nb_zeros_c*'0'+str(checkpoint)+'.ckpt' # TODO: convert to f-strings

print(f'load weights from {fold_folder}/{checkpoint_i}')
vae.load_weights(f'{fold_folder}/{checkpoint_i}')
      

_,_,z_tr = vae.encoder.predict(X_tr)
_,_,z_va = vae.encoder.predict(X_va)
print(f"{z_tr.shape = }, {z_tr.shape = }" )

logreg = LogisticRegression(solver='liblinear',C=1e5)
logreg.fit(z_tr, Y_tr)
Y_pr = logreg.predict(z_va) 
print(f"{Y_pr.shape = }, {z_tr.shape = }" )

TP, TN, FP, FN, MCC = ef.ComputeMCC(Y_va, Y_pr, 'True')

#Z_DIM = z_tr.shape[1] #200 # Dimension of the latent vector (z)
"""
