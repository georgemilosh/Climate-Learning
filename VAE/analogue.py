# George Miloshevich 2022
# This routine is written for two parameters: input folder for VAE weights and the given epoch. It computes the analogs based on the projected states using VAE
# The new usage analogue.py <folder> <epochs> <PCAwhitening>
#   example <epochs> = [10,100,1000] true
import os, sys
import shutil
import json
import pickle
from pathlib import Path
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set


folder = Path(sys.argv[1])  # The name of the folder where the weights have been stored
     
checkpoints = sys.argv[2].split(',') # The checkpoint at which the weights have been stored
checkpoints = [int(checkpoint_el) for checkpoint_el in checkpoints]

PCAwhitening = False
if len(sys.argv) > 3:
    PCAwhitening = sys.argv[3] == 'true' # this decides if we normalize the pca components prior to computing the analogs and can only be used if the autoencoder is actually a pca

import logging
from colorama import Fore # support colored output in terminal
from colorama import Style
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO

import pandas as pd 
from scipy.spatial import cKDTree
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
ut = foo.ut


print("==Importing tensorflow packages===")
import random as rd  
from scipy.stats import norm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.metrics import log_loss

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

def classify(fold_folder, evaluate_epoch, vae, X_tr, z_tr, Y_tr, X_va, z_va, Y_va, u=1): # Overwriting the functionality so that it saves analogues
    '''
    Parameters
    -----------------
    X_tr : np.ndarray
        original train set
    z_tr : np.ndarray
        latent train set
    Y_tr:  np.ndarray
        train labels
    X_va : np.ndarray
        original validation set
    z_tr : np.ndarray
        latent validation set
    Y_va:  np.ndarray
        validation labels
    u: float
        prescribed undersampling rate for classification
    '''
    logger.info(f"{vae = }, {fold_folder = }")

    logger.info(f"{Fore.BLUE}")
    logger.info(f"==classify of classification.py==")
    #logger.info(f"{X_va[23,15,65,0] = }, {z_va[23,14] = }, {Y_va[23] = }") # Just testing if data is processed properly (potentially remove this line)
    #logger.info(f"{X_tr[23,15,65,0] = }, {z_tr[23,14] = }, {Y_tr[23] = }") # Just testing if data is processed properly (potentially remove this line)
    score = []

    
    dist_tr = {}
    ind_new_tr = {}
    ind_tr = {}
    dist_va = {}
    ind_new_va = {}
    ind_va = {}
    for checkpoint in checkpoints: # The original intent for this line was to loop over checkpoints of the autoencoder training. 
        found = 0
        while found == 0:
            checkpoint_path = str(fold_folder)+f"/cp_vae-{checkpoint:04d}.ckpt" # TODO: convert checkpoints to f-strings
            checkpoint_path_check = checkpoint_path+".index" 
            if not os.path.exists(checkpoint_path_check) and not PCAwhitening: # it could be that the training was unstable and we have to look for a checkpoint just before (also we should not be working in PCAwhitening mode)
                found = 0
                logger.info(f"there is no {checkpoint_path_check}")
                checkpoint = checkpoint - 1
            else:
                found = 1
                
                # vae = tf.keras.models.load_model(fold_folder, compile=False) # i commented this because vae is already supplied as the input to the classifier
                if PCAwhitening: # We want covariance matrix to be normalized
                    z_tr = (np.diag(np.sqrt(len(X_tr))/vae.encoder.singular_values_) @ vae.encoder.predict(X_tr)[2].T).T
                    z_tr = (np.diag(np.sqrt(len(X_tr))/vae.encoder.singular_values_) @ vae.encoder.predict(X_va)[2].T).T
                else:
                    logger.info(f"==loading the model: {checkpoint_path}")
                    vae.load_weights(f'{checkpoint_path}').expect_partial()
                    logger.info(f'{checkpoint_path} weights loaded')
                    _,_,z_tr = vae.encoder.predict(X_tr)
                    _,_,z_va = vae.encoder.predict(X_va)
                #z = np.concatenate((z_tr,z_va),axis=0) # This structure will be preserved for each fold
                dim = z_tr.shape[1] #dim = z.shape[1]
                siz = z_tr.shape[0] #siz = z.shape[0]
                logger.info(f"{z_tr.shape = }, {siz = }, {T = }, {time_start = }, {time_end = }")
                #Zminus3 = z.reshape(siz//(time_end-time_start-T+1),time_end-time_start-T+1,-1)[:,:-3,:] # Remove last 3 days that makrov chain cannot access
                Zminus3 = z_tr.reshape(siz//(time_end-time_start-T+1),time_end-time_start-T+1,-1)[:,:-3,:] # Remove last 3 days that makrov chain cannot access
                logger.info(f"{Zminus3.shape = }")
                tree = cKDTree(Zminus3.reshape(-1,dim))
                #dist[checkpoint], ind[checkpoint] = tree.query(z, k=1000,n_jobs = 3)
                dist_tr[checkpoint], ind_tr[checkpoint] = tree.query(z_tr, k=1000,n_jobs = 3)
                logger.info(f"{ind_tr[checkpoint] = }")
                ind_new_tr[checkpoint] = ind_tr[checkpoint] // (time_end-time_start-T+1 - 3)*(3) + ind_tr[checkpoint]
                logger.info(f"{ind_new_tr[checkpoint] = }")

                dist_va[checkpoint], ind_va[checkpoint] = tree.query(z_va, k=1000,n_jobs = 3)
                logger.info(f"{ind_va[checkpoint] = }")
                ind_new_va[checkpoint] = ind_va[checkpoint] // (time_end-time_start-T+1 - 3)*(3) + ind_va[checkpoint]
                logger.info(f"{ind_new_va[checkpoint] = }")
                
                logger.info(f"{z_tr.shape = }, {z_va.shape = }, {Zminus3.shape = }" )
                logger.info(f"{ind_new_va[checkpoint].shape = },  {ind_new_tr[checkpoint].shape = }")

                logger.info(f"Before undersampling: {len(Y_tr) = }, {len(Y_va) = }, {np.sum(Y_tr==1) = }, {np.sum(Y_va==1) = }")    
                #z_tr, Y_tr = ln.undersample(z_tr, Y_tr, u=u)  
                #logger.info(f"After undersampling: {len(Y_tr) = }, {len(Y_va) = }, {np.sum(Y_tr==1) = }, {np.sum(Y_va==1) = }")    
                """ # It seems strange I was dumping the matrices each checkpoint, they are growing dictionaries after all...
                open_file = open(f'{fold_folder}/analogues.pkl', "wb")
                pickle.dump({'dist_tr' : dist_tr, 'ind_new_tr' : ind_new_tr, 'dist_va' : dist_va, 'ind_new_va' : ind_new_va}, open_file)
                open_file.close()
                """
        open_file = open(f'{fold_folder}/analogues.pkl', "wb")
        pickle.dump({'dist_tr' : dist_tr, 'ind_new_tr' : ind_new_tr, 'dist_va' : dist_va, 'ind_new_va' : ind_new_va}, open_file)
        open_file.close()
    logger.info(f"{Style.RESET_ALL}")
    return 0

#z_tr[23,24], Y_tr[23], z_va[23,24], Y_va[23]#


run_vae_kwargs = ut.json2dict(f"{folder}/config.json")
T = ut.extract_nested(run_vae_kwargs, 'T')
if (ut.keys_exists(run_vae_kwargs, 'label_period_start') and ut.keys_exists(run_vae_kwargs, 'label_period_end')):
    label_period_start = ut.extract_nested(run_vae_kwargs, 'label_period_start')
    label_period_end = ut.extract_nested(run_vae_kwargs, 'label_period_end')
    time_start = ut.extract_nested(run_vae_kwargs, 'time_start')
    time_end = ut.extract_nested(run_vae_kwargs, 'time_end')
    #if label_period_start is not None:
    #    time_start = label_period_start
    #if label_period_end is not None: We comment this because the idea is to keep extra X's so that we get also the X's for the period that is normally deprecated since those heat waves end in september
    #    time_end = label_period_end
    run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'myinput' : 'N', 'evaluate_epoch' :checkpoints[-1], 'time_start' : time_start}) #, 'time_end' : time_end})
else:
    run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'myinput' : 'N', 'evaluate_epoch' :checkpoints[-1]}) # backward compatibiity where there was no month of may
if not os.path.exists(ut.extract_nested(run_vae_kwargs, 'mylocal')): # we are assuming that training was not run on R740server5
    run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'mylocal' : '/ClimateDynamics/MediumSpace/ClimateLearningFR/gmiloshe/PLASIM/'})
logger.info(f"{run_vae_kwargs = }")
foo.classify = classify
logger.info(f"{Style.RESET_ALL}")


history, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, LAT, LON, vae, X_va, Y_va, X_tr, Y_tr, analogues = foo.run_vae(folder, **run_vae_kwargs)

logger.info(f"{Fore.BLUE}") #  indicates we are inside the routine 
# the rest of the code goes here
logger.info(f"{len(analogues) = }")
logger.info(f"{type(analogues[0]) = }")



logger.info(f"{Style.RESET_ALL}")

# Construct 2D array for lon-lat:
