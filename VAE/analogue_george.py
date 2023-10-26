# George Miloshevich 2022
# This routine is written for one parameter: input folder for VAE weights, coefficients that control the weight of geopotential referred to as checkpoints and the max number of nearest neighbors  
# It computes the analogs using the distance based on local temperature, soil moisture and geopotential where only the latter passes through an autoencoder
# The new usage analogue_george.py <folder> <coefficients> <NN>
#   example <coefficients> = [0,1,10]
# example: python analogue_george.py ./xforanalogsL2loss/ZGonlyNA24by48/Z16/yrs500/interT15fw20.1.20skip2epochs100 0.1,1,10 100
# <NN> = 100
import os, sys
import shutil
import json
import pickle
from pathlib import Path
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set


folder = Path(sys.argv[1])  # The name of the folder where the weights have been stored
     
checkpoints = sys.argv[2].split(',') # The coefficient which multiplies the geopotential field
checkpoints = [float(checkpoint_el) for checkpoint_el in checkpoints]

if len(sys.argv) < 4:
    NN = 1000 # max number of nearest neighbors. Large numbers make very large matrices but if we want to remove self analogs this could be necessary
else:
    NN = int(sys.argv[3])

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
print(f"{ut = }")
print("==Checking GPU==")
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

print("==Checking CUDA==")
tf.test.is_built_with_cuda()

import multiprocessing
cpucount = multiprocessing.cpu_count()

print(f'detected {cpucount} cores')

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
    logger.info(f"{extra_day = }")
    #logger.info(f"{X_va[23,15,65,0] = }, {z_va[23,14] = }, {Y_va[23] = }") # Just testing if data is processed properly (potentially remove this line)
    #logger.info(f"{X_tr[23,15,65,0] = }, {z_tr[23,14] = }, {Y_tr[23] = }") # Just testing if data is processed properly (potentially remove this line)
    
    time_series_tr = np.load(f"{fold_folder}/time_series_tr.npy")
    time_series_va = np.load(f"{fold_folder}/time_series_va.npy")
    # Normalizing:
    time_series_tr_mean = np.mean(time_series_tr,0)
    time_series_tr_std = np.std(time_series_tr,0)
    time_series_tr_std[time_series_tr_std==0] = 1 # If there is no variance we shouldn't divide by zero
    time_series_tr = (time_series_tr-time_series_tr_mean)/time_series_tr_std
    time_series_va = (time_series_va-time_series_tr_mean)/time_series_tr_std
    time_series_tr = np.delete(time_series_tr, 1, axis=1) # removing geopotential from the series, assuming field # 1
    time_series_va = np.delete(time_series_va, 1, axis=1)
    
    logger.info(f'{time_series_tr.shape = }, {X_tr.shape = }, {time_series_va.shape = }, {X_va.shape = }')
    dist_tr = {}
    ind_new_tr = {}
    ind_tr = {}
    dist_va = {}
    ind_new_va = {}
    ind_va = {}
    for checkpoint in checkpoints: #[1]: # This way we maintain the datastructure found in files generated by analogue.py (The original intent for this line was to loop over checkpoints of the autoencoder training. )
        logger.info(f"{checkpoint = }") # Here geopotential has been conveniently extracted while training
        zz_tr = np.concatenate((checkpoint*z_tr/z_tr.shape[1], time_series_tr),axis=1) # we can use checkpoint as a parameter
        zz_va = np.concatenate((checkpoint*z_va/z_va.shape[1], time_series_va),axis=1)
        logger.info(f'{zz_tr.shape = }, {zz_va.shape = }')        
        dim = zz_tr.shape[1] #dim = z.shape[1]
        siz = zz_tr.shape[0] #siz = z.shape[0]
        logger.info(f"{zz_tr.shape = }, {siz = }, {T = }, {time_start = }, {time_end = }")
        #Zminus3 = z.reshape(siz//(time_end-time_start-T+1),time_end-time_start-T+1,-1)[:,:-extra_day,:] # Remove last 3 days that makrov chain cannot access
        Zminus3 = zz_tr.reshape(siz//(time_end-time_start-T+1),time_end-time_start-T+1,-1)[:,:-extra_day,:] # Remove last 3 days that makrov chain cannot access
        logger.info(f"{Zminus3.shape = }")
        logger.info("computing KDTree...")
        tree = cKDTree(Zminus3.reshape(-1,dim))
        #dist[checkpoint], ind[checkpoint] = tree.query(z, k=NN,n_jobs = 3)
        try:
            dist_tr[checkpoint], ind_tr[checkpoint] = tree.query(zz_tr, k=NN,n_jobs = cpucount//2)
        except TypeError:
            dist_tr[checkpoint], ind_tr[checkpoint] = tree.query(zz_tr, k=NN, workers = cpucount//2)
             
        logger.info(f"{ind_tr[checkpoint] = }")
        ind_new_tr[checkpoint] = ind_tr[checkpoint] // (time_end-time_start-T+1 - extra_day)*(extra_day) + ind_tr[checkpoint]
        logger.info(f"{ind_new_tr[checkpoint] = }")
        try:
            dist_va[checkpoint], ind_va[checkpoint] = tree.query(zz_va, k=NN,n_jobs = cpucount//2)
        except TypeError:
            dist_va[checkpoint], ind_va[checkpoint] = tree.query(zz_va, k=NN, workers = cpucount//2)
        logger.info(f"{ind_va[checkpoint] = }")
        ind_new_va[checkpoint] = ind_va[checkpoint] // (time_end-time_start-T+1 - extra_day)*(extra_day) + ind_va[checkpoint]
        logger.info(f"{ind_new_va[checkpoint] = }")
        
        logger.info(f"{zz_tr.shape = }, {zz_va.shape = }, {Zminus3.shape = }" )
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
    run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'myinput' : 'N', 'evaluate_epoch' :1, 'time_start' : time_start}) #, 'time_end' : time_end}) # it doesn't matter which checkpoint we choose since we do not need the output of the VAE
else:
    run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'myinput' : 'N', 'evaluate_epoch' :1}) # backward compatibiity where there was no month of may
if not os.path.exists(ut.extract_nested(run_vae_kwargs, 'mylocal')): # we are assuming that training was not run on R740server5
    run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'mylocal' : '/ClimateDynamics/MediumSpace/ClimateLearningFR/gmiloshe/PLASIM/'})

extra_day = 1
if ut.keys_exists(run_vae_kwargs, 'A_weights'):
    A_weights = ut.extract_nested(run_vae_kwargs, 'A_weights')
    if A_weights is not None:
        extra_day = A_weights[0] # We need to see if the labels were interpolated to see how much the algorithm should jump each summer


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
