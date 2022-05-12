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

import pandas as pd 
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

def classify(fold_folder, evaluate_epoch, vae, X_tr, z_tr, Y_tr, X_va, z_va, Y_va, u=1):
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
    u=1 #10
    percent = ut.extract_nested(run_vae_kwargs, 'percent')
    logger.info(f"{Fore.BLUE}")
    logger.info(f"==classify of classification.py==")
    #logger.info(f"{X_va[23,15,65,0] = }, {z_va[23,14] = }, {Y_va[23] = }") # Just testing if data is processed properly (potentially remove this line)
    #logger.info(f"{X_tr[23,15,65,0] = }, {z_tr[23,14] = }, {Y_tr[23] = }") # Just testing if data is processed properly (potentially remove this line)
    if evaluate_epoch>999:
        checkpoints = list(range(0,10,1))+list(range(10,100,10))+list(range(100,1000,100))+list(range(1000,evaluate_epoch+1,1000))
    elif evaluate_epoch>99: # because we want range to reach the last evaluate_epoch
        checkpoints = list(range(0,10,1))+list(range(10,100,10))+list(range(100,evaluate_epoch+1,100))
    elif evaluate_epoch>9: # because we want range to reach the last evaluate_epoch
        checkpoints = list(range(0,10,1))+list(range(10,evaluate_epoch+1,10))
    else:
        checkpoints = list(range(0,evaluate_epoch+1,1))
    checkpoints = checkpoints[1:]
    score = []
    labels = ['Log_L2','kNN_uni','kNN_dst']
    L_parameter = [1e-1, 5e-1, 1e0, 1e1, 1e2, 1e3, 1e5] # Logistic regularization coefficients (L2 by default)
    K_parameter = [] # Number of KNN
    #K_parameter2 = [1, 2, 3, 5, 10, 15, 20, 40, 100] # Number of KNN

    for checkpoint in checkpoints:
        score_method = []
        checkpoint_path = str(fold_folder)+f"/cp_vae-{checkpoint:04d}.ckpt" # TODO: convert checkpoints to f-strings
        logger.info(f"==loading the model: {checkpoint_path}")
        vae = tf.keras.models.load_model(fold_folder, compile=False)
        vae.load_weights(f'{checkpoint_path}').expect_partial()
        logger.info(f'{checkpoint_path} weights loaded')
        _,_,z_tr = vae.encoder.predict(X_tr)
        _,_,z_va = vae.encoder.predict(X_va)
        logger.info(f"{z_tr.shape = }, {z_va.shape = }" )

        logger.info(f"Before undersampling: {len(Y_tr) = }, {len(Y_va) = }, {np.sum(Y_tr==1) = }, {np.sum(Y_va==1) = }")    
        z_tr, Y_tr = ln.undersample(z_tr, Y_tr, u=u)  
        logger.info(f"After undersampling: {len(Y_tr) = }, {len(Y_va) = }, {np.sum(Y_tr==1) = }, {np.sum(Y_va==1) = }")    
        maxskill = -(percent/100.)*np.log(percent/100.)-(1-percent/100.)*np.log(1-percent/100.)
        logisticclassifier = [LogisticRegression(solver='liblinear',C=index_i, n_jobs=len(L_parameter)) for index_i in L_parameter]
        kNNclassifier = [neighbors.KNeighborsClassifier(n_neighbors, weights="uniform",n_jobs=32) for n_neighbors in K_parameter]
        # kNNclassifier2 = [neighbors.KNeighborsClassifier(n_neighbors, weights="distance") for n_neighbors in K_parameter]
        for classifier, C_parameter in zip([logisticclassifier, kNNclassifier], [L_parameter,K_parameter]):
            logger.info(f"{classifier = }")
            entropy, skill= [np.zeros(len(classifier),) for x in range(2)]
            for i in range(len(classifier)):
                classifier[i].fit(z_tr, Y_tr)
                logger.info("model fit")
                # Y_pr = classifier[i].predict(z_va) 
                Y_pr_prob = classifier[i].predict_proba(z_va)
                logger.info("model predict")
                entropy[i] = log_loss(Y_va, Y_pr_prob[:, 1])
                skill[i] = (maxskill-entropy[i])/maxskill
                logger.info(f"{skill.shape = }")
                # TP[i], TN[i], FP[i], FN[i], MCC[i] = ef.ComputeMCC(Y_va, Y_pr, 'True')
                # _, entropy[i], skill[i], BS[i], _, freq[i] = ef.ComputeMetrics(np.array(Y_va), Y_pr_prob, percent, reundersampling_factor=u, print_output=False) 

            score_method.append(pd.DataFrame(np.array([C_parameter, entropy, skill]).transpose(), columns =['C', 'entropy', 'skill'])) 
        score.append(pd.concat(score_method, keys=labels,names=['method', None]))

    score = pd.concat(score, keys=checkpoints,names=['checkpoint','method', None])
    logger.info('score:')
    logger.info(f'{score}')
    logger.info(f"{Style.RESET_ALL}")
    return score

#z_tr[23,24], Y_tr[23], z_va[23,24], Y_va[23]#
run_vae_kwargs = ut.json2dict(f"{folder}/config.json")

foo.classify = classify
logger.info(f"{Style.RESET_ALL}")
run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'myinput' : 'N', 'evaluate_epoch' :checkpoint})
history, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, LAT, LON, vae, X_va, Y_va, X_tr, Y_tr, score = foo.run_vae(folder, **run_vae_kwargs)

logger.info(f"{Fore.BLUE}") #  indicates we are inside the routine 
# the rest of the code goes here
#logger.info('score = ')
#logger.info(f'{score}')
score = pd.concat(score, keys=range(len(score)),names=['fold','checkpoint','method', None])
#logger.info('score:')
#logger.info(f'{score}')
score.to_csv(f'{folder}/score.csv')

logger.info(f"{Style.RESET_ALL}")

# Construct 2D array for lon-lat:
