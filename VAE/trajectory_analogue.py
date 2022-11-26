# George Miloshevich 2022

import os, sys
import pickle
from pathlib import Path

from functools import partial # allows us to create a function with arguments passed
folder = Path(sys.argv[1])  # The name of the folder where the weights have been stored
     
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

from importlib import import_module
#foo = import_module(fold_folder+'/Funs.py', package=None)


#folder = './xforanalogs/NA24by48/Z8/yrs500/interT15fw20.1.20lrs4'
foo = module_from_file("foo", f'{folder}/Funs.py')
import pickle
import random as rd  
from scipy.stats import norm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.metrics import log_loss

tff = foo.tff # tensorflow routines 
ut = foo.ut # utilities
ln = foo.ln #Learn2_new.py
print(ln)
print(ut)
print(tff)
import committor_analogue as ca
print(ca)

ca.tff = tff
ca.ut = ut
ca.ln = ln


import numba as nb
from numba import jit,guvectorize,set_num_threads
@guvectorize([(nb.int64,nb.int64,nb.int64[:],nb.int64[:],nb.int64,nb.int64[:,:],nb.int64[:,:])],
             '(),(),(n),(m),(),(k,o)->(n,m)',nopython=True,target="parallel") # to enable numba parallelism
def TrajOnePoint(day, nn, n_Traj, numsteps, Markov_step, Matr_tr,res): # a day implies the temporal coordinates in days of the input from the 0-th year of the 1000 year long dataset
    """ Compute `n_Traj` trajectories starting from `day` that take `numsteps` steps
    Args:
        day (_int_):            day in the validation set where the Markov chain will start
        nn (_int_):             number of nearest neighbors to look for
        n_Traj (_int array_):         Number of MC samples that start from the same day (number of trajectories).
        numsteps (_int array_):       numbe of steps to take in each trajectory
        Markov_step (_int_):    step in the Markov chain (how many days)
        Matr_tr (_ndarray_):    T matrix inside the training set
        res (_float_):          stores the committor (return), this is how numba vectorization forces the output to be treated
    """
    wrong_index = 0 # This checks that during input or execution we were always working with indecies that exist in the considered matrices and we don't go below or above
    if (day >= Matr_tr.shape[0]) or (day < 0): # We don't allow inputs that are outside of the range of Matr_va. 
        #print("day > Matr_va.shape[0]")
        wrong_index = 1 # manual debugging (unfortunately numba does not capture this)
        print("We don't allow inputs that are outside of the range of Matr_va")
        for l_1 in n_Traj:
            for l_2 in numsteps:
                res[l_1][l_2] = np.nan # we simply  don't have corresponding index
    if nn > Matr_tr.shape[1]:
        wrong_index = 1 # manual debugging: use to monitor if we get out of the Matr_tr allowed set
        print("We don't allow inputs that are outside of the range of Matr_va")
    else:
        #print("day <= Matr_va.shape[0]")
        for i in n_Traj:   
            s = day # We initialize trajectory at the first day
            res[i][0] = s
            #print("output: ", day,app,s, Matr_va.shape)
            if (s >= Matr_tr.shape[0]) or (s < 0):
                wrong_index = 1
            for j in numsteps[1:]: 
                    app = rd.randint(0,nn-1) #analogue selection
                    s = Matr_tr[s][app] + Markov_step         #analog state s is evolved in time
                    res[i][j] = s
                    if (s >= Matr_tr.shape[0]) or (s < 0):
                        wrong_index = 1
                    if nn > Matr_tr.shape[1]:
                        wrong_index = 1 # manual debugging: use to monitor if we get out of the Matr_tr allowed set
        if wrong_index == 1:
            print("Somewhere inside the code there was an input outside of the range of matrices/vectors")
            for i in n_Traj:
                for j in numsteps:
                    res[l_1][l_2] = np.nan # we simply  don't have the corresponding index
        
def DressTrajOnePoint(day, ther,dela, Temp_va, Temp_tr, nn, n_Traj, numsteps, Markov_step, Matr_va, Matr_tr):
    """This function calls TrajOnePoint by selecting the right inputs provided to ca.CommOnePoint()
    """
    n_Traj = np.arange(n_Traj)
    numsteps = np.arange(numsteps) # Note that this overwrites what was prescribed in ca.RunNeighbors(). We want to simulate a day in summer
    print(f'{n_Traj = }')
    return TrajOnePoint(day, nn, n_Traj, numsteps, Markov_step, Matr_tr)

ca.CommOnePoint = DressTrajOnePoint

ln.RunCheckpoints = ca.RunCheckpoints
ln.RunNeighbors = ca.RunNeighbors
ln.RunFolds = ca.RunFolds


#from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.insert(1, '../ERA')
year_permutation = np.load(f'{folder}/year_permutation.npy')
run_vae_kwargs = ut.json2dict(f"{folder}/config.json")
T = ut.extract_nested(run_vae_kwargs, 'T')
if (ut.keys_exists(run_vae_kwargs, 'label_period_start') and ut.keys_exists(run_vae_kwargs, 'label_period_end')):
    label_period_start = ut.extract_nested(run_vae_kwargs, 'label_period_start')
    label_period_end = ut.extract_nested(run_vae_kwargs, 'label_period_end')
    time_start = ut.extract_nested(run_vae_kwargs, 'time_start')
    time_end = ut.extract_nested(run_vae_kwargs, 'time_end')
threshold = np.array([np.load(f'{folder}/threshold.npy')]) #Threshold defining committor. This parameter I don't need, I shall perhaps transform it into epochs for variational autoencoder 
percent = ut.extract_nested(run_vae_kwargs, 'percent')
nfolds = ut.extract_nested(run_vae_kwargs, 'nfolds')
n_days = time_end-time_start-T+1   
logger.info(f"{Style.RESET_ALL}")

extra_day=1
if ut.keys_exists(run_vae_kwargs, 'A_weights'):
    A_weights = ut.extract_nested(run_vae_kwargs, 'A_weights')
    if A_weights is not None:
        extra_day = A_weights[0] # We need to see if the labels were interpolated to see how much the algorithm should jump each summer
if extra_day == 3:
    delay = np.arange(6)
else:
    delay = 3*np.arange(6)

RunFolds_kwargs_default = ln.get_default_params(ca.RunFolds, recursive=True)
RunFolds_kwargs_default = ut.set_values_recursive(
    RunFolds_kwargs_default, {'num_Traj' : 10, 'chain_step' : extra_day,  'threshold' : threshold,
                                'delay' : delay, 'neighbors' : [3,5,10,20,40], 'num_steps' : n_days - (label_period_start-time_start),
                              'T' : T, 'allowselfanalogs' : True, 'input_set' : 'va', 'bulk_set' : 'tr',
                              'start_calendar_day' :(label_period_start-time_start), 'start_day_set' : 'tr'}  )

chain_step = ut.extract_nested(RunFolds_kwargs_default, 'chain_step')  
logger.info(RunFolds_kwargs_default)


logger.info(f"{Fore.BLUE}") #  indicates we are inside the routine 

with open(f'{folder}/fold_{0}/analogues.pkl', "rb") as open_file:
    analog = pickle.load(open_file)
analogues_tr = list(analog['ind_new_tr'].values())[0] # Here we need to load just random analogs for the compilation below
time_series_tr = np.load(f"{folder}/fold_{0}/time_series_tr.npy")[:,0]
#sec_1 = TrajOnePoint(33, [2,3,5,10,20,50],10, 5, chain_step, analogues_tr) # compiling (maybe we only need this once)
sec = ca.RunFolds(folder,1, threshold, n_days, **RunFolds_kwargs_default)[0]   # We only run 1 fold
logger.info(f"{Style.RESET_ALL}")

#logger.info(f'{sec[10][10].shape = }')

#logger.info(sec[10][10][0,0])

#logger.info(sec[10][10][0,0]%n_days)
time_series_synth = {k: {j:time_series_tr[u] for j, u in v.items()} for k, v in sec.items()}
#logger.info(f'{time_series_synth[10][10].shape = }')
convolve_vec = np.vectorize(partial(np.convolve, **{'mode':'valid'}), signature='(n),(m)->(k)')
A_synth = {k: {j:convolve_vec(u, np.ones(T)/T) for j, u in v.items()} for k, v in time_series_synth.items()}
logger.info(f'{A_synth[10][10].shape = }')
logger.info('Saving the synthetic time series in ')


#logger.info(time_series_tr[sec[10][10]][0,0])




# Construct 2D array for lon-lat:
