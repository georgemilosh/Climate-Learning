# George Miloshevich 2022
# inspired by https://keras.io/examples/generative/vae/
# merged with (and upgraded to tensorflow 2) https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8
# Adapted some routines from Learn2_new.py of Alessandro Lovo


### IMPORT LIBRARIES #####

## general purpose
from cmath import log
from copy import deepcopy
import os as os
from pathlib import Path
from stat import S_IREAD
import sys
from tkinter.messagebox import NO
import traceback
from unittest import skip
import warnings
import time
import shutil
import gc
from matplotlib import path
from matplotlib.pyplot import hist, loglog
import psutil
import numpy as np
import inspect
import ast
import logging
import pickle
import itertools



## machine learning
# from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

sys.path.insert(1, '../ERA')

import ERA_Fields as ef # general routines
import TF_Fields as tff # tensorflow routines
import utilities as ut



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set
# separators to create the run name from the run arguments
arg_sep = '--' # separator between arguments
value_sep = '__' # separator between an argument and its value

########## USAGE ###############################


############################################
########## DATA PREPROCESSING ##############
############################################

fields_infos = {
    't2m': { # temperature
        'name': 'tas',
        'filename_suffix': 'tas',
        'label': 'Temperature',
    },
    'mrso': { # soil moisture
        'name': 'mrso',
        'filename_suffix': 'mrso',
        'label': 'Soil Moisture',
    },
}

for h in [200,300,500,850]: # geopotential heights
    fields_infos[f'zg{h}'] = {
        'name': 'zg',
        'filename_suffix': f'zg{h}',
        'label': f'{h} mbar Geopotential',
    }

    
def load_data(dataset_years=8000, year_list=None, sampling='', Model='Plasim', area='France', filter_area='France',
              lon_start=0, lon_end=128, lat_start=0, lat_end=24, mylocal='/local/gmiloshe/PLASIM/',fields=['t2m_filtered','zg500','mrso_filtered']):
    '''
    Loads the data into Plasim_Fields objects

    Parameters
    ----------
    dataset_years : int, optional
        number of years of the dataset, for now 8000 or 1000.
    year_list : array-like or str or int or tuple or None, optional
        list of years to load from the dataset. If None all years are loaded
        if str must be in the format 'range([<start>],<end>,[<step>])', where square brackets mean the argument is optional. It will be interpreted as np.range([<start>],<end>,[<step>])
        if tuple must be in format ([<start>],<end>,[<step>])
        if int is just like providing only <end>
    sampling : str, optional
        '' (dayly) or '3hrs'
    Model : str, optional
        'Plasim', 'CESM', ... For now only Plasim is implemented
    area : str, optional
        region of interest, e.g. 'France'
    filter_area : str, optional
        area over which to keep filtered fields, usually the same of `area`. `filter` implies a mask
    lon_start, lon_end, lat_start, lat_end : int
        longitude and latitude extremes of the data expressed in indices (model specific)
    mylocal : str or Path, optional
        path the the data storage. For speed it is better if it is a local path.
    fields : list, optional
        list of field names to be loaded. Add '_filtered' to the name to have the values of the field outside `filter_area` set to zero.

    Returns
    -------
    _fields: dict
        dictionary of ERA_Fields.Plasim_Field objects
    '''

    if area != filter_area:
        warnings.warn(f'Fields will be filtered on a different area ({filter_area}) than the region of interest ({area})')

    if dataset_years == 1000:
        dataset_suffix = ''
    elif dataset_years == 8000:
        dataset_suffix = '_LONG'
    else:
        raise ValueError(f'Invalid number of {dataset_years = }')

    if isinstance(year_list, str):
        if '(' not in year_list or ')' not in year_list:
            raise ValueError(f'Unable to parse {year_list = }')
        year_list = f"({year_list.split('(',1)[1].split(')',1)[0]})" # get just the arguments
        year_list = ast.literal_eval(year_list) # now year_list is int or tuple

    if isinstance(year_list,int):
        year_list = np.arange(year_list)
    elif isinstance(year_list, tuple):
        year_list = np.arange(*year_list) # unpack the arguments of the tuple
    
    mask, cell_area, lsm = ef.ExtractAreaWithMask(mylocal,Model,area) # extract land-sea mask and multiply it by cell area

    if sampling == '3hrs': 
        prefix = ''
        file_suffix = f'../Climate/Data_Plasim{dataset_suffix}/'
    else:
        prefix = f'ANO{dataset_suffix}_'
        file_suffix = f'Data_Plasim{dataset_suffix}/'

    # load the fields
    _fields = {}
    for field_name in fields:
        do_filter = False
        if field_name.endswith('_filtered'): # TO IMPROVE: if you have to filter the data load just the interesting part
            field_name = field_name.rsplit('_', 1)[0] # remove '_filtered'
            do_filter = True
        if field_name not in fields_infos:
            raise KeyError(f'Unknown field {field_name}')
        f_infos = fields_infos[field_name]
        # create the field object
        field = ef.Plasim_Field(f_infos['name'], prefix+f_infos['filename_suffix'], f_infos['label'],
                                Model=Model, lat_start=lat_start, lat_end=lat_end, lon_start=lon_start, lon_end=lon_end,
                                myprecision='single', mysampling=sampling, years=dataset_years)
        # load the data
        field.load_field(mylocal+file_suffix, year_list=year_list)
        # Set area integral
        field.abs_area_int, field.ano_area_int = field.Set_area_integral(area,mask,containing_folder=None) # don't save area integrals in order to avoid conflicts between different runs
        # filter
        if do_filter: # set to zero all values outside `filter_area`
            filter_mask = ef.create_mask(Model, filter_area, field.var, axes='last 2', return_full_mask=True)
            field.var *= filter_mask

        _fields[field_name] = field  
    
    return _fields

def prepare_data(load_data_kwargs=None, prepare_XY_kwargs=None):
    # GM: since the kwargs are passed in a recursive manner it makes it difficult to keep track of how the function such as prepare_data can be used in isolation from Trainer class, or for example prepare_XY. Perhaps some short totorial would be appropriate
    '''
    Combines all the steps from loading the data to the creation of X and Y

    Parameters
    ----------
    load_data_kwargs: dict
        arguments to pass to the function `load_data`
    prepare_XY_kwargs: dict
        arguments to pass to the function `prepare_XY`
        
    Returns
    -------
    X : np.ndarray
        data. If flatten_time_axis with shape (days, lat, lon, fields), else (years, days, lat, lon, fields)
    Y : np.ndarray 
        labels. If flatten_time_axis with shape (days,), else (years, days)
    year_permutation : np.ndarray
        with shape (years,), final permutaion of the years that reproduces X and Y once applied to the just loaded data
    '''
    if load_data_kwargs is None:
        load_data_kwargs = {}
    if prepare_XY_kwargs is None:
        prepare_XY_kwargs = {}
    # load data
    fields = load_data(**load_data_kwargs)

    return prepare_XY(fields, **prepare_XY_kwargs)  

def make_XY(fields, label_field='t2m', time_start=30, time_end=120, T=14, tau=0, percent=5, threshold=None):
    '''
    Combines `make_X` and `assign_labels`

    Parameters:
    -----------
    fields : dict of Plasim_Field objects
    label_field : str, optional
        key for the field used for computing labels
    time_start : int, optional
        first day of the period of interest
    time_end : int, optional
        first day after the end of the period of interst
    T : int, optional
        width of the window for the running average
    tau : int, optional
        delay between observation and prediction
    percent : float, optional
        percentage of the most extreme heatwaves
    threshold : float, optional
        if provided overrides `percent`

    Returns:
    --------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    Y : np.ndarray
        with shape (years, days)
    '''
    X = make_X(fields, time_start=time_start, time_end=time_end, T=T, tau=tau)
    Y = assign_labels(fields[label_field], time_start=time_start, time_end=time_end, T=T, percent=percent, threshold=threshold)
    return X,Y

def assign_labels(field, time_start=30, time_end=120, T=14, percent=5, threshold=None):
    '''
    Given a field of anomalies it computes the `T` days forward convolution of the integrated anomaly and assigns label 1 to anomalies above a given `threshold`.
    If `threshold` is not provided, then it is computed from `percent`, namely to identify the `percent` most extreme anomalies.

    Parameters
    ----------
    field : Plasim_Field object
    time_start : int, optional
        first day of the period of interest
    time_end : int, optional
        first day after the end of the period of interst
    T : int, optional
        width of the window for the running average
    percent : float, optional
        percentage of the most extreme heatwaves
    threshold : float, optional
        if provided overrides `percent`.

    Returns:
    --------
    labels : np.ndarray
        2D array with shape (years, days) and values 0 or 1
    '''
    A, A_flattened, threshold =  field.ComputeTimeAverage(time_start, time_end, T=T, percent=percent, threshold=threshold)[:3]
    return np.array(A >= threshold, dtype=int)

def roll_X(X, roll_axis='lon', roll_steps=64):
    '''
    Rolls `X` along a given axis. useful for example for moving France away from the Greenwich meridian.
    In other words this allows one, for example, to shift the grid so that desired areas are not found at the boundary.
    In principle this function allows us to roll along arbitrary axis, including days or years.

    Parameters
    ----------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    axis : str, optional
        'year' (or 'y'), 'day' (or 'd'), 'lat', 'lon', 'field' (or 'f')
    steps : int, optional
        number of gridsteps to roll: a positive value for 'steps' means that the elements of the array are moved forward in it,
        e.g. `steps` = 1 means that the old first element is now in the second place
        This means that for every axis a positive value of `steps` yields a shift of the array
        'year', 'day' : forward in time
        'lat' : southward
        'lon' : eastward
        'field' : forward in the numbering of the fields
    
    Returns
    -------
    X_rolled : np.ndarray
        of the same shape of `X`
    '''
    if roll_steps == 0:
        return X
    if roll_axis.startswith('y'):
        roll_axis = 0
    elif roll_axis.startswith('d'):
        roll_axis = 1
    elif roll_axis == 'lat':
        roll_axis = 2
    elif roll_axis == 'lon':
        roll_axis = 3
    elif roll_axis.startswith('f'):
        roll_axis = 4
    else:
        raise ValueError(f'Unknown valur for axis: {roll_axis}')
    return np.roll(X,roll_steps,axis=roll_axis)


def prepare_XY(fields, make_XY_kwargs=None, roll_X_kwargs=None,
               do_premix=False, premix_seed=0, do_balance_folds=True, nfolds=10, year_permutation=None, flatten_time_axis=True):
    '''
    Performs all operations to extract from the fields X and Y ready to be fed to the neural network.

    Parameters
    ----------
    fields : dict of ef.Plasim_Field objects
    make_XY_kwargs : dict
        arguments to pass to the function `make_XY`
    roll_X_kwargs : dict
        arguments to pass to the function `roll_X`
    do_premix : bool, optional
        whether to perform premixing, by default False
    premix_seed : int, optional
        seed for premixing, by default 0
    do_balance_folds : bool, optional
        whether to balance folds
    nfolds : int, optional
        necessary for balancing folds
    year_permutation : np.ndarray, optional
        if provided overrides both premixing and fold balancing, useful for transfer learning as avoids contaminating test sets. By default None
    flatten_time_axis : bool, optional
        whether to flatten the time axis consisting of years and days

    Returns
    -------
    X : np.ndarray
        data. If flatten_time_axis with shape (days, lat, lon, fields), else (years, days, lat, lon, fields)
    Y : np.ndarray 
        labels. If flatten_time_axis with shape (days,), else (years, days)
    tot_permutation : np.ndarray
        with shape (years,), final permutaion of the years that reproduces X and Y once applied to the just loaded data
    '''
    if make_XY_kwargs is None:
        make_XY_kwargs = {}
    if roll_X_kwargs is None:
        roll_X_kwargs = {}
    X,Y = make_XY(fields, **make_XY_kwargs)
    
    # move greenwich_meridian
    X = roll_X(X, **roll_X_kwargs)

    # mixing
    print('Mixing')
    start_time = time.time()
    
    if year_permutation is None:
        # premixing
        if do_premix:
            premix_permutation = shuffle_years(X, seed=premix_seed, apply=False)
            Y = Y[premix_permutation]
            year_permutation = premix_permutation

        # balance folds:
        if do_balance_folds:
            weights = np.sum(Y, axis=1) # get the number of heatwave events per year
            balance_permutation = balance_folds(weights,nfolds=nfolds, verbose=True)
            Y = Y[balance_permutation]
            if year_permutation is None:
                year_permutation = balance_permutation
            else:
                year_permutation = ut.compose_permutations([year_permutation, balance_permutation])
    else:
        year_permutation = np.array(year_permutation)
        Y = Y[year_permutation]
        print('Mixing overriden by provided permutation')

    # apply permutation to X
    if year_permutation is not None:    
        X = X[year_permutation]
    print(f'Mixing completed in {ut.pretty_time(time.time() - start_time)}\n')
    print(f'{X.shape = }, {Y.shape = }')

    # flatten the time axis dropping the organizatin in years
    if flatten_time_axis:
        X = X.reshape((X.shape[0]*X.shape[1],*X.shape[2:]))
        Y = Y.reshape((Y.shape[0]*Y.shape[1]))
        print(f'Flattened time: {X.shape = }, {Y.shape = }')

    return X, Y, year_permutation

def shuffle_years(X, permutation=None, seed=0, apply=False):
    '''
    Permutes `X` along the first axis

    Parameters
    ----------
    X : np.ndarray
        array with the data to permute
    permutation : None or 1D np.ndarray, optional,
        must be a permutation of an array of `np.arange(X.shape[0])`
    seed : int, optional
        if `permutation` is None, then it is computed using the provided seed.
    apply : bool, optional
        if True the function returns the permuted data, otherwise the permutation is returned
    
    Returns
    -------
    if apply:
        np.ndarray of the same shape of X
    else:
        np.ndarray of shape (X.shape[0],)
    '''
    if permutation is None:
        if seed is not None:
            np.random.seed(seed)
        permutation = np.random.permutation(X.shape[0])
    if len(permutation) != X.shape[0]:
        raise ValueError(f'Shape mismatch between X ({X.shape[0] = }) and permutation ({len(permutation) = })')
    if apply:
        return X[permutation,...]
    return permutation


def make_X(fields, time_start=30, time_end=120, T=14, tau=0):
    '''
    Cuts the fields in time and stacks them. The original fields are not modified

    Parameters
    ----------
    fields : dict of Plasim_Field objects
    time_start : int, optional
        first day of the period of interest
    time_end : int, optional
        first day after the end of the period of interst
    T : int, optional
        width of the window for the running average
    tau : int, optional
        delay between observation and prediction (meaningful when negative)

    Returns
    -------
    X : np.ndarray
        with shape (years, days, lat, lon, field)
    '''
    # stack the fields
    X = np.array([field.var[:, time_start+tau:time_end+tau-T+1, ...] for field in fields.values()]) # NOTE: maybe chenge to -tau
    # now transpose the array so the field index becomes the last
    X = X.transpose(*range(1,len(X.shape)), 0)
    return X

def balance_folds(weights, nfolds=10, verbose=False):
    '''
    Returns a permutation that, once applied to `weights` would make the consecutive `nfolds` pieces of equal length have their sum the most similar to each other.

    Parameters
    ----------
    weights : 1D array-like
    nfolds : int, optional
        must be a divisor of `len(weights)`
    verbose : bool, optional

    Returns
    -------
    permutation : permutation of `np.arange(len(weights))`
    '''
    class Fold():
        def __init__(self, target_length, target_sum, name=None):
            self.indexs = []
            self.length = target_length
            self.target_sum = target_sum
            self.sum = 0
            self.hunger = np.infty # how much a fold 'wants' the next datapoint
            self.name = name
        
        def add(self, a):
            self.indexs.append(a[1])
            self.sum += a[0]
            if self.length == len(self.indexs):
                if verbose:
                    print(f'fold {self.name} done!')
                return True
            self.hunger = (self.target_sum - self.sum)/(self.length - len(self.indexs))
            return False

    fold_length = len(weights)//nfolds
    if len(weights) != fold_length*nfolds:
        raise ValueError(f'Cannot make {nfolds} folds of equal lenght out of {len(weights)} years of data')
    target_sum = np.sum(weights)/nfolds

    # create nfolds Folds that will redistribute the data
    folds = [Fold(fold_length, target_sum, name=i) for i in range(nfolds)]
    permutation = []

    # sort the weights keeping track of the original order
    ws = [(a,i) for i,a in enumerate(weights)]
    ws.sort()
    ws = ws[::-1]
    
    sums = []
    if verbose:
        print('Balancing folds')
    # run over the weights and distribute data to the folds
    for a in ws:
        # determine the hungriest fold, i.e. the one that has its sum the furthest from its target
        j = np.argmax([f.hunger for f in folds])
        if folds[j].add(a): # add datapoint and check if the fold is full
            f = folds.pop(j)
            sums.append(f.sum)
            permutation += f.indexs

    if len(permutation) != len(weights):
        raise ValueError('balance_folds: Something went wrong during balancing: either missing or duplicated data')

    if verbose:
        sums = np.array(sums)
        print(f'Sums of the balanced {nfolds} folds:\n{sums}\nstd/avg = {np.std(sums)/target_sum}\nmax relative deviation = {np.max(np.abs(sums - target_sum))/target_sum*100}\%')

    return permutation



def PrepareParameters(creation):
    print("==Preparing Parameters==")
    WEIGHTS_FOLDER = './models/'
    
    RESCALE_TYPE =  'rescale' # 'nomralize' # 
    Z_DIM = 64 #8 #16 #256 # Dimension of the latent vector (z)
    BATCH_SIZE = 128#512
    LEARNING_RATE = 1e-3#5e-4# 1e-3#5e-6
    N_EPOCHS = 20#600#200
    SET_YEARS = range(8000) # the set of years that variational autoencoder sees
    SET_YEARS_LABEL = 'range8000'
    K1 = 0.9 # 1#100
    K2 = 0.1 #1
    
    data_path='../../gmiloshe/PLASIM/'
    
    Model = 'Plasim'
    
    lon_start = 0
    lon_end = 128
    lat_start = 0 # latitudes start from 90 degrees North Pole
    lat_end = 24
    Months1 = [0, 0, 0, 0, 0, 0, 30, 30, 30, 30, 30, 0, 0, 0] 
    Tot_Mon1 = list(itertools.accumulate(Months1))
    checkpoint_name = WEIGHTS_FOLDER+Model+'_t2mzg500mrso_yrs-'+SET_YEARS_LABEL+'_'+RESCALE_TYPE+'_k1_'+str(K1)+'_k2_'+str(K2)+'_LR_'+str(LEARNING_RATE)+'_ZDIM_'+str(Z_DIM)
    return WEIGHTS_FOLDER, RESCALE_TYPE, Z_DIM, BATCH_SIZE, LEARNING_RATE, N_EPOCHS, SET_YEARS, K1, K2, checkpoint_name, data_path, Model, lon_start, lon_end, lat_start, lat_end, Tot_Mon1
    
def CreateFolder(creation,checkpoint_name):
    myinput = "Y" # default input
    print("==Creating folders if they didn't exist==")
    print("parameter creation = ", creation)
    if creation == None:
        if os.path.exists(checkpoint_name): # Create the directory
            print('folder '+checkpoint_name+' exists. Should I overwrite?')
            
            myinput = input(" write Y to overwrite, N to stop execution, C to continue the run: ")
            if myinput == "N":
                sys.exit("User has aborted the program")
            if myinput == "Y":
                os.system("rm "+checkpoint_name+"/*.ckpt.*")
        else:
            print('folder '+checkpoint_name+' created')
            os.mkdir(checkpoint_name)

        sys.stdout = ef.Logger(checkpoint_name+'/logger.log')  # Keep a copy of print outputs there
        shutil.copy(__file__, checkpoint_name+'/Funs.py') # Copy this file to the directory of the training
        shutil.copy('history.py', checkpoint_name)  # Also copy a version of the files we work with to analyze the results of the training
        shutil.copy('reconstruction.py', checkpoint_name)
    return myinput

def RescaleNormalize(X,RESCALE_TYPE, creation,checkpoint_name):
    if RESCALE_TYPE == 'normalize':
        print("===Normalizing X===")
        if creation == None: # mean and std have to be computed
            X_mean = np.mean(X,0)
            X_std = np.std(X,0)
            X_std[X_std==0] = 1 # If there is no variance we shouldn't divide by zero
            np.save(checkpoint_name+'/X_mean', X_mean)
            np.save(checkpoint_name+'/X_std', X_std)
        else:
            X_mean = np.load(checkpoint_name+'/X_mean.npy')
            X_std = np.load(checkpoint_name+'/X_std.npy')

        return 1./(1.+np.exp(-(X-X_mean)/X_std)) # we have apply sigmoid because variational autoencoder reconstructs with this activation
    else:
        print("===Rescaling X===")
        if creation == None:
            maxX = np.max(X,tuple(list(range(len(X.shape)-1)))) # Equivalent to np.max(X,(0,1,...,last-1))
            minX = np.min(X,tuple(list(range(len(X.shape)-1))))
            np.save(checkpoint_name+'/maxX', maxX)
            np.save(checkpoint_name+'/minX', minX)
        else:
            maxX = np.load(checkpoint_name+'/maxX.npy')
            minX = np.load(checkpoint_name+'/minX.npy')
    return (X - minX) / (maxX - minX)

def ConstructVAE(INPUT_DIM, Z_DIM, checkpoint_name, N_EPOCHS, myinput, K1, K2):
    print("==Building encoder==")
    encoder_inputs, encoder_outputs, shape_before_flattening, encoder  = tff.build_encoder(input_dim = INPUT_DIM, 
                                                output_dim = Z_DIM, 
                                                conv_filters = [32, 64, 64, 64],
                                                conv_kernel_size = [3,3,3,3],
                                                conv_strides = [2,2,2,1],
                                                conv_padding = ["same","same","same","valid"], use_batch_norm=True, use_dropout=True)
    encoder.summary()
    print("==Building decoder==")      
    # Decoder
    decoder_input, decoder_output, decoder = tff.build_decoder(input_dim = Z_DIM,  
                                        shape_before_flattening = shape_before_flattening,
                                        conv_filters = [64,64,32,3],
                                        conv_kernel_size = [3,3,3,3],
                                        conv_strides = [1,2,2,2],
                                        conv_padding = ["valid","same","same","same"])
    decoder.summary()


    print("==Attaching decoder and encoder and compiling==")

    vae = tff.VAE(encoder, decoder, k1=K1, k2=K2)
    print("vae.k1 = ", vae.k1, " , vae.k2 = ", vae.k2)
    if myinput == 'Y':
        INITIAL_EPOCH = 0
        history = []
        checkpoint = []
    else: # the run has to be continued
        history = np.load(checkpoint_name+'/history', allow_pickle=True)
        INITIAL_EPOCH = len(history['loss'])
        print("==loading the model: ", checkpoint_name)
        N_EPOCHS = N_EPOCHS + INITIAL_EPOCH 
        #vae = tf.keras.models.load_model(checkpoint_name, compile=False)
        checkpoint = tf.train.latest_checkpoint(checkpoint_name)
        print("checkpoint = ", checkpoint)
        vae.load_weights(checkpoint)

    print("INITIAL_EPOCH = ", INITIAL_EPOCH)

    INPUT_DIM_withnone = list(INPUT_DIM)
    INPUT_DIM_withnone.insert(0,None)
    
    vae.build(tuple(INPUT_DIM_withnone)) 
    vae.compute_output_shape(tuple(INPUT_DIM_withnone))
    vae.summary()

    checkpoint_path = checkpoint_name+"/cp-{epoch:04d}.ckpt"
    
    return vae, history, N_EPOCHS, INITIAL_EPOCH, checkpoint, checkpoint_path

def PrepareDataAndVAE(creation=None, DIFFERENT_YEARS=None):
    WEIGHTS_FOLDER, RESCALE_TYPE, Z_DIM, BATCH_SIZE, LEARNING_RATE, N_EPOCHS, SET_YEARS, K1, K2, checkpoint_name, data_path, Model, lon_start, lon_end, lat_start, lat_end, Tot_Mon1 = PrepareParameters(creation)

    if isinstance(DIFFERENT_YEARS, np.ndarray): # Need to check because otherwise comparing array to None would give an error
        SET_YEARS = DIFFERENT_YEARS # for benchmark runs we don't need all years or the same years, with different years we can load some other data.
    else: # might be a list
        if DIFFERENT_YEARS!=None: #check that the parameter is not None before overwriting the years
            SET_YEARS = DIFFERENT_YEARS # for benchmark runs we don't need all years or the same years, with different years we can load some other data.
    
    myinput = CreateFolder(creation,checkpoint_name)
    
    _fields = load_data(dataset_years=8000, year_list=SET_YEARS) # Fix support for different years
  
    X, _Y, _year_permutation = prepare_XY(_fields)
    #X = X.reshape(-1,*X.shape[2:])
    
    INPUT_DIM = X.shape[1:]  # Image dimension
    
    X = RescaleNormalize(X,RESCALE_TYPE, creation, checkpoint_name)
    print("X.shape = ", X.shape,  " ,np.mean(X[:,5,5,0]) = ", np.mean(X[:,5,5,0]), " ,np.std(X[:,5,5,0]) = ", np.std(X[:,5,5,0]), " , np.min(X) = ", np.min(X), " , np.max(X) = ", np.max(X))

    vae, history, N_EPOCHS, INITIAL_EPOCH, checkpoint, checkpoint_path = ConstructVAE(INPUT_DIM, Z_DIM, checkpoint_name, N_EPOCHS, myinput, K1, K2)
    
    return X, vae, Z_DIM, N_EPOCHS, INITIAL_EPOCH, BATCH_SIZE, LEARNING_RATE, checkpoint_path, checkpoint_name, myinput, history

if __name__ == '__main__': # we do this so that we can then load this file as a module in reconstruction.py
    print("==Checking GPU==")
    import tensorflow as tf
    tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )

    print("==Checking CUDA==")
    tf.test.is_built_with_cuda()

    start = time.time()
    X, vae, Z_DIM, N_EPOCHS, INITIAL_EPOCH, BATCH_SIZE, LEARNING_RATE, checkpoint_path, checkpoint_name, myinput, history = PrepareDataAndVAE()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

    vae.compile(optimizer=tf.keras.optimizers.Adam(lr = LEARNING_RATE))
    #vae.summary()

    print("==fit ==")
    my_history = vae.fit(X, epochs=N_EPOCHS, initial_epoch=INITIAL_EPOCH, batch_size=BATCH_SIZE, shuffle=True, callbacks=[cp_callback])
    if myinput == 'C':
        print("we merge the history dictionaries")
        print("len(history['loss'] = ", len(history['loss']))
        print("len(my_history.history['loss']) = ", len(my_history.history['loss']))
        for key in history:
            history[key] = history[key]+my_history.history[key]
    else:
        history = my_history.history
    print("len(history['loss']) = ", len(history['loss']))

    end = time.time()
    print("Learning time = ",end - start)

    vae.save(checkpoint_name)
    with open(checkpoint_name+'/history', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    #np.save(checkpoint_name+'/history.npy',my_history.history)


    print("==saving the model ==")
