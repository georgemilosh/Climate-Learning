# Created in 2022

# @author: George Miloshevich 
"""
This file contains the functions that are used to train the VAE (Variational Autoencoder)
it is also used to create the file/folder structure prepared for training Stochastic Weather Generator (SWG).

Usage
-----
To create the folder structure and the relevant config.json modify the parameters in the function called `kwargator` and run the file:

    python vae_learn2.py <folder>

This will create a folder with the name <folder> copy the vae_learn2.py (along with its dependencies) and rename it into Funs.py
It will also create config.json with the parameters specified in the function `kwargator`.

How training proceeds will depend on these parameters:

Parameters
----------

    We recommend the following parameters (by default Plasim dataset will be used):
        'return_time_series' : True,    #  time series integrated over the area
        'return_threshold' : True,      #  returns the threshold for heatwaves
        'myinput':'Y',                  # This should always be `Y` when training the model
        'validation_data' : True,       # whether to also measure the reconstruction loss on the validation data
        'k1': 0.9 , 'k2':0.1,           # the weights of the reconstruction and KL loss respectively that we usually set for VAE
        
        
       

        'checkpoint_every': 1,          # how often to save the model. Typically if running VAE for the first time we also set:
        'N_EPOCHS': 100,                # how many epochs to train. Next if the training did not converge (measured by the validation loss)
                                        # training can be continued, by `cd`-ing into the folder, increasing `config.json`-s parameter 'N_EPOCHS' 
                                        # and running:
                            ```
                                python Funs.py .
                            ```

        'field_weights': [20., 1., 20.] # the weights of the fields in the reconstruction loss that we typically set, i.e. we want to 
                                                    # weight t2m and mrso more than zg500. This parameter exists for historical reasons
                                                    # when we were training autoencoder on all three fields at once. This approach
                                                    # was later abandoned due to inefficiency, however the parameter remains the same
        `usemask` : False,             # whether to use the mask of the area impacted by the heatwave to only reconstruct that part of the fields,
                                        # e.g. fields such as `t2m` and `mrso` will be masked with the mask of the area impacted by the heatwave
                                        # this parameter should only be set to `true` if all three fields are being reconstructed (this only matters
                                        # if we will choose 'use_autoencoder' = True, otherwise autoencoder is omitted). As we have found
                                        # that reconstructing geopotential only is more efficient for SWG we never use `usemask` =  True anymore

        'keep_dims' : [1],              # which dimensions to keep when reconstructing the fields. In this case `1` implies we are selecting 
                                        # geopotential. If, however, you intend to use <folder> for training vanilla SWG (without dimensionality reduction)
                                        # you should use default fvalue of `keep_dims`.
                            
    # If you only need to create <folder> to train vanilla SWG (without dimensionality reduction) you should set:
        'normalization_mode' : 'global_logit',   # vanilla SWG does not use the same normalization as VAE
        'use_autoencoder' : False, # whether to use VAE or not. SWG training will work regardeless but you will not use the resources for the projection

    # If you intend to train VAE  here is the architecture we found suitable for the problem of projecting geopotential:
        'Z_DIM': 16, 
            'encoder_conv_filters':             [16, 16, 16, 32, 32,  32,   64, 64],
            'encoder_conv_kernel_size':[5,  5,  5,  5,   5,   5,   5,  3], #  [5,  5,  5,  5,   5,   5,   5,  3, 64]
            'encoder_conv_strides'    :[2,  1,  1,  2,   1,   1,   2,  1],
            'encoder_conv_padding':["same","same","same","same","same","same","same","valid"],
            'encoder_conv_activation':["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu"], 
            'encoder_conv_skip': [[0,2],[3,5]], 
            'encoder_use_batch_norm' : [True,True,True,True,True,True,True,True], 
            'encoder_use_dropout' : [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],
    'decoder_conv_filters':[64,32,32,32,16,16,16,1], #3], # Use 3 if working with 3 fields
            'decoder_conv_kernel_size':[3, 5, 5, 5, 5, 5, 5, 5],
                'decoder_conv_strides':[1, 2, 1, 1, 2, 1, 1, 2],
                'decoder_conv_padding':["valid","same","same","same","same","same","same","same"],
                'decoder_conv_activation':["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","sigmoid"], 
                'decoder_use_batch_norm' : [True,True,True,True,True,True,True,True],
                'decoder_use_dropout' : [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]



    # If you want to work with 3 day running means you should set:
        'label_field' : 't2m_inter',       # as opposed to 't2m' which is the default. `t2m_inter` is 
                                                                    the 3 day running mean of `t2m`
        'A_weights' : [3,0,0, 3,0,0, 3,0,0, 3,0,0, 3,0,0], 
        'fields': ['t2m_inter_filtered','zg500_inter','mrso_inter_filtered']    # `inter` implies that we take 3 day running mean fields
                        # and `filter` implies that we masked the corresponding fields with the mask of the area impacted by the heatwave

    # Concerning the time of interest, the tas.nc, mrso.nc and zg500.nc have been extracted from May to the end of September.
        yet we are only interested in predicting heatwaves in June, July and August. Meanwhile, we would like to 
        be able to predict based on the series of the previous 15 days. This is why we set:
         'time_start' : 15,              # 15 days after May 1 is when our dataset X starts (for the purposes of
                                                                        training to project climatic states)
         'label_period_start' : 30,     # 30 days after May 1 is when we start predicting the labels Y (June 1)
         'time_end' :           134,    # 134 days after May 1 (September 15) is the last date that we use to train VAE and/or SWG
         'label_period_end' : 120,      # 120 days after May 1 is when we stop predicting  labels Y(August 30)

    # The domain of predictors is typically North Atlantic Europe
        'lat_start' :   0, 
        'lat_end' :     24, 
        'lon_start' :   98, 
        'lon_end' :     18, 


    
"""
# 


### IMPORT LIBRARIES #####

# log to stdout
import logging
import sys
import os
import time
import itertools
import numpy as np
import shutil
import psutil
import gc
import pickle
import traceback
from pathlib import Path
from colorama import Fore # support colored output in terminal
from colorama import Style
from functools import partial # this one we use for the scheduler

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO

## user defined modules
# I go back to absolute paths becuase otherwise I have trouble using reconstruction.py on this file later when it will be called from the folder it creates
sys.path.insert(1, '/homedata/gmiloshe/Climate-Learning/')       
        
import PLASIM.Learn2_new as ln
ut = ln.ut # utilities
ef = ln.ef # ERA_Fields_New
tff = ln.tff # TF_Fields

logger.info("==Checking GPU==")
import tensorflow as tf
from tensorflow.keras.callbacks import TerminateOnNaN
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

logger.info("==Checking CUDA==")
tf.test.is_built_with_cuda()

# set spacing of the indentation
ut.indentation_sep = '    '



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set
# separators to create the run name from the run arguments


######################################
########## COPY SOURCE FILES #########
######################################

def move_to_folder(folder):
    '''
    Copies this file and its dependencies to a given folder. 
    Parameters
    ----------
    folder: string
        name of the folder that accepts the copies
        
    '''
    logger.info(f"==move_to_folder==")

    sys.stdout = ef.Logger(folder)  # Keep a copy of print outputs there
    shutil.copy(__file__, folder+'/Funs.py') # Copy this file to the directory of the training
    shutil.copy('history.py', folder)  # Also copy a version of the files we work with to analyze the results of the training
    shutil.copy('reconstruction.py', folder)
    shutil.copy('classification.py', folder)
    shutil.copy('config.json', folder)
    shutil.copy('../ERA/ERA_Fields_New.py', folder)
    shutil.copy('../ERA/TF_Fields.py', folder)
    shutil.copy('../PLASIM/Learn2_new.py', folder)


############################################
########## DATA PREPROCESSING ##############
############################################

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
def normalize_X(X,folder, myinput='N',mode='pointwise'):
    '''
    Performs data normalization

    Parameters
    ----------
    X : np.ndarray of shape (N, ...)
        data
    mode : 'pointwise', optional
        how to perform the normalization. If any other mode used 
        it will default to rescaling
    Returns
    -------
    X_n : np.ndarray of same shape as X
        normalized/rescaled data passed through a sigmoid
    
    Saves
    -----
    if mode == 'pointwise':
        X_mean : np.ndarray of shape (...)
            mean of X along the first axis
        X_std : np.ndarray of shape (...)
            std of X along the first axis
    else:
        maxX : field-wise maximum of X 
        minX : field-wise minimum of X 
    '''
    if mode == 'pointwise':
        logger.info("===Normalizing X===")
        if myinput != 'N': # mean and std have to be computed
            X_mean = np.mean(X,0)
            X_std = np.std(X,0)
            X_std[X_std==0] = 1 # If there is no variance we shouldn't divide by zero
            np.save(f'{folder}/X_mean', X_mean)
            np.save(f'{folder}/X_std', X_std)
        else: # mean and std are expected to be already computed
            logger.info(f'loading {folder}/X_mean.npy')
            X_mean = np.load(f'{folder}/X_mean.npy')
            X_std = np.load(f'{folder}/X_std.npy')

        return   1./(1.+np.exp(-(X-X_mean)/X_std)) # (X-X_mean)/X_std # # we have applied sigmoid because variational autoencoder reconstructs with this activation
    elif mode == 'global_logit': # normalizing by global variance but the output is not passed through the sigmoid
        logger.info("===Normalizing X===")
        if myinput != 'N': # mean and std have to be computed
            X_mean = np.mean(X,tuple(list(range(len(X.shape)-1)))) # Equivalent to np.max(X,(0,1,...,last-1))
            X_std = np.std(X,tuple(list(range(len(X.shape)-1)))) # Equivalent to np.max(X,(0,1,...,last-1))
            X_std[X_std==0] = 1 # If there is no variance we shouldn't divide by zero
            np.save(f'{folder}/X_mean', X_mean)
            np.save(f'{folder}/X_std', X_std)
        else: # mean and std are expected to be already computed
            logger.info(f'loading {folder}/X_mean.npy')
            X_mean = np.load(f'{folder}/X_mean.npy')
            X_std = np.load(f'{folder}/X_std.npy') 
        return (X-X_mean)/X_std 
    elif mode == 'rescale':
        logger.info("===Rescaling X===")
        if myinput != 'N':
            maxX = np.max(X,tuple(list(range(len(X.shape)-1)))) # Equivalent to np.max(X,(0,1,...,last-1))
            minX = np.min(X,tuple(list(range(len(X.shape)-1))))
            np.save(f'{folder}/maxX', maxX)
            np.save(f'{folder}/minX', minX)
        else:
            maxX = np.load(f'{folder}/maxX.npy')
            minX = np.load(f'{folder}/minX.npy')
        return (X - minX) / (maxX - minX) # 2*(X - minX)/(maxX - minX)-1  #
    return X

################################################
########## NEURAL NETWORK CONSTRUCTION ###########
################################################

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
def create_or_load_vae(folder, INPUT_DIM, myinput, filter_mask, VAE_kwargs=None, build_encoder_skip_kwargs=None, build_decoder_skip_kwargs=None, create_classifier_kwargs=None, checkpoint_every=1):
    '''
    Creates a Variational AutoEncoder Model or loads the existing one 
        from the weights given in the model
        
    Parameters
    ----------
    folder
        the name of the file where we store the model(s)
    INPUT_DIM : tuple
        shape of a single input datapoint, i.e. not counting 
        the axis corresponding to iteration through the datapoints (batch axis)
    Z_DIM: int
        dimension of the latent space
    VAE_kwargs: dictionary (optional)
        contains some extra parameters such as:
            mask_area: str
                which geographical area we put a mask on
            Z_DIM: int
                dimension of the latent space
            N_EPOCHS: int
                number of epochs to train
        the rest are passed to tff.VAE constructor (see TF_Fields.py)
    build_encoder_skip_kwargs: dictionary (optional)
        see tff.build_encoder_skip kwargs (TF_Fields.py)
    build_decoder_skip_kwargs: dictionary (optional)
        see tff.build_decoder_skip kwargs (TF_Fields.py)
    create_classifier_kwargs: dictionary (optional)
    

    Returns
    -------
    model : keras.models.Model
    N_EPOCHS: 
        it used to change if the myinput='C' thus there used to be need to return
    INITIAL_EPOCH: int
        it may change as well because of myinput='C'
    ckpt_path_callback: keras.callbacks.ModelCheckpoint
        contains the style of the checkpoint file
    '''
    
    ### preliminary operations
    ##########################
    if VAE_kwargs is None:
        VAE_kwargs = {}
    if build_encoder_skip_kwargs is None:
        build_encoder_skip_kwargs = {}
    if build_decoder_skip_kwargs is None:
        build_decoder_skip_kwargs = {}
    if create_classifier_kwargs is None:
        create_classifier_kwargs = {}
    VAE_kwargs_local = VAE_kwargs.copy()
    mask_area = VAE_kwargs_local.pop('mask_area') # because vae_kwargs_local must be passed to tff.VAE constructor and we still need vae_kwargs for other folds
    Z_DIM = VAE_kwargs_local.pop('Z_DIM')
    N_EPOCHS = VAE_kwargs_local.pop('N_EPOCHS')
    k1 = VAE_kwargs_local.pop('k1')
    print_summary = VAE_kwargs_local.pop('print_summary')
    
    print(f"{INPUT_DIM[:-1] = }")
    
    ones_dim = np.ones(INPUT_DIM[:-1])
    
    logger.info(f'{filter_mask.shape = }')
    logger.info(f'{ones_dim.shape = }')
    logger.info(f'{np.array([filter_mask,ones_dim,filter_mask], dtype=bool).shape = }')
    
    filter_mask = np.array([filter_mask,ones_dim,filter_mask], dtype=bool).transpose(1,2,0) 
    
    logger.info(f'{filter_mask.dtype}')
    
    
    logger.info("==Building encoder==")
    if k1 == 'pca':
        vae = tff.PCAer(**VAE_kwargs) 
    else: # Not PCA so we do the standard autoencoder
        _, _, shape_before_flattening, encoder  = tff.build_encoder_skip(input_dim = INPUT_DIM, output_dim = Z_DIM, **build_encoder_skip_kwargs)
        classifier = tff.create_classifier(Z_DIM, **create_classifier_kwargs)
        if print_summary:
            encoder.summary()
        logger.info("==Building decoder==") 
        logger.info(f"{filter_mask.shape = }")
        # logger.info(f"{filter_mask = }")
        _, _, decoder = tff.build_decoder_skip(mask=filter_mask, input_dim = Z_DIM, shape_before_flattening = shape_before_flattening, **build_decoder_skip_kwargs)
        if print_summary:
            decoder.summary()
        logger.info("==Attaching decoder and encoder and compiling==")
        vae = tff.VAE(encoder, decoder, classifier, **VAE_kwargs) #, mask_weights=mask_weights)
        
    if print_summary:
        logger.info(f'{vae.k1 = },{vae.k2 = } ')
    
    #todo: consider breaking into a second function from here

    if myinput == 'Y': # The run is to be performed from scratch
        INITIAL_EPOCH = 0
        history_vae = []
    else: # the run has to be continued
        history_vae = np.load(f'{folder}/history_vae', allow_pickle=True)
        if hasattr(vae,'load_weights'):
            INITIAL_EPOCH = len(history_vae['loss'])
            logger.info(f'==loading the model: {folder}')
            #N_EPOCHS = N_EPOCHS + INITIAL_EPOCH 
            checkpoint_path = tf.train.latest_checkpoint(folder)
            logger.info(f'loading weights {checkpoint_path = }')
            vae.load_weights(checkpoint_path).expect_partial()
        else:
            INITIAL_EPOCH = 0 # there are no epochs if autoencoder is not used, e.g. if computing PCA components
            vae.encoder = pickle.load(open(f'{folder}/encoder.pkl', 'rb')) # see description of the class PCAer for PCAer.save method
        
    logger.info(f'{INITIAL_EPOCH = }')

    INPUT_DIM_withnone = list(INPUT_DIM)
    INPUT_DIM_withnone.insert(0,None)
    
    if hasattr(vae,'build'):
        vae.build(tuple(INPUT_DIM_withnone)) 
        vae.compute_output_shape(tuple(INPUT_DIM_withnone))
    if print_summary:
        vae.summary()
    ckpt_path_callback=ln.make_checkpoint_callback(str(folder)+"/cp_vae-{epoch:04d}.ckpt", checkpoint_every=checkpoint_every)
    #ckpt_path_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(folder)+"/cp_vae-{epoch:04d}.ckpt",save_weights_only=True,verbose=1)# TODO: convert checkpoints to f-strings
    logger.info(f'{ckpt_path_callback = }') 
    return vae, history_vae, N_EPOCHS, INITIAL_EPOCH, ckpt_path_callback

@ut.execution_time
@ut.indent_logger(logger)
def classify(fold_folder, evaluate_epoch, vae, X_tr, z_tr, Y_tr, X_va, z_va, Y_va, u=1):
    '''
    At the moment is void
    '''
    return None

@ut.execution_time
@ut.indent_logger(logger)   
def k_fold_cross_val(folder, myinput, mask, X, Y, time_series, year_permutation, create_or_load_vae_kwargs=None, train_vae_kwargs=None, keep_dims=None, nfolds=10, val_folds=1, 
                     range_nfolds=None, u=1, normalization_mode='pointwise', classification=True, evaluate_epoch='last', repeat_nan=5, use_autoencoder=True):
    '''
    Performs k fold cross validation on a model architecture.

    Parameters
    ----------
    folder : str
        folder in which to save data related to the folds
    X : np.ndarray
        all data (train + val)
    Y : np.ndarray
        all labels
    time_series : list
        a list of area integrals masked over X
    mask : np.ndarray
       
    year_permutation : np.ndarray
        provided here to be stored in individual fold during training
    create_or_load_vae_kwargs : dict
        dictionary with the parameters to create a vae model, see create_or_load_vae()
    keep_dims: list, optional
        if provided the autoencoder will receive only the fields corresponding to the elements of the list
    nfolds : int, optional
        number of folds
    val_folds : int, optional
        number of folds to be used for the validation set for every split
    u : float, optional
        undersampling factor (>=1). If = 1 no undersampling is performed (NOT OPERATIONAL AT THE MOMENT)
    normalization_mode: str
        type of normalization we will use, if 'pointwise' grid-point normalization will be used (see normalize_X())
    classification: bool
        whether to perform classification or not
    evaluate_epoch: str or int
        epoch at which we wish to perform dimensionality reduction for classification. if 'last' then last checkpoint of vae will be used
    repeat_nan: int
        tells do while to repeat the fold that many times if the successive outputs of history['loss'] are nan. Basically if the run has failed we repeat that many times
    use_autoencoder: bool (Defaults to True)
        If True dimensionality reduction will be performed
        
    Returns
    ----------
    
    N_EPOCHS: int
        number of epochs the training consists of
    INITIAL_EPOCH: int
        where the training restarts
    checkpoint_path: str
        path to the last checkpoint (check!)
    vae:  keras.models.Model
        variational autoencoder (pre-trained at the last fold which 
        is useful for reconstruction when we call a specific fold)
    X_tr : np.ndarray
        training data from the last fold
    Y_tr : np.ndarray
        training labels from the last fold
    X_va : np.ndarray
        validation data from the last fold
    Y_va : np.ndarray
        validation labels from the last fold
    score: None or specified by classify()
        intended for reporting classification score
    '''
    
    if create_or_load_vae_kwargs is None:
        create_or_load_vae_kwargs = {}
    if train_vae_kwargs is None:
        train_vae_kwargs = {}
    #folder = folder.rstrip('/')
    reconstruction=False # by default we are not doing reconstruction, we are either doing training and/or classification
    # k fold cross validation
    scores = []
    if range_nfolds is None:
        range_nfolds=range(nfolds) #range(2)
    if myinput == 'N': # In training regime, otherwise default value for reconstruction
        if os.path.exists(f'{folder}/fold_num.npy'): # # we are inside of one of the folds
            logger.log(35,f'{folder}/fold_num.npy exists')
            range_nfolds=[int(np.load(f'{folder}/fold_num.npy'))]  #the other option would be parsing the fold_N string to get i in future
            reconstruction=True # This option is useful for reconstruction, so that we don't need to load all the data
        else: # we are outside of the folds
            logger.info(f'{folder}/fold_num.npy does not exist')
    logger.info(f'{range_nfolds = }')    
    my_memory = [] # monitor RAM storage
    score = [] # for classification if need be  
    for i in range_nfolds:
        repeat_nan_local = repeat_nan # create an integer copy
        while repeat_nan_local > -1: # try to train the model
            logger.info('=============')
            logger.log(35, f'fold {i} ({i+1}/{nfolds})')
            logger.info('=============')
            logger.info(f'{repeat_nan_local = }')
            fold_folder=folder # this allows us to work in a fold by default
            if os.path.exists(f'{folder}/fold_num.npy'): # # we are inside of one of the folds (we are running the script in the reconstruction mode)
                logger.info(f'{folder}/fold_num.npy exists')
            else: # we are not inside of one of the folds, either this is a new run, or we need to iterate through the runs
                logger.info(f'{folder}/fold_num.npy does not exist')
                fold_folder = f'{folder}/fold_{i}'
                if myinput == 'Y': # If 'C' we don't need to change anything
                    if not os.path.exists(fold_folder):
                        os.mkdir(fold_folder)
                        np.save(f'{fold_folder}/fold_num',i) # the other option would be parsing the fold_N string to get i in future

            # split data
            logger.info(f"{i = }, {X.shape = }, {Y.shape = }, {nfolds=}, {val_folds=}")
            logger.info(f"We are working in the mode {reconstruction = }")
            if reconstruction: # in this case we have already specified the precise set of years
                X_tr = []
                Y_tr = []
                X_va = X
                Y_va = Y
                classification=False # we will not do classification in this case (not compatible with current set-up of reconstruction)
            else:
                _, time_series_tr, _, time_series_va = ln.k_fold_cross_val_split(i, time_series, time_series, nfolds=nfolds, val_folds=val_folds)
                if (not os.path.exists(f"{fold_folder}/time_series_tr.npy"))and(myinput != 'N'):
                    np.save(f'{fold_folder}/time_series_tr',time_series_tr)
                    np.save(f'{fold_folder}/time_series_va',time_series_va)
                    logger.info(f'saved {fold_folder}/time_series_tr.npy')
                    logger.info(f'saved {fold_folder}/time_series_va.npy')
                X_tr, Y_tr, X_va, Y_va = ln.k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)
                if myinput == 'Y': # we also store the year permuation in individual folds so that we can easily access the appropriate years
                    _,_,_,year_permutation_va = ln.k_fold_cross_val_split(i, np.array(year_permutation), np.array(year_permutation)
                                                                          , nfolds=nfolds, val_folds=val_folds) # we want years that correspond to validation
                    np.save(f'{fold_folder}/year_permutation_va',year_permutation_va)
                    logger.info(f'saved {fold_folder}/year_permutation_va.npy')
                    np.save(f'{fold_folder}/Y_va',Y_va)
                    np.save(f'{fold_folder}/Y_tr',Y_tr)
                    logger.info(f'saved {fold_folder}/Y_va.npy')
                    logger.info(f'saved {fold_folder}/Y_tr.npy')
            INPUT_DIM = X_va.shape[1:]  # Image dimension
            logger.info(f"{INPUT_DIM = }")
            # perform undersampling
            #X_tr, Y_tr = undersample(X_tr, Y_tr, u=u)
            if not reconstruction: # either training or classification, in both cases we need training set
                n_pos_tr = np.sum(Y_tr)
                n_neg_tr = len(Y_tr) - n_pos_tr
                logger.info(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')


                X_tr = normalize_X(X_tr, fold_folder, myinput=myinput, mode=normalization_mode)
                logger.info(f'{X_tr.shape = },{np.mean(X_tr[:,5,5,0]) = },{np.std(X_tr[:,5,5,0]) = }')
            X_va = normalize_X(X_va, fold_folder, myinput='N', mode=normalization_mode) #validation is always normalized passively (based on already computed normalization)
            if not reconstruction: # either training or classification, in both cases we need training set
                logger.info(f"{X_va[5,5,5,1] = }, {Y_va[5] = }, {X_tr[5,5,5,1] = }, {Y_tr[5] = }")
            logger.info(f"{create_or_load_vae_kwargs = }")
            
            if keep_dims is not None: # In this case we only apply autoencoder to a single field, the other fields will be treated as area integrals
                logger.info(f"{type(keep_dims) = }, {keep_dims = }")
                if not reconstruction: # either training or classification, in both cases we need training set
                    logger.info(f"{type(X_tr) = }, {X_tr.shape = }")
                    X_tr = X_tr[...,keep_dims]
                X_va = X_va[...,keep_dims]
                logger.info(f"{X_va.shape = }")
                INPUT_DIM = X_va.shape[1:]  # Image dimension
                logger.info(f"recomputing: {INPUT_DIM = }")
            
            if use_autoencoder: # Note that if this is not exercized there will be no dimensionality reduction and scripts such as reconstruction.py will not work
                vae, history_vae, N_EPOCHS, INITIAL_EPOCH, ckpt_path_callback = create_or_load_vae(fold_folder, INPUT_DIM, myinput, mask, **create_or_load_vae_kwargs)
                if hasattr(vae, 'trainable_weights'):
                    logger.info(f"{len(vae.trainable_weights) = }, {len(vae.encoder.trainable_weights) = }, {len(vae.decoder.trainable_weights) = }")
                
                if hasattr(vae.encoder, 'layers'):
                    for inner_layer in vae.encoder.layers:
                        if inner_layer.name == 'encoder_conv_0':
                            logger.info(f"vae.encoder layers: {inner_layer.name = }")
                            logger.info(f"{inner_layer.weights[0][0,0,0,:] = }")
                if hasattr(vae, 'classifier'):
                    for inner_layer in vae.classifier.layers:
                        logger.info(f"vae.classifier layers: {inner_layer.name = }")
                        logger.info(f"{inner_layer.weights = }")
                #logger.info(f"Before training: {vae_trainable_weights = }")
                # logger.info(f"Before training: {vae.classifier.trainable_weights = }")
                if myinput!='N': 
                    history_loss = train_vae(X_tr, Y_tr, X_va, Y_va, vae, ckpt_path_callback, fold_folder, myinput, N_EPOCHS, INITIAL_EPOCH, history_vae, **train_vae_kwargs)
                else: # myinput='N' is useful when loading this function in reconstruction.py or classification for instance
                    history_loss = np.load(f"{fold_folder}/history_vae", allow_pickle=True)['loss']
                # logger.info(f"After training: {vae.classifier.trainable_weights = }")
                if hasattr(vae, 'trainable_weights'):
                    logger.info(f"{len(vae.trainable_weights) = }, {len(vae.encoder.trainable_weights) = }, {len(vae.decoder.trainable_weights) = }")
                
                #for trainable_weights in vae.encoder.trainable_weights:
                #    logger.info(f"vae.encoder layers: {trainable_weights = }")
                if hasattr(vae.encoder, 'layers'):
                    for inner_layer in vae.encoder.layers:
                        if inner_layer.name == 'encoder_conv_0':
                            logger.info(f"vae.encoder layers: {inner_layer.name = }")
                            logger.info(f"{inner_layer.weights[0][0,0,0,:] = }")
                """
                for layer in vae.layers:
                    logger.info(f"for layer in vae.layers: {layer.name = }, {layer = }")
                    if layer.name == 'encoder':
                        for inner_layer in layer.layers:
                            if inner_layer.name == 'encoder_conv_0':
                                logger.info(f"encoder layers: {inner_layer.name = }")
                                logger.info(f"{inner_layer.weights[0][0,0,0,:] = }")
                    if layer.name == 'classifier':
                        for inner_layer in layer.layers:
                            logger.info(f"classifier layers: {inner_layer.name = }")
                            logger.info(f"{inner_layer.weights = }")
                """
                if hasattr(vae, 'classifier'):
                    #for trainable_weights in vae.classifier.trainable_weights:
                    #    logger.info(f"vae.classifier layers: {trainable_weights = }")
                    for inner_layer in vae.classifier.layers:
                        logger.info(f"vae.classifier layers: {inner_layer.name = }")
                        logger.info(f"{inner_layer.weights = }")
                
            

                if not reconstruction: # either training or classification, in both cases we need training set

                    z_mean_tr,_,z_tr = vae.encoder.predict(X_tr)
                    z_mean_va,_,z_va = vae.encoder.predict(X_va)
                    logger.info("Evaluating classification")
                    if hasattr(vae, 'classifier'):
                        logger.info("vae.classifier fit")
                        if vae.class_type is not None:
                            if vae.class_type == "stochastic":
                                logger.info("Y_pr_prob_va = vae.classifier.predict(z_va)")
                                Y_pr_prob_va = vae.classifier.predict(z_va)[:, 0]
                                logger.info("Y_pr_prob_tr = vae.classifier.predict(z_tr)")
                                Y_pr_prob_tr = vae.classifier.predict(z_tr)[:, 0]
                            else: # i.e. "mean"
                                logger.info("Y_pr_prob_va = vae.classifier.predict(z_mean_va)")
                                Y_pr_prob_va = vae.classifier.predict(z_mean_va)[:, 0]
                                logger.info("Y_pr_prob_tr = vae.classifier.predict(z_mean_tr)")
                                Y_pr_prob_tr = vae.classifier.predict(z_mean_tr)[:, 0]
                        else:
                            logger.info("Y_pr_prob_va = vae.classifier.predict(z_va)")
                            Y_pr_prob_va = vae.classifier.predict(z_va)[:, 0]
                            logger.info("Y_pr_prob_tr = vae.classifier.predict(z_tr)")
                            Y_pr_prob_tr = vae.classifier.predict(z_tr)[:, 0]
                        logger.info(f"{Y_tr.shape = }, {Y_pr_prob_tr.shape = } ")
                        vaebce_tr = vae.bce(Y_tr,Y_pr_prob_tr)
                        vaebce_va = vae.bce(Y_va,Y_pr_prob_va)
                        logger.info(f"{vaebce_va = }, {vaebce_tr = } ")
                    else:
                        logger.info("vae.classifier does not exist")
                # Now we decide whether to use a different epoch for the projection
                checkpoint_path = tf.train.latest_checkpoint(fold_folder)
                logger.info(f"{checkpoint_path = }")
                if myinput == 'N': # if running this code in passive mode we have to re-load the weights         
                    if evaluate_epoch != 'last': # we load a specific checkpoint
                        checkpoint_path = str(fold_folder)+f"/cp_vae-{evaluate_epoch:04d}.ckpt" # TODO: convert checkpoints to f-strings
                    logger.info(f"==loading the model: {checkpoint_path}")
                    # vae = tf.keras.models.load_model(fold_folder, compile=False)
                    if hasattr(vae, 'load_weights'):
                        vae.load_weights(f'{checkpoint_path}').expect_partial()
                        logger.info(f'{checkpoint_path} weights loaded')
            else: # i.e. use_autoencoder = false
                vae, z_tr, z_va, history_vae, checkpoint_path = None, None, None, None, None
                
            if classification:
                score.append(classify(fold_folder, evaluate_epoch, vae, X_tr, z_tr, Y_tr, X_va, z_va, Y_va, u)) 
            else:
                score=None
            my_memory.append(psutil.virtual_memory())
            logger.info(f'RAM memory: {my_memory[-1][3]:.3e}') # Getting % usage of virtual_memory ( 3rd field)

            tf.keras.backend.clear_session()
            gc.collect() # Garbage collector which removes some extra references to the objects. This is an attempt to micromanage the python handling of RAM
            if use_autoencoder: 
                if np.isinf( np.array(history_loss)).any() or np.isnan( np.array(history_loss)).any(): # check if there was a 'nan' entry in the loss (it failed)
                    logger.log(35, f'fold {i} had loss = nan and/or inf')
                    repeat_nan_local = repeat_nan_local-1
                else:
                    # sucessful fold, so terminating while
                    repeat_nan_local = -1
            else: # If the autoencdoer is not used we don't need to check the above condition
                # sucessful fold, so terminating while
                repeat_nan_local = -1
                N_EPOCHS = 0
                INITIAL_EPOCH = 0
                
            if myinput == 'N': # In passive mode we don't need to run while multiple times
                repeat_nan_local = -1
            logger.info(f' fold to terminate with {repeat_nan_local = }')
            
    return history_vae, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, vae, X_va, Y_va, X_tr, Y_tr, score 

########################################
###### TRAINING THE NETWORK ############
########################################

def scheduler(epoch, lr=5e-4, epoch_tol=None, lr_min=5e-4):
    '''
    This function keeps the initial learning rate for the first `epoch_tol` epochs
      and decreases it exponentially after that.
    Parameters
    ----------
    epoch_tol: int
        epoch until which we apply flat lr learning rate, if None learning rate will be fixed
    lr: float
        learning rate
    lr_min: float
        minimal learning rate
  '''
    if epoch_tol is None:
        return lr
    elif epoch < epoch_tol:
        return lr
    else:
        new_lr = lr*tf.math.exp(-0.1*(epoch-epoch_tol+1))
        if new_lr < lr_min:
            new_lr = lr_min
        return new_lr


class PrintLR(tf.keras.callbacks.Callback):
    '''
        Prints learning rate given the input model
    '''
    def __init__(self, model):
        self.model = model
    def on_epoch_end(self, epoch, logs=None):
        logger.info('\nLearning rate for epoch {} is {}'.format(        epoch + 1, self.model.optimizer.lr.numpy()))
"""
class scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    This function keeps the initial learning rate for the first ten epochs
    and decreases it exponentially after that.
    Parameters
    ----------
    epoch_tol: int
        epoch until which we apply flat lr learning rate, if None learning rate will be fixed
    lr: float
        learning rate
    lr_min: float
        minimal learning rate
    '''
    def __init__(self, lr=5e-4, epoch_tol=None, lr_min=5e-4):
        super(scheduler, self).__init__()
        self.lr = lr
        self.epoch_tol = epoch_tol
        self.lr_min = lr_min

    def __call__(self, step):
        if self.epoch_tol is None:
            return self.lr
        if step < self.epoch_tol:
            return self.lr
        else:
            new_lr = self.lr*tf.math.exp(-0.1)
            if new_lr < self.lr_min:
                new_lr = self.lr_min
            return new_lr
"""

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function: logger indent causes: IndexError: string index out of range
def train_vae(X_tr, Y_tr, X_va, Y_va, vae, cp_callback, folder, myinput, N_EPOCHS, INITIAL_EPOCH, history_vae, batch_size=128, scheduler_kwargs=None, validation_data=None):
    '''
    Trains the model

    Parameters
    ----------
    batch_size: (int)
    validation_data: 
        if None, the validation data will be empty, if not, the X_va and Y_va will actually be used
    
    '''
    if validation_data is not None:
        validation_data=(X_va,Y_va)
    term = TerminateOnNaN()  # fail during training on NaN loss
    if scheduler_kwargs == None:
        scheduler_kwargs = {}
    logger.info(f"{np.max(X_tr) = }, {np.min(X_tr) = }")
    logger.info(f"{scheduler_kwargs = }")
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_callback,save_weights_only=True,verbose=1)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(partial(scheduler, **scheduler_kwargs))
    
    if hasattr(vae,'compile'): # Not pca
        vae.compile(optimizer=tf.keras.optimizers.Adam())

    logger.info(f"== fit ==")
    logger.info(f'{X_tr.shape = }, {N_EPOCHS = }, {INITIAL_EPOCH = }, {batch_size = }')
    logger.info(f'{cp_callback = }')
    
    my_history_vae = vae.fit(X_tr, Y_tr, validation_data=validation_data, epochs=N_EPOCHS, initial_epoch=INITIAL_EPOCH, batch_size=batch_size, shuffle=True, callbacks=[cp_callback,scheduler_callback, PrintLR(**dict(model=vae)),term], verbose=2) # train on the last 9 folds
    # Note that we need verbose=2 statement or else @ut.indent_logger(logger) causes errors
    
    if myinput == 'C':
        if ('loss' in my_history_vae.history.keys()): # problems if the fold already contains the checkpoint = N_EPOCHS
            logger.info("we merge the history_vae dictionaries")
            logger.info(f" {len(history_vae['loss']) = }")
            logger.info(f" {len(my_history_vae.history['loss']) = }")
            for key in history_vae:
                history_vae[key] = history_vae[key]+my_history_vae.history[key]
    else:
        if vae.k1 != 'pca':
            history_vae = my_history_vae.history
        else:
            history_vae = { 'loss' : vae.score(X_tr) } # To compute average log likelihood if sklearn was used instead of autoencoder
    # logger.info(f" { len(history_vae['loss']) = }")

    vae.save(folder)
    with open(folder+'/history_vae', 'wb') as file_pi:
        pickle.dump(history_vae, file_pi)
    
    return history_vae['loss']

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
def run_vae(folder, myinput='N', XY_run_vae_keywargs=None, k_fold_cross_val_kwargs = None, load_data_kwargs = None, prepare_XY_kwargs = None, log_level=logging.INFO):# SET_YEARS=range(8000), XY_run_vae_kwargs=None):
    '''
    Loads the data and Creates a Variational AutoEncoder 
    
    Parameters
    ----------
    myinput: string
        'N': used for post-processing mode
    SET_YEARS: 
        initial data complement to a hold-out. Should be set to range(8000) for no hold-out
    reconst_reduce_years_set: int
        number of years to be taken randomly for reconstruction
    XY_run_vae_keywargs: 
        X, Y, year_permutation that are supplied in case we need to run this function 
        in succession to avoid loading the data all the time. Never tested this
    evaluate_epoch: str or int
        epoch at which we wish to perform dimensionality reduction for classification. 
        if 'last' then last checkpoint of vae will be used
        
    Returns
    ----------
    TODO: remove Y as it is already contained in Y_tr + Y_va
    N_EPOCHS: int
        number of epochs the training consists of
    INITIAL_EPOCH: int
        where the training restarts
    checkpoint_path: str
        path to the last checkpoint (check!)
    vae:  keras.models.Model
        variational autoencoder (pre-trained at the last fold which 
        is useful for reconstruction when we call a specific fold)
    Y: np.ndarray
        labels
    X_tr : np.ndarray
        training data from the last fold
    Y_tr : np.ndarray
        training labels from the last fold
    X_va : np.ndarray
        validation data from the last fold
    Y_va : np.ndarray
        validation labels from the last fold
    score: None or specified by classify()
        intended for reporting classification score
    LON:
        Meshgridded longitude
    LAT:
        Meshgridded latitude
    '''
    # setup logger to file
    folder = Path(folder) # converting string to path
    
    if myinput != 'N': # in passive mode we don't write to log.log
        fh = logging.FileHandler(f'{folder}/log.log')
        fh.setLevel(log_level)
        logger.handlers.append(fh)
    
    logger.info(f"{myinput = }")
    if k_fold_cross_val_kwargs is None:
        k_fold_cross_val_kwargs = {}
    if load_data_kwargs is None:
        load_data_kwargs = {}
    if prepare_XY_kwargs is None:
        prepare_XY_kwargs = {}
        
    nfolds = k_fold_cross_val_kwargs['nfolds']
    val_folds = k_fold_cross_val_kwargs['val_folds']
    lat_start = ut.extract_nested(load_data_kwargs, 'lat_start')
    lat_end = ut.extract_nested(load_data_kwargs, 'lat_end')
    lon_start = ut.extract_nested(load_data_kwargs, 'lon_start')
    lon_end = ut.extract_nested(load_data_kwargs, 'lon_end')

    lat_W = lat_end - lat_start
    if lon_start > lon_end:
        lon_W = lon_end - lon_start + 128
    else:
        lon_W = lon_end - lon_start
    logger.info(f" inputs {lat_W = }, {lon_W = }")
    encoder_conv_filters = ut.extract_nested(k_fold_cross_val_kwargs, 'encoder_conv_filters')
    encoder_conv_kernel_size = ut.extract_nested(k_fold_cross_val_kwargs, 'encoder_conv_kernel_size')
    encoder_conv_strides = ut.extract_nested(k_fold_cross_val_kwargs, 'encoder_conv_strides')
    encoder_conv_padding = ut.extract_nested(k_fold_cross_val_kwargs, 'encoder_conv_padding')
    decoder_conv_filters = ut.extract_nested(k_fold_cross_val_kwargs, 'decoder_conv_filters')
    decoder_conv_kernel_size = ut.extract_nested(k_fold_cross_val_kwargs, 'decoder_conv_kernel_size')
    decoder_conv_strides = ut.extract_nested(k_fold_cross_val_kwargs, 'decoder_conv_strides')
    decoder_conv_padding = ut.extract_nested(k_fold_cross_val_kwargs, 'decoder_conv_padding')

    for encoder_conv_filters1, encoder_conv_kernel_size1, encoder_conv_strides1, encoder_conv_padding1 in zip(encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides, encoder_conv_padding):
        logger.info(f"{encoder_conv_filters1 = }, {encoder_conv_kernel_size1 = }, {encoder_conv_strides1 = }, {encoder_conv_padding1 = }")
        if encoder_conv_padding1 == "same":
            lat_W = np.ceil(float(lat_W) / float(encoder_conv_strides1))
            lon_W = np.ceil(float(lon_W) / float(encoder_conv_strides1))
            logger.info(f" processing layer results in the dimension {lat_W = }, {lon_W = }")
        if encoder_conv_padding1 == "valid":
            lat_W = np.ceil(float(lat_W - encoder_conv_kernel_size1 + 1) / float(encoder_conv_strides1))
            lon_W = np.ceil(float(lon_W - encoder_conv_kernel_size1 + 1) / float(encoder_conv_strides1))
            logger.info(f" processing layer results in the dimension {lat_W = }, {lon_W = }")
    for decoder_conv_filters1, decoder_conv_kernel_size1, decoder_conv_strides1, decoder_conv_padding1 in zip(decoder_conv_filters, decoder_conv_kernel_size, decoder_conv_strides, decoder_conv_padding):
        logger.info(f"{decoder_conv_filters1 = }, {decoder_conv_kernel_size1 = }, {decoder_conv_strides1 = }, {decoder_conv_padding1 = }")
        if decoder_conv_padding1 == "same":
            lat_W = lat_W*decoder_conv_strides1
            lon_W = lon_W*decoder_conv_strides1
            logger.info(f" processing layer results in the dimension {lat_W = }, {lon_W = }")
        if decoder_conv_padding1 == "valid":
            lat_W = (lat_W-1)*decoder_conv_strides1 + decoder_conv_kernel_size1
            lon_W = (lon_W-1)*decoder_conv_strides1 + decoder_conv_kernel_size1
            logger.info(f" processing layer results in the dimension {lat_W = }, {lon_W = }")

    logger.info(f" pausing for 2 seconds...")
    time.sleep(2) 
    try:
        if XY_run_vae_keywargs is None: # we don't have X and Y yet, need to load them (may take a lot of time!)
        # loading full X can be heavy and unnecessary for reconstruction.py so we choose to work with validation automatically provided that folder already involves a fold: 
            (X, Y, year_permutation, lat, lon, time_series, threshold), mask  = ln.prepare_data_and_mask(load_data_kwargs=load_data_kwargs, prepare_XY_kwargs=prepare_XY_kwargs) # Here I follow the same structure as Alessandro has, otherwise we could use prepare_data_kwargs
            LON, LAT = np.meshgrid(lon,lat)
            np.save(f'{folder}/threshold',threshold)
        else: # we already have X and Y yet, no need to load them
            logger.info(f"loading from provided XY_run_vae_keywargs")
            X = XY_run_vae_keywargs['X']
            Y = XY_run_vae_keywargs['Y']
            year_permutation = XY_run_vae_keywargs['year_permutation']
            LAT = XY_run_vae_keywargs['LAT']
            LON = XY_run_vae_keywargs['LON']

        print(f'{X.shape = }')
        if myinput != 'N':
            np.save(f'{folder}/year_permutation',year_permutation)
            np.save(f'{folder}/Y',Y)
        else:
            if os.path.exists(f'{folder}/reconstruction.py'): # We are outside the folds
                year_permutation_load = np.load(f'{folder}/year_permutation.npy')
                Y_load = np.load(f'{folder}/Y.npy')
            else: # We are likely inside the folds (reconstruction.py is indentended to be called like that)
                year_permutation_load = np.load(f'{folder.parent}/year_permutation.npy')
                Y_load = np.load(f'{folder.parent}/Y.npy')
            # TODO: Check optionally that the files are consistent

        history_vae, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, vae, X_va, Y_va, X_tr, Y_tr, score = k_fold_cross_val(folder, myinput, mask, X, Y, time_series, year_permutation,**k_fold_cross_val_kwargs)
        
    except Exception as e:
            logger.critical(f'Run on {folder = } failed due to {repr(e)}')
            tb = traceback.format_exc() # log the traceback to the log file
            logger.error(tb)
            raise RuntimeError('Run failed') from e
            
    finally:
            if myinput != 'N': # in passive mode we don't write to log.log
                logger.handlers.remove(fh) # stop writing to the log file

    
    return history_vae, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, LAT, LON, vae, X_va, Y_va, X_tr, Y_tr, score 


ln.k_fold_cross_val = k_fold_cross_val # so that we can use functions such as ln.get_default_params without confusion with Learn2_new.py
ln.normalize_X = normalize_X
ln.move_to_folder = move_to_folder
ln.create_or_load_vae = create_or_load_vae # We also need this for kwargator() to work
ln.train_vae = train_vae
ln.VAE = tff.VAE
ln.build_encoder_skip = tff.build_encoder_skip
ln.build_decoder_skip = tff.build_decoder_skip
ln.create_classifier = tff.create_classifier
ln.scheduler = scheduler

def kwargator(thefun):
    '''
    an option to write kwargs from __main__ to json
    '''
    thefun_kwargs_default = ln.get_default_params(thefun, recursive=True)
    thefun_kwargs_default = ut.set_values_recursive(thefun_kwargs_default,
                                            {'return_time_series' : True, 'return_threshold' : True,'myinput':'Y', 'normalization_mode' : 'global_logit', 'use_autoencoder' : False, 'nfolds' : 5, 'mylocal' : '/scratchx/gmiloshe/Data/PLASIM/',
                                             'fields': ['t2m_inter_filtered','zg500_inter','mrso_inter_filtered'], 'label_field' : 't2m_inter', 'year_list': 'range(100)', 'T' : 15, 'A_weights' : [3,0,0, 3,0,0, 3,0,0, 3,0,0, 3,0,0],
                                               'print_summary' : False, 'k1': 0.9 , 'k2':0.1, 'field_weights': [2., 1., 2.], 'mask_area': 'France', 'area' : 'France', 'filter_area' : 'France', 'usemask' : False, 'Z_DIM': 8, #16, #8, #64,
                                                'N_EPOCHS': 10,'batch_size': 128, 'checkpoint_every': 1, 'lr': 5e-4, 'epoch_tol': None, 'lr_min' : 5e-4, 'lat_start' : 0, 'lat_end' : 24, 'lon_start' : 98, 'lon_end' : 18, 
                                                #'lat_start' : 4, 'lat_end' : 22, 'lon_start' : 101, 'lon_end' : 15, 
                                                'time_start' : 15, 'label_period_start' : 30,  'time_end' : 134, 'label_period_end' : 120,
                                                #'lat_0' : 0, 'lat_1' : 24, 'lon_0' : (64-28), 'lon_1' : (64+15), 'coef_out' : 0.1, 'coef_in' : 1, 
                                                # 'coef_class' : 0.1, 'class_type' : 'mean', 'L2factor' : 1e-9,
                                                'print_summary' : True
                                              })

    logger.info(ut.dict2str(thefun_kwargs_default)) # a nice way of printing nested dictionaries
    ut.dict2json(thefun_kwargs_default,'config.json')
    

if __name__ == '__main__': # we do this so that we can then load this file as a module in reconstruction.py
    
    #folder = './models/test5'
    folder = sys.argv[1]
    
    

    # folder name where weights will be stored
    myinput='Y' # default value
    if os.path.exists(folder): 
        logger.info(f'folder {folder} already exists. Should I overwrite?') 
        myinput = input(" write Y to delete the contains of the folder and start from scratch, N to stop execution, C to continue the run: ")
        if myinput == "N": # cancel
            sys.exit("User has aborted the program")
        if myinput != 'C': # if the run has been continued we want to uniquely use config.json already stored
            kwargator(run_vae) 
        if myinput == "Y": # overwrite
            os.system("rm -rf "+folder+"/*")
            move_to_folder(folder) 
        
    else: # Create the directory
        if myinput != 'C': # if the run has been continued we want to uniquely use config.json already stored
            kwargator(run_vae) 
            logger.info(f'folder {folder} created') 
            os.mkdir(folder)
            move_to_folder(folder)
    run_vae_kwargs_default = ut.json2dict('config.json')
    if myinput=='C': # If the run is to be continued import the kwargs from there
        if folder != '.': 
            raise ValueError("Only continue the run inside the existing folder containing training with `Funs.py .`")
        run_vae_kwargs_default = ut.set_values_recursive(run_vae_kwargs_default, {'myinput' : myinput})
    
    run_vae(folder, **run_vae_kwargs_default)
