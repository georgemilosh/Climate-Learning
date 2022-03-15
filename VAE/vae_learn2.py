# George Miloshevich 2022
# inspired by https://keras.io/examples/generative/vae/
# merged with (and upgraded to tensorflow 2) https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8
# Adapted some routines from Learn2_new.py of Alessandro Lovo


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
from pathlib import Path
from colorama import Fore # support colored output in terminal
from colorama import Style
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO



## user defined modules
# I go back to absolute paths becuase otherwise I have trouble using reconstruction.py on this file later when it will be called from the folder it creates
sys.path.insert(1, '/ClimateDynamics/MediumSpace/ClimateLearningFR/gmiloshe/')       
        
import PLASIM.Learn2_new as ln
ut = ln.ut # utilities
ef = ln.ef # ERA_Fields_New
tff = ln.tff # TF_Fields

logger.info("==Checking GPU==")
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

logger.info("==Checking CUDA==")
tf.test.is_built_with_cuda()

# set spacing of the indentation
ut.indentation_sep = '  '



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

    sys.stdout = ef.Logger(folder)  # Keep a copy of print outputs there
    shutil.copy(__file__, folder+'/Funs.py') # Copy this file to the directory of the training
    shutil.copy('history.py', folder)  # Also copy a version of the files we work with to analyze the results of the training
    shutil.copy('reconstruction.py', folder)
    shutil.copy('../ERA/ERA_Fields.py', folder)
    shutil.copy('../ERA/TF_Fields.py', folder)


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
        how to perform the normalization.

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
            X_mean = np.load(f'{folder}/X_mean.npy')
            X_std = np.load(f'{folder}/X_std.npy')

        return   1./(1.+np.exp(-(X-X_mean)/X_std)) # (X-X_mean)/X_std # # we have applied sigmoid because variational autoencoder reconstructs with this activation
    else:
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

################################################
########## NEURAL NETWORK CONSTRUCTION ###########
################################################

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
def create_or_load_vae(folder, INPUT_DIM, myinput, filter_area='France', Z_DIM=64, N_EPOCHS=2, vae_kwargs=None, encoder_kwargs=None, decoder_kwargs=None):
    '''
    Creates a Variational AutoEncoder Model or loads the existing one from the weights given in the model
        
    Parameters
    ----------
    folder
        the name of the file where we store the model
    INPUT_DIM : tuple
        shape of a single input datapoint, i.e. not counting the axis corresponding to iteration through the datapoints (batch axis)
    Z_DIM: int
        dimension of the latent space
    vae_kwargs:
    encoder_kwargs:
    decoder_kwargs:

    

    Returns
    -------
    model : keras.models.Model
    N_EPOCHS: 
        it may change if the myinput='C' thus the need to return
    INITIAL_EPOCH: int
        it may change as well because of myinput='C'
    '''
    
    ### preliminary operations
    ##########################
    if vae_kwargs is None:
        vae_kwargs = {}
    if encoder_kwargs is None:
        encoder_kwargs = {}
    if decoder_kwargs is None:
        decoder_kwargs = {}
    vae_kwargs_local = vae_kwargs.copy()
    filter_area = vae_kwargs_local.pop('filter_area') # because vae_kwargs_local must be passed to tff.VAE constructor and we still need vae_kwargs for other folds
    Z_DIM = vae_kwargs_local.pop('Z_DIM')
    N_EPOCHS = vae_kwargs_local.pop('N_EPOCHS')
    
    print(f"{Fore.GREEN}{INPUT_DIM[:-1] = }{Style.RESET_ALL}")
    
    ones_dim = np.ones(INPUT_DIM[:-1])
    
    filter_mask = ln.roll_X(ef.create_mask('Plasim',filter_area, ones_dim, axes='last 2', return_full_mask=True),1)
    logger.info(f'{filter_mask.shape = }')
    logger.info(f'{ones_dim.shape = }')
    logger.info(f'{np.array([filter_mask,ones_dim,filter_mask], dtype=bool).shape = }')
    
    filter_mask = np.array([filter_mask,ones_dim,filter_mask], dtype=bool).transpose(1,2,0) 
    
    logger.info(f'{filter_mask.dtype}')
    
    
    logger.info("==Building encoder==")
    encoder_inputs, encoder_outputs, shape_before_flattening, encoder  = tff.build_encoder_skip(input_dim = INPUT_DIM, output_dim = Z_DIM, **encoder_kwargs)
    encoder.summary()
    logger.info("==Building decoder==") 
    decoder_input, decoder_output, decoder = tff.build_decoder_skip(mask=filter_mask, input_dim = Z_DIM, shape_before_flattening = shape_before_flattening, **decoder_kwargs)
    decoder.summary()


    logger.info("==Attaching decoder and encoder and compiling==")
    vae = tff.VAE(encoder, decoder, **vae_kwargs_local) #, mask_weights=mask_weights)
    logger.info(f'{vae.k1 = },{vae.k2 = } ')
    if myinput == 'Y': # The run is to be performed from scratch
        INITIAL_EPOCH = 0
        history_vae = []
        checkpoint = []
    else: # the run has to be continued
        history_vae = np.load(f'{folder}/history_vae', allow_pickle=True)
        INITIAL_EPOCH = len(history_vae['loss'])
        logger.info(f'==loading the model: {folder}')
        N_EPOCHS = N_EPOCHS + INITIAL_EPOCH 
        #vae = tf.keras.models.load_model(folder, compile=False)
        checkpoint = tf.train.latest_checkpoint(folder)
        logger.info(f'checkpoint =  {checkpoint}')
        vae.load_weights(checkpoint)
        
    logger.info(f'INITIAL_EPOCH =  {INITIAL_EPOCH}')

    INPUT_DIM_withnone = list(INPUT_DIM)
    INPUT_DIM_withnone.insert(0,None)
    
    vae.build(tuple(INPUT_DIM_withnone)) 
    vae.compute_output_shape(tuple(INPUT_DIM_withnone))
    vae.summary()

    checkpoint_path = str(folder)+"/cp_vae-{epoch:04d}.ckpt" # TODO: convert checkpoints to f-strings
    
    return vae, history_vae, N_EPOCHS, INITIAL_EPOCH, checkpoint, checkpoint_path



@ut.execution_time
#@ut.indent_logger(logger)   # -> Causes error:   File "/ClimateDynamics/MediumSpace/ClimateLearningFR/gmiloshe/ERA/utilities.py", line 88, in wrapper
#    message = (indentation_sep+f'\n{indentation_sep}'.join(message[:-1].split('\n')) + message[-1])
#IndexError: string index out of range
def k_fold_cross_val(folder, myinput, X, Y, create_vae_kwargs=None, nfolds=10, val_folds=1, range_nfolds=None, u=1, normalization_mode='pointwise'):
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
    create_vae_kwargs : dict
        dictionary with the parameters to create a vae model
    nfolds : int, optional
        number of folds
    val_folds : int, optional
        number of folds to be used for the validation set for every split
    u : float, optional
        undersampling factor (>=1). If = 1 no undersampling is performed (NOT OPERATION AT THE MOMENT)
    
    '''
    if create_vae_kwargs is None:
        create_vae_kwargs = {}

    #folder = folder.rstrip('/')

    # k fold cross validation
    scores = []
    if myinput != 'N': # In training regime, otherwise default value for reconstruction
        range_nfolds=range(nfolds)
    else:
        if not os.path.exists(f'{folder}/reconstruction.py'): # we are inside of one of the folds
            range_nfolds=[int(np.load(f'{folder}/fold_num.npy'))]  #the other option would be parsing the fold_N string to get i in future
        else: # we are outside of the folds
            range_nfolds=range(nfolds)
        
    my_memory = [] # monitor RAM storage
    
    for i in range_nfolds:
        logger.info('=============')
        logger.log(35, f'fold {i} ({i+1}/{nfolds})')
        logger.info('=============')
        # create fold_folder
       
        if not os.path.exists(f'{folder}/reconstruction.py'): # # we are inside of one of the folds
            logger.info(f'{folder}/reconstruction.py does not exist')
            fold_folder=folder
        else: # we are not inside of one of the folds, either this is a new run, or we need to iterate through the runs
            logger.info(f'{folder}/reconstruction.py exists')
            if myinput != 'N': # if 'N' do not create a new folder (just loading) 
                fold_folder = f'{folder}/fold_{i}'
                if myinput == 'Y': # If 'C' we con't need to change anything
                    os.mkdir(fold_folder)
                    np.save(f'{fold_folder}/fold_num',i) # the other option would be parsing the fold_N string to get i in future

        # split data
        logger.info(f"{Fore.RED}{i = }, {X.shape = }, {Y.shape = }, {nfolds=}, {val_folds=}{Style.RESET_ALL}")
        X_tr, Y_tr, X_va, Y_va = ln.k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)
        
        # perform undersampling
        #X_tr, Y_tr = undersample(X_tr, Y_tr, u=u)

        n_pos_tr = np.sum(Y_tr)
        n_neg_tr = len(Y_tr) - n_pos_tr
        logger.info(f'number of training data: {len(Y_tr)} of which {n_neg_tr} negative and {n_pos_tr} positive')

        INPUT_DIM = X_tr.shape[1:]  # Image dimension
        logger.info(f"{Fore.RED}{INPUT_DIM = }{Style.RESET_ALL}")
    
        X_tr = normalize_X(X_tr, fold_folder, myinput=myinput, mode=normalization_mode)
        X_va = normalize_X(X_va, fold_folder, myinput='N', mode=normalization_mode) #validation is always normalized passively (based on already computed normalization)
        logger.info(f'{X_tr.shape = },{np.mean(X_tr[:,5,5,0]) = },{np.std(X_tr[:,5,5,0]) = }')
        
        logger.info(f"{Fore.YELLOW}{create_vae_kwargs = }{Style.RESET_ALL}")
        vae, history_vae, N_EPOCHS, INITIAL_EPOCH, checkpoint, checkpoint_path = create_or_load_vae(fold_folder, INPUT_DIM, myinput, filter_area='France', Z_DIM=64, N_EPOCHS=2,
                                                **create_vae_kwargs)
        if myinput!='N': 
            history_loss = train_vae(X_tr, vae, checkpoint_path, fold_folder, myinput, N_EPOCHS, INITIAL_EPOCH, history_vae, batch_size=128, lr=1e-3)
        else: # myinput='N' is useful when loading this function in reconstruction.py for instance
            history_loss = []
            
        my_memory.append(psutil.virtual_memory())
        logger.info(f'RAM memory: {my_memory[-1][3]:.3e}') # Getting % usage of virtual_memory ( 3rd field)

        tf.keras.backend.clear_session()
        gc.collect() # Garbage collector which removes some extra references to the objects. This is an attempt to micromanage the python handling of RAM


    return history_vae, history_loss, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, vae, X_va, X_tr


@ut.execution_time  # prints the time it takes for the function to run
#@ut.indent_logger(logger)   # indents the log messages produced by this function
# logger indent causes: IndexError: string index out of range
def run_vae(folder, myinput='N', SET_YEARS=range(100)):
    '''
    Loads the data and Creates a Variational AutoEncoder 
    
    Parameters
    ----------
    myinput: string
        'N': used for post-processing mode
    SET_YEARS: 
        initial data hold-out. Should be set to range(8000) for a real run
    '''
    folder = Path(folder)
    logger.info(f"{myinput = }")
    X, Y, _year_permutation, lat, lon = ln.prepare_data(load_data_kwargs = {'fields': ['t2m_filtered','zg500','mrso_filtered'], 'lat_end': 24, 'dataset_years': 8000, 'year_list': SET_YEARS},
                           prepare_XY_kwargs = {'roll_X_kwargs': {'roll_steps': 64}}) # That's the version that fails
    LON, LAT = np.meshgrid(lon,lat) 
    print(f'{X.shape = }')
    if myinput != 'N':
        np.save(f'{folder}/year_permutation',_year_permutation)
        np.save(f'{folder}/Y',Y)
    else:
        if os.path.exists(f'{folder}/reconstruction.py'): # We are outside the folds
            year_permutation_load = np.load(f'{folder}/year_permutation.npy')
            Y_load = np.load(f'{folder}/Y.npy')
        else: # We are likely inside the folds (reconstruction.py is indentended to be called like that)
            year_permutation_load = np.load(f'{folder.parent}/year_permutation.npy')
            Y_load = np.load(f'{folder.parent}/Y.npy')
        # TODO: Check optionally that the files are consistent
    
    history_vae, history_loss, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, vae, X_va, X_tr = k_fold_cross_val(folder, myinput, X, Y,
                            create_vae_kwargs={'vae_kwargs':{'k1': 0.9, 'k2': 0.1, 'from_logits': False, 'field_weights': [2.0, 1.0, 2.0], 'filter_area':'France', 'Z_DIM': 64, 'N_EPOCHS':2},
                                            'encoder_kwargs':{'conv_filters':[16, 16, 16, 32, 32,  32,   64, 64],
                                                        'conv_kernel_size':[5,  5,  5,  5,   5,   5,   5,  3], 
                                                        'conv_strides'    :[2,  1,  1,  2,   1,   1,   2,  1],
                                                        'conv_padding':["same","same","same","same","same","same","same","valid"], 
                                                        'conv_activation':["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu"], 
                                                        'conv_skip':dict({(0,2),(3,5)}), 
                                                        'use_batch_norm' : [True,True,True,True,True,True,True,True], 
                                                        'use_dropout' : [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]},
                                            'decoder_kwargs':{'conv_filters':[64,32,32,32,16,16,16,3], 
                                                        'conv_kernel_size':[3, 5, 5, 5, 5, 5, 5, 5], 
                                                            'conv_strides':[1, 2, 1, 1, 2, 1, 1, 2],
                                                            'conv_padding':["valid","same","same","same","same","same","same","same"], 
                                                         'conv_activation':["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","sigmoid"], 
                                                               'conv_skip':dict({(1,3),(4,6)}),
                                                            'use_batch_norm' : [True,True,True,True,True,True,True,True], 
                                                            'use_dropout' : [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], 'usemask' : True}})
    
    return history_vae, history_loss, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, LAT, LON, Y, vae, X_va, X_tr


@ut.execution_time  # prints the time it takes for the function to run
#@ut.indent_logger(logger)   # indents the log messages produced by this function
# logger indent causes: IndexError: string index out of range
def train_vae(X, vae, checkpoint_path, folder, myinput, N_EPOCHS, INITIAL_EPOCH, history_vae, batch_size=128, lr=1e-3):
    
    logger.info(f" {np.max(X) = }, {np.min(X) = }")
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

    vae.compile(optimizer=tf.keras.optimizers.Adam(lr = lr))

    logger.info(f"{Fore.GREEN}== fit =={Style.RESET_ALL}")
    logger.info(f'{ X.shape = }, {N_EPOCHS = }, {INITIAL_EPOCH = }, {batch_size = }')
    logger.info(f'{cp_callback = }')
    vae.summary()
    my_history_vae = vae.fit(X, epochs=N_EPOCHS, initial_epoch=INITIAL_EPOCH, batch_size=batch_size, shuffle=True, callbacks=[cp_callback]) # train on the last 9 folds

    if myinput == 'C':
        logger.info("we merge the history_vae dictionaries")
        logger.info(f" {len(history_vae['loss']) = }")
        logger.info(f" {len(my_history_vae.history['loss']) = }")
        for key in history_vae:
            history_vae[key] = history_vae[key]+my_history_vae.history[key]
    else:
        history_vae = my_history_vae.history
    logger.info(f" { len(history_vae['loss']) = }")

    vae.save(folder)
    with open(folder+'/history_vae', 'wb') as file_pi:
        pickle.dump(history_vae, file_pi)
    return history_vae['loss']


if __name__ == '__main__': # we do this so that we can then load this file as a module in reconstruction.py
    
    #folder = './models/test5'
    folder = sys.argv[1]
    
    
    # folder name where weights will be stored
    myinput='Y' # default value
    if os.path.exists(folder): 
        logger.info(f'folder {folder} already exists. Should I overwrite?') 
        myinput = input(" write Y to overwrite, N to stop execution, C to continue the run: ")
        if myinput == "N": # cancel
            sys.exit("User has aborted the program")
        if myinput == "Y": # overwrite
            os.system("rm -rf "+folder+"/*")
            move_to_folder(folder) 
    else: # Create the directory
        logger.info(f'folder {folder} created') 
        os.mkdir(folder)
        move_to_folder(folder)
 
    run_vae(folder=folder,myinput=myinput)