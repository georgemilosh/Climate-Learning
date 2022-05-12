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



## user defined modules. small test
# I go back to absolute paths becuase otherwise I have trouble using reconstruction.py on this file later when it will be called from the folder it creates
sys.path.insert(1, '/ClimateDynamics/MediumSpace/ClimateLearningFR/gmiloshe/')       
        
import PLASIM.Learn2_new as ln
ut = ln.ut # utilities
ef = ln.ef # ERA_Fields_New
tff = ln.tff # TF_Fields

print("==Checking GPU==")
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

print("==Checking CUDA==")
tf.test.is_built_with_cuda()

# set spacing of the indentation
ut.indentation_sep = '  '



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set
# separators to create the run name from the run arguments


######################################
########## COPY SOURCE FILES #########
######################################

def move_to_folder(checkpoint_name):
    '''
    Copies this file and its dependencies to a given folder. 
    Parameters
    ----------
    checkpoint_name: string
        name of the folder that accepts the copies
        
    '''

    sys.stdout = ef.Logger(checkpoint_name)  # Keep a copy of print outputs there
    shutil.copy(__file__, checkpoint_name+'/Funs.py') # Copy this file to the directory of the training
    shutil.copy('history.py', checkpoint_name)  # Also copy a version of the files we work with to analyze the results of the training
    shutil.copy('reconstruction.py', checkpoint_name)
    shutil.copy('../ERA/ERA_Fields.py', checkpoint_name)
    shutil.copy('../ERA/TF_Fields.py', checkpoint_name)


############################################
########## DATA PREPROCESSING ##############
############################################

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
def normalize_X(X,checkpoint_name, myinput='N',mode='pointwise'):
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
            np.save(checkpoint_name+'/X_mean', X_mean)
            np.save(checkpoint_name+'/X_std', X_std)
        else: # mean and std are expected to be already computed
            X_mean = np.load(checkpoint_name+'/X_mean.npy')
            X_std = np.load(checkpoint_name+'/X_std.npy')

        return   1./(1.+np.exp(-(X-X_mean)/X_std)) # (X-X_mean)/X_std # # we have applied sigmoid because variational autoencoder reconstructs with this activation
    else:
        logger.info("===Rescaling X===")
        if myinput != 'N':
            maxX = np.max(X,tuple(list(range(len(X.shape)-1)))) # Equivalent to np.max(X,(0,1,...,last-1))
            minX = np.min(X,tuple(list(range(len(X.shape)-1))))
            np.save(checkpoint_name+'/maxX', maxX)
            np.save(checkpoint_name+'/minX', minX)
        else:
            maxX = np.load(checkpoint_name+'/maxX.npy')
            minX = np.load(checkpoint_name+'/minX.npy')
    return (X - minX) / (maxX - minX) # 2*(X - minX)/(maxX - minX)-1  #

################################################
########## NEURAL NETWORK CONSTRUCTION ###########
################################################

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
def create_or_load_vae(checkpoint_name, mask_weights, INPUT_DIM, myinput, Z_DIM=64, N_EPOCHS=2, vae_kwargs=None, encoder_kwargs=None, decoder_kwargs=None):
    '''
    Creates a Variational AutoEncoder Model or loads the existing one from the weights given in the model
        
    Parameters
    ----------
    checkpoint_name
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
    '''
    
    ### preliminary operations
    ##########################
    if vae_kwargs is None:
        vae_kwargs = {}
    if encoder_kwargs is None:
        encoder_kwargs = {}
    if decoder_kwargs is None:
        decoder_kwargs = {}
    
    logger.info("==Building encoder==")
    encoder_inputs, encoder_outputs, shape_before_flattening, encoder  = tff.build_encoder_skip(input_dim = INPUT_DIM, output_dim = Z_DIM, **encoder_kwargs)
    encoder.summary()
    logger.info("==Building decoder==") 
    decoder_input, decoder_output, decoder = tff.build_decoder_skip(mask=mask_weights, input_dim = Z_DIM, shape_before_flattening = shape_before_flattening, **decoder_kwargs)
    decoder.summary()


    logger.info("==Attaching decoder and encoder and compiling==")
    vae = tff.VAE(encoder, decoder, **vae_kwargs) #, mask_weights=mask_weights)
    logger.info(f'{vae.k1 = },{vae.k2 = } ')
    if myinput == 'Y': # The run is to be performed from scratch
        INITIAL_EPOCH = 0
        history = []
        checkpoint = []
    else: # the run has to be continued
        history = np.load(checkpoint_name+'/history', allow_pickle=True)
        INITIAL_EPOCH = len(history['loss'])
        logger.info(f'==loading the model: {checkpoint_name}')
        N_EPOCHS = N_EPOCHS + INITIAL_EPOCH 
        #vae = tf.keras.models.load_model(checkpoint_name, compile=False)
        checkpoint = tf.train.latest_checkpoint(checkpoint_name)
        logger.info(f'checkpoint =  {checkpoint}')
        vae.load_weights(checkpoint)
        
    logger.info(f'INITIAL_EPOCH =  {INITIAL_EPOCH}')

    INPUT_DIM_withnone = list(INPUT_DIM)
    INPUT_DIM_withnone.insert(0,None)
    
    vae.build(tuple(INPUT_DIM_withnone)) 
    vae.compute_output_shape(tuple(INPUT_DIM_withnone))
    vae.summary()

    checkpoint_path = checkpoint_name+"/cp-{epoch:04d}.ckpt"
    
    return vae, history, N_EPOCHS, INITIAL_EPOCH, checkpoint, checkpoint_path

@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
def PrepareDataAndVAE(checkpoint_name, myinput='Y', SET_YEARS=range(100)):
    '''
    Loads the data and Creates a Variational AutoEncoder 
    
    Parameters
    ----------
    myinput: 
    '''
    X, _Y, _year_permutation, lat, lon = ln.prepare_data(load_data_kwargs = {'fields': ['t2m_filtered','zg500','mrso_filtered'], 'lat_end': 24, 'dataset_years': 8000, 'year_list': SET_YEARS},
                           prepare_XY_kwargs = {'roll_X_kwargs': {'roll_steps': 64}}) # That's the version that fails
    LON, LAT = np.meshgrid(lon,lat) 
    if myinput != 'N':
        np.save(checkpoint_name+'/year_permutation',_year_permutation)
        np.save(checkpoint_name+'/Y',_Y)
    else:
        year_permutation_load = np.load(checkpoint_name+'/year_permutation.npy')
        Y_load = np.load(checkpoint_name+'/Y.npy')
        # TODO: Check optionally that the files are consistent
    
    INPUT_DIM = X.shape[1:]  # Image dimension
    
    X = normalize_X(X, checkpoint_name, myinput=myinput, mode='pointwise')
    logger.info(f'{X.shape = },{np.mean(X[:,5,5,0]) = },{np.std(X[:,5,5,0]) = }')
    
    filter_mask = ln.roll_X(ef.create_mask('Plasim','France', LON, axes='last 2', return_full_mask=True),1)
    logger.info(f'{Fore.RED}{filter_mask.shape = }')
    logger.info(f'{np.ones(LON.shape).shape = }')
    logger.info(f'{np.array([filter_mask,np.ones(LON.shape),filter_mask], dtype=bool).shape = }{Style.RESET_ALL}')
    
    filter_mask = np.array([filter_mask,np.ones(LON.shape),filter_mask], dtype=bool).transpose(1,2,0) 
    
    logger.info(f'{X.dtype = }, {filter_mask.dtype}')
   
    vae, history, N_EPOCHS, INITIAL_EPOCH, checkpoint, checkpoint_path = create_or_load_vae(checkpoint_name, filter_mask, INPUT_DIM, myinput, 
                                            vae_kwargs={'k1': 0.9, 'k2': 0.1, 'from_logits': False, 'field_weights': [2.0, 1.0, 2.0]},
                                        encoder_kwargs={'conv_filters':[16, 16, 16, 32, 32,  32,   64, 64],
                                                    'conv_kernel_size':[5,  5,  5,  5,   5,   5,   5,  3], 
                                                    'conv_strides'    :[2,  1,  1,  2,   1,   1,   2,  1],
                                                    'conv_padding':["same","same","same","same","same","same","same","valid"], 
                                                    'conv_activation':["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu"], 
                                                    'conv_skip':dict({(0,2),(3,5)}), 
                                                    'use_batch_norm' : [True,True,True,True,True,True,True,True], 
                                                    'use_dropout' : [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]},
                                        decoder_kwargs={'conv_filters':[64,32,32,32,16,16,16,3], 
                                                    'conv_kernel_size':[3, 5, 5, 5, 5, 5, 5, 5], 
                                                        'conv_strides':[1, 2, 1, 1, 2, 1, 1, 2],
                                                        'conv_padding':["valid","same","same","same","same","same","same","same"], 
                                                     'conv_activation':["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","sigmoid"], 
                                                           'conv_skip':dict({(1,3),(4,6)}),
                                                        'use_batch_norm' : [True,True,True,True,True,True,True,True], 
                                                        'use_dropout' : [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25], 'usemask' : True})
    
    return X, lat, lon, vae, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, history


@ut.execution_time  # prints the time it takes for the function to run
@ut.indent_logger(logger)   # indents the log messages produced by this function
# logger indent causes: IndexError: string index out of range
def train_vae(X, vae, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, checkpoint_name, myinput, history, batch_size=128, lr=1e-3):
    
    logger.info(f" {np.max(X) = }, {np.min(X) = }")
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

    vae.compile(optimizer=tf.keras.optimizers.Adam(lr = lr))

    logger.info(f"{Fore.GREEN}== fit =={Style.RESET_ALL}")
    logger.info(f'{ X[X.shape[0]//10:,...].shape = }, {N_EPOCHS = }, {INITIAL_EPOCH = }, {batch_size = }')
    logger.info(f'{cp_callback = }')
    vae.summary()
    my_history = vae.fit(X[X.shape[0]//10:,...], epochs=N_EPOCHS, initial_epoch=INITIAL_EPOCH, batch_size=batch_size, shuffle=True, callbacks=[cp_callback]) # train on the last 9 folds

    if myinput == 'C':
        logger.info("we merge the history dictionaries")
        logger.info(f" {len(history['loss']) = }")
        logger.info(f" {len(my_history.history['loss']) = }")
        for key in history:
            history[key] = history[key]+my_history.history[key]
    else:
        history = my_history.history
    logger.info(f" { len(history['loss']) = }")

    vae.save(checkpoint_name)
    with open(checkpoint_name+'/history', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    return history['loss']


if __name__ == '__main__': # we do this so that we can then load this file as a module in reconstruction.py
    
    #checkpoint_name = './models/test5'
    checkpoint_name = sys.argv[1]
    
    
    # folder name where weights will be stored
    myinput='Y' # default value
    if os.path.exists(checkpoint_name): 
        logger.info(f'folder {checkpoint_name} already exists. Should I overwrite?') 
        myinput = input(" write Y to overwrite, N to stop execution, C to continue the run: ")
        if myinput == "N": # cancel
            sys.exit("User has aborted the program")
        if myinput == "Y": # overwrite
            os.system("rm -rf "+checkpoint_name+"/*")
            move_to_folder(checkpoint_name) 
    else: # Create the directory
        logger.info(f'folder {checkpoint_name} created') 
        os.mkdir(checkpoint_name)
        move_to_folder(checkpoint_name)
 
    X, lon, lat, vae, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, history = PrepareDataAndVAE(checkpoint_name=checkpoint_name,myinput=myinput)
    
    
    history_loss = train_vae(X, vae, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, checkpoint_name, myinput, history, batch_size=128, lr=1e-3)
