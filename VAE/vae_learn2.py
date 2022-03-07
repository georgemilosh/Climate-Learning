# George Miloshevich 2022
# inspired by https://keras.io/examples/generative/vae/
# merged with (and upgraded to tensorflow 2) https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8
# Adapted some routines from Learn2_new.py of Alessandro Lovo


### IMPORT LIBRARIES #####

## general purpose



# log to stdout
import logging
import sys
import os
import time
import itertools
import numpy as np
import shutil
from pathlib import Path
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO


## user defined modules
this_module = sys.modules[__name__]
path_to_here = Path(__file__).resolve().parent
path_to_PLASIM = path_to_here / 'PLASIM' # when absolute path, so you can run the script from another folder (outside VAE)
if not os.path.exists(path_to_PLASIM):
    path_to_PLASIM = path_to_here.parent / 'PLASIM'
    if not os.path.exists(path_to_PLASIM):
        raise FileNotFoundError('Could not find PLASIM folder')

# go to the parent so vscode is happy with code completion :)
path_to_PLASIM = path_to_PLASIM.parent
path_to_PLASIM = str(path_to_PLASIM)
logger.info(f'{path_to_PLASIM = }/PLASIM/')
if not path_to_PLASIM in sys.path:
    sys.path.insert(1, path_to_PLASIM)        
        
import PLASIM.Learn2_new as ln
ut = ln.ut # utilities
ef = ln.ef # ERA_Fields_New
tff = ln.tff # TF_Fields

# set spacing of the indentation
ut.indentation_sep = '  '



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set
# separators to create the run name from the run arguments


def PrepareParameters(creation):
    print("==Preparing Parameters==")
    WEIGHTS_FOLDER = './models/'
    
    RESCALE_TYPE =    'normalize' #    'rescale'  
    Z_DIM = 64 #8 #16 #256 # Dimension of the latent vector (z)
    BATCH_SIZE = 128#512
    LEARNING_RATE = 1e-3#5e-4# 1e-3#5e-6
    N_EPOCHS = 2 #10#600#200
    SET_YEARS =   range(100)  #range(8000) #range(1000)   # the set of years that variational autoencoder sees
    SET_YEARS_LABEL =   'range100' #'range8000' #'range1000' # 
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
    checkpoint_name = WEIGHTS_FOLDER+Model+'vae_learn2_weight212custlosswithfilter_t2mzg500mrso_resdeep_filt5_yrs-'+SET_YEARS_LABEL+'_last9folds_'+RESCALE_TYPE+'_k1_'+str(K1)+'_k2_'+str(K2)+'_LR_'+str(LEARNING_RATE)+'_ZDIM_'+str(Z_DIM)
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

        sys.stdout = ef.Logger(checkpoint_name)  # Keep a copy of print outputs there
        shutil.copy(__file__, checkpoint_name+'/Funs.py') # Copy this file to the directory of the training
        shutil.copy('history.py', checkpoint_name)  # Also copy a version of the files we work with to analyze the results of the training
        shutil.copy('reconstruction.py', checkpoint_name)
        shutil.copy('../ERA/ERA_Fields.py', checkpoint_name)
        shutil.copy('../ERA/TF_Fields.py', checkpoint_name)
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

        return   1./(1.+np.exp(-(X-X_mean)/X_std)) # (X-X_mean)/X_std # # we have apply sigmoid because variational autoencoder reconstructs with this activation
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
    return (X - minX) / (maxX - minX) # 2*(X - minX)/(maxX - minX)-1  #

def ConstructVAE(INPUT_DIM, Z_DIM, checkpoint_name, N_EPOCHS, myinput, K1, K2, from_logits=False, mask_weights=None):
    print("==Building encoder==")
    """
    encoder_inputs, encoder_outputs, shape_before_flattening, encoder  = tff.build_encoder2(input_dim = INPUT_DIM, 
                                                output_dim = Z_DIM, 
                                                conv_filters = [32,64,64,64],
                                                conv_kernel_size = [3,3,3,3],
                                                conv_strides = [2,2,2,1],
                                                conv_padding = ["same","same","same","valid"], 
                                                conv_activation = ["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu"],use_batch_norm=True, use_dropout=True)
    """
    encoder_inputs, encoder_outputs, shape_before_flattening, encoder  = tff.build_encoder_skip(input_dim = INPUT_DIM, 
                                                output_dim = Z_DIM, 
                                                conv_filters =     [16, 16, 16, 32, 32,  32,   64, 64],
                                                conv_kernel_size = [5,  5,  5,  5,   5,   5,   5,  3],
                                                conv_strides =     [2,  1,  1,  2,   1,   1,   2,  1],
                                                conv_padding =     ["same","same","same","same","same","same","same","valid"], 
                                                conv_activation =  ["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu"],
                                                #conv_skip = dict({}),use_batch_norm=True, use_dropout=True)
                                                conv_skip = dict({(0,2),(3,5)}),use_batch_norm=True, use_dropout=True)
    encoder.summary()
    print("==Building decoder==")      
    """
    decoder_input, decoder_output, decoder = tff.build_decoder2(input_dim = Z_DIM,  
                                        shape_before_flattening = shape_before_flattening,
                                        conv_filters = [64,64,32,3],
                                        conv_kernel_size = [3,3,3,3],
                                        conv_strides = [1,2,2,2],
                                        conv_padding = ["valid","same","same","same"], 
                                        conv_activation = ["LeakyRelu","LeakyRelu","LeakyRelu","sigmoid"])
    """
    decoder_input, decoder_output, decoder = tff.build_decoder_skip(input_dim = 64,  
                                        shape_before_flattening = shape_before_flattening,
                                        conv_filters =     [64,32,32,32,16,16,16,3],
                                        conv_kernel_size = [3, 5, 5, 5, 5, 5, 5, 5],
                                        conv_strides =     [1, 2, 1, 1, 2, 1, 1, 2],
                                        conv_padding =     ["valid","same","same","same","same","same","same","same"], 
                                        conv_activation =  ["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu","sigmoid"], 
                                        #conv_skip = dict({}), use_batch_norm = True, use_dropout = True)
                                        conv_skip = dict({(1,3),(4,6)}), use_batch_norm = True, use_dropout = True, mask = mask_weights)
    decoder.summary()


    print("==Attaching decoder and encoder and compiling==")
    
    
    
    #vae = tff.VAE(encoder, decoder, k1=K1, k2=K2, from_logits=from_logits, field_weights=None) 
    vae = tff.VAE(encoder, decoder, k1=K1, k2=K2, from_logits=from_logits, field_weights=[2.0, 1.0, 2.0]) #, mask_weights=mask_weights)
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
    
    X, Y, yp, LON, LAT = ln.prepare_data(load_data_kwargs = {'fields': ['t2m_filtered','zg500','mrso_filtered'],'lat_end': 24, 'dataset_years': 8000, 'year_list': SET_YEARS},
                           prepare_XY_kwargs = {'roll_X_kwargs': {'roll_steps': 64}})
    
    print("LON.shape = ", LON.shape, " ; LAT.shape = ", LAT.shape)
    np.save(checkpoint_name+'/year_permutation',yp)
    np.save(checkpoint_name+'/Y',Y)
    #X = X.reshape(-1,*X.shape[2:])
    
    INPUT_DIM = X.shape[1:]  # Image dimension
    
    X = RescaleNormalize(X,RESCALE_TYPE, creation, checkpoint_name)
    

    print("X.shape = ", X.shape,  " ,np.mean(X[:,5,5,0]) = ", np.mean(X[:,5,5,0]), " ,np.std(X[:,5,5,0]) = ", np.std(X[:,5,5,0]), " , np.min(X) = ", np.min(X), " , np.max(X) = ", np.max(X))
    
    """
    BATCH_SIZE2 = 128 #testing an idea to put a mask dependent weight on the reconstruction loss. This idea caused errors so i am giving up
    # The idea is to create the mask in accordance to what we believe are relevant fields for the proper classification of the heat waves
    filter_mask = roll_X(ef.create_mask(Model,'France', X[:BATCH_SIZE2,...,0], axes='first 2', return_full_mask=True),1)  # this mask can be used in filtering the weights of reconstruction loss in VAE
    print('filter_mask.shape = ', filter_mask.shape)
    print('np.ones(X[:BATCH_SIZE2,...,0].shape).shape = ', np.ones(X[:BATCH_SIZE2,...,0].shape).shape)
    print('np.array([filter_mask,np.ones(X[:BATCH_SIZE2,...,0].shape),filter_mask], dtype=bool).shape = ', np.array([filter_mask,np.ones(X[:BATCH_SIZE2,...,0].shape),filter_mask], dtype=bool).shape)
    filter_mask = np.array([filter_mask,np.ones(X[:BATCH_SIZE2,...,0].shape),filter_mask], dtype=bool).transpose(1,2,3,0) # Stack truth mask (for zg500) between two layers that have a mask
    print('filter_mask.shape = ', filter_mask.shape)
    """
    filter_mask = ln.roll_X(ef.create_mask(Model,'France', X[0,...,0], axes='first 2', return_full_mask=True),1)
    print('filter_mask.shape = ', filter_mask.shape)
    print('np.ones(X[0,...,0].shape).shape = ', np.ones(X[0,...,0].shape).shape)
    print('np.array([filter_mask,np.ones(X[0,...,0].shape),filter_mask], dtype=bool).shape = ', np.array([filter_mask,np.ones(X[0,...,0].shape),filter_mask], dtype=bool).shape)
    filter_mask = np.array([filter_mask,np.ones(X[0,...,0].shape),filter_mask], dtype=bool).transpose(1,2,0) 
    
    print("X.dtype = ", X.dtype, " ,filter_mask.dtype = ", filter_mask.dtype)
    
    vae, history, N_EPOCHS, INITIAL_EPOCH, checkpoint, checkpoint_path = ConstructVAE(INPUT_DIM, Z_DIM, checkpoint_name, N_EPOCHS, myinput, K1, K2, from_logits=False, mask_weights=filter_mask)
    
    return X, LON, LAT, vae, Z_DIM, N_EPOCHS, INITIAL_EPOCH, BATCH_SIZE, LEARNING_RATE, checkpoint_path, checkpoint_name, myinput, history

if __name__ == '__main__': # we do this so that we can then load this file as a module in reconstruction.py
    print("==Checking GPU==")
    import tensorflow as tf
    tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )

    print("==Checking CUDA==")
    tf.test.is_built_with_cuda()

    start = time.time()
    X, LON, LAT, vae, Z_DIM, N_EPOCHS, INITIAL_EPOCH, BATCH_SIZE, LEARNING_RATE, checkpoint_path, checkpoint_name, myinput, history = PrepareDataAndVAE()
    print("np.max(X) = ", np.max(X),"np.min(X) = ", np.min(X))
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

    vae.compile(optimizer=tf.keras.optimizers.Adam(lr = LEARNING_RATE))
    #vae.summary()

    print("==fit ==")
    print("X[X.shape[0]//10:,...].shape = ", X[X.shape[0]//10:,...].shape)
    my_history = vae.fit(X[X.shape[0]//10:,...], epochs=N_EPOCHS, initial_epoch=INITIAL_EPOCH, batch_size=BATCH_SIZE, shuffle=True, callbacks=[cp_callback]) # train on the last 9 folds
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
