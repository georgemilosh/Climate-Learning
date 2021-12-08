# George Miloshevich 2021
# inspired by https://keras.io/examples/generative/vae/
# merged with (and upgraded to tensorflow 2) https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8

import time
import os, sys
from glob import glob
import pickle
import shutil
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set
import numpy as np
sys.path.insert(1, '../ERA')
import ERA_Fields as ef # general routines
import TF_Fields as tff # tensorflow routines 
import itertools 


def PrepareParameters(creation):
    print("==Preparing Parameters==")
    WEIGHTS_FOLDER = './models/'
    
    RESCALE_TYPE = 'normalize'#'rescale' #'nomralize'
    Z_DIM = 6#2 #16 #64 #200 # Dimension of the latent vector (z)
    BATCH_SIZE = 128#512
    LEARNING_RATE = 1e-3#5e-4# 1e-3#5e-6
    N_EPOCHS = 20#600#200
    NUM_IMAGES = 4000 # number of years that variational autoencoder sees
    K1 = 1#100
    K2 = 1
    
    data_path='../../gmiloshe/PLASIM/'
    
    Model = 'Plasim'
    
    lon_start = 0
    lon_end = 128
    lat_start = 0 # latitudes start from 90 degrees North Pole
    lat_end = 24
    Months1 = [0, 0, 0, 0, 0, 0, 30, 30, 30, 30, 30, 0, 0, 0] 
    Tot_Mon1 = list(itertools.accumulate(Months1))
    checkpoint_name = WEIGHTS_FOLDER+Model+'_'+RESCALE_TYPE+'_k1_'+str(K1)+'_k2_'+str(K2)+'_LR_'+str(LEARNING_RATE)+'_ZDIM_'+str(Z_DIM)
    return WEIGHTS_FOLDER, RESCALE_TYPE, Z_DIM, BATCH_SIZE, LEARNING_RATE, N_EPOCHS, NUM_IMAGES, K1, K2, checkpoint_name, data_path, Model, lon_start, lon_end, lat_start, lat_end, Tot_Mon1
    
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
        
    return myinput

def LoadData(creation, Model, lat_start, lat_end, lon_start, lon_end, NUM_IMAGES):
    print("==Reading data==")

    ###### THIS FILEDS ARE ANOMALIES!
    zg500 = ef.Plasim_Field('zg','ANO_LONG_zg500','500 mbar Geopotential', Model, lat_start, lat_end, lon_start, lon_end,'single','')
    
    if creation == None:
        zg500.years = NUM_IMAGES  
    else: # if we are running this to plot reconstruction rather than for training we don't need to load all the images which takes too much time
        zg500.years = NUM_IMAGES//10
    
    zg500.load_field('/local/gmiloshe/PLASIM/Data_Plasim_LONG/')
    X = zg500.var.reshape(zg500.var.shape[0]*zg500.var.shape[1],zg500.var.shape[2],zg500.var.shape[3],1)

    INPUT_DIM = X.shape[1:]  # Image dimension
    print("X.shape = ", X.shape, "np.mean(X) = ", np.mean(X), " ,np.std(X) = ", np.std(X))
    
    return X, INPUT_DIM

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
            maxX = np.max(X)
            minX = np.min(X)
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
                                                conv_padding = ["same","same","same","valid"], use_batch_norm=False, use_dropout=False)
    encoder.summary()
    print("==Building decoder==")      
    # Decoder
    decoder_input, decoder_output, decoder = tff.build_decoder(input_dim = Z_DIM,  
                                        shape_before_flattening = shape_before_flattening,
                                        conv_filters = [64,64,32,1],
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

def PrepareDataAndVAE(creation=None):
    WEIGHTS_FOLDER, RESCALE_TYPE, Z_DIM, BATCH_SIZE, LEARNING_RATE, N_EPOCHS, NUM_IMAGES, K1, K2, checkpoint_name, data_path, Model, lon_start, lon_end, lat_start, lat_end, Tot_Mon1 = PrepareParameters(creation)
    
    myinput = CreateFolder(creation,checkpoint_name)
    
    X, INPUT_DIM = LoadData(creation, Model, lat_start, lat_end, lon_start, lon_end, NUM_IMAGES)
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
