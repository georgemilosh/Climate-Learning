# Importation des librairies

import tensorflow as tf
import numpy as np
#tf.enable_eager_execution() # This command is deprecated, use the one below
#tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, Dropout, SpatialDropout2D, Input, concatenate, Softmax, Reshape, Add, Conv2DTranspose, LeakyReLU # Vallerian's approach
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import datasets, layers, models 
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle



def Custom_BCE(y_true,y_pred):
    '''
        binary cross entropy
    '''
    p1 = y_true * tf.math.log(  tf.clip_by_value( y_pred , tf.keras.backend.epsilon() , 1 - tf.keras.backend.epsilon() ) + tf.keras.backend.epsilon() )
    p2 = ( 1 - y_true ) * tf.math.log( 1 -  tf.clip_by_value( y_pred , tf.keras.backend.epsilon() , 1 - tf.keras.backend.epsilon() ) + tf.keras.backend.epsilon() )
    return -tf.reduce_mean( p1 + p2 )

#####################################################
########### PCA Autoencoder  ################        
##################################################### 

class PCAencoder(PCA):
    """_summary_

    Args:
        PCA (_type_): _description_
    """
    
    def predict(self,*args,**kwargs):
        _X = self.transform(args[0].reshape(args[0].shape[0],-1))
        return _X, _X, _X # PCA expects the input of type fit(X) such that X is 2 dimensional and encoder generally has three outputs that we will set to the same number. This is done to be consistent with the functionality of class VAE(tf.keras.Model)
    
    def summary(self):
        print(f'We are computing PCA')

class PCAer:
    """_summary_
        Essentially decorator class that keeps the inputs and outputs maximally similar to autoencoder so that we could using the same routines
    """
    def __init__(self, *args, k1=1, k2=1, from_logits=False, field_weights=None, 
            lat_0=None, lat_1=None, lon_0=None, lon_1=None, coef_out=1, coef_in=0, coef_class=0, loss_type=None, class_type='stochastic', mask_area=None, Z_DIM=2, N_EPOCHS=2, print_summary=True, **kwargs):
        self.k1 = 'pca'
        self.k2 = 'pca'
        self.Z_DIM = Z_DIM
        self.encoder = PCAencoder(n_components=Z_DIM, svd_solver="randomized", whiten=True)
        self.shape = None # is created when calling method fit()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
        self.field_weights = field_weights
        self.lat_0 = lat_0 # parameters for weighting the reconstruction loss geographically
        self.lat_1 = lat_1 # parameters for weighting the reconstruction loss geographically
        self.lon_0 = lon_0 # parameters for weighting the reconstruction loss geographically
        self.lon_1 = lon_1 
        self.coef_out = coef_out # The full grid coefficient for reconstruction loss
        self.coef_in = coef_in # The inner grid coefficient (lat_0:lat_1, lon_0:lon_1) for reconstruction loss
        if loss_type is None: # Assuming Bernoulli variables
            self.rec_loss_form = self.bce
        else: # assuming Gaussian variables, for future compatibility write 'L2'
            self.rec_loss_form = tf.losses.MeanSquaredError()
    def fit(self,*args, **kwargs):
        print(f'{args[0].shape = }')
        result_fit = self.encoder.fit(args[0].reshape(args[0].shape[0],-1)) # PCA expects the input of type fit(X) such that X is 2 dimensional
        self.shape = self.shape = args[0].shape[0]
        print(f'{np.sum(self.encoder.explained_variance_ratio_) = }')
        return result_fit 
    def score(self,*args,**kwargs):
        return self.encoder.score(args[0].reshape(args[0].shape[0],-1))
    def save(self,folder):
        with open(folder+'/encoder.pkl', 'wb') as file_pi:
            pickle.dump(self.encoder, file_pi)
    def decoder(self,X):
        return self.encoder.inverse_transform(X).reshape(self.shape)
    def compute_loss(self,data,factor=None):
        if factor == None:
            factor = 18*data.shape[1]*data.shape[2] # 18 because we don't have access to k1 value any more and for consistency we choose 18
        reconstruction = self.decoder(data)
        return compute_loss(data,self.rec_loss_form,self.coef_out,self.coef_in,self.field_weights,self.lat_0,self.lat_1,self.lon_0,self.lon_1,factor)
    def summary(self):
        print(f'PCA with {self.Z_DIM} components')

#####################################################
########### Variational Autoencoder  ################        
##################################################### 


class ConstMul(tf.keras.layers.Layer):
    '''
        A layer of constant values. Modified from
            see https://stackoverflow.com/questions/61211101/how-to-multiply-a-fixed-weight-matrix-to-a-keras-layer-output

        Additional Parameters
        ----------
        const_val : 
            Either a scalar or a numpy array that contains the values that will multiply the input and add an intercept.
            
        Example Usage:
        ----------
            inputs = tf.keras.Input(shape=(2,2))
            outputs = ConstMul(np.array([[3,2],[0,0]]),0.5)(inputs)
            mymodel = tf.keras.Model(inputs, outputs)
            test = np.random.rand(2,2,2)
            mymodel(test)
        
        '''
    def __init__(self, const_a, const_b, *args, **kwargs):
        super(ConstMul, self).__init__(**kwargs)
        self.const_a = const_a
        self.const_b = const_b

    def call(self, inputs, **kwargs):
        return inputs * self.const_a + self.const_b

class Sampling(tf.keras.layers.Layer):  # Normal distribution sampling for the encoder output of the variational autoencoder
    '''
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    '''
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def compute_loss(data,reconstruction,rec_loss_form,coef_out,coef_in,field_weights,lat_0,lat_1,lon_0,lon_1,factor):
    if field_weights is None: # I am forced to use this awkward way to apply field weights since I cannot use the new version of tensorflow where axis parameter can be given
        reconstruction_loss = coef_out*factor*tf.reduce_mean([tf.reduce_mean(rec_loss_form(data[...,i][..., np.newaxis], reconstruction[...,i][..., np.newaxis])) for i in range(reconstruction.shape[3])] ) 
        if coef_in != 0: # This only matters if we want loss which depends on geographical areas
            reconstruction_loss += coef_in*factor*tf.reduce_mean([tf.reduce_mean(rec_loss_form(data[...,lat_0:lat_1,lon_0:lon_1,i][..., np.newaxis], reconstruction[...,lat_0:lat_1,lon_0:lon_1,i][..., np.newaxis])) for i in range(reconstruction.shape[3])] )
    else:  # The idea behind adding [..., np.newaxis] is to be able to use sample_weight in self.rec_loss_form on three dimensional input
        reconstruction_loss = coef_out*factor*tf.reduce_mean([field_weights[i]*tf.reduce_mean(rec_loss_form(data[...,i][..., np.newaxis], reconstruction[...,i][..., np.newaxis])) for i in range(reconstruction.shape[3])])
        if coef_in != 0: # This only matters if we want loss which depends on geographical areas
            reconstruction_loss +=  coef_in*factor*tf.reduce_mean([field_weights[i]*tf.reduce_mean(rec_loss_form(data[...,lat_0:lat_1,lon_0:lon_1,i][..., np.newaxis], reconstruction[...,lat_0:lat_1,lon_0:lon_1,i][..., np.newaxis])) for i in range(reconstruction.shape[3])])
    return reconstruction_loss
    
class VAE(tf.keras.Model): # Class of variational autoencoder
    '''
    Class Variation Autoencoder
        inherits : keras.models.Model

    Parameters
    ----------
    
    encoder : 
    decoder:
    k1 : int
        weight of reconstruction loss
    k2 : int
        weight of KL loss
    from_logits : bool
        Whether to use logits in binary cross entropy
    field_weights: list of floats or NaN
        if not None weights will be applied to the reconstructed field (last axis) when computing cross-entropy. 
        The idea is to prioritize some fields, for instance the fields that will be filtered (masked) in the decoder
        so that they contribute more. We know that soil moisture matters a lot for the heat waves, yet it is highly local
    lat_0: int
        latitude from which the loss is conditioned with coefficient coef_in
    lat_1: int
        latitude up to which the loss is conditioned  with coefficient coef_in
    lon_0: int
        longitude from which the loss is conditioned with coefficient coef_in
    lon_1: int
        longitude up to which the loss is conditioned  with coefficient coef_in
    coef_in: float
        coefficient of the reconstruction loss in the box (lat_0:lat1,lon0:lon1)
    coef_in: float
        coefficient of the reconstruction loss of the full box
    coef_class 
        coefficient of the classifier which compares the labels to the first component of the latent space
    '''
    def __init__(self, *args, k1=1, k2=1, from_logits=False, field_weights=None, 
            lat_0=None, lat_1=None, lon_0=None, lon_1=None, coef_out=1, coef_in=0, coef_class=0, loss_type=None, class_type='stochastic', mask_area=None, Z_DIM=2, N_EPOCHS=2, print_summary=True, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = args[0]
        self.decoder = args[1]
        if len(args) > 2:
            self.classifier = args[2]
        else:
            self.classifier = None
        print(f"{self.classifier = }")
        
        self.k1 = k1 # Reconstruction weight
        self.k2 = k2 # K-L divergence weight
        self.lat_0 = lat_0 # parameters for weighting the reconstruction loss geographically
        self.lat_1 = lat_1 # parameters for weighting the reconstruction loss geographically
        self.lon_0 = lon_0 # parameters for weighting the reconstruction loss geographically
        self.lon_1 = lon_1 
        self.coef_out = coef_out # The full grid coefficient for reconstruction loss
        self.coef_in = coef_in # The inner grid coefficient (lat_0:lat_1, lon_0:lon_1) for reconstruction loss
        self.coef_class = coef_class # Coefficient which sets the importance of classification comparison between the data[1] - assumed to be a label and z[0] output 
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        #self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss") # It is possible to define my own val metric, though tensorflow seems to take care of that
        #self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="val_reconstruction_loss")
        #self.val_kl_loss_tracker = tf.keras.metrics.Mean(name="val_kl_loss")
        #self.val_class_loss_tracker = tf.keras.metrics.Mean(name="val_class_loss")
        self.from_logits = from_logits
        self.class_type = class_type  # Decided if mean or the stochastic term is used for zz to condition based on classification
        self.encoder_input_shape = self.encoder.input.shape   # i.e. TensorShape([None, 24, 128, 3])
        self.field_weights = field_weights # Choose which fields the reconstruction loss cares about
        #self.mask_weights = mask_weights # Choose which grid points the reconstruction loss cares about  # This idea didn't work due to some errors
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
        if loss_type is None: # Assuming Bernoulli variables
            self.rec_loss_form = self.bce
        else: # assuming Gaussian variables, for future compatibility write 'L2'
            self.rec_loss_form = tf.losses.MeanSquaredError()

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.class_loss_tracker #, # It is possible to define my own val metric, though tensorflow seems to take care of that
            #self.val_total_loss_tracker,
            #self.val_reconstruction_loss_tracker,
            #self.val_kl_loss_tracker,
            #self.val_class_loss_tracker
        ]
    def call(self, inputs):
        _, _, z, zz = self.call_encoder_classifier(inputs)
        return self.decoder(z), zz

    def call_encoder_classifier(self,inputs):
        z_mean, z_log_var, z =  self.encoder(inputs)
        if self.classifier is not None:
            if self.class_type == 'stochastic':
                zz = self.classifier(z)
            else:
                zz = self.classifier(z_mean)
        else:
            zz = z
        return z_mean, z_log_var, z, zz
    
    def compute_losses(self, data):
        if isinstance(data, tuple): # If model.fit receives both X and Y
            print('both X and Y are provided')
            label = data[1]
            data = data[0]
        else:
            print('only X is provided')
        z_mean, z_log_var, z, zz = self.call_encoder_classifier(data)

        reconstruction = self.decoder(z)
        factor = self.k1*self.encoder_input_shape[1]*self.encoder_input_shape[2] # this factor is designed for consistency with the previous defintion of the loss
        # We should try tf.reduce_mean([0.1,0.1,0.4]*tf.cast([bce(data[...,i][..., np.newaxis], reconstruction[...,i][..., np.newaxis],sample_weight=np.ones((2,4,3))) for i in range(3)], dtype=np.float32))
        reconstruction_loss = compute_loss(data,reconstruction,self.rec_loss_form,self.coef_out,self.coef_in,self.field_weights,self.lat_0,self.lat_1,self.lon_0,self.lon_1,factor)
        # there is probably a more efficient way to do the line above: we are adding the full region plus a sub region.
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.k2*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        if self.classifier is not None: # the idea is to have the first coordinate approximate the committor
            class_loss = self.coef_class*self.bce(label,zz[:,0])
        else: # without labels we cannot say what is the class error
            class_loss = 0
        total_loss = reconstruction_loss + kl_loss + class_loss
        return total_loss, reconstruction_loss, kl_loss, class_loss
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss, class_loss = self.compute_losses(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss" : self.class_loss_tracker.result()
        }
    def test_step(self, data):
        val_total_loss, val_reconstruction_loss, val_kl_loss, val_class_loss = self.compute_losses(data)
        self.total_loss_tracker.update_state(val_total_loss) # # It is possible to define my own val metric, though tensorflow seems to take care of that
        self.reconstruction_loss_tracker.update_state(val_reconstruction_loss)
        self.kl_loss_tracker.update_state(val_kl_loss)
        self.class_loss_tracker.update_state(val_class_loss)
        return { # tensorflow will automatically attach 'val' to the names here
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss" : self.class_loss_tracker.result()
        }

def create_classifier(mysize, L2factor=None): # Logistic regression
    classifier_inputs = tf.keras.Input(shape=(mysize,), name ='classifier input')
    if L2factor is None:
        classifier_outputs = tf.keras.layers.Dense(1, kernel_regularizer=l2(L2factor))(classifier_inputs)
    else:
        classifier_outputs = tf.keras.layers.Dense(1)(classifier_inputs)
    return tf.keras.Model(classifier_inputs, classifier_outputs, name="classifier")

def build_encoder_skip(input_dim, output_dim, encoder_conv_filters = [32,64,64,64],
                                                encoder_conv_kernel_size = [3,3,3,3],
                                                encoder_conv_strides = [2,2,2,1],
                                                encoder_conv_padding = ["same","same","same","valid"], 
                                                encoder_conv_activation = ["LeakyRelu","LeakyRelu","LeakyRelu","LeakyRelu"],
                                                encoder_conv_skip = None,
                                                encoder_use_batch_norm=[False,False,False,False], 
                                                encoder_use_dropout=[0,0,0,0]):
    
    '''
    builds encoder with potential skip connections

    Parameters
    ----------
    encoder_conv_filters : list of int, optional
        number of filters corresponding to the convolutional layers
    encoder_conv_kernel_size : int, 2-tuple or list of ints or 2-tuples, optional
        If list must be of the same size of `conv_channels`
    encoder_conv_strides : int, 2-tuple or list of ints or 2-tuples, optional
        same as kernel_sizes
    encoder_conv_padding : str
        the type of padding used for each layer
    encoder_use_batch_norm : bool or list of bools, optional
        whether to add a BatchNormalization layer after each Conv2D layer
    encoder_conv_activations : str or list of str, optional
        activation functions after each convolutional layer
    encoder_use_dropout : float in [0,1] or list of floats in [0,1], optional
        dropout to be applied after the BatchNormalization layer. If 0 no dropout is applied
    encoder_conv_skip: list of lists to be converted to a dictionary
        creates a skip connection between two layers given by key and value entries in the dictionary. 
        If empty no skip connections are included. The skip connection will not work if 
        the dimensions of layers mismatch. For this convolutional architecture should be implemented in future
        
    Returns
    --------
    encoder_inputs:
    encoder_outputs:
    shape_before_flattening:
    encoder: 
    '''
    if encoder_conv_skip is not None:
        encoder_conv_skip_dict = dict(tuple(map(tuple, encoder_conv_skip)))
    else:
        encoder_conv_skip_dict = dict({})
    # Number of Conv layers
    n_layers = len(encoder_conv_filters)
    encoder_inputs = tf.keras.Input(shape=input_dim, name ='encoder input')
    
    x = []
    x.append(encoder_inputs)
    # Add convolutional layers
    for i in range(n_layers):
        # print(i, f"Conv2D, filters = {encoder_conv_filters[i]}, kernel_size = {encoder_conv_kernel_size[i]}, strides = {encoder_conv_strides[i]}, padding = {encoder_conv_padding[i]}")
        conv = Conv2D(filters = encoder_conv_filters[i], 
                kernel_size = encoder_conv_kernel_size[i],
                strides = encoder_conv_strides[i], 
                padding = encoder_conv_padding[i],
                name = 'encoder_conv_' + str(i))(x[i])

        if encoder_use_batch_norm[i]:
            conv = BatchNormalization()(conv)
            # print("conv = BatchNormalization()(conv)")
            
        if encoder_conv_activation[i] == 'LeakyRelu':
            actv = LeakyReLU()(conv)
            # print("actv = LeakyReLU()(conv)")
        else:
            actv = Activation(encoder_conv_activation[i])(conv)
            # print("actv = Activation(conv_activation[i])(conv)")

        if encoder_use_dropout[i]>0:
            actv = Dropout(rate=encoder_use_dropout[i])(actv)
            # print("actv = Dropout(rate=0.25)(actv)")
        
        if i in encoder_conv_skip_dict.values(): # The arrow of the skip connection end here
            # print('conv = keras.layers.add([conv, arrow_start])')
            actv = keras.layers.add([actv, arrow_start])
            if encoder_use_batch_norm:
                actv = BatchNormalization()(actv)
                # print("actv = BatchNormalization()(actv)")
        
        if i in encoder_conv_skip_dict.keys(): # The arrow of the skip connection starts here
            # print('arrow_start = actv')
            arrow_start = actv
        
        x.append(actv)
        

    shape_before_flattening = K.int_shape(x[-1])[1:] 
    print("shape_before_flattening = ", shape_before_flattening)
    x.append(tf.keras.layers.Flatten()(x[-1]))

    for i in range(len(encoder_conv_filters),len(encoder_conv_kernel_size)): # if lengths are the same there will be no extra layer
        dense = tf.keras.layers.Dense(encoder_conv_kernel_size[i], name="dense")(x[-1])
        if encoder_use_batch_norm[i]:
            dense = BatchNormalization()(dense)
        if encoder_conv_activation[i] == 'LeakyRelu':
            actv = LeakyReLU()(dense)
        else:
            actv = Activation(encoder_conv_activation[i])(dense)
        if encoder_use_dropout[i]>0:
            actv = Dropout(rate=encoder_use_dropout[i])(actv)
        x.append(actv)


    z_mean = tf.keras.layers.Dense(output_dim, name="z_mean")(x[-1])
    z_log_var = tf.keras.layers.Dense(output_dim, name="z_log_var")(x[-1])
    z = Sampling()([z_mean, z_log_var])
    encoder_outputs = [z_mean, z_log_var, z]
    encoder = tf.keras.Model(encoder_inputs, encoder_outputs, name="encoder")
    return encoder_inputs, encoder_outputs,  shape_before_flattening, encoder    




def build_decoder_skip(mask,input_dim, shape_before_flattening, decoder_conv_filters = [64,64,32,3],
                                        decoder_conv_kernel_size = [3,3,3,3],
                                        decoder_conv_strides = [1,2,2,2],
                                        decoder_conv_padding = ["valid","same","same","same"], 
                                        decoder_conv_activation = ["LeakyRelu","LeakyRelu","LeakyRelu","sigmoid"],
                                        decoder_conv_skip = None, 
                                        decoder_use_batch_norm = [False,False,False,False], 
                                        decoder_use_dropout = [0,0,0,0], usemask=False, reshape_activation="relu"):
    '''
    builds decoder

     Parameters
    ----------
    mask: np.ndarray
        A mask (filter) to be applied to the output of the decoder (provided that usemask=True)
    shape_before_flattening: tuple, int
        shape of the latent space before flattening. 
    decoder_conv_filters : list of int, optional
        number of filters corresponding to the convolutional layers
    decoder_conv_kernel_size : int, 2-tuple or list of ints or 2-tuples, optional
        If list must be of the same size of `conv_channels`
    decoder_conv_strides : int, 2-tuple or list of ints or 2-tuples, optional
        same as kernel_sizes
    decoder_conv_padding : str
        the type of padding used for each layer
    decoder_use_batch_norm : bool or list of bools, optional
        whether to add a BatchNormalization layer after each Conv2D layer
    decoder_conv_activations : str or list of str, optional
        activation functions after each convolutional layer
    decoder_use_dropout : float in [0,1] or list of floats in [0,1], optional
        dropout to be applied after the BatchNormalization layer. If 0 no dropout is applied
    decoder_conv_skip: list of lists to be converted to a dictionary
        creates a skip connection between two layers given by key and value entries in the dictionary. 
        If empty no skip connections are included. The skip connection will not work if 
        the dimensions of layers mismatch. For this convolutional architecture should be implemented in future
    usemask: bool
        If True then `mask` will be applied to the output, so that we can ignore the values set to 0
        
    Returns
    --------
    decoder_inputs:
    decoder_outputs:
    encoder: 
    '''
    if decoder_conv_skip is not None:
        decoder_conv_skip_dict = dict(tuple(map(tuple, decoder_conv_skip)))
    else:
        decoder_conv_skip_dict = dict({})
        
    # Number of Conv layers
    n_layers = len(decoder_conv_filters)
    decoder_inputs = tf.keras.Input(shape=input_dim)
    
    x = [decoder_inputs]

    for i in range(len(decoder_conv_filters),len(decoder_conv_kernel_size)): # if lengths are the same there will be no extra layer
        dense = tf.keras.layers.Dense(decoder_conv_kernel_size[i], name="dense")(x[-1])
        if decoder_use_batch_norm[i]:
            dense = BatchNormalization()(dense)
        if decoder_conv_activation[i] == 'LeakyRelu':
            actv = LeakyReLU()(dense)
        else:
            actv = Activation(decoder_conv_activation[i])(dense)
        if decoder_use_dropout[i]>0:
            actv = Dropout(rate=decoder_use_dropout[i])(actv)
        x.append(actv)
        
    x.append(tf.keras.layers.Dense(tf.math.reduce_prod(shape_before_flattening), activation=reshape_activation)(x[-1]))

    x[-1] = tf.keras.layers.Reshape(shape_before_flattening)(x[-1])
    
    preconvlen_x = len(x)
    #print(f"{preconvlen_x = }")
    
    # Add convolutional layers
    for i in range(n_layers):
        #print(i, f"Conv2D, filters = {decoder_conv_filters[i]}, kernel_size = {decoder_conv_kernel_size[i]}, strides = {decoder_conv_strides[i]}, padding = {decoder_conv_padding[i]}")
        conv = Conv2DTranspose(filters = decoder_conv_filters[i], 
                            kernel_size = decoder_conv_kernel_size[i],
                            strides = decoder_conv_strides[i], 
                            padding = decoder_conv_padding[i],
                            name = 'decoder_conv_' + str(i))(x[i+preconvlen_x-1])
        
        if decoder_use_batch_norm[i]:
            conv = BatchNormalization()(conv)
            #print("conv = BatchNormalization()(conv)")
            
        if decoder_conv_activation[i] == 'LeakyRelu':
            actv = LeakyReLU()(conv)
            #print("actv = LeakyReLU()(conv)")
        else:
            actv = Activation(decoder_conv_activation[i])(conv)
            #print("actv = Activation(conv_activation[i])(conv)")
            
        if decoder_use_dropout[i]:
            actv = Dropout(rate=decoder_use_dropout[i])(actv)
            #print("actv = Dropout(rate=0.25)(actv)")
        
        if i in decoder_conv_skip_dict.values(): # The arrow of the skip connection end here
            #print('conv = keras.layers.add([conv, arrow_start])')
            actv = keras.layers.add([actv, arrow_start])
            if decoder_use_batch_norm:
                actv = BatchNormalization()(actv)
                #print("actv = BatchNormalization()(actv)")
        
        if i in decoder_conv_skip_dict.keys(): # The arrow of the skip connection starts here
            #print('arrow_start = actv')
            arrow_start = actv
        
        x.append(actv)
    #print(f"{usemask = }")    
    if usemask: # a tensorflow array that will typically contain 
        decoder_outputs = ConstMul(mask,(~mask)*0.5)(x[-1])  # This will multiply the input by mask consisting of 0's (False) and 1's (True). Because the decoder is expected to reconstruct sigmoid function we add 0.5 where there were 0's
    else:
        decoder_outputs = x[-1]
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")
    
    return decoder_inputs, decoder_outputs, decoder  




#####################################################
#### Old unused routines to be suppressed soon ######        
##################################################### 

def plot_compare(model, images=None, add_noise=False): # Plot images generated by the autoencoder
    model.encoder(images)
    mean, logvar, z_sample = model.encoder(images)
    reconst_images = model.decoder(z_sample).numpy()
    #print("reconst_images.shape=",reconst_images.shape)
    n_to_show = images.shape[0]

    fig = plt.figure(figsize=(30, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')        
        sub.imshow(img)

    for i in range(n_to_show):
        #print(reconst_images[i].shape)
        img = reconst_images[i].squeeze()
        #print(img.shape)
        sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
        sub.axis('off')
        sub.imshow(img)

def vae_generate_images(vae,Z_DIM,n_to_show=10):
    reconst_images = vae.decoder.predict(np.random.normal(0,1,size=(n_to_show,Z_DIM)))

    fig = plt.figure(figsize=(30, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')        
        sub.imshow(img)

def build_encoder2(input_dim, output_dim, conv_filters, conv_kernel_size, conv_strides, conv_padding, conv_activation, use_batch_norm = False, use_dropout = False):
    # Number of Conv layers
    n_layers = len(conv_filters)
    encoder_inputs = tf.keras.Input(shape=input_dim, name ='encoder input')
    x = encoder_inputs
    # Add convolutional layers
    for i in range(n_layers):
        print(i)
        x = Conv2D(filters = conv_filters[i], 
                kernel_size = conv_kernel_size[i],
                strides = conv_strides[i], 
                padding = conv_padding[i],
                name = 'encoder_conv_' + str(i))(x)

        if use_batch_norm:
            x = BatchNormalization()(x)
        if conv_activation[i] == 'LeakyRelu':
            x = LeakyReLU()(x)
        else:
            x = Activation(conv_activation[i])(x)

        if use_dropout:
            x = Dropout(rate=0.25)(x)

    shape_before_flattening = K.int_shape(x)[1:] 
    print("shape_before_flattening = ", shape_before_flattening)
    x = tf.keras.layers.Flatten()(x)
    z_mean = tf.keras.layers.Dense(output_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(output_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder_outputs = [z_mean, z_log_var, z]
    encoder = tf.keras.Model(encoder_inputs, encoder_outputs, name="encoder")
    return encoder_inputs, encoder_outputs,  shape_before_flattening, encoder

def build_encoder(input_dim, output_dim, conv_filters, conv_kernel_size, conv_strides, conv_padding, use_batch_norm = False, use_dropout = False):
    # For backward compatibility we keep the old version that used to have prescribed activation
    conv_activation = ['LeakyRelu' for i in range(len(conv_filters))]
    
    return build_encoder2(input_dim, output_dim, conv_filters, conv_kernel_size, conv_strides, conv_padding, conv_activation, use_batch_norm, use_dropout)

def build_decoder2(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, conv_strides,conv_padding, conv_activation):
    # Number of Conv layers
    n_layers = len(conv_filters)
    decoder_inputs = tf.keras.Input(shape=input_dim)
    
    x = tf.keras.layers.Dense(tf.math.reduce_prod(shape_before_flattening), activation="relu")(decoder_inputs)
    x = tf.keras.layers.Reshape(shape_before_flattening)(x)

    # Add convolutional layers
    for i in range(n_layers):
        x = Conv2DTranspose(filters = conv_filters[i], 
                            kernel_size = conv_kernel_size[i],
                            strides = conv_strides[i], 
                            padding = conv_padding[i],
                            name = 'decoder_conv_' + str(i))(x)
        if conv_activation[i] == 'LeakyRelu':
            x = LeakyReLU()(x)
        else:
            x = Activation(conv_activation[i])(x)
        # Adding a sigmoid layer at the end to restrict the outputs 
        # between 0 and 1
        #if i < n_layers - 1:
        #    x = LeakyReLU()(x)
        #else:
        #    x = Activation('sigmoid')(x)
    decoder_outputs = x
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    return decoder_inputs, decoder_outputs, decoder  


def build_decoder(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, conv_strides,conv_padding):
    # For backward compatibility we keep the old version that used to have prescribed activation
    conv_activation = ['LeakyRelu' for i in range(len(conv_filters))]
    conv_activation[-1] = 'sigmoid'
    return build_decoder2(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, conv_strides,conv_padding, conv_activation) 


#### Custom Metrics ######

class UnbiasedMetric(keras.metrics.Metric):
    def __init__(self, name, undersampling_factor=1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.undersampling_factor = undersampling_factor
        self.r = tf.cast([0.5*np.log(undersampling_factor), -0.5*np.log(undersampling_factor)], tf.float32)

    
class MCCMetric(UnbiasedMetric): # This function is designed to produce confusion matrix during training each epoch
    def __init__(self, num_classes=2, threshold=None, undersampling_factor=1, name='MCC', **kwargs):
        '''
        Mathews correlation coefficient metric

        Parameters
        ----------
        num_classes : int, optional
            number of classes, by default 2
        threshold : float, optional
            If num_classes == 2 allows to choose a threshold over which to consider an event positive. If None the event is positive if it has probability higher than 0.5. By default None
        '''
        super().__init__(name=name, undersampling_factor=undersampling_factor, **kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.threshold = threshold
        if self.num_classes != 2:
            raise NotImplementedError('MCC works only with 2 classes')
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
    
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
            
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
    
    @tf.autograph.experimental.do_not_convert
    def result(self):
        #return self.process_confusion_matrix()
        cm=self.total_cm
        
        #return cm
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        TP = cm[1,1]
        MCC_den = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        MCC = (TP * TN - FP *FN)/tf.sqrt(MCC_den)
        return tf.cond(MCC_den == 0, lambda: tf.constant(0, dtype=tf.float32), lambda: MCC)
    
    def confusion_matrix(self,y_true, y_pred): # Make a confusion matrix
        if self.undersampling_factor > 1 or self.threshold is not None:
            y_pred = keras.layers.Softmax()(y_pred + self.r) # apply shift of logits and softmax to convert to balanced probabilities
        if self.threshold is None:
            y_pred=tf.argmax(y_pred,1)
        else:
            y_pred = tf.cast(y_pred[:,1] > self.threshold, tf.int8)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def fill_output(self,output):
        results=self.result()
        
class ConfusionMatrixMetric(UnbiasedMetric): # This function is designed to produce confusion matrix during training each epoch
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes, undersampling_factor=1, name='confusion_matrix', **kwargs):
        super().__init__(name=name, undersampling_factor=undersampling_factor, **kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
            
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
        
    def result(self):
        #return self.process_confusion_matrix()
        cm=self.total_cm
        return cm
    
    def confusion_matrix(self,y_true, y_pred): # Make a confusion matrix
        if self.undersampling_factor > 1:
            y_pred = keras.layers.Softmax()(y_pred + self.r) # apply shift of logits and softmax to convert to balanced probabilities
        
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def fill_output(self,output):
        results=self.result()

class BrierScoreMetric(UnbiasedMetric):
    def __init__(self, undersampling_factor=1, name='BrierScore', **kwargs):
        super().__init__(name=name, undersampling_factor=undersampling_factor, **kwargs)
        self.mse = keras.metrics.MeanSquaredError()
        self.my_metric = self.add_weight(name='BScore', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        _ = self.mse.update_state(y_true, keras.layers.Softmax()(y_pred + self.r))
        self.my_metric.assign(self.mse.result())

    def result(self):
        return self.my_metric

class UnbiasedCrossentropyMetric(UnbiasedMetric):

  def __init__(self, undersampling_factor=1, name='UnbiasedCrossentropy', **kwargs):
    super().__init__(name=name, undersampling_factor=undersampling_factor, **kwargs)
    self.my_metric = self.add_weight(name='CLoss', initializer='zeros')
    self.m = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)

  def update_state(self, y_true, y_pred, sample_weight=None):
    #_ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    _ = self.m.update_state(y_true, y_pred+self.r) # the idea is to add the weight factor inside the logit so that we effectively change the probabilities
    self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric


# same as above but less elegant
class CustomLoss(tf.keras.metrics.Metric):

  def __init__(self, r, name='CustomLoss', **kwargs):
    super().__init__(name=name, **kwargs)
    self.my_metric = self.add_weight(name='CLoss', initializer='zeros')
    self.r = r # undersampling_factor array (we expect the input as tf.cast(-0.5*np.log(undersampling_factor), 0.5*np.log(undersampling_factor))
    self.m = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)


  def update_state(self, y_true, y_pred, sample_weight=None):
    #_ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    _ = self.m.update_state(y_true, y_pred+self.r) # the idea is to add the weight factor inside the logit so that we effectively change the probabilities
    self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric

  """ 
  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.my_metric.assign(0.) 
  """

class UnbiasedCrossentropyLoss(keras.losses.SparseCategoricalCrossentropy):
    '''
    This is the same as the UnbiasedCrossentropyMetric but can be used as a loss
    '''
    def __init__(self, undersampling_factor=1, name='unbiased_crossentropy_loss'):
        super().__init__(from_logits=True, name=name)
        self.r = tf.cast([0.5*np.log(undersampling_factor), -0.5*np.log(undersampling_factor)], tf.float32)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred + self.r)

class MyMetrics_layer(tf.keras.metrics.Metric):

  def __init__(self, name='MyMetrics_layer', **kwargs):
    super(MyMetrics_layer, self).__init__(name=name, **kwargs)
    self.my_metric = self.add_weight(name='my_metric1', initializer='zeros')
    self.m = tf.keras.metrics.SparseCategoricalAccuracy()


  def update_state(self, y_true, y_pred, sample_weight=None):
    _ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.my_metric.assign(0.)
"""
class CustomLoss(tf.keras.metrics.Metric):

  def __init__(self, name='custom_loss', **kwargs):
    super(CustomLoss, self).__init__(name=name, **kwargs)
    #self.custom_loss = self.add_weight(name='closs', initializer='zeros')
    self.scce=tf.keras.losses.SparseCategoricalCrossentropy()#from_logits=True), #If the predicted labels are not converted to a probability distribution by the last layer of the model (using sigmoid or softmax activation functions), we need to inform these t

  def update_state(self, y_true, y_pred, sample_weight=None):
    update_state_out = self.scce(y_true,y_pred)
    update_state_out = y_true.shape
    self.custom_loss.assign_add(update_state_out)

  def result(self):
    #return self.custom_loss
    return self.scce(y_true,y_pred)
"""
class MySequential(keras.Sequential): # Here we design 
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        output={m.name: m.result() for m in self.metrics[:-1]}
        #if 'confusion_matrix_metric' in self.metrics_names:
        #    self.metrics[-1].fill_output(output)
        return output
    
    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x, y = data

        y_pred = self(x, training=False)  # Forward pass
        # Compute the loss value.
        # The loss function is configured in `compile()`.
        loss = self.compiled_loss(
            y,
            y_pred,
            regularization_losses=self.losses,
        )

        self.compiled_metrics.update_state(y, y_pred)
        output={m.name: m.result() for m in self.metrics[:-1]}
        #if 'confusion_matrix_metric' in self.metrics_names:
        #    self.metrics[-1].fill_output(output)    
        return output
    
class MyModel(keras.Sequential): # Here we design 
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        output={m.name: m.result() for m in self.metrics[:-1]}
        #if 'confusion_matrix_metric' in self.metrics_names:
        #    self.metrics[-1].fill_output(output)
        return output
    
    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x, y = data

        y_pred = self(x, training=False)  # Forward pass
        # Compute the loss value.
        # The loss function is configured in `compile()`.
        loss = self.compiled_loss(
            y,
            y_pred,
            regularization_losses=self.losses,
        )

        self.compiled_metrics.update_state(y, y_pred)
        output={m.name: m.result() for m in self.metrics[:-1]}
        #if 'confusion_matrix_metric' in self.metrics_names:
        #    self.metrics[-1].fill_output(output)    
        return output
    
    
def my_create_logistic_model2(factor, my_input_shape, regularization='none'):  # This is the leftover from the old routine
    if regularization == 'none':
        model = MySequential([
            tf.keras.layers.Flatten(input_shape=my_input_shape),     # if the model has a tensor input
            tf.keras.layers.Dense(2)])
    elif regularization == 'l2':
        model = MySequential([
            tf.keras.layers.Flatten(input_shape=my_input_shape),
            tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l2(factor))])
    else:
        model = MySequential([
            tf.keras.layers.Flatten(input_shape=my_input_shape),
            tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l1(factor))])
    return model

def my_create_logistic_model(factor, my_input_shape, regularization='none'):  # This is the type of the logistic model we use in training
    if type(my_input_shape) == tuple: # use a rectangular input that needs to be flattened
        layer1 = tf.keras.layers.Flatten(input_shape=my_input_shape)
    else: # the input is already flat (the choice for the input shape is historical)
        layer1 = tf.keras.layers.Input(shape=(my_input_shape,))
    if regularization == 'none':
        layer2 = tf.keras.layers.Dense(2)
    elif regularization == 'l2':
        layer2 = tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l2(factor))
    else:
        layer2 = tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l1(factor))
    model = MySequential([layer1, layer2])
    
    return model

    inputs = keras.Input(shape=model_input_dim) # Create a standard model (alternatively we can use our function my_create_logistic_model which builds a custom sequential model
    x = tf.keras.layers.Flatten(input_shape=model_input_dim)(inputs)
    outputs = tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l2(regularizer_coef))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)   # alternatively we can write:     model = MyModel(inputs=inputs, outputs=outputs)


def create_Logistic(model_input_dim, regularizer='l2', regularizer_coef=1e-9):
    inputs = keras.Input(shape=model_input_dim) # Create a standard model (alternatively we can use our function my_create_logistic_model which builds a custom sequential model
    x = tf.keras.layers.Flatten(input_shape=model_input_dim)(inputs)
    if regularizer == 'none':
        outputs = tf.keras.layers.Dense(2)(x)
    elif regularizer == 'l2':
        outputs = tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l2(regularizer_coef))(x)
    else:
        outputs = tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l1(regularizer_coef))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)   # alternatively we can write:     model = MyModel(inputs=inputs, outputs=outputs)
    return model

def committor_predictor_correct_sigmoid(model_input_dim):
    #global loss_list
    model = Sequential()
    model.add(Conv2D(32, (12,12), input_shape=model_input_dim))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (12,12)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (9,9)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (9,9)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model

def committor_predictor_correct(model_input_dim):
    #global loss_list
    model = Sequential()
    model.add(Conv2D(32, (12,12), input_shape=model_input_dim))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (12,12)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (9,9)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (9,9)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2))
    return model


def committor_predictor_shallow(model_input_dim):
    model = Sequential()
    model.add(Conv2D(32, (16,16), input_shape=model_input_dim))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Conv2D(64, (16,16)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    model.add(MaxPooling2D(2,2))
    
    model.add(Conv2D(128, (13,13)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2))
    return model

def minimal_CNN(model_input_dim): # This CNN I took from https://www.tensorflow.org/tutorials/images/cnn
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=model_input_dim))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
    return model

def small_CNN(model_input_dim): # This CNN I took from https://www.tensorflow.org/tutorials/images/cnn
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=model_input_dim))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
    return model

def make_ds(features, labels, BUFFER_SIZE=[]): # if you're using tf.data the easiest way to produce balanced examples is to start with a positive and a negative dataset, and merge them
    if BUFFER_SIZE == []:
        BUFFER_SIZE=features.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds


class History_trained_model(object): # This function can be used to open history of a trained model
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params

def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    return - tf.math.log(1. / x - 1.)/2

def probability_model(inputs,input_model): # This function is used to apply softmax to the output of the neural network
    x = input_model(inputs)
    outputs = layers.Softmax()(x)
    return keras.Model(inputs, outputs)


def ModelToProb(X,X_test,model): # compute probabilities based on the model and the input X_test
    if isinstance(X, list): # In the new system X consists of lists (useful for fused or combined CNNs)
        inputs = []
        for myindex in range(len(X)): # preparing inputs for the probability softmax
            model_input_dim = X[myindex].shape[1:]
            inputs.append(layers.Input(shape=model_input_dim))
        my_probability_model = probability_model(inputs,model)
    else:
        my_probability_model=(tf.keras.Sequential([ # softmax output to make a prediction
              model,
              tf.keras.layers.Softmax()
            ]))
    Y_pred = model.predict(X_test)
    Y_pred_prob = my_probability_model.predict(X_test)
    return Y_pred, Y_pred_prob
    
