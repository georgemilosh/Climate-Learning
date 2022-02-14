# Importation des librairies

import tensorflow as tf
import numpy as np
#tf.enable_eager_execution() # This command is deprecated, use the one below
#tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, Dropout, SpatialDropout2D, Input, concatenate, Softmax, Reshape, Add, Conv2DTranspose, LeakyReLU # Vallerian's approach
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import datasets, layers, models # from https://www.tensorflow.org/tutorials/images/cnn
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

class Sampling(tf.keras.layers.Layer):  # Normal distribution sampling for the encoder output of the variational autoencoder
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class VAE(tf.keras.Model): # Class of variational autoencoder
    def __init__(self, encoder, decoder, k1=1, k2=1, from_logits=False, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.k1 = k1
        self.k2 = k2
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.from_logits = from_logits
        print("VAE: self.from_logits = ", self.from_logits)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def call(self, inputs):
        _, _, z =  self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = self.k1*tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction, from_logits=self.from_logits), axis=(1, 2) # -Y_n Log( P_n) - (1 - Y_n) Log( 1 - P_n) is the expression for binary entropy
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.k2*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def build_encoder_skip(input_dim, output_dim, conv_filters, conv_kernel_size, conv_strides, conv_padding, conv_activation, conv_skip, use_batch_norm = False, use_dropout = False):
    # Number of Conv layers
    n_layers = len(conv_filters)
    encoder_inputs = tf.keras.Input(shape=input_dim, name ='encoder input')
    
    x = []
    x.append(encoder_inputs)
    # Add convolutional layers
    for i in range(n_layers):
        print(i, f"Conv2D, filters = {conv_filters[i]}, kernel_size = {conv_kernel_size[i]}, strides = {conv_strides[i]}, padding = {conv_padding[i]}")
        conv = Conv2D(filters = conv_filters[i], 
                kernel_size = conv_kernel_size[i],
                strides = conv_strides[i], 
                padding = conv_padding[i],
                name = 'encoder_conv_' + str(i))(x[i])

        if use_batch_norm:
            conv = BatchNormalization()(conv)
            print("conv = BatchNormalization()(conv)")
            
        if conv_activation[i] == 'LeakyRelu':
            actv = LeakyReLU()(conv)
            print("actv = LeakyReLU()(conv)")
        else:
            actv = Activation(conv_activation[i])(conv)
            print("actv = Activation(conv_activation[i])(conv)")

        if use_dropout:
            actv = Dropout(rate=0.25)(actv)
            print("actv = Dropout(rate=0.25)(actv)")
        
        if i in conv_skip.values(): # The arrow of the skip connection end here
            print('conv = keras.layers.add([conv, arrow_start])')
            actv = keras.layers.add([actv, arrow_start])
            if use_batch_norm:
                actv = BatchNormalization()(actv)
                print("actv = BatchNormalization()(actv)")
        
        if i in conv_skip.keys(): # The arrow of the skip connection starts here
            print('arrow_start = actv')
            arrow_start = actv
        
        x.append(actv)
        

    shape_before_flattening = K.int_shape(x[-1])[1:] 
    print("shape_before_flattening = ", shape_before_flattening)
    x.append(tf.keras.layers.Flatten()(x[-1]))
    z_mean = tf.keras.layers.Dense(output_dim, name="z_mean")(x[-1])
    z_log_var = tf.keras.layers.Dense(output_dim, name="z_log_var")(x[-1])
    z = Sampling()([z_mean, z_log_var])
    encoder_outputs = [z_mean, z_log_var, z]
    encoder = tf.keras.Model(encoder_inputs, encoder_outputs, name="encoder")
    return encoder_inputs, encoder_outputs,  shape_before_flattening, encoder    
    
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
    encoder.summary()
    return encoder_inputs, encoder_outputs,  shape_before_flattening, encoder

def build_encoder(input_dim, output_dim, conv_filters, conv_kernel_size, conv_strides, conv_padding, use_batch_norm = False, use_dropout = False):
    # For backward compatibility we keep the old version that used to have prescribed activation
    conv_activation = ['LeakyRelu' for i in range(len(conv_filters))]
    
    return build_encoder2(input_dim, output_dim, conv_filters, conv_kernel_size, conv_strides, conv_padding, conv_activation, use_batch_norm, use_dropout)

def build_decoder_skip(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, conv_strides,conv_padding, conv_activation, conv_skip, use_batch_norm = False, use_dropout = False):
    # Number of Conv layers
    n_layers = len(conv_filters)
    decoder_inputs = tf.keras.Input(shape=input_dim)
    
    x = []
    
    x.append(tf.keras.layers.Dense(tf.math.reduce_prod(shape_before_flattening), activation="relu")(decoder_inputs))
    x[0] = tf.keras.layers.Reshape(shape_before_flattening)(x[0])
    
    
    # Add convolutional layers
    for i in range(n_layers):
        print(i, f"Conv2D, filters = {conv_filters[i]}, kernel_size = {conv_kernel_size[i]}, strides = {conv_strides[i]}, padding = {conv_padding[i]}")
        conv = Conv2DTranspose(filters = conv_filters[i], 
                            kernel_size = conv_kernel_size[i],
                            strides = conv_strides[i], 
                            padding = conv_padding[i],
                            name = 'decoder_conv_' + str(i))(x[i])
        
        if use_batch_norm:
            conv = BatchNormalization()(conv)
            print("conv = BatchNormalization()(conv)")
            
        if conv_activation[i] == 'LeakyRelu':
            actv = LeakyReLU()(conv)
            print("actv = LeakyReLU()(conv)")
        else:
            actv = Activation(conv_activation[i])(conv)
            print("actv = Activation(conv_activation[i])(conv)")
            
        if use_dropout:
            actv = Dropout(rate=0.25)(actv)
            print("actv = Dropout(rate=0.25)(actv)")
        
        if i in conv_skip.values(): # The arrow of the skip connection end here
            print('conv = keras.layers.add([conv, arrow_start])')
            actv = keras.layers.add([actv, arrow_start])
            if use_batch_norm:
                actv = BatchNormalization()(actv)
                print("actv = BatchNormalization()(actv)")
        
        if i in conv_skip.keys(): # The arrow of the skip connection starts here
            print('arrow_start = actv')
            arrow_start = actv
        
        x.append(actv)
        

    decoder_outputs = x[-1]
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder_inputs, decoder_outputs, decoder  

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
