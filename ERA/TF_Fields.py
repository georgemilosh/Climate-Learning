# Importation des librairies

import tensorflow as tf
#tf.enable_eager_execution() # This command is deprecated, use the one below
#tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, Dropout, SpatialDropout2D, Input, concatenate, Softmax, Reshape, Add # Vallerian's approach
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import datasets, layers, models # from https://www.tensorflow.org/tutorials/images/cnn



    
    
class MCCMetric(tf.keras.metrics.Metric): # This function is designed to produce confusion matrix during training each epoch
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes, **kwargs):
        super(MCCMetric,self).__init__(name='MCC',**kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
            
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
        
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
        
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def fill_output(self,output):
        results=self.result()
        
class ConfusionMatrixMetric(tf.keras.metrics.Metric): # This function is designed to produce confusion matrix during training each epoch
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrixMetric,self).__init__(name='confusion_matrix',**kwargs) # handles base args (e.g., dtype)
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
        
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def fill_output(self,output):
        results=self.result()


class CustomLoss(tf.keras.metrics.Metric):

  def __init__(self, r, **kwargs):
    super(CustomLoss, self).__init__(name='CustomLoss', **kwargs)
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
