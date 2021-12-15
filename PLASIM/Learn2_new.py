# George Miloshevich 2021
# Train a neural network

# Import librairies
import os as os
from pathlib import Path
import sys
import warnings
import time
import shutil
import gc
from numpy.random.mtrand import permutation
import psutil
import numpy as np

path_to_ERA = Path(__file__).resolve().parent.parent / 'ERA' # when absolute path, so you can run the script from another folder (outside plasim)
sys.path.insert(1, str(path_to_ERA))
# sys.path.insert(1, '../ERA/')
import ERA_Fields as ef # general routines
import TF_Fields as tff # tensorflow routines 

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from operator import mul
from functools import reduce

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

########## NEURAL NETWORK DEFINITION ###########

def custom_CNN(model_input_dim): # This CNN I took from https://www.tensorflow.org/tutorials/images/cnn
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=model_input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.SpatialDropout2D(0.2))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.SpatialDropout2D(0.2))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.SpatialDropout2D(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2))
    return model


def probability_model(inputs,input_model): # This function is used to apply softmax to the output of the neural network
    x = input_model(inputs)
    outputs = layers.Softmax()(x)
    return keras.Model(inputs, outputs)



########## COPY SOURCE FILES #########

def move_to_folder(folder):
    '''
    Copies this file and its dependencies to a given folder.
    '''
    folder = Path(folder).resolve()
    ERA_folder = folder / 'ERA'

    if os.path.exists(ERA_folder):
        raise FileExistsError(f'Cannot copy scripts to {folder}: you already have some there')
    ERA_folder.mkdir(parents=True,exist_ok=True)

    # copy this file
    path_to_here = Path(__file__).resolve() # path to this file
    shutil.copy(path_to_here, folder)

    # copy other files in the same directory as this one
    path_to_here = path_to_here.parent
    shutil.copy(path_to_here / 'config', folder)

    # copy files in ../ERA/
    path_to_here = path_to_here.parent / 'ERA'
    shutil.copy(path_to_here / 'cartopy_plots.py', ERA_folder)
    shutil.copy(path_to_here / 'ERA_Fields.py', ERA_folder)
    shutil.copy(path_to_here / 'TF_fields.py', ERA_folder)

    # copy additional files
    # History.py
    # Metrics.py
    # Recalc_Tau_Metrics.py
    # Recalc_History.py

    print(f'Now you can go to {folder} and run the learning from there')
    
    

########## DATA PREPROCESSING ##############

fields_infos = {
    't2m': {
        'name': 'tas',
        'filename_suffix': 'tas',
        'label': 'Temperature',
    },
    'mrso': {
        'name': 'mrso',
        'filename_suffix': 'mrso',
        'label': 'Soil Moisture',
    },
}

for h in [200,300,500,850]:
    fields_infos[f'zg{h}'] = {
        'name': 'zg',
        'filename_suffix': f'zg{h}',
        'label': f'{h} mbar Geopotential',
    }


def load_data(dataset_years=8000, year_list=None, sampling='', Model='Plasim', area='France', filter_area='France',
              lon_start=0, lon_end=128, lat_start=0, lat_end=22, mylocal='/local/gmiloshe/PLASIM/',fields=['t2m','zg500','mrso_filtered']):
    '''
    Loads the data.

    Parameters:
    -----------
        dataset_years: number of years of te dataset, 8000 or 1000
        year_list: list of years to load from the dataset
        sampling: '' (dayly) or '3hrs'
        Model: 'Plasim', 'CESM', ...
        area: region of interest, e.g. 'France'
        filter_area: area over which to keep filtered fields
        lon_start, lon_end, lat_start, lat_end: longitude and latitude extremes of the data expressed in indices (model specific)
        mylocal: path the the data storage. For speed it is better if it is a local path.
        fields: list of field to be loaded. Add '_filtered' to the name to have the velues of the field outside `area` set to zero.

    Returns:
    --------
        _fields: dictionary of ERA_Fields.Plasim_Field objects


    TO IMPROVE:
        possibility to load less years from a dataset
    '''

    if area != filter_area:
        warnings.warn(f'Fields will be filtered on a different area ({filter_area}) than the region of interest ({area})')

    if dataset_years == 1000:
        dataset_suffix = ''
    elif dataset_years == 8000:
        dataset_suffix = '_LONG'
    else:
        raise ValueError('Invalid number of dataset years')
   

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
        field.load_data(mylocal+file_suffix, year_list=year_list)
        # Set area integral
        field.abs_area_int, field.ano_area_int = field.Set_area_integral(area,mask,'Postproc')
        # filter
        if do_filter: # set to zero all values outside `filter_area`
            filter_mask = ef.create_mask(Model, filter_area, field.var, axes='last 2', return_full_mask=True)
            field.var *= filter_mask

        _fields[field_name] = field  
    
    return _fields


def assign_labels(field, time_start, time_end, T=14, percent=5, threshold=None):
    '''
    Given a field of anomalies it computes the `T` days forward convolution of the integrated anomaly and assigns label 1 to anomalies above a given `threshold`.
    If `threshold` is not provided, then it is computed from `percent`, namely to identify the `percent` most extreme anomalies.

    Returns:
    --------
        labels: 2D array with shape (years, days) and values 0 or 1
    '''
    A, A_flattened, threshold =  field.ComputeTimeAverage(time_start, time_end, T=T, percent=percent, threshold=threshold)[:3]
    return np.array(A >= threshold, dtype=int)

def make_X(fields, time_start, time_end, T=14, tau=0):
    '''
    Cuts the fields in time and stacks them. The original fields are not modified

    Returns:
    --------
        X: array with shape (years, days, lat, lon, field)
    '''
    # stack the fields
    X = np.array([field.var[:, time_start+tau:time_end+tau-T+1, ...] for field in fields.values()])
    # now transpose the array so the field index becomes the last
    X = X.transpose(*range(1,len(X.shape)), 0)
    return X

def roll_X(X, axis='lon', steps=0):
    '''
    Rolls `X` along a given axis. useful for example for moving France away from the Greenwich meridian

    Parameters:
    -----------
        X: array with shape (years, days, lat, lon, field)
        axis: 'year' (or 'y'), 'day' (or 'd'), 'lat', 'lon', 'field' (or 'f')
        steps: number of gridsteps to roll
            a positive value for 'steps' means that the elements of the array are moved forward in it, e.g. `steps` = 1 means that the old first element is now in the second place
            This means that for every axis a positive value of `steps` yields a shift of the array
            'year', 'day' : forward in time
            'lat' : southward
            'lon' : eastward
            'field' : forward in the numbering of the fields
    '''
    if steps == 0:
        return X
    if axis.startswith('y'):
        axis = 0
    elif axis.startswith('d'):
        axis = 1
    elif axis == 'lat':
        axis = 2
    elif axis == 'lon':
        axis = 3
    elif axis.startswith('f'):
        axis = 4
    else:
        raise ValueError(f'Unknown valur for axis: {axis}')
    return np.roll(X,steps,axis=axis)

####### MIXING ########

def invert_permutation(permutation):
    '''
    Inverts a permutation.
    e.g.:
        a = np.array([3,4,2,5])
        p = np.random.permutation(np.arange(4))
        a_permuted = a[p]
        p_inverse = invert_permutation(p)

        `a` and `a_permuted[p_inverse]` will be equal

    Parameters:
    -----------
        permutation: 1D array that must be a permutation of an array of the kind `np.arange(n)` with `n` integer
    '''
    return np.argsort(permutation)

def compose_permutations(permutations):
    '''
    Composes a series of permutations
    e.g.:
        a = np.array([3,4,2,5])
        p1 = np.random.permutation(np.arange(4))
        p2 = np.random.permutation(np.arange(4))
        p_composed = compose_permutations([p1,p2])
        a_permuted1 = a[p1]
        a_permuted2 = a_permuted1[p2]
        a_permuted_c = a[p_composed]

        `a_permuted_c` and `a_permuted2` will be equal

    Parameters:
    -----------
        permutations: list of 1D arrays that must be a permutation of an array of the kind `np.arange(n)` with `n` integer and the same for every permutation
    '''
    l = len(permutations[0])
    for p in permutations[1:]:
        if len(p) != l:
            raise ValueError('All permutations must have the same length')
    ps = permutations[::-1]
    p = ps[0]
    for _p in ps[1:]:
        p = _p[p]
    return p
    

def shuffle_years(X, permutation=None, seed=0, apply=False):
    '''
    Permutes `X` along the first axis

    Parameters:
    -----------
        X: array with the data to permute
        permutation: None or 1D array that must be a permutation of an array of `np.arange(X.shape[0])`
        seed: int, if `permutation` is None, then it is computed using the provided seed.
        apply: bool, if True the function returns the permuted data, otherwise the permutation is returned
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

def balance_folds(weights, nfolds=10):
    '''
    Returns a permutation that, once applied to `weights` would make the consecutive `nfolds` pieces of equal length have their sum the most similar to each other.

    Parameters:
    -----------
        weights: 1D array
        nfolds: int, must be a divisor of `len(weights)`

    Returns:
    --------
        permutation: permutation of `np.arange(len(weights))`
    '''
    class Fold():
        def __init__(self, target_length, target_sum, name=None):
            self.indexs = []
            self.length = target_length
            self.target_sum = target_sum
            self.sum = 0
            self.hunger = np.infty
            self.name = name
        
        def add(self, a):
            self.indexs.append(a[1])
            self.sum += a[0]
            if self.length == len(self.indexs):
                print(f'fold {self.name} done!')
                return True
            self.hunger = (self.target_sum - self.sum)/(self.length - len(self.indexs))
            return False

    fold_length = len(weights)//nfolds
    if len(weights) != fold_length*nfolds:
        raise ValueError(f'Cannot make {nfolds} folds of equal lenght out of {len(weights)} years of data')
    target_sum = np.sum(weights)/nfolds


    folds = [Fold(fold_length, target_sum, name=i) for i in range(nfolds)]
    permutation = []

    ws = [(a,i) for i,a in enumerate(weights)]
    ws.sort()
    ws = ws[::-1]
        
    for a in ws:
        # determine the hungriest fold
        j = np.argmax([f.hunger for f in folds])
        if folds[j].add(a):
            f = folds.pop(j)
            permutation += f.indexs

    if len(permutation) != len(weights):
        raise ValueError('balance_folds: Something went wrong during balancing: either missing or duplicated data')

    return permutation

    
    

def Prepare(creation = None):  # if we do not specify creation it automacially creates new folder. If we specify the creation, it should correspond to the folder we are running the file from

    percent = 1 # 5  # Percent of the anomalies that will be treated as heat waves 

    
    tau = 0 #-5  # lag
    usepipelines = False # This variable used to have the following meaning: if True => Dataset.from_tensor_slices will be used. This is a more advanced method but it takes more RAM and there is a possiblity for memory leaks when repeating training for cross-validation. Now it is used to pass extra parameters from the function PrepareData()
    fullmetrics = True # If True MCC and confusion matrix will be evaluated during training. This makes training slower!
    

    # If an integer >= 1 is chosen we simply undersample by this rate
    # If a float between 0 and 1 is chosen we select each state with the probability given by this float
    undersampling_factor = 20 # 1 #15 #10 #5 #1 #0.25 #0.15 #0.25    # How much we want to reduce our dataset
    oversampling_factor = 1 # oversampling_factor = 1 means that oversampling will not be performed
    thefield = 't2m' # Important: this is the field that is used to determine the extrema (important for undersampling) and therefore the label space
    BATCH_SIZE = 1024 # choose this for training so that the chance of encountering positive batches is nonnegligeable
    if tau < 0: # we plan to take advantage of transfer learning
        NUM_EPOCHS = 10
    else:
        NUM_EPOCHS = 40 #1000 #20 #200 #50 # number of epochs the training involves
    saveweightseveryblaepoch = 1 # If set to 0 the model will not save weights as it is being trained, otherwise this number will tell us how many epochs it waits until saving
    if saveweightseveryblaepoch > 0:
        ckpt = 'ckpt'
    else:
        cktp = ''
    

    checkpoint_name_root = f'{myscratch}training/stack_CNN_equalmixed_{ckpt}_{thefield}France__with_zg500_t2mmrsoFrance_{sampling}_22by128_u{undersampling_factor}o{oversampling_factor}_LONG{num_years}yrs__per_{percent}_tau_'
    checkpoint_name = checkpoint_name_root+str(tau) # current filename, we also need filename of the previous tau for transfer learning (previous tau is obtained by adding to tau an integer)
    checkpoint_name_previous = checkpoint_name_root+str(tau+5) # THIS HAS TO BE CHANGED IF THE STEP IN TAU IS DIFFERENT!!

    print("creation = ", creation)
    if creation == None: # If we are not running from the same directory (in other words, if we are running for the first time). Otherwise nothing needs to be copied. We just need to follow the same procedures to load the data as were done when the model was trained. Basically the parameter is not None (I choose the folder itself as a parameter always) if we need to load the model and the corresponding data to run some tests or make plots) The alrnative is to load the data by hand but then spatial care has to be made that the same operations are performed to the data, as the ones that were performed when the model was trained. 
        if os.path.exists(checkpoint_name): 
            print(f'folder {checkpoint_name} exists. Should I overwrite?')
            if input(" write Y to overwrite, else the execution will stop: ") != "Y":
                sys.exit("User has aborted the program")
        else: # Create the directory
            print('folder '+checkpoint_name+' created')
            os.mkdir(checkpoint_name)

        sys.stdout = Logger(checkpoint_name+'/logger.log')  # Keep a copy of print outputs there
        shutil.copy(__file__, checkpoint_name) # Copy this file to the directory of the training
        dest = shutil.copy(__file__, checkpoint_name+'/Funs.py')
        shutil.copy(myscratch+'../ERA/ERA_Fields.py', checkpoint_name)
        shutil.copy(myscratch+'../ERA/TF_Fields.py', checkpoint_name)
        shutil.copy(myscratch+'History.py', checkpoint_name)
        shutil.copy(myscratch+'Recalc_History.py', checkpoint_name)
        shutil.copy(myscratch+'Recalc_Tau_Metrics.py', checkpoint_name)
        shutil.copy(myscratch+'Metrics.py', checkpoint_name)
    # specify the precise input rectangle that will enter inside the neural network. These will be concatenated, which is useful when we need to include Greenwich meridian. 
    #           The procedure below will concatenate a rectangle with latitude range lat_from[0] - lat_to[0] and longtitude range lon_from[0] - lon_to[0] with
    #                       a rectangle with latitude range lat_from[1] - lat_to[1] and longitude range lon_from[1] - lon_from[1].
    #lat_from = [4,4]     # 18x42
    #lat_to   = [22,22]
    #lon_from = [101,0]
    #lon_to   = [128,15]
    lat_from =  [0,0]   # 22x128
    lat_to =    [22,22]
    lon_from =  [64, 0]
    lon_to =    [128, 64]


    print(f'{percent = }, {T = }, {Model = }, {area = }, {undersampling_factor = }, {lat_from = }, {lat_to = }, {lon_from = }, {lon_to = }, {thefield = }')
    
    

    del t2m.var
    gc.collect() # Garbage collector which removes some extra references to the object
    usepipelines = A_reshape, threshold, checkpoint_name_previous, tau
    undersampling_factor = [undersampling_factor, oversampling_factor]

                                
    return X, list_extremes, thefield, sampling, percent, usepipelines, undersampling_factor, new_mixing,  saveweightseveryblaepoch, NUM_EPOCHS, BATCH_SIZE, checkpoint_name, fullmetrics



######## TRAIN THE NETWORK #########

if __name__ == '__main__':
    print(f"====== running {__file__} ====== ")  
    print(f"{tf.__version__ = }")
    if int(tf.__version__[0]) < 2:
        print(f"{tf.test.is_gpu_available() = }")
    else:
        print(f"{tf.config.list_physical_devices('GPU') = }")

    start = time.time()



    X, list_extremes, thefield, sampling, percent, usepipelines, undersampling_factor, new_mixing,  saveweightseveryblaepoch, NUM_EPOCHS, BATCH_SIZE, checkpoint_name, fullmetrics = PrepareData() # The reason it is written as a function is because we want to re-use it when loading the data
    
    oversampling_factor = undersampling_factor[1]
    undersampling_factor = undersampling_factor[0]
                                
    print("full dimension of the data is X[0].shape = ", X[0].shape) # do the previous statement in steps so that first we get a list (I extract the necessary sizes)
    end = time.time()

    # Getting % usage of virtual_memory ( 3rd field)
    print(f'RAM memory used: {psutil.virtual_memory()[3]}')
    print(f'Reading time = {end - start}')
    start = time.time()

    mylabels = np.array(list_extremes)
    checkpoint_name_previous = usepipelines[2]
    tau = usepipelines[3]

    if tau < 0: # we can import previous weights
        # Here we insert analysis of the previous tau with the assessment of the ideal checkpoint
        history = np.load(checkpoint_name_previous+'/batch_'+str(0)+'_history.npy', allow_pickle=True).item()
        if ('val_CustomLoss' in history.keys()):
            print( "'val_CustomLoss' in history.keys()")
            historyCustom = []
            for i in range(10): # preemptively compute the optimal score
                historyCustom.append(np.load(checkpoint_name_previous+'/batch_'+str(i)+'_history.npy', allow_pickle=True).item()['val_CustomLoss'])
            historyCustom = np.mean(np.array(historyCustom),0)
            opt_checkpoint = np.argmin(historyCustom) # We will use optimal checkpoint in this case!
        else:
            print( "'val_CustomLoss' not in history.keys()")
            sys.exit("Aborting the program!")

    
    # do the training 10 times to with 10-fold cross validation
    my_MCC = np.zeros(10,)
    my_entropy = np.zeros(10,)
    my_skill = np.zeros(10,)
    my_BS = np.zeros(10,)
    my_WBS = np.zeros(10,)
    my_freq = np.zeros(10,)
    my_memory = []

    for i in range(10):
        print("===============================")
        print("cross validation i = ", str(i))
        test_indices, train_indices, train_true_labels_indices, train_false_labels_indices, filename_permutation = TrainTestSplitIndices(i,X, mylabels, 1, sampling, new_mixing, thefield, percent) # 1 implies undersampling_rate=1 indicating that we supress the manual undersampling
        print("# events in the train sample after TrainTestSplitIndices: ",  len(train_indices))
        print("original proportion of positive events in the train sample: ",  np.sum(mylabels[train_indices])/len(train_indices))
        
        oversampling_strategy = oversampling_factor/(100/percent-1)
        if oversampling_factor > 1:
            print("oversampling_strategy = ", oversampling_strategy )
            over = RandomOverSampler(random_state=42, sampling_strategy=oversampling_strategy) # first oversample the minority class to have 15 percent the number of examples of the majority class
            #over = SMOTEENN(random_state=42, sampling_strategy=oversampling_strategy) # first oversample the minority class to have 15 percent the number of examples of the majority class
        
        undersampling_strategy = undersampling_factor*oversampling_strategy
        print("undersampling_strategy = ", undersampling_strategy )
        under = RandomUnderSampler(random_state=42, sampling_strategy=undersampling_strategy)
                            
        if oversampling_factor > 1:
                steps = [('o', over), ('u', under)]
        else:
                steps = [('u', under)]
        pipeline = Pipeline(steps=steps) # The Pipeline can then be applied to a dataset, performing each transformation in turn and returning a final dataset with the accumulation of the transform applied to it, in this case oversampling followed by undersampling.
        # To make use of the in-built pipelines we need to transform the X into the required dimensions
        XTrain_indicesShape = X[train_indices].shape
        print("Original dimension of the train set is X[train_indices].shape = ", XTrain_indicesShape)
        X_train, Y_train = pipeline.fit_resample(X[train_indices].reshape(XTrain_indicesShape[0],XTrain_indicesShape[1]*XTrain_indicesShape[2]*XTrain_indicesShape[3]),  mylabels[train_indices])
        X_train = X_train.reshape(X_train.shape[0],XTrain_indicesShape[1],XTrain_indicesShape[2],XTrain_indicesShape[3])

        Y_test = mylabels[test_indices]
        neg = train_false_labels_indices.shape[0]
        pos = train_true_labels_indices.shape[0]
        
        print("====Dimensions of the data before entering the neural net===")
        print("dimension of the train set is X[train_indices].shape = ", X_train.shape)
        
        # normalize the data with pointwise mean and std
        X_mean = np.mean(X_train,0)
        X_std = np.std(X_train,0)
        
        print(f'{np.sum(X_std < 1e-5)/np.product(X_std.shape)*100}\% of the data have std below 1e-5')
        X_std[X_std==0] = 1 # If there is no variance we shouldn't divide by zero ### hmmm: this may create discontinuities

        X_test = (X[test_indices]-X_mean)/X_std
        Y_test = mylabels[test_indices]

        X_train = (X_train-X_mean)/X_std


         
        print("Y_train.shape = ", Y_train.shape)
        print("Y_test.shape = ", Y_test.shape)
         
        print(f"Train set: # of true labels = {np.sum(Y_train)}, # of false labels = {Y_train.shape[0] - np.sum(Y_train)}")
        print(f"Train set: effective sampling rate for rare events is {np.sum(Y_train)/Y_train.shape[0]}")
        
        np.save(f'{checkpoint_name}/batch_{i}_X_mean', X_mean) # this values must be saved if the neural network is to be tested again, by reloading some other data
        np.save(f'{checkpoint_name}/batch_{i}_X_std', X_std)
        
        if tau < 0: # engagge transfer learning
            print(f"opt_checkpoint: {opt_checkpoint} ,loading model: {checkpoint_name_previous}")
            model = (tf.keras.models.load_model(f'{checkpoint_name_previous}/batch_{i}', compile=False)) # if we just want to train

            nb_zeros_c = 4-len(str(opt_checkpoint))
            cp_checkpoint_name = '/cp-'+nb_zeros_c*'0'+str(opt_checkpoint)+'.ckpt'
            print(f'loading weights from {checkpoint_name_previous}/batch_{i}{cp_checkpoint_name}')
            model.load_weights(f'{checkpoint_name_previous}/batch_{i}{cp_checkpoint_name}')
        else:
            model_input_dim = X.shape[1:] 
            model = custom_CNN(model_input_dim) 
       
        tf_sampling = tf.cast([0.5*np.log(undersampling_factor), -0.5*np.log(undersampling_factor)], tf.float32)
        #print("model_input_dim = ",model_input_dim)
        model.summary()
        if fullmetrics:
            METRICS=['accuracy',MCCMetric(2),ConfusionMatrixMetric(2),CustomLoss(tf_sampling)]#tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)]#CustomLoss()]   # the last two make the code run longer but give precise discrete prediction benchmarks
        else:
            METRICS=['loss']
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 2e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #If the predicted labels are not converted to a probability distribution by the last layer of the model (using sigmoid or softmax activation functions), we need to inform these three Cross-Entropy functions by setting their from_logits = True.
            #One advantage of using sparse categorical cross-entropy is it saves storage in memory as well as time in computation because it simply uses a single integer for a class, rather than a whole one-hot vector. This works despite the fact that the neural network has an one-hot vector output  
            metrics=METRICS   # the last two make the code run longer but give precise discrete prediction benchmarks
        )
        # Create a callback that saves the model's weights every saveweightseveryblaepoch epochs
        checkpoint_path = checkpoint_name+'/batch_'+str(i)+"/cp-{epoch:04d}.ckpt"
        if saveweightseveryblaepoch > 0:
            print("cp_callback save option on")
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
        else:
            cp_callback=None

        model.save_weights(checkpoint_path.format(epoch=0))

        my_history=model.fit(X_train, Y_train, batch_size=BATCH_SIZE, validation_data=(X_test,Y_test), shuffle=True, callbacks=[cp_callback], epochs=NUM_EPOCHS,verbose=2, class_weight=None)

        model.save(f'{checkpoint_name}/batch_{i}')
        np.save(f'{checkpoint_name}/batch_{i}_history.npy',my_history.history)
        
        my_probability_model=(tf.keras.Sequential([ # softmax output to make a prediction
              model,
              tf.keras.layers.Softmax()
            ]))

        print("======================================")
        my_memory.append(psutil.virtual_memory())
        print('RAM memory:', my_memory[i][3])

        tf.keras.backend.clear_session()
        gc.collect() # Garbage collector which removes some extra references to the object

        # Getting % usage of virtual_memory ( 3rd field)

    np.save(f'{checkpoint_name}/RAM_stats.npy', my_memory)

    end = time.time()
    print(f'files saved in  {checkpoint_name}')
    print(f'Learning time = {end - start}')



