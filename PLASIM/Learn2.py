# George Miloshevich 2021
# Train a neural network

# Importation des librairies
import os as os
import sys
sys.path.insert(1, '../ERA')
from ERA_Fields import* # general routines
from TF_Fields import* # tensorflow routines 
import time
import shutil
import gc
import psutil
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from operator import mul
from functools import reduce

########## NEURAL NETWORK DEFINITION ###########

def custom_CNN(model_input_dim): # This CNN I took from https://www.tensorflow.org/tutorials/images/cnn
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=model_input_dim))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.2))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(SpatialDropout2D(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2))
    return model


def probability_model(inputs,input_model): # This function is used to apply softmax to the output of the neural network
    x = input_model(inputs)
    outputs = layers.Softmax()(x)
    return keras.Model(inputs, outputs)



########## DATA PREPROCESSING ##############

def PrepareData(creation = None):  # if we do not specify creation it automacially creates new folder. If we specify the creation, it should correspond to the folder we are running the file from

    sampling='' #'3hrs' # This chooses whether we want say daily sampling or 3 hour one. Notice that the corresponding NetCDF files are kept in different places
    percent = 1 # 5  # Percent of the anomalies that will be treated as heat waves 

    timesperday = 8 # 3 hour long periods in case we choose this sampling
    T = 14
    if sampling == '3hrs':
        T *= timesperday

    tau = 0 #-5  # lag
    usepipelines = False # This variable used to have the following meaning: if True => Dataset.from_tensor_slices will be used. This is a more advanced method but it takes more RAM and there is a possiblity for memory leaks when repeating training for cross-validation. Now it is used to pass extra parameters from the function PrepareData()
    fullmetrics = True # If True MCC and confusion matrix will be evaluated during training. This makes training slower!

    Model = 'Plasim'
    area = 'France'
    lon_start = 0
    lon_end = 128
    lat_start = 0 # latitudes start from 90 degrees North Pole
    lat_end = 22


    #myscratch='/scratch/gmiloshe/PLASIM/'  # where files used to be
    mylocal='/local/gmiloshe/PLASIM/' #'/local/gmiloshe/PLASIM/'      # where we keep large datasets that need to be loaded
    myscratch=TryLocalSource(mylocal)        # Check if the data is not there and can be found in some other source
    new_mixing = False                     # if set to True the undersampling will also follow the suit
    
    num_years = 8000                       # Select the number of years from the simulation for the analysis


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
        if os.path.exists(checkpoint_name): # Create the directory
            print(f'folder {checkpoint_name} exists. Should I overwrite?')
            if input(" write Y to overwrite, else the execution will stop: ") != "Y":
                sys.exit("User has aborted the program")
        else:
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
    #                       a rectangle with latitude range lat_from[1] = lat_to[1] and longitude range lon_from[1] - lon_from[1].
    #lat_from = [4,4]     # 18x42
    #lat_to   = [22,22]
    #lon_from = [101,0]
    #lon_to   = [128,15]
    lat_from =  [0,0]   # 22x128
    lat_to =    [22,22]
    lon_from =  [64, 0]
    lon_to =    [128, 64]


    print(f'{percent = }, {T = }, {Model = }, {area = }, {undersampling_factor = }, {lat_from = }, {lat_to = }, {lon_from = }, {lon_to = }, {thefield = }')
    
    Months1 = [0, 0, 0, 0, 0, 0, 30, 30, 30, 30, 30, 0, 0, 0] # number of days per month with two leading 0s so that index 5 corresponds to May
    if sampling == '3hrs': # The dataset will be large
        Months1 = list(np.array(Months1)*timesperday)
    Tot_Mon1 = list(itertools.accumulate(Months1))

    time_start = Tot_Mon1[6]
    time_end = Tot_Mon1[9] #+(Tot_Mon1[10]-Tot_Mon1[9])//2   # uncomment this if we are to use full summer (including the portion with september due to T days window)

    if sampling == '3hrs': 
        prefix = ''
        file_prefix = '../Climate/'
    else:
        prefix = 'ANO_LONG_'
        file_prefix = ''

    t2m = Plasim_Field('tas',prefix+'tas','Temperature', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling, myprecision='single', years=8000)
    zg500 = Plasim_Field('zg',prefix+'zg500','500 mbar Geopotential', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling, myprecision='single', years=8000)
    mrso = Plasim_Field('mrso',prefix+'mrso','soil moisture', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling, myprecision='single', years=8000)
    
    
    
    t2m.load_field(mylocal+file_prefix+'Data_Plasim_LONG/')  # load the data
    zg500.load_field(mylocal+file_prefix+'Data_Plasim_LONG/')
    mrso.load_field(mylocal+file_prefix+'Data_Plasim_LONG/')
    
    LON = t2m.LON
    LAT = t2m.LAT
    print(t2m.var.dtype,zg500.var.dtype,mrso.var.dtype)

    mask, cell_area, lsm = ExtractAreaWithMask(mylocal,Model,area) # extract land sea mask and multiply it by cell area
    print(mask)

    t2m.abs_area_int, t2m.ano_area_int = t2m.Set_area_integral(area,mask,'PostprocLONG')
    zg500.abs_area_int, zg500.ano_area_int = zg500.Set_area_integral(area,mask,'PostprocLONG') 
    mrso.abs_area_int, mrso.ano_area_int = mrso.Set_area_integral(area,mask,'PostprocLONG')

    
    # ===Below we filter out just the area of France for mrso====
    filter_mask = np.zeros((t2m.var.shape[2],t2m.var.shape[3])) # a mask which sets to zero all values
    filter_lat_from = [13, 13]  # defining the domain of 1's
    filter_lat_to = [17, 17] 
    filter_lon_from = [-1, 0] 
    filter_lon_to =  [128, 3] 

    for myiter in range(len(filter_lat_from)): # seting values to 1 in the desired domain
        filter_mask[filter_lat_from[myiter]:filter_lat_to[myiter],filter_lon_from[myiter]:filter_lon_to[myiter]] = 1
                
    mrso.var = mrso.var*filter_mask # applying the filter to set to zero all values outside the domain
    # ==== Premixing =====
    print('==== PreMixing ====')
    if creation == None: # If we are not running from the same directory
        filename_mixing = t2m.PreMixing(new_mixing, 'PostprocLONG',num_years)  # perform mixing (mix batches and years but not days of the same year!)  # NEW MIXING MEANS ALSO NEW UNDERSAMPLING!
        shutil.copy(filename_mixing, checkpoint_name) # move the permutation file that was used to mix 
        zg500.PreMixing(False, 'PostprocLONG',num_years) # IT IS IMPORTANT THAT ALL SUBSEQUENT FIELDS BE MIXED (SHUFFLED) THE SAME WAY, otherwise no synchronization!
        mrso.PreMixing(False, 'PostprocLONG',num_years)
    else:
        filename_mixing = t2m.PreMixing(new_mixing,creation,num_years) # load from the folder that we are calling this file from   # NEW MIXING MEANS ALSO NEW UNDERSAMPLING!
        zg500.PreMixing(False,creation,num_years) # IT IS IMPORTANT THAT ALL SUBSEQUENT FIELDS BE MIXED (SHUFFLED) THE SAME WAY, otherwise no synchronization!
        mrso.PreMixing(False,creation,num_years)
    print(f"{t2m.var.shape = }")
    print(f"{time_end = } ,{time_start = } ,{T = }")
    
    A, A_reshape, threshold, list_extremes, convseq =  t2m.ComputeTimeAverage(time_start,time_end,T,tau, percent)
    print(f"{threshold = }")
    # ==== Equal mixing =====
    #   it ensures that there is equal number of heatwaves (or nearly equal) per what we call batch (century if we are dealing with 1000 years of data). This way the skill fluctuates less per ``batch''. This is not the same definition of batch we find in machine learning!
    print('==== EqualMixing ====')
    if creation == None: # If we are not running from the same directory
        filename_mixing = t2m.EqualMixing(A, threshold, new_mixing, 'PostprocLONG',num_years)
        shutil.copy(filename_mixing, checkpoint_name) # move the permutation file that was used to mix 
        zg500.EqualMixing(A, threshold, False, 'PostprocLONG',num_years) #IT IS IMPORTANT THAT ALL SUBSEQUENT FIELDS BE MIXED (SHUFFLED) THE SAME WAY, otherwise no synchronization!
        mrso.EqualMixing(A, threshold, False, 'PostprocLONG',num_years)
    else:
        filename_mixing = t2m.EqualMixing(A, threshold, new_mixing,creation,num_years)
        zg500.EqualMixing(A, threshold, False,creation,num_years)
        mrso.EqualMixing(A, threshold, False,creation,num_years)
    # Now we have to recompute the extremes:
    
    A, A_reshape, threshold, list_extremes, convseq =  t2m.ComputeTimeAverage(time_start,time_end,T,tau, percent)
    
    print(f"{threshold = }")
    print(A.dtype)
    
    # ==== Just to be safe temperature filter is applied after the computation of A(t)
    t2m.var = t2m.var*filter_mask # applying the filter to set to zero all values outside the domain

    # Below we reshape into time by flattened array, these are only needed if we require them for training the network
    t2m.abs_area_int_reshape = t2m.ReshapeInto1Dseries(area, mask, Tot_Mon1[6], Tot_Mon1[9], T, tau)
    mrso.abs_area_int_reshape = mrso.ReshapeInto1Dseries(area, mask, Tot_Mon1[6], Tot_Mon1[9], T, tau)
    print("mrso.abs_area_int_reshape.shape = ", mrso.abs_area_int_reshape.shape)

    print("t2m.var.shape = ", t2m.var.shape)

    # Below we reshape into time by flattened array
    #Xs = [t2m.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    #                   If only 2m temperature is used the following command is to be used: (1D representation)
    #
    #X = np.concatenate(Xs, axis=1)
    #                   If we want to include soil moisture integrated over the area the following command is to be used:
    #   (1D representation)
    #X = np.c_[mrso.abs_area_int_reshape[:,np.newaxis],np.concatenate(Xs, axis=1)]
    #
    # Below we reshape into a timex 2D array (space)

    #Xs = [t2m.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    #X = np.concatenate(Xs, axis=2)



    #============== Use this if only one field needs to be used:===========
    ##Xs = [zg500.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    ##Xs = [mrso.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    ##Xs = [t2m.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    ##X = np.concatenate(Xs, axis=2)
    ##X = X[:,:,:,np.newaxis]

    # =================Use this if many fields need to be used:============

    Xs = [t2m.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    X = np.concatenate(Xs, axis=2)
    

    ## Without Graining:
    Xs = [zg500.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    X= np.concatenate([X[:,:,:,np.newaxis], np.concatenate(Xs, axis=2)[:,:,:,np.newaxis]], axis=3)
    

    
    Xs = [mrso.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    X= np.concatenate([X, np.concatenate(Xs, axis=2)[:,:,:,np.newaxis]], axis=3)

    # ========= Use Previous times ===========

    ##Xs = [t2m.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau-4,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    ##X= np.concatenate([X, np.concatenate(Xs, axis=2)[:,:,:,np.newaxis]], axis=3)
        
    ##Xs = [zg500.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau-4,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    ##X= np.concatenate([X, np.concatenate(Xs, axis=2)[:,:,:,np.newaxis]], axis=3)
    
    ##Xs = [mrso.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau-4,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    #X= np.concatenate([X, np.concatenate(Xs, axis=2)[:,:,:,np.newaxis]], axis=3)
    # ========== Coarse Graining: ============
    #X1 = t2m.DownScale(time_start,time_end,lat_start,lat_end,lon_start,lon_end,T,tau, (18,42))
    #X2 = zg500.DownScale(time_start,time_end,lat_start,lat_end,lon_start,lon_end,T,tau, (18,42))
    #X3 = mrso.DownScale(time_start,time_end,lat_start,lat_end,lon_start,lon_end,T,tau, (18,42))
    #X = np.concatenate([X1[:,:,:,np.newaxis],  X2[:,:,:,np.newaxis],  X3[:,:,:,np.newaxis]], axis=3)

    # if we want to use FFT space (2D representation)
    ##X = t2m.ComputeFFT(time_start, time_end,T,tau,'real')
    ##X = t2m.ComputeFFThalf(time_start, time_end,T,tau,'real')
    ##X = zg500.ComputeFFThalf(time_start, time_end,T,tau,'real')
    ##X = zg500.ComputeFFT(time_start, time_end,T,tau,'real')
    
    #X = zg500.ComputeFFTnoPad(time_start, time_end,T,tau,'real',(18,42))
    del t2m.var
    gc.collect() # Garbage collector which removes some extra references to the object
    usepipelines = A_reshape, threshold, checkpoint_name_previous, tau
    undersampling_factor = [undersampling_factor, oversampling_factor]

                                
    return X, list_extremes, thefield, sampling, percent, usepipelines, undersampling_factor, new_mixing,  saveweightseveryblaepoch, NUM_EPOCHS, BATCH_SIZE, checkpoint_name, fullmetrics



######## TRAIN THE NETWORK #########

if __name__ == '__main__':
    print("====== running Learn.py ====== ")  
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
    print('RAM memory used:', psutil.virtual_memory()[3])
    print("Reading time = ",end - start)
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
        print("# events in the train sample after TrainTestSp;litIndices: ",  len(train_indices))
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
        X_mean = np.mean(X_train,0)
        X_std = np.std(X_train,0)
        X_std[X_std==0] = 1 # If there is no variance we shouldn't divide by zero

        X_test = (X[test_indices]-X_mean)/X_std
        Y_test = mylabels[test_indices]

        X_train = (X_train-X_mean)/X_std


         
        print("Y_train.shape = ", Y_train.shape)
        print("Y_test.shape = ", Y_test.shape)
         
        print("Train set: # of true labels = ", np.sum(Y_train), " ,# of false labels = ", Y_train.shape[0] - np.sum(Y_train))
        print("Train set: effective sampling rate for rare events is ", np.sum(Y_train)/Y_train.shape[0])
        
        np.save(checkpoint_name+'/batch_'+str(i)+'_X_mean', X_mean) # this values must be saved if the neural network is to be tested again, by reloading some other data
        np.save(checkpoint_name+'/batch_'+str(i)+'_X_std', X_std)
        
        if tau < 0: # engagge transfer learning
            print("opt_checkpoint: ", opt_checkpoint, " ,loading model: ", checkpoint_name_previous)
            model = (tf.keras.models.load_model(checkpoint_name_previous+'/batch_'+str(i), compile=False)) # if we just want to train

            nb_zeros_c = 4-len(str(opt_checkpoint))
            cp_checkpoint_name = '/cp-'+nb_zeros_c*'0'+str(opt_checkpoint)+'.ckpt'
            print("loading weights from ",checkpoint_name_previous+'/batch_'+str(i)+cp_checkpoint_name)
            model.load_weights(checkpoint_name_previous+'/batch_'+str(i)+cp_checkpoint_name)
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

        model.save(checkpoint_name+'/batch_'+str(i))
        np.save(checkpoint_name+'/batch_'+str(i)+'_history.npy',my_history.history)
        
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

    np.save(checkpoint_name+'/RAM_stats.npy', my_memory)

    end = time.time()
    print("files saved in ", checkpoint_name)
    print("Learning time = ",end - start)



