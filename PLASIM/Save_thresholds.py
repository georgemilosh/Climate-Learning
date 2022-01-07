
# Potential inputs:
#python Save_thresholds.py training/__folder.France_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u20o1_LONG8000yrs__per_1_tau_ 1 min
# the first argument is the location of the files, second argument is the percent of heat waves and the third is opt_checkpoint, if min specified then optimal checkpoing will be obtained by minimizing the CustomLoss

# The script can also be used with
#python Save_thresholds.py training/__folder.France14_equalmixed_22by128__/CNN_eqlmxd_ckpt_t2mT14France__with_zg500_t2mmrsoFrance__u1o1_8000yrs__per_5_finetune_tau_ 5 min
#python Save_thresholds.py training/__folder.France_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u20o1_LONG8000yrs__per_1_tau_ 1 min
#python Save_thresholds.py training/__folder.France_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u40o1_LONG8000yrs__per_0.1_tau_ 0.1 min

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

def CNN_layers(input_1):# This CNN is inspired by https://www.tensorflow.org/tutorials/images/cnn
    x = layers.Conv2D(32, (3, 3))(input_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SpatialDropout2D(0.2)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SpatialDropout2D(0.2)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SpatialDropout2D(0.2)(x)
    return layers.Flatten()(x)

def bottom_layers(input_1):
    x = layers.Dense(64, activation='relu')(input_1)
    x = layers.Dropout(0.2)(x)
    return layers.Dense(2)(x)


from tensorflow.keras.regularizers import l2
import math
def create_regularized_model(factor, rate, inputshape=(1,)):
    model = tf.keras.models.Sequential([
        #tf.keras.layers.Flatten(input_shape=(8, 8)),     # if the model has a tensor input
        tf.keras.layers.Input(shape=inputshape),                 # if the model has a flat input
        tf.keras.layers.Dense(2, kernel_regularizer=l2(factor))
    ])
    return model

start = time.time()

sampling='' #'3hrs' # This chooses whether we want say daily sampling or 3 hour one. Notice that the corresponding NetCDF files are kept in different places
creation = sys.argv[1]+'0'
percent = float(sys.argv[2])
opt_checkpoint = sys.argv[3]
if opt_checkpoint != 'min':
    opt_checkpoint = int(opt_checkpoint)

timesperday = 8 # 3 hour long periods in case we choose this sampling
if sampling == '3hrs':
    T = 14*timesperday
else:
    T = 14

tau = 0 #-5  # lag
usepipelines = False # if True => Dataset.from_tensor_slices will be used. This is a more advanced method but it takes more RAM and there is a possiblity for memory leaks when repeating training for cross-validation
fullmetrics = True # If True MCC and confusion matrix will be evaluated during training. This makes training slower!

Model = 'Plasim'
area = 'France'
lon_start = 0
lon_end = 128
lat_start = 0 # latitudes start from 90 degrees North Pole
lat_end = 22


#myscratch='/scratch/gmiloshe/PLASIM/'  # where we acess .py files and save output
mylocal='/local/gmiloshe/PLASIM/' #'/local/gmiloshe/PLASIM/'      # where we keep large datasets that need to be loaded
myscratch=TryLocalSource(mylocal)        # Check if the data is not there and can be found in some other source
#myscratch=mylocal

new_mixing = False                     # if set to True the undersampling will also follow the suit

num_years = 8000                       # Select the number of years from the simulation for the analysis


# If an integer >= 1 is chosen we simply undersample by this rate
# If a float between 0 and 1 is chosen we select each state with the probability given by this float
undersampling_factor = 1 # 1 #15 #10 #5 #1 #0.25 #0.15 #0.25
oversampling_factor = 1 # oversampling_factor = 1 means that oversampling will not be performed
thefield = 't2m' # Important: this is the field that is used to determine the extrema (important for undersampling) and therefore the label space
BATCH_SIZE = 1024 # choose this for training so that the chance of encountering positive batches is nonnegligeable
NUM_EPOCHS = 20 #100 #1000 #20 #200 #50 # number of epochs the training involves
saveweightseveryblaepoch = 1 # If set to 0 the model will not save weights as it is being trained, otherwise this number will tell us how many epochs it weights until saving
if saveweightseveryblaepoch > 0:
    ckpt = 'ckpt'
else:
    cktp = ''

    
print("creation = ", creation)

#lat_from = [4,4]     # 18x42
#lat_to   = [22,22]
#lon_from = [101,0]
#lon_to   = [128,15]
lat_from =  [0,0]   # 22x128
lat_to =    [22,22]
lon_from =  [64, 0]
lon_to =    [128, 64]


print([percent, T, Model, area, undersampling_factor, lat_from, lat_to, lon_from, lon_to, thefield])

if sampling == '3hrs':
    Months1 = [0, 0, 0, 0, 0, 0, timesperday*30, timesperday*30, timesperday*30, timesperday*30, timesperday*30, 0, 0, 0]
else: # if sampling == 'daily'
    Months1 = [0, 0, 0, 0, 0, 0, 30, 30, 30, 30, 30, 0, 0, 0] 
Tot_Mon1 = list(itertools.accumulate(Months1))

time_start = Tot_Mon1[6]
time_end = Tot_Mon1[9] #+(Tot_Mon1[10]-Tot_Mon1[9])//2   # uncomment this if we are to use full summer (including the portion with september due to T days window)

if sampling == '3hrs': 
    prefix = ''
    file_prefix = '../Climate/'
else:
    prefix = 'ANO_LONG_'
    file_prefix = ''

t2m = Plasim_Field('tas',prefix+'tas','Temperature', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling)     # if we want to use surface tempeature
zg500 = Plasim_Field('zg',prefix+'zg500','500 mbar Geopotential', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling)
#zg300 = Plasim_Field('zg',prefix+'zg300','300 mbar Geopotential', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling)
mrso = Plasim_Field('mrso',prefix+'mrso','soil moisture', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling)
#ua300 = Plasim_Field('ua',prefix+'ua300','eastward wind', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling)
#va300 = Plasim_Field('va',prefix+'va300','northward wind', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling)
#hfls = Plasim_Field('hfls',prefix+'hfls','surface latent heat flux', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling)
#hfss = Plasim_Field('hfss',prefix+'hfss','surface sensible heat flux', Model, lat_start, lat_end, lon_start, lon_end,'single',sampling)

t2m.years=8000
zg500.years=8000
mrso.years=8000

#ts.load_field(mylocal+file_prefix+'Data_Plasim/')  # load the data
t2m.load_field(mylocal+file_prefix+'Data_Plasim_LONG/')  # load the data
zg500.load_field(mylocal+file_prefix+'Data_Plasim_LONG/')
#zg300.load_field(mylocal+file_prefix+'Data_Plasim/')
mrso.load_field(mylocal+file_prefix+'Data_Plasim_LONG/')
#ua300.load_field(mylocal+file_prefix+'Data_Plasim/')
#va300.load_field(mylocal+file_prefix+'Data_Plasim/')
#hfls.load_field(mylocal+file_prefix+'Data_Plasim/')
#hfss.load_field(mylocal+file_prefix+'Data_Plasim/')

LON = t2m.LON
LAT = t2m.LAT
print(t2m.var.dtype,t2m.var.dtype,t2m.var.dtype)

mask, cell_area, lsm = ExtractAreaWithMask(mylocal,Model,area) # extract land sea mask and multiply it by cell area
print(mask)

#ts.abs_area_int, ts.ano_area_int = ts.Set_area_integral(area,mask)
t2m.abs_area_int, t2m.ano_area_int = t2m.Set_area_integral(area,mask,'PostprocLONG')
zg500.abs_area_int, zg500.ano_area_int = zg500.Set_area_integral(area,mask,'PostprocLONG') 
#zg300.abs_area_int, zg300.ano_area_int = zg300.Set_area_integral(area,mask) 
mrso.abs_area_int, mrso.ano_area_int = mrso.Set_area_integral(area,mask,'PostprocLONG')
#ua300.abs_area_int, ua300.ano_area_int = ua300.Set_area_integral(area,mask) 
#va300.abs_area_int, va300.ano_area_int = va300.Set_area_integral(area,mask)
#hfls.abs_area_int, hfls.ano_area_int = hfls.Set_area_integral(area,mask) 
#hfss.abs_area_int, hfss.ano_area_int = hfss.Set_area_integral(area,mask)


# ===Below we filter out just the area of France for mrso====

filter_mask = np.zeros((t2m.var.shape[2],t2m.var.shape[3])) # a mask which sets to zero all values
filter_lat_from = [13, 13]  # defining the domain of 1's
filter_lat_to = [17, 17] 
filter_lon_from = [-1, 0] 
filter_lon_to =  [128, 3] 

for myiter in range(len(filter_lat_from)): # seting values to 1 in the desired domain
        filter_mask[filter_lat_from[myiter]:filter_lat_to[myiter],filter_lon_from[myiter]:filter_lon_to[myiter]] = 1

mrso.var = mrso.var*filter_mask # applying the filter to set to zero all values outside the domain
    
    

filename_mixing = t2m.PreMixing(new_mixing,creation,num_years) # load from the folder that we are calling this file from   # NEW MIXING MEANS ALSO NEW UNDERSAMPLING!
zg500.PreMixing(False,creation,num_years) # IT IS IMPORTANT THAT ALL SUBSEQUENT FIELDS BE MIXED (SHUFFLED) THE SAME WAY, otherwise no synchronization!
#zg300.PreMixing(False,creation,num_years)
mrso.PreMixing(False,creation,num_years)
print("t2m.var.shape = ", t2m.var.shape)
print("time_end = ", time_end, " ,time_start = ", time_start, " ,T = ", T)

A, A_reshape, threshold, list_extremes, convseq =  t2m.ComputeTimeAverage(time_start,time_end,T,tau, percent)
print("threshold = ",threshold)


filename_mixing = t2m.EqualMixing(A, threshold, new_mixing,creation,num_years)
zg500.EqualMixing(A, threshold, False,creation,num_years)
#zg300.EqualMixing(A, threshold, False,creation,num_years)
mrso.EqualMixing(A, threshold, False,creation,num_years)
# Now we have to recompute the extremes:

A, A_reshape, threshold, list_extremes, convseq =  t2m.ComputeTimeAverage(time_start,time_end,T,tau, percent)

# ===== Applying filter to the temperature field: ====
t2m.var = t2m.var*filter_mask # applying the filter to set to zero all values outside the domain

print("threshold = ",threshold)
print(A.dtype)
# Below we reshape into time by flattened array
t2m.abs_area_int_reshape = t2m.ReshapeInto1Dseries(area, mask, Tot_Mon1[6], Tot_Mon1[9], T, tau)
mrso.abs_area_int_reshape = mrso.ReshapeInto1Dseries(area, mask, Tot_Mon1[6], Tot_Mon1[9], T, tau)
print("mrso.abs_area_int_reshape.shape = ", mrso.abs_area_int_reshape.shape)

print("t2m.var.shape = ", t2m.var.shape)

checkpoint_name = creation[:-1]+str(tau)

maxskill = -(percent/100.)*np.log(percent/100.)-(1-percent/100.)*np.log(1-percent/100.)


undersampling_factors=10**np.arange(-3,2,.1)
taus = range(0,-35,-5)
new_MCC = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_entropy = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_BS = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_WBS = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_freq = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_skill = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_TP = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_FP = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_TN = np.zeros((len(taus),10,undersampling_factors.shape[0]))
new_FN = np.zeros((len(taus),10,undersampling_factors.shape[0]))
for itau, tau in enumerate(taus):
    checkpoint_name = creation[:-1]+str(tau)
    print("\t checkpoint_name  = ", checkpoint_name)
     # =================Use this if many fields need to be used:============

    Xs = [t2m.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    X = np.concatenate(Xs, axis=2)


    ## Without Coarse Graining:
    Xs = [zg500.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    X= np.concatenate([X[:,:,:,np.newaxis], np.concatenate(Xs, axis=2)[:,:,:,np.newaxis]], axis=3)

    Xs = [mrso.ReshapeInto2Dseries(time_start, time_end,lat_from1,lat_to1,lon_from1,lon_to1,T,tau,dim=2) for lat_from1,lat_to1, lon_from1, lon_to1 in zip(lat_from,lat_to,lon_from,lon_to)] # here we extract the portion of the globe
    X= np.concatenate([X, np.concatenate(Xs, axis=2)[:,:,:,np.newaxis]], axis=3)
    print("\tX.shape = ", X.shape)
    maxskill = -(percent/100.)*np.log(percent/100.)-(1-percent/100.)*np.log(1-percent/100.)
    history = np.load(checkpoint_name+'/batch_'+str(0)+'_history.npy', allow_pickle=True).item()
    print("\tlen(history) = ", len(history))
    if sys.argv[3] == 'min': # Otherwise we specity what the opt_checkpoint should be
        print('\t\tmin will be used')
        if ('val_CustomLoss' in history.keys()):
            print( "\t\t\t'val_CustomLoss' in history.keys()")
            historyCustom = []
            for i in range(10): # preemptively compute the optimal score
                historyCustom.append(np.load(checkpoint_name+'/batch_'+str(i)+'_history.npy', allow_pickle=True).item()['val_CustomLoss'])
            historyCustom = np.mean(np.array(historyCustom),0)
            opt_checkpoint = np.argmin(historyCustom)+1 # We will use optimal checkpoint in this case!
        else:
            print('\t\t\tval_CustomLoss is not there')

    print("\t\t========Optimal checkpoint = ", opt_checkpoint)

    mylabels = np.array(list_extremes)
    model = tf.keras.models.load_model(checkpoint_name+'/batch_'+str(0), compile=False) # if we just want to train

    for i in range(10):
        print("\t\t\t===============================")
        print("\t\t\tcross validation i = ", str(i))

        X_test, Y_test, X_mean, X_std, test_indices = NormalizeAndX_test(i, X, mylabels, 1, sampling, new_mixing, thefield, percent, checkpoint_name)

        nb_zeros_c = 4-len(str(opt_checkpoint))
        checkpoint_name_batch = '/cp-'+nb_zeros_c*'0'+str(opt_checkpoint)+'.ckpt'
        model.load_weights(checkpoint_name+'/batch_'+str(i)+checkpoint_name_batch)
        my_probability_model=(tf.keras.Sequential([ # softmax output to make a prediction
                  model,
                  tf.keras.layers.Softmax()
                ]))
        Y_pred_prob = my_probability_model.predict(X_test)
        #Y_pred, Y_pred_prob = ModelToProb(X,X_test,model)

        for j in range(undersampling_factors.shape[0]):
            undersampling_factor = undersampling_factors[j]
            new_MCC[itau,i,j], new_entropy[itau,i,j], new_skill[itau,i,j], new_BS[itau,i,j], new_WBS[itau,i,j], new_freq[itau,i,j]  = ComputeMetrics(Y_test, Y_pred_prob, percent, undersampling_factor)


            Y_pred_prob_renorm = ReNormProbability(Y_pred_prob, undersampling_factor)
            label_assignment = np.argmax(Y_pred_prob_renorm,1)

            new_TP[itau,i,j], new_TN[itau,i,j], new_FP[itau,i,j], new_FN[itau,i,j], MCC = ComputeMCC(Y_test, label_assignment, False)
new_TPR = new_TP/(new_TP+new_FN)
new_PPV = new_TP/(new_TP+new_FP)
#new_PPV[new_TP==0] = 0
new_FPR = new_FP/(new_FP+new_TN)
new_F1 = 2*new_TP/(2*new_TP+new_FP+new_FN)
np.savez(creation+'/new_vars_'+str(sys.argv[3]), new_MCC=new_MCC, new_entropy=new_entropy, new_BS=new_BS, new_WBS=new_WBS, new_freq=new_freq, new_skill=new_skill, new_TP=new_TP, new_FP=new_FP, new_TN=new_TN, new_FN=new_FN, new_TPR=new_TPR, new_PPV=new_PPV, new_FPR=new_FPR, new_F1=new_F1, undersampling_factors=undersampling_factors, taus=taus)

end = time.time()
print("Reading and processing time = ",end - start)
