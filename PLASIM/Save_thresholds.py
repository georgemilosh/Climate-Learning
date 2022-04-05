
# Potential inputs:
#python Save_thresholds.py training/__folder.France_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u20o1_LONG8000yrs__per_1_tau_ min
# the first argument is the location of the files, second argument is the percent of heat waves and the third is opt_checkpoint, if min specified then optimal checkpoing will be obtained by minimizing the CustomLoss

# The script can also be used with
#python Save_thresholds.py training/__folder.France14_equalmixed_22by128__/CNN_eqlmxd_ckpt_t2mT14France__with_zg500_t2mmrsoFrance__u1o1_8000yrs__per_5_finetune_tau_ min
#python Save_thresholds.py training/__folder.France_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u20o1_LONG8000yrs__per_1_tau_ min
#python Save_thresholds.py training/__folder.France_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u40o1_LONG8000yrs__per_0.1_tau_ min

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
import importlib.util
def module_from_file(module_name, file_path): #The code that imports the file which originated the training with all the instructions
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

start = time.time()


creation = sys.argv[1]+'0'
opt_checkpoint = sys.argv[2]
if opt_checkpoint != 'min':
    opt_checkpoint = int(opt_checkpoint)

tau = 0
checkpoint_name = creation[:-1]+str(tau)


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
    foo = module_from_file("foo", checkpoint_name+'/Funs.py')
    X, list_extremes, thefield, sampling, percent, usepipelines, undersampling_factor_load, new_mixing,  saveweightseveryblaepoch, NUM_EPOCHS, BATCH_SIZE, checkpoint_name_load, fullmetrics = foo.PrepareData(checkpoint_name)
    
    print("\tX.shape = ", X.shape)
    history = np.load(checkpoint_name+'/batch_'+str(0)+'_history.npy', allow_pickle=True).item()
    print("\tlen(history) = ", len(history))
    if sys.argv[2] == 'min': # Otherwise we specity what the opt_checkpoint should be
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
np.savez(creation+'/new_vars_'+str(sys.argv[2]), new_MCC=new_MCC, new_entropy=new_entropy, new_BS=new_BS, new_WBS=new_WBS, new_freq=new_freq, new_skill=new_skill, new_TP=new_TP, new_FP=new_FP, new_TN=new_TN, new_FN=new_FN, new_TPR=new_TPR, new_PPV=new_PPV, new_FPR=new_FPR, new_F1=new_F1, undersampling_factors=undersampling_factors, taus=taus)

end = time.time()
print("Reading and processing time = ",end - start)
