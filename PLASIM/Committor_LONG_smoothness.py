# George Miloshevich 2021
# Plot the composite maps conditioned to TP< TP, FN, FP

# Importation des librairies

import sys

#training_name0 = 'training/__folder.IntegratedArea__/stack_CNN_equalmixed_ckpt_t2mFrance__with_mrsoArea__u10o1_LONG8000yrs__per_5_tau_'
#training_name0 = 'training/__folder.France14_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mT14France__with_zg500_t2mmrsoFrance__u10o1_8000yrs__per_5_tau_'
training_name0 = 'training/__folder.France14_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mT14France__with_zg500_t2mmrsoFrance__u10o1_8000yrs__per_5_finetune_tau_'
#training_name0 = 'training/__folder.France_equalmixed_22by128__/stack_CNN_ckpt_t2mFrance__with_zg500__22by128_u10o1_LONG8000yrs__per_5_tau_'
tau = 0
training_name = training_name0+str(tau)

#sys.path.insert(1, '../ERA')
sys.path.insert(1, training_name)
from ERA_Fields import*
from TF_Fields import*
import importlib.util
import time
start = time.time()
def module_from_file(module_name, file_path): #The code that imports the file which originated the training with all the instructions
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
list_extremes_tau = []
committor_tau = []
A_reshape_tau = []
for tau in range(0,-31,-1):
    training_name = training_name0+str(tau)
    filename = training_name+"/Funs.py"
    r = str(10)
    checkpoint = str(50)
    undersampling_rate_input = float(ReadStringFromFileRaw(filename, 'undersampling_factor')) # 
    print("===UNDERSAMPLING RATE INPUT====", undersampling_rate_input)
    #checkpoint = np.argmin(np.mean(np.load(training_name+'/my_entropy_r'+str(undersampling_rate_input)+'.npy'),0))

    historyCustom = []
    for i in range(10): # preemptively compute the optimal score
        historyCustom.append(np.load(training_name+'/batch_'+str(i)+'_history.npy', allow_pickle=True).item()['val_CustomLoss'])
    historyCustom = np.mean(np.array(historyCustom),0)
    checkpoint = np.argmin(historyCustom) # We will use optimal checkpoint in this case!
    print("========Optimal checkpoint = ", checkpoint)

    #sys.path.insert(1, training_name)
    foo = module_from_file("foo", training_name+'/Funs.py')
    import gc

    #from Funs import*

    scratch = '/ClimateDynamics/MediumSpace/ClimateLearningFR/gmiloshe/PLASIM/'
    markers = ['|', '.', ',', 'x', '+', '*', '1', '2', '3', '4']
    X, list_extremes, thefield, sampling, percent, usepipelines, undersampling_factor_load, new_mixing,  saveweightseveryblaepoch, NUM_EPOCHS, BATCH_SIZE, checkpoint_name_load, fullmetrics = foo.PrepareData(training_name)
    A_reshape = usepipelines[0].reshape((10,-1))
    threshold = usepipelines[1]
    #print(f"{X[232,12,12,1] = }")
    #lat_from, lat_to, lon_from, lon_to = usepipelines[2:6]

    lat_from = [4,4]     # 18x42
    lat_to   = [22,22]
    lon_from = [101,0]
    lon_to   = [128,15]
    print("A_reshape.shape = ",A_reshape.shape)
    if list(usepipelines[0]>=threshold)==list_extremes:
        print("A_reshape and threshold consistent with list_extremes")
    else:
        print("A_reshape and threshold not consistent with list_extremes")
    undersampling_factor = r

    mylabels = np.array(list_extremes)


    committor = np.zeros(A_reshape.shape)            
    label_assignment = np.zeros(A_reshape.shape)
    my_metric_name = ['MCC_r'+str(undersampling_factor),'entropy_r'+str(undersampling_factor), 'skill_r'+str(undersampling_factor), 'BS_r'+str(undersampling_factor), 'WBS_r'+str(undersampling_factor), 'freq_r'+str(undersampling_factor)]
    undersampling_factor = int(undersampling_factor) # because before it can take values like 1.0 or 10.0
    for i in range(10):
        print("===============================")
        print("cross validation i = ", str(i))

        X_test, Y_test, X_mean, X_std, test_indices = NormalizeAndX_test(i, X, mylabels, undersampling_factor, sampling, new_mixing, thefield, percent, scratch+training_name)

        model = tf.keras.models.load_model(scratch+training_name+'/batch_'+str(i), compile=False) # if we just want to train


        nb_zeros_c = 4-len(str(checkpoint))
        checkpoint_name = '/cp-'+nb_zeros_c*'0'+str(checkpoint)+'.ckpt'

        model.load_weights(scratch+training_name+'/batch_'+str(i)+checkpoint_name)

        Y_pred, Y_pred_prob = ModelToProb(X,X_test,model)
        committor[i,:] = ReNormProbability(Y_pred_prob, undersampling_factor)[:,1]
        label_assignment[i,:] = np.argmax(Y_pred,1)
    A_reshape = A_reshape.flatten()
    committor = committor.flatten()
    label_assignment = label_assignment.flatten()
    list_extremes = np.array(list(map(int, list_extremes)))
    A_reshape_tau.append(A_reshape)
    committor_tau.append(committor)
    list_extremes_tau.append(list_extremes)
    print("X.shape = ",X.shape)
    print("label_assignment = ", label_assignment)
    print("list_extremes = ", list_extremes)
    TP_list =((label_assignment == 1)& (list_extremes == 1))
    TN_list = ((label_assignment == 0)& (list_extremes == 0))
    FP_list = ((label_assignment == 1)& (list_extremes == 0))
    FN_list = ((label_assignment == 0)& (list_extremes == 1))

    print(np.sum(TP_list), np.sum(TN_list), np.sum(FP_list), np.sum(FN_list))
    #TP, TN, FP, FN, new_MCC = ComputeMCC(list_extremes, label_assignment, 'True')

    #print("X[TP_list,:,:,:].shape = ", X[TP_list,:,:,:].shape)
    np.save(training_name+'/A_preshaped2', np.array(A_reshape))
    np.save(training_name+'/committor2', np.array(committor))
    np.save(training_name+'/list_extremes_preshaped2', np.array(list_extremes))
    end = time.time()
    print("Computation time = ",end - start)
    
    
list_extremes = np.array(list_extremes).reshape(8000,-1)
np.save(training_name+'/list_extremes2',np.array(list_extremes))
A = A_reshape.reshape(8000,-1)
np.save(training_name+'/A2',A)
np.save(training_name+'/threshold2',np.array(threshold))
committor_tau = np.array(committor_tau)
committor_tau = committor_tau.reshape(31,8000,-1)
np.save(training_name+'/committor_tau2',committor_tau)
