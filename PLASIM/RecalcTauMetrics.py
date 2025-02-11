# George Miloshevich 2021
# Compute benchmarks for the agregate of the files at a given checkpoint
# Example of usage: python RecalcTauMetrics.py training/__folder.France_equalmixed_22by128__/stack_CNN_equalmix_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u10o1_LONG8000yrs__per_5_tau_ training/__folder.France_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u20o1_LONG8000yrs__per_1_tau_  training/__folder.France_equalmixed_22by128__/CNN_eqlmxd_t2mFrance__with_zg500_t2mmrsoFrance__u10o1_LONG8000yrs__per_5_tau_20_finetine_tau_ training/__folder.France_equalmixed_22by128__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__22by128_u40o1_LONG8000yrs__per_0.1_tau_  0 1
# Importation des librairies

import numpy as np
import random as rnd
import os
import importlib.util
import time
start = time.time()

import sys
sys.path.insert(1, '../ERA')
from ERA_Fields import*

from TF_Fields import* 
import glob
from itertools import cycle

scratch = '/scratch/gmiloshe/PLASIM/'
def module_from_file(module_name, file_path): #The code that imports the file which originated the training with all the instructions
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
fig = plt.figure(figsize=(20,5))
plt.style.use('seaborn-whitegrid')
ax1 = plt.subplot(141)
ax2 = plt.subplot(142)
ax3 = plt.subplot(143)
ax4 = plt.subplot(144)

fig2 = plt.figure(figsize=(20,5))
plt.style.use('seaborn-whitegrid')
ax21 = plt.subplot(141)
ax22 = plt.subplot(142)
ax23 = plt.subplot(143)
ax24 = plt.subplot(144)

linestyles=['-', '--', '-.', ':']
linestylescycle = cycle(linestyles)
linestylesnext = next(linestylescycle)

markerstyles=['1','2','3','4']
markerstylescycle = cycle(markerstyles)
markerstylesnext = next(markerstylescycle)
print('Number of curves to output ',len(sys.argv) - 2)

if '[' in sys.argv[-1]: #checking for multiple inputs
    #if len(sys.argv[-1]) > 1: # if undersampling_factor is different for each input
    str_undersampling_factors = sys.argv[-1].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    undersampling_factors = [float(i) for i in str_undersampling_factors]#[int(i) for i in str_undersampling_factors]
    print("\t undersampling factors = ", undersampling_factors)
else:
    undersampling_factor = float(sys.argv[-1]) #int(sys.argv[-1]) # undersampling_factor is used to re-normalize probabilities
    print("\t undersampling factor = ", undersampling_factor)
    

if '[' in sys.argv[-2]:
    str_checkpoints = sys.argv[-2].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    checkpoints = [int(i) for i in str_checkpoints]
    print("\t checkpoints = ", checkpoints)
else:
    checkpoint = int(sys.argv[-2])
    print("\t checkpoint = ", checkpoint)

    
for check_num1, checkpoint_name1 in enumerate(sys.argv[1:-2]):
#checkpoint_name1 = sys.argv[1]
    print("    ")
    print('\t checkpoint_name1=========',checkpoint_name1,'=========')
    
    if '[' in sys.argv[-1]:
        #if len(sys.argv[-1]) > 1:
        undersampling_factor = undersampling_factors[check_num1]
    if '[' in sys.argv[-2]:
        checkpoint = checkpoints[check_num1]
    
    print("\t undersampling_factor (r) = ", undersampling_factor)
    print("\t checkpoint = ", checkpoint)
    mean_new_MCC = []
    std_new_MCC = []
    mean_entropy = []
    std_entropy = []
    mean_skill = []
    std_skill = []
    mean_BS = []
    std_BS = []
    mean_WBS = []
    std_WBS = []
    tau = []
    
    mean_new_TPR = []
    std_new_TPR = []
    mean_new_PPV = []
    std_new_PPV = []
    mean_new_FPR = []
    std_new_FPR = []
    mean_new_F1 = []
    std_new_F1 = []
    counter = 0
    print("\t Looking for filenames ", checkpoint_name1+'*')
    for checkpoint_name in glob.glob(checkpoint_name1+'*'):
        print("    ")
        print("\t\t counter = ", counter)
        counter = counter + 1
        print('\t\t >>>>>>>>checkpoint_name = ', checkpoint_name)
        
        if os.path.exists(checkpoint_name+"/Funs.py"): # we have checkpoints
            print("\t\t\t Checkpoints were stored during the training")
            foo = module_from_file("foo", checkpoint_name+'/Funs.py')
            filename = checkpoint_name+"/Funs.py"
        else: # we only have the last output, this method has been phased out at least since July 29. So most experiments in Google Slides "Committor discovery in climate models" don't need this any more. We will soon remove this backward compatibility feature
            print("\t\t\t Checkpoints were not stored during the training")
            if os.path.exists(checkpoint_name+"/Run_TF_Logistic.py"):
                print("\t\t\t\t File Run_TF_Logistic.py exists")
                filename = checkpoint_name+"/Run_TF_Logistic.py"
            else:
                print("\t\t\t\t File Run_TF_Logistic doesn't exist, looking for Learn.py")
                filename = checkpoint_name+"/Learn.py"
                #file1 = open(checkpoint_name+"/Learn.py", "r")
        percent = float(ReadStringFromFileRaw(filename, 'percent', False))
        
        print("\t\t===PERCENT====", percent) # we need percent to compute the skill
        
        sampling = ReadStringFromFileRaw(filename, ' sampling', False) # basically samples per day
        if (sampling=="''") or (sampling=="'' "):
            sampling = 1
        elif (sampling == "'3hrs'") or (sampling == "'3hrs' "):
            sampling = 8
        undersampling_rate_input = float(ReadStringFromFileRaw(filename, 'undersampling_factor', False)) # 
        print("\t\t===UNDERSAMPLING RATE when training ====", undersampling_rate_input)
        print("\t\t===User defioned Undersampling rate input====", undersampling_factor)

        #else:
        #    sampling = 1
        print("\t\t===SAMPLING====", sampling)
        current_tau = (float(ReadStringFromFileRaw(filename, 'tau', False))/sampling)
        print("\t\t>>>>>>>>>>>>>>>>===tau=========", current_tau)
        tau.append(current_tau)
        print("\t\tchecking for ",checkpoint_name+'/my_MCC_r'+str(undersampling_factor)+'.npy')
        if (os.path.exists(checkpoint_name+'/my_MCC_r'+str(undersampling_factor)+'.npy'))and(undersampling_factor>1): # If benchmarks were computed before usually with Recalc_History.py. This feature was necessary before the normalized cross-entropy skill got included directly in the fit metric, in other words now it is computed during the training, which is much faster. Still if we want to compute the metrics with some new undersampling rate (undersampling_factor) we might need it. It takes a lot of time to compute if we want the whole History so some other approach is needed rather Recalc_History.py. This is because Recalc_History.py computes benchmarks for each epoch, which means the weights have to be loaded for each epoch and tested on the data. This is particularly costly if we want to work with 8000 year long dataset. So starting from October 5 in training/__folder.France_equalmixed_18by42__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__18by42_u10o1_LONG8000yrs__per_5_tau_0/ for example I stopped computation of things like my_MCC_r10.npy. Instead the function will be defaulted to else, since these files are abscent. This feature (the whole if statement) should be phased out in future, as we will determine optimal checkpoint by looking at the normalized skill maximum (or normalized cross-entropy skill), thus if we want to change the user defined undersampling rate we will only compute it for a given optimal checkpoint
            print("\t\t\t loading ", checkpoint_name+'/my_MCC_r'+str(undersampling_factor)+'.npy')
            if checkpoint == 0: # Use custom checkpoint defined at the input
                opt_checkpoint = checkpoint
            else: # take the minimum
                opt_checkpoint = np.argmin(np.load(checkpoint_name+'/my_entropy_r'+str(undersampling_factor)+'.npy'),1)
            print("\t\t\t ========Optimal checkpoint = ", opt_checkpoint)
            new_MCC = np.load(checkpoint_name+'/my_MCC_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_entropy = np.load(checkpoint_name+'/my_entropy_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_skill = np.load(checkpoint_name+'/my_skill_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_BS = np.load(checkpoint_name+'/my_BS_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_WBS = np.load(checkpoint_name+'/my_WBS_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_freq = np.load(checkpoint_name+'/my_freq_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
        else:
            print("\t\t\t We don't have the postcomputed files such as my_MCC_r"+str(undersampling_factor)+'.npy')
            new_MCC = np.zeros(10,)
            new_entropy = np.zeros(10,)
            new_BS = np.zeros(10,)
            new_WBS = np.zeros(10,)
            new_freq = np.zeros(10,)
            new_skill = np.zeros(10,)
            new_TP = np.zeros(10,)
            new_FP = np.zeros(10,)
            new_TN = np.zeros(10,)
            new_FN = np.zeros(10,)
            if (undersampling_factor>1)and(undersampling_factor != undersampling_rate_input): # 
                print("\t\t\t\t The r we want to compute is undersampling_factor and is >1 ")
                print("\t\t\t\t and it is not equal to undersampling_rate_input, which is the value we expect to be evaluated during training (I have started doing this roughly from Oct 4) ")
                X, list_extremes, thefield, sampling_load, percent, usepipelines, undersampling_factor_load, new_mixing,  saveweightseveryblaepoch, NUM_EPOCHS, BATCH_SIZE, checkpoint_name_load, fullmetrics = foo.PrepareData(checkpoint_name)
                mylabels = np.array(list_extremes)
                model = tf.keras.models.load_model(scratch+checkpoint_name+'/batch_'+str(0), compile=False) # if we just want to train

                for i in range(10):
                    print("===============================")
                    print("\t\t\t\t\t >>>>>>>>>>>>>>>>>>>>>>>>>>cross validation i = ", str(i))
                    if os.path.exists(checkpoint_name+"/Funs.py"): # we have checkpoints
                       
                        X_test, Y_test, X_mean, X_std, test_indices = NormalizeAndX_test(i, X, mylabels, undersampling_factor, sampling, new_mixing, thefield, percent, scratch+checkpoint_name)
                        
                        nb_zeros_c = 4-len(str(checkpoint))
                        checkpoint_name_batch = '/cp-'+nb_zeros_c*'0'+str(checkpoint)+'.ckpt'
                        model.load_weights(scratch+checkpoint_name+'/batch_'+str(i)+checkpoint_name_batch)
                        
                        Y_pred, Y_pred_prob = ModelToProb(X,X_test,model)
                    else: # we only have the last output
                        Y_test = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_test.npy')
                        Y_pred = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_pred.npy')

                        Y_pred_prob = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_pred_prob.npy')
                    new_MCC[i], new_entropy[i], new_skill[i], new_BS[i], new_WBS[i], new_freq[i]  = ComputeMetrics(Y_test, Y_pred_prob, percent, undersampling_factor)
                    print("\t\t\t\t\t MCC = " , new_MCC[i]," ,entropy = ", new_entropy[i], " ,entropy = ", -np.sum(np.c_[1-Y_test,Y_test]*np.log(Y_pred_prob))/Y_test.shape[0], " ,BS = ", new_BS[i], " , WBS = ", new_WBS[i], " , freq = ", new_freq[i])
            else: # undersampling_rate=1 or user defined undersampling factor is equal to the one used during training (we are assuming that undersampling_rate < 1 is not provided any more). This part of the if statement will be involved in all old training folders starting from October 5.
                # We have already computed this during training (no need to re-normalize the probabilities)
                print("\t\t\t\t (undersampling_factor not > 1 or undersampling_factor = undersampling_rate_input")
                maxskill = -(percent/100.)*np.log(percent/100.)-(1-percent/100.)*np.log(1-percent/100.) # climatological skill that we know in advance (based on the percent of the heat waves)
                history = np.load(checkpoint_name+'/batch_'+str(0)+'_history.npy', allow_pickle=True).item()
                if ('val_CustomLoss' in history.keys()): # we look for the minimum of the normalized skill score (CustomLoss)
                    print( "\t\t\t\t\t 'val_CustomLoss' in history.keys()")
                    historyCustom = []
                    for i in range(10): # preemptively compute the optimal score
                        historyCustom.append(np.load(checkpoint_name+'/batch_'+str(i)+'_history.npy', allow_pickle=True).item()['val_CustomLoss'])
                    historyCustom = np.mean(np.array(historyCustom),0)
                    if checkpoint != 0:
                        opt_checkpoint = checkpoint
                    else:
                        opt_checkpoint = np.argmin(historyCustom)+1 # We will use optimal checkpoint in this case!
                else: # If somehow the customLoss is missing (this could be if we are running this on a folder generated before October 5 where the files my_MCC_r.... when not computed 
                    print( "'val_CustomLoss' not in history.keys()")
                    opt_checkpoint = checkpoint # then opt_checkpoint will be just the one we provide when calling Recalc_Tau_metrics.py
                for i in range(10):
                    #print("===============================")
                    #print("cross validation i = ", str(i))
                    history = np.load(checkpoint_name+'/batch_'+str(i)+'_history.npy', allow_pickle=True).item()
                    #if i == 0:
                        #print(history.keys())
                    #print("len(history['val_MCC']) = ", len(history['val_MCC']), " , checkpoint = ", checkpoint)
                    #print("history['val_MCC'][checkpoint] = ", history['val_MCC'][checkpoint], " , checkpoint = ", checkpoint)
                    
                    if (undersampling_factor > 1.0): #use custom entropy since we don't care about normal entropy in this case
                        #print("undersampling_factor > 1 so normally we need CustomLoss")
                        if ('val_CustomLoss' in history.keys()):
                            #print("We have val_CustomLoss")
                            #opt_checkpoint = np.argmin(history['val_CustomLoss']) # We will use optimal checkpoint in this case!
                            new_entropy[i] = history['val_CustomLoss'][opt_checkpoint]#[checkpoint]
                        else: 
                            #print("We don't have val_CustomLoss so what can only use the last Y_test, Y_pred which we shall call optimal")
                            opt_checkpoint = len(history['val_loss']) #-1
                            Y_test = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_test.npy')
                            Y_pred = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_pred.npy')

                            Y_pred_prob = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_pred_prob.npy')
                            new_MCC[i], new_entropy[i], new_skill[i], new_BS[i], new_WBS[i], new_freq[i]  = ComputeMetrics(Y_test, Y_pred_prob, percent, undersampling_factor)
                            print("\t\t\t\t\t MCC = " , new_MCC[i]," ,entropy = ", new_entropy[i], " ,entropy = ", -np.sum(np.c_[1-Y_test,Y_test]*np.log(Y_pred_prob))/Y_test.shape[0], " ,BS = ", new_BS[i], " , WBS = ", new_WBS[i], " , freq = ", new_freq[i])
                    else: # use normal entropy because if the data was trained with undersampling factor 1 the probabilities do not need to be estimated
                        new_entropy[i] = history['val_loss'][opt_checkpoint]#[checkpoint]
                    new_skill[i] =  (maxskill-new_entropy[i])/maxskill
                    #print("opt_checkpoint = ", opt_checkpoint, "len(history['val_MCC'])",len(history['val_MCC']))
                    new_MCC[i] = history['val_MCC'][opt_checkpoint]#[checkpoint]
                    [[new_TN[i], new_FP[i]],[new_FN[i], new_TP[i]]] = history['val_confusion_matrix'][opt_checkpoint]#[checkpoint]
                    
                print("\t\t\t\t ========Optimal checkpoint = ", opt_checkpoint)
        print("\t\t\t", checkpoint_name1+f" TOTAL MCC  = {np.mean(new_MCC):.3f} +- {np.std(new_MCC):.3f} , entropy = {np.mean(new_entropy):.3f} +- {np.std(new_entropy):.3f} , skill = {np.mean(new_skill):.3f} +- {np.std(new_skill):.3f}, Brier = {np.mean(new_BS):.3f} +- {np.std(new_BS):.3f} , Weighted Brier = {np.mean(new_WBS):.3f} +- {np.std(new_WBS):.3f} , frequency = {np.mean(new_freq):.3f} +- {np.std(new_freq):.3f}")
        
        print(f"TP = {np.mean(new_TP,0):.0f} +- {np.std(new_TP,0):.0f}, TN = {np.mean(new_TN,0):.0f} +- {np.std(new_TN,0):.0f}, FP = {np.mean(new_FP,0):.0f} +- {np.std(new_FP,0):.0f}, FN = {np.mean(new_FN,0):.0f} +- {np.std(new_FN,0):.0f}")
        print(f"TP = {new_TP}")
        print(f"TN = {new_TN}")
        print(f"FP = {new_FP}")
        print(f"FN = {new_FN}")
        print(f"TPR = {new_TP/(new_TP+new_FN)}")
        print(f"PPV = {new_TP/(new_TP+new_FP)}")
        print(f"FPR = {new_FP/(new_FP+new_TN)}")
        print(f"F1 = {2*new_TP/(2*new_TP+new_FP+new_TN)}")
        
        mean_new_TPR.append(np.mean(new_TP/(new_TP+new_FN),0))
        std_new_TPR.append(np.std(new_TP/(new_TP+new_FN),0))
        mean_new_PPV.append(np.mean(new_TP/(new_TP+new_FP),0))
        std_new_PPV.append(np.std(new_TP/(new_TP+new_FP),0))
        mean_new_FPR.append(np.mean(new_FP/(new_FP+new_TN),0))
        std_new_FPR.append(np.std(new_FP/(new_FP+new_TN),0))
        mean_new_F1.append(np.mean(2*new_TP/(2*new_TP+new_FP+new_TN),0))
        std_new_F1.append(np.std(2*new_TP/(2*new_TP+new_FP+new_TN),0))
        
        mean_new_MCC.append(np.mean(new_MCC))
        std_new_MCC.append(np.std(new_MCC))
        mean_entropy.append(np.mean(new_entropy))
        std_entropy.append(np.std(new_entropy))
        mean_skill.append(np.mean(new_skill))
        std_skill.append(np.std(new_skill))
        mean_BS.append(np.mean(new_BS))
        std_BS.append(np.std(new_BS))
        mean_WBS.append(np.mean(new_WBS))
        std_WBS.append(np.std(new_WBS))

    
    print("tau = ", tau, " ,mean_new_MCC = ", mean_new_MCC)
    metrics = dict(zip(['tau','MCC','MCC_std','entropy','entropy_std','BS','BS_std','WBS', 'WBS_std'], [tau, mean_new_MCC, std_new_MCC, mean_entropy, std_entropy, mean_BS, std_BS, mean_WBS, std_WBS]))
    #sorted_metrics = dict(sorted(metrics.items(), key=lambda item: item[1]))

    metrics = {}  # D  is of a type dictionary
    for i in range(len(tau)):
        metrics[tau[i]] = [mean_new_MCC[i], std_new_MCC[i], mean_entropy[i], std_entropy[i], mean_BS[i], std_BS[i], mean_skill[i], std_skill[i], mean_new_TPR[i], std_new_TPR[i], mean_new_PPV[i], std_new_PPV[i], mean_new_FPR[i], std_new_FPR[i], mean_new_F1[i], std_new_F1[i]]
        
    sorted_metrics = sorted(metrics.items(), key=lambda kv: kv[0], reverse=True)

   
    
    for i in range(len(sorted_metrics)):
        #print(len(sorted_metrics[i][1]))
        templist = [sorted_metrics[i][1][ii] for ii in range(len(sorted_metrics[0][1]))]
        #print("templist = ",templist)
        templist.insert(0, sorted_metrics[i][0])
        #print("templist = ",templist)
        #print("len(templist) = ",len(templist))
        tau[i], mean_new_MCC[i], std_new_MCC[i], mean_entropy[i], std_entropy[i], mean_BS[i], std_BS[i], mean_skill[i], std_skill[i], mean_new_TPR[i], std_new_TPR[i], mean_new_PPV[i], std_new_PPV[i], mean_new_FPR[i], std_new_FPR[i], mean_new_F1[i], std_new_F1[i] = templist 
    print("tau = ", tau)
    print("mean MCC = ", mean_new_MCC)
    print("std  MCC = ", std_new_MCC)
    print("mean PPV = ", mean_new_PPV)
    print("std  PPV = ", std_new_PPV)
    
    #ax1.plot(-np.array(tau), mean_new_MCC, linestyle=linestylesnext, marker=markerstylesnext)
    ax1.errorbar(-np.array(tau), mean_new_MCC, yerr = std_new_MCC, capsize = 3, elinewidth = 1, capthick = 1, linestyle=linestylesnext, marker=markerstylesnext)
    ax1.fill_between(-np.array(tau), np.array(mean_new_MCC) - np.array(std_new_MCC), np.array(mean_new_MCC) + np.array(std_new_MCC), alpha=0.1)
    ax2.errorbar(-np.array(tau), mean_skill, yerr = std_skill, capsize = 3,  elinewidth = 1, capthick = 1, linestyle=linestylesnext, marker=markerstylesnext)
    #ax2.plot(-np.array(tau), mean_skill, linestyle=linestylesnext, marker=markerstylesnext)
    ax2.fill_between(-np.array(tau), np.array(mean_skill) - np.array(std_skill), np.array(mean_skill) + np.array(std_skill), alpha=0.1)
    ax3.plot(-np.array(tau), mean_BS, linestyle=linestylesnext, marker=markerstylesnext)
    ax3.fill_between(-np.array(tau), np.array(mean_BS) - np.array(std_BS), np.array(mean_BS) + np.array(std_BS), alpha=0.1)
    ax4.plot(-np.array(tau), mean_entropy, linestyle=linestylesnext, marker=markerstylesnext)
    ax4.fill_between(-np.array(tau), np.array(mean_entropy) - np.array(std_entropy), np.array(mean_entropy) + np.array(std_entropy), alpha=0.1)
    
    # Here we plot recall, precision etc
    ax21.errorbar(-np.array(tau), mean_new_TPR, yerr = std_new_TPR, capsize = 3, elinewidth = 1, capthick = 1, linestyle=linestylesnext, marker=markerstylesnext)
    ax21.fill_between(-np.array(tau), np.array(mean_new_TPR) - np.array(std_new_TPR), np.array(mean_new_TPR) + np.array(std_new_TPR), alpha=0.1)
    ax22.errorbar(-np.array(tau), mean_new_PPV, yerr = std_new_PPV, capsize = 3,  elinewidth = 1, capthick = 1, linestyle=linestylesnext, marker=markerstylesnext)
    ax22.fill_between(-np.array(tau), np.array(mean_new_PPV) - np.array(std_new_PPV), np.array(mean_new_PPV) + np.array(std_new_PPV), alpha=0.1)
    ax23.plot(-np.array(tau), mean_new_FPR, linestyle=linestylesnext, marker=markerstylesnext)
    ax23.fill_between(-np.array(tau), np.array(mean_new_FPR) - np.array(std_new_FPR), np.array(mean_new_FPR) + np.array(std_new_FPR), alpha=0.1)
    ax24.plot(-np.array(tau), mean_new_F1, linestyle=linestylesnext, marker=markerstylesnext)
    ax24.fill_between(-np.array(tau), np.array(mean_new_F1) - np.array(std_new_F1), np.array(mean_new_F1) + np.array(std_new_F1), alpha=0.1)
    
    linestylesnext = next(linestylescycle)
    markerstylesnext = next(markerstylescycle)
   

ax1.set_title('MCC')
ax1.set_xlabel(r'$\tau$ (days)')
ax2.set_xlabel(r'$\tau$ (days)')
ax2.set_title('normalized cross-entropy skill')
ax3.set_xlabel(r'$\tau$ (days)')
ax3.set_title('Brier Score')
ax4.set_title('categorical cross-entropy')
ax4.set_xlabel(r'$\tau$ (days)')
ax1.set_ylim([-0, .5])
ax2.set_ylim([-0, .5])
#plt.legend(loc='best')
#plt.tight_layout()


ax21.set_title('TPR (Recall) = TP/(TP+FN)')
ax21.set_xlabel(r'$\tau$ (days)')
ax22.set_xlabel(r'$\tau$ (days)')
ax22.set_title('PPV (Precision) = TP/(TP+FP)')
ax23.set_xlabel(r'$\tau$ (days)')
ax23.set_title('FPR = FP/(FP+TN)')
ax24.set_title('F1 = 2(PPV+TPR)/(PPV+TPR)')
ax24.set_xlabel(r'$\tau$ (days)')
end = time.time()
print("Computation time = ",end - start)

plt.show()
bbox = ax1.get_tightbbox(fig.canvas.get_renderer())
fig.savefig("Images/Tau_scan_MCC_norm"+sys.argv[-1]+".png", bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()), dpi=200)
bbox = ax2.get_tightbbox(fig.canvas.get_renderer())
fig.savefig("Images/Tau_scan_skill_norm"+sys.argv[-1]+".png", bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()), dpi=200)
print("saved Images/Tau_scan_MCC_norm"+sys.argv[-1]+".png and Images/Tau_scan_skill_norm"+sys.argv[-1]+".png")
