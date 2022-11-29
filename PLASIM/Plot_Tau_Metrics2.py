# George Miloshevich 2021
# Compute benchmarks for the agregate of the files at a given checkpoint
# Example of usage: python Recalc_Tau_Metrics.py training/__folder.small_slow_CNN_premixed.Real__/small_CNN_premixed_ckpt_t2m__with_t2m_zg500_mrso__sampl_Real_10_per_5_tau_ 24 10
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

extra_parameter=1 # extra parameters I add in sys.argv

fig = plt.figure(figsize=(8,4))
#plt.style.use('seaborn-whitegrid')
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
#ax3 = plt.subplot(143)
#ax4 = plt.subplot(144)


ax1.grid(axis='x', color='0.9')
ax1.grid(axis='y', color='0.9')
ax2.grid(axis='x', color='0.9')
ax2.grid(axis='y', color='0.9')

title = sys.argv[-extra_parameter]
extra_parameter=extra_parameter+1
facealpha=0.1
#widthstyles=[4,3,2,1]
print(" processing sys.argv[-extra_parameter] = ", sys.argv[-extra_parameter])
str_widthstyles = sys.argv[-extra_parameter].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
widthstyles = [str(s) for s in str_widthstyles]
extra_parameter=extra_parameter+1
print("widthstyles = ", widthstyles)

#labels=['$z_G$','$t_M$','$s_M$','$s_L$']
print(" processing sys.argv[-extra_parameter] = ", sys.argv[-extra_parameter])
str_labels = sys.argv[-extra_parameter].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
labels = [str(s) for s in str_labels]
extra_parameter=extra_parameter+1
print("labels = ", labels)


#markerstyles=['1','2','3','4']

print(" processing sys.argv[-extra_parameter] = ", sys.argv[-extra_parameter])
str_markerstyles = sys.argv[-extra_parameter].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
markerstyles = [str(s) for s in str_markerstyles]
extra_parameter=extra_parameter+1
print("markerstyles = ", markerstyles)

#hatches=['/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
print(" processing sys.argv[-extra_parameter] = ", sys.argv[-extra_parameter])
str_hatches = sys.argv[-extra_parameter].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
hatches = [str(s) for s in str_hatches]
extra_parameter=extra_parameter+1
print("hatches = ", hatches)

#linestyles=['-','--','-.',':']
print(" processing sys.argv[-extra_parameter] = ", sys.argv[-extra_parameter])
str_linestyles = sys.argv[-extra_parameter].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
linestyles = [str(s) for s in str_linestyles]
extra_parameter=extra_parameter+1
print("linestyles = ", markerstyles)

#colors=['tab:blue','tab:orange','tab:green','tab:red']
print(" processing sys.argv[-extra_parameter] = ", sys.argv[-extra_parameter])
str_colors = sys.argv[-extra_parameter].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
colors = [str(s) for s in str_colors]
extra_parameter=extra_parameter+1
print("colors = ", colors)

print(len(labels),len(markerstyles),len(hatches),len(linestyles),len(colors),len(widthstyles))

print('Number of curves to output ',len(sys.argv) - 1 - extra_parameter)

if '[' in sys.argv[-extra_parameter]: #checking for multiple inputs
    #if len(sys.argv[-1]) > 1: # if undersampling_factor is different for each input
    str_undersampling_factors = sys.argv[-extra_parameter].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    undersampling_factors = [float(i) for i in str_undersampling_factors]#[int(i) for i in str_undersampling_factors]
    print("undersampling factors = ", undersampling_factors)
else:
    undersampling_factor = float(sys.argv[-extra_parameter]) #int(sys.argv[-1]) # undersampling_factor is used to re-normalize probabilities
    print("undersampling factor = ", undersampling_factor)
    

if '[' in sys.argv[-1-extra_parameter]:
    str_checkpoints = sys.argv[-1-extra_parameter].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    checkpoints = [int(i) for i in str_checkpoints]
    print("checkpoints = ", checkpoints)
else:
    checkpoint = int(sys.argv[-1-extra_parameter])
    print("checkpoint = ", checkpoint)

    
for check_num1, checkpoint_name1 in enumerate(sys.argv[1:-1-extra_parameter]):
#checkpoint_name1 = sys.argv[1]
    print('=========',checkpoint_name1,'=========')
    
    if '[' in sys.argv[-extra_parameter]:
        #if len(sys.argv[-1]) > 1:
        undersampling_factor = undersampling_factors[check_num1]
    if '[' in sys.argv[-1-extra_parameter]:
        checkpoint = checkpoints[check_num1]
    
    print("undersampling_factor (r) = ", undersampling_factor)
    print("checkpoint = ", checkpoint)
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
    counter = 0
    print("Looking for filenames ", checkpoint_name1+'*')
    for checkpoint_name in glob.glob(checkpoint_name1+'*'):
        print("counter = ", counter)
        counter = counter + 1
        print('\t checkpoint_name = ', checkpoint_name)
        
        if os.path.exists(checkpoint_name+"/Funs.py"): # we have checkpoints
            print(" Checkpoints were stored during the training")
            foo = module_from_file("foo", checkpoint_name+'/Funs.py')
            filename = checkpoint_name+"/Funs.py"
        else: # we only have the last output, this method has been phased out at least since July 29. So most experiments in Google Slides "Committor discovery in climate models" don't need this any more. We will soon remove this backward compatibility feature
            print(" Checkpoints were not stored during the training")
            if os.path.exists(checkpoint_name+"/Run_TF_Logistic.py"):
                print("File Run_TF_Logistic.py exists")
                filename = checkpoint_name+"/Run_TF_Logistic.py"
            else:
                print("File Run_TF_Logistic doesn't exist, looking for Learn.py")
                filename = checkpoint_name+"/Learn.py"
                #file1 = open(checkpoint_name+"/Learn.py", "r")
        percent = float(ReadStringFromFileRaw(filename, 'percent'))
        
        print("===PERCENT====", percent) # we need percent to compute the skill
        
        sampling = ReadStringFromFileRaw(filename, ' sampling') # basically samples per day
        if (sampling=="''") or (sampling=="'' "):
            sampling = 1
        elif (sampling == "'3hrs'") or (sampling == "'3hrs' "):
            sampling = 8
        undersampling_rate_input = float(ReadStringFromFileRaw(filename, 'undersampling_factor')) # 
        print("===UNDERSAMPLING RATE when training ====", undersampling_rate_input)
        print("===User defioned Undersampling rate input====", undersampling_factor)

        #else:
        #    sampling = 1
        print("===SAMPLING====", sampling)
        current_tau = (float(ReadStringFromFileRaw(filename, 'tau'))/sampling)
        print("===tau=========", current_tau)
        tau.append(current_tau)
        print("checking for ",checkpoint_name+'/my_MCC_r'+str(undersampling_factor)+'.npy')
        if (os.path.exists(checkpoint_name+'/my_MCC_r'+str(undersampling_factor)+'.npy'))and(undersampling_factor>1): # If benchmarks were computed before usually with Recalc_History.py. This feature was necessary before the normalized cross-entropy skill got included directly in the fit metric, in other words now it is computed during the training, which is much faster. Still if we want to compute the metrics with some new undersampling rate (undersampling_factor) we might need it. It takes a lot of time to compute if we want the whole History so some other approach is needed rather Recalc_History.py. This is because Recalc_History.py computes benchmarks for each epoch, which means the weights have to be loaded for each epoch and tested on the data. This is particularly costly if we want to work with 8000 year long dataset. So starting from October 5 in training/__folder.France_equalmixed_18by42__/stack_CNN_equalmixed_ckpt_t2mFrance__with_t2m_zg500_mrsoFrance__18by42_u10o1_LONG8000yrs__per_5_tau_0/ for example I stopped computation of things like my_MCC_r10.npy. Instead the function will be defaulted to else, since these files are abscent. This feature (the whole if statement) should be phased out in future, as we will determine optimal checkpoint by looking at the normalized skill maximum (or normalized cross-entropy skill), thus if we want to change the user defined undersampling rate we will only compute it for a given optimal checkpoint
            print("loading ", checkpoint_name+'/my_MCC_r'+str(undersampling_factor)+'.npy')
            opt_checkpoint = np.argmin(np.load(checkpoint_name+'/my_entropy_r'+str(undersampling_factor)+'.npy'),1)
            print("========Optimal checkpoint = ", opt_checkpoint)
            new_MCC = np.load(checkpoint_name+'/my_MCC_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_entropy = np.load(checkpoint_name+'/my_entropy_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_skill = np.load(checkpoint_name+'/my_skill_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_BS = np.load(checkpoint_name+'/my_BS_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_WBS = np.load(checkpoint_name+'/my_WBS_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
            new_freq = np.load(checkpoint_name+'/my_freq_r'+str(undersampling_factor)+'.npy')[:,opt_checkpoint]#checkpoint]
        else:
            print("We don't have the postcomputed files such as my_MCC_r"+str(undersampling_factor)+'.npy')
            new_MCC = np.zeros(10,)
            new_entropy = np.zeros(10,)
            new_BS = np.zeros(10,)
            new_WBS = np.zeros(10,)
            new_freq = np.zeros(10,)
            new_skill = np.zeros(10,)
            if (undersampling_factor>1)and(undersampling_factor != undersampling_rate_input): # 
                print("The r we want to compute is undersampling_factor and is >1 ")
                print(" and it is not equal to undersampling_rate_input, which is the value we expect to be evaluated during training (I have started doing this roughly from Oct 4) ")
                X, list_extremes, thefield, sampling_load, percent, usepipelines, undersampling_factor_load, new_mixing,  saveweightseveryblaepoch, NUM_EPOCHS, BATCH_SIZE, checkpoint_name_load, fullmetrics = foo.PrepareData(checkpoint_name)
                mylabels = np.array(list_extremes)
                model = tf.keras.models.load_model(scratch+checkpoint_name+'/batch_'+str(0), compile=False) # if we just want to train

                for i in range(10):
                    print("===============================")
                    print("cross validation i = ", str(i))
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
                    print("MCC = " , new_MCC[i]," ,entropy = ", new_entropy[i], " ,entropy = ", -np.sum(np.c_[1-Y_test,Y_test]*np.log(Y_pred_prob))/Y_test.shape[0], " ,BS = ", new_BS[i], " , WBS = ", new_WBS[i], " , freq = ", new_freq[i])
            else: # undersampling_rate=1 or user defined undersampling factor is equal to the one used during training (we are assuming that undersampling_rate < 1 is not provided any more). This part of the if statement will be involved in all old training folders starting from October 5.
                # We have already computed this during training (no need to re-normalize the probabilities)
                print("(undersampling_factor not > 1 or undersampling_factor = undersampling_rate_input")
                maxskill = -(percent/100.)*np.log(percent/100.)-(1-percent/100.)*np.log(1-percent/100.) # climatological skill that we know in advance (based on the percent of the heat waves)
                history = np.load(checkpoint_name+'/batch_'+str(0)+'_history.npy', allow_pickle=True).item()
                if ('val_CustomLoss' in history.keys()): # we look for the minimum of the normalized skill score (CustomLoss)
                    print( "'val_CustomLoss' in history.keys()")
                    historyCustom = []
                    for i in range(10): # preemptively compute the optimal score
                        historyCustom.append(np.load(checkpoint_name+'/batch_'+str(i)+'_history.npy', allow_pickle=True).item()['val_CustomLoss'])
                    historyCustom = np.mean(np.array(historyCustom),0)
                    opt_checkpoint = np.argmin(historyCustom)+1 # We will use optimal checkpoint in this case! we add one because history doesn't save the 0 state
                else: # If somehow the customLoss is missing (this could be if we are running this on a folder generated before October 5 where the files my_MCC_r.... when not computed 
                    print( "'val_CustomLoss' not in history.keys()")
                    opt_checkpoint = checkpoint # then opt_checkpoint will be just the one we provide when calling Recalc_Tau_metrics.py
                if opt_checkpoint == len(history['val_CustomLoss']):
                    opt_checkpoint = opt_checkpoint - 1
                print("===opt_checkpoint = ", opt_checkpoint)
                for i in range(10):
                    #print("===============================")
                    #print("cross validation i = ", str(i))
                    history = np.load(checkpoint_name+'/batch_'+str(i)+'_history.npy', allow_pickle=True).item()
                    if i == 0:
                        print(history.keys())
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
                            opt_checkpoint = len(history['val_loss'])  #-1 the length of history doesn't include the 0 initial state
                            Y_test = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_test.npy')
                            Y_pred = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_pred.npy')

                            Y_pred_prob = np.load(checkpoint_name+'/batch_'+str(i)+'_Y_pred_prob.npy')
                            new_MCC[i], new_entropy[i], new_skill[i], new_BS[i], new_WBS[i], new_freq[i]  = ComputeMetrics(Y_test, Y_pred_prob, percent, undersampling_factor)
                            print("MCC = " , new_MCC[i]," ,entropy = ", new_entropy[i], " ,entropy = ", -np.sum(np.c_[1-Y_test,Y_test]*np.log(Y_pred_prob))/Y_test.shape[0], " ,BS = ", new_BS[i], " , WBS = ", new_WBS[i], " , freq = ", new_freq[i])
                    else: # use normal entropy because if the data was trained with undersampling factor 1 the probabilities do not need to be estimated
                        new_entropy[i] = history['val_loss'][opt_checkpoint]#[checkpoint]
                    new_skill[i] =  (maxskill-new_entropy[i])/maxskill
                    #print("opt_checkpoint = ", opt_checkpoint, "len(history['val_MCC'])",len(history['val_MCC']))
                    new_MCC[i] = history['val_MCC'][opt_checkpoint]#[checkpoint]
                    
                    #print("MCC = " , new_MCC[i]," ,entropy = ", new_entropy[i],  " ,BS = ", new_BS[i], " , WBS = ", new_WBS[i], " , freq = ", new_freq[i])
                print("========Optimal checkpoint = ", opt_checkpoint)
        print(checkpoint_name1+f" TOTAL MCC  = {np.mean(new_MCC):.3f} +- {np.std(new_MCC):.3f} , entropy = {np.mean(new_entropy):.3f} +- {np.std(new_entropy):.3f} , skill = {np.mean(new_skill):.3f} +- {np.std(new_skill):.3f}, Brier = {np.mean(new_BS):.3f} +- {np.std(new_BS):.3f} , Weighted Brier = {np.mean(new_WBS):.3f} +- {np.std(new_WBS):.3f} , frequency = {np.mean(new_freq):.3f} +- {np.std(new_freq):.3f}")
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
        metrics[tau[i]] = [mean_new_MCC[i], std_new_MCC[i], mean_entropy[i], std_entropy[i], mean_BS[i], std_BS[i], mean_skill[i], std_skill[i]]
        
    sorted_metrics = sorted(metrics.items(), key=lambda kv: kv[0], reverse=True)

    print(sorted_metrics)

    for i in range(len(sorted_metrics)):
        tau[i], mean_new_MCC[i], std_new_MCC[i], mean_entropy[i], std_entropy[i], mean_BS[i], std_BS[i], mean_skill[i], std_skill[i] = sorted_metrics[i][0], sorted_metrics[i][1][0], sorted_metrics[i][1][1], sorted_metrics[i][1][2], sorted_metrics[i][1][3], sorted_metrics[i][1][4], sorted_metrics[i][1][5], sorted_metrics[i][1][6], sorted_metrics[i][1][7]

    print(f">>>>>{3*int(widthstyles[check_num1]) = }, {widthstyles[check_num1] = }, {colors[check_num1] = }, {linestyles[check_num1] = }, {markerstyles[check_num1] = }, {labels[check_num1] = } ")
    ax1.errorbar(-np.array(tau), mean_new_MCC, yerr = std_new_MCC, capsize = 3*int(widthstyles[check_num1]),  elinewidth = int(widthstyles[check_num1]), capthick = 1, color=colors[check_num1], linestyle=linestyles[check_num1], marker=markerstyles[check_num1], label=labels[check_num1])
    ax1.fill_between(-np.array(tau), np.array(mean_new_MCC) - np.array(std_new_MCC), np.array(mean_new_MCC) + np.array(std_new_MCC), color=colors[check_num1], hatch=hatches[check_num1], alpha=facealpha)
    ax2.errorbar(-np.array(tau), mean_skill, yerr = std_skill, capsize = 3*int(widthstyles[check_num1]),  elinewidth = int(widthstyles[check_num1]), capthick = 1, color=colors[check_num1], linestyle=linestyles[check_num1], marker=markerstyles[check_num1], label=labels[check_num1])
    ax2.fill_between(-np.array(tau), np.array(mean_skill) - np.array(std_skill), np.array(mean_skill) + np.array(std_skill), color=colors[check_num1], hatch=hatches[check_num1], alpha=facealpha)
   

#ax1.set_title('MCC')
ax1.set_xlabel(r'$\tau$ (days)')
ax2.set_xlabel(r'$\tau$ (days)')
ax1.set_ylabel('MCC')
ax2.set_ylabel('NLS')
#ax2.set_title('normalized cross-entropy skill')
#ax3.set_xlabel(r'$\tau$ (days)')
#ax3.set_title('Brier Score')
#ax4.set_title('categorical cross-entropy')
#ax4.set_xlabel(r'$\tau$ (days)')
ax1.set_ylim([-0, .5])
ax2.set_ylim([-0, .5])

if labels[0]!='':
	ax1.legend(loc='best', fancybox=True, shadow=True)
	ax2.legend(loc='best', fancybox=True, shadow=True)
#plt.tight_layout()
end = time.time()
print("Computation time = ",end - start)

plt.show()
bbox = ax1.get_tightbbox(fig.canvas.get_renderer())
fig.savefig("Images/"+title+"_MCC.png", bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()), dpi=200)
bbox = ax2.get_tightbbox(fig.canvas.get_renderer())
fig.savefig("Images/"+title+"_Skill.png", bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()), dpi=200)

print("saved Images/"+title+"_MCC.png and Images/"+title+"_Skill.png")
