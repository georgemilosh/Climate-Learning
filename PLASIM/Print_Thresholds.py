import sys
import numpy as np
percent = float(sys.argv[2])

opt_checkpoint = sys.argv[3]
if opt_checkpoint != 'min':
    opt_checkpoint = int(opt_checkpoint)

output = np.load(sys.argv[1]+"0/new_vars_"+str(opt_checkpoint)+".npz")
[new_MCC, new_entropy, new_BS, new_WBS, new_freq, new_skill, new_TP, new_FP, new_TN, new_FN, new_TPR, new_PPV, new_FPR, new_F1, undersampling_factors, taus]=[output[key] for key in output]


maxskill = -(percent/100.)*np.log(percent/100.)-(1-percent/100.)*np.log(1-percent/100.)
new_Skill = maxskill-new_entropy
new_Normal_Skill = new_Skill/maxskill

new_Brier_skill = ((percent/100.)-(percent/100.)**2 - new_BS)/((percent/100.)-(percent/100.)**2)

new_TPR = new_TP/(new_TP+new_FN)
new_PPV = new_TP/(new_TP+new_FP)
#new_PPV[new_TP==0] = 0
new_FPR = new_FP/(new_FP+new_TN)
new_F1 = 2*new_TP/(2*new_TP+new_FP+new_FN)

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def fmt(x):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

#fig2 = plt.figure(figsize=(40,20))
#spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig2)
ax = []
itrt = 0
ktrt = 0
jtrt = 0
iui = 20 # initial undersampling index
for new_metric, metric_label in zip([new_MCC[:,:,iui:],new_Skill[:,:,iui:],new_Normal_Skill[:,:,iui:], new_F1[:,:,iui:],new_TPR[:,:,iui:], new_PPV[:,:,iui:], new_Brier_skill[:,:,iui:], new_BS[:,:,iui:]],['MCC','cross-entropy skill','normalized cross-entropy skill', 'F1','recall','precision', 'Brier Skill','Brier Score']):
    #ax.append(fig2.add_subplot(spec2[jtrt,ktrt]))
    fig = plt.figure(figsize=(6,6))
    ax.append(fig.add_subplot(111))
    print(itrt)

    colorparams = new_metric.shape[2]
    cmap = plt.get_cmap('hsv', colorparams)
    color_idx =  np.arange(0,len(undersampling_factors[iui:]),1) #undersampling_factors #


    for i, iterate in enumerate(color_idx):
        ax[itrt].plot(range(0,35,5),np.mean(new_metric[:,:,i],1), c = cmap(iterate), linewidth=2, marker='o')
    print(i,metric_label,np.mean(new_metric[:,:,i],1))
    # Normalizer
    norm = mpl.colors.Normalize(vmin=color_idx.min(), vmax=color_idx.max())

    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    #cbar = fig2.colorbar(sm, format=ticker.FuncFormatter(fmt))
    cbar = fig.colorbar(sm, format=ticker.FuncFormatter(fmt))
    cbar.set_ticks(color_idx)
    cbar.set_ticklabels(list(map(fmt,undersampling_factors[iui:])))
    ax[itrt].set_facecolor((0, 0, 0))
    ax[itrt].set_xlabel('$\\tau$ (days)')
    ax[itrt].set_title(metric_label+' vs days and undersampling')
    if (metric_label=='recall' or metric_label=='precision'):
        ax[itrt].set_ylim([0, 1])
    else:
        ax[itrt].set_ylim([0,0.5])
    if (metric_label=='cross-entropy skill'):
        ax[itrt].set_ylim([0,0.1])
    if (metric_label=='Brier Skill'):
        ax[itrt].set_ylim([0,0.35])
    ax[itrt].set_xlabel(r'$\tau$ (days)')
    #bbox = ax[itrt].get_tightbbox(fig2.canvas.get_renderer())
    #fig2.savefig("Images/Tau_scan_"+metric_label+"_"+str(percent)+"_"+str('min')+".png", bbox_inches=bbox.transformed(fig2.dpi_scale_trans.inverted()), dpi=200)
    fig.savefig("Images/Tau_scan_"+metric_label+"_"+str(percent)+"_"+str('min')+".png", dpi=200, bbox_inches='tight')
    itrt = itrt + 1
    #ktrt = ktrt + 1
    #if ktrt > 2:
    #    ktrt = 0
    #    jtrt = jtrt+1

#plt.show()
