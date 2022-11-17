# George Miloshevich 2022
# Plot the development of metrics during training
# Importation des librairies

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
nfolds = int(sys.argv[2])

# we must now number of folds in advance

folder = sys.argv[1]
if len(sys.argv)>3:
    limiting_fraction = int(sys.argv[3]) # tells the plot what fraction of x to show
else:
    limiting_fraction = 10

fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.68)
color_idx = np.linspace(0, 1, nfolds)
ax2 = ax1.twinx() 
ax3 = ax1.twinx() 
ax4 = ax1.twinx()
# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
ax3.spines.right.set_position(("axes", 1.2))
ax4.spines.right.set_position(("axes", 1.4))
history = np.load(f'{folder}/fold_{0}/history_vae', allow_pickle=True)
metric = {}
for key in history:
    metric[key] = []
    

for i in range(nfolds):
    history = np.load(f'{folder}/fold_{i}/history_vae', allow_pickle=True)#.item()
    print(history.keys())
    
    
    for key in history:
        epochs = range(1, len(history[key])+1)
        metric[key].append(history[key])
        print(epochs)
ln1, ln2, ln3, ln4, ln1_va, ln2_va, ln3_va, ln4_va = [], [], [], [], [], [], [], []
epochs = epochs[len(epochs)//limiting_fraction:]
for key in history:
    metric[key] = np.array(metric[key])
    print(f'{metric[key].shape = }')
    meanskilarray = np.mean(metric[key],0)
    stdskillarray = np.std(metric[key],0)
    meanskilarray = meanskilarray[len(meanskilarray)//limiting_fraction:]
    stdskillarray = stdskillarray[len(stdskillarray)//limiting_fraction:]
    print(f'{meanskilarray.shape = }')
    

    label =f'{key}_{i}'
    if key=='loss':
        cmap = 'k'
        ln1 = ax1.plot(epochs, meanskilarray, label =label, linestyle='solid', color = cmap)
        ax1.fill_between(epochs,meanskilarray-stdskillarray,meanskilarray+stdskillarray,alpha=0.1, color = cmap)
    elif key=='val_loss':
        cmap = 'k'
        ln1_va = ax1.plot(epochs, meanskilarray, label =label, linestyle='solid', color = cmap)
        ax1.fill_between(epochs,meanskilarray-stdskillarray,meanskilarray+stdskillarray,alpha=0.1, color = cmap)
    elif key=='reconstruction_loss':
        cmap = 'r'
        ln2 = ax2.plot(epochs, meanskilarray, label =label, linestyle='dashed',color = cmap)
        ax2.fill_between(epochs,meanskilarray-stdskillarray,meanskilarray+stdskillarray,alpha=0.1, color = cmap)
    elif key=='val_reconstruction_loss':
        cmap = 'm'
        ln2_va = ax2.plot(epochs, meanskilarray, label =label, linestyle='dashed',color = cmap)
        ax2.fill_between(epochs,meanskilarray-stdskillarray,meanskilarray+stdskillarray,alpha=0.1, color = cmap)
    elif key=='kl_loss':
        cmap = 'b'
        ln3 = ax3.plot(epochs, meanskilarray, label =label, linestyle='dotted',color = cmap)
        ax3.fill_between(epochs,meanskilarray-stdskillarray,meanskilarray+stdskillarray,alpha=0.1, color = cmap)
        ln4 = ln3.copy()
    elif key=='val_kl_loss':
        cmap = 'c'
        ln3_va = ax3.plot(epochs, meanskilarray, label =label, linestyle='dotted',color = cmap)
        ax3.fill_between(epochs,meanskilarray-stdskillarray,meanskilarray+stdskillarray,alpha=0.1, color = cmap)
        ln4_va = ln3_va.copy()
    elif key=='class_loss':
        cmap = 'g'
        ln4 = ax4.plot(epochs, meanskilarray, label =label, linestyle='dashdot',color = cmap)
        ax4.fill_between(epochs,meanskilarray-stdskillarray,meanskilarray+stdskillarray,alpha=0.1, color = cmap)
    elif key=='val_class_loss':
        cmap = 'y'
        ln4_va = ax4.plot(epochs, meanskilarray, label =label, linestyle='dashdot',color = cmap)
        ax4.fill_between(epochs,meanskilarray-stdskillarray,meanskilarray+stdskillarray,alpha=0.1, color = cmap)
    if i ==nfolds-1:
        lns = ln1+ln2+ln3+ln4+ln1_va+ln2_va+ln3_va+ln4_va
            
        
labs = [l.get_label() for l in lns]
print(f'{labs = }')
ax4.legend(lns, labs, loc=0)

ax1.set_xlabel("Epochs")
ax1.set_ylabel("loss")
ax2.set_ylabel("reconstruction_loss")
ax3.set_ylabel("kl_loss")
ax1.set_xlim([epochs[len(epochs)//10],epochs[-1]])
ax2.grid()
fig.savefig(f"Images/history.png", bbox_inches='tight', dpi=200)
plt.show()
