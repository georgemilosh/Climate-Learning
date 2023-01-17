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
for i in range(nfolds):
    history = np.load(f'{folder}/fold_{i}/history_vae', allow_pickle=True)#.item()
    print(history.keys())


    for key in history:
        epochs = range(1, len(history[key])+1)
        print(epochs)
        if i ==nfolds-1:
            label =f'{key}_{i}'
        else:
            label =''
        if key=='loss':
            cmap = plt.cm.Reds(color_idx[i])
            ln1 = ax1.plot(epochs, history[key], label =label, linestyle='solid', marker='.', color = cmap)
        elif key=='reconstruction_loss':
            cmap = plt.cm.Blues(color_idx[i])
            ln2 = ax2.plot(epochs, history[key], label =label, linestyle='dashed', marker='*',color = cmap)
        elif key=='kl_loss':
            cmap = plt.cm.Greens(color_idx[i])
            ln3 = ax3.plot(epochs, history[key], label =label, linestyle='dotted', marker='o',color = cmap)
            ln4 = ln3.copy()
        elif key=='class_loss':
            cmap = plt.cm.Greys(color_idx[i]) 
            ln4 = ax4.plot(epochs, history[key], label =label, linestyle='dashdot', marker='x',color = cmap)
        if i ==nfolds-1:
            lns = ln1+ln2+ln3+ln4


labs = [l.get_label() for l in lns]
ax4.legend(lns, labs, loc=0)

ax1.set_xlabel("Epochs")
ax1.set_ylabel("loss")
ax2.set_ylabel("reconstruction_loss")
ax3.set_ylabel("kl_loss")
ax2.grid()
fig.savefig(f"Images/history.png", bbox_inches='tight', dpi=200)
plt.show()

