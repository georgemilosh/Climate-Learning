# George Miloshevich 2022
# Plot the development of metrics during training
# Importation des librairies

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
nfolds = int(sys.argv[2])

# we must now number of folds in advance

def plot_metric(i,train_metrics, val_metrics, name_metrics,color_idx):
    epochs = range(1, len(train_metrics) + 1)
    if i == 9:
        plt.plot(epochs, val_metrics, label='test '+str(i), color = plt.cm.Reds(color_idx))
        plt.plot(epochs, train_metrics, label='train '+str(i), color = plt.cm.Blues(color_idx))
    else:
        plt.plot(epochs, val_metrics, color = plt.cm.Reds(color_idx))
        plt.plot(epochs, train_metrics, color = plt.cm.Blues(color_idx))
    plt.title('Training and validation '+name_metrics)
    plt.xlabel("Epochs")
    plt.ylabel('metrics')
    return epochs

folder = sys.argv[1]

fig = plt.figure()
for i in range(nfolds):
    history = np.load(f'{folder}/fold_{i}/history_vae', allow_pickle=True)#.item()
    print(history.keys())

    for key in history:
        epochs = range(1, len(history[key])+1)
        print(epochs)
        plt.plot(epochs, history[key], label =f'{key}_{i}')
    #loss.append(history[metric_name])
    #val_loss.append(history['val_'+metric_name])
     #   epochs = plot_metric(i,history[metric_name], history['val_'+metric_name], metric_name, color_idx[i])

    #loss = np.array(loss)
    #plt.plot(epochs, np.mean(loss,0), 'g--', linewidth = 2, label ='train average')
    #plt.plot(epochs, np.mean(val_loss,0), 'g-.', linewidth = 2, label ='test average')

plt.legend(loc='best')
plt.show()
