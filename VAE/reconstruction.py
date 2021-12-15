# George Miloshevich 2021
# This routine is written for two parameters: input folder for VAE weights and the given epoch. It shows us how good the reconstruction of the VAE works
import os, sys
from glob import glob
import shutil
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set

print("==Importing tensorflow packages===")
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.insert(1, '../ERA')
import TF_Fields as tff # tensorflow routines 



print("==Checking GPU==")
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

print("==Checking CUDA==")
tf.test.is_built_with_cuda()

import importlib.util
def module_from_file(module_name, file_path): #The code that imports the file which originated the training with all the instructions
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

checkpoint_name = sys.argv[1]
checkpoint = sys.argv[2]

print("checkpoint_name = ", checkpoint_name)
print("loading module from ", checkpoint_name+'/Funs.py')
foo = module_from_file("foo", checkpoint_name+'/Funs.py')
print("==Reading data==")
X, vae, Z_DIM, N_EPOCHS, INITIAL_EPOCH, BATCH_SIZE, LEARNING_RATE, checkpoint_path, checkpoint_name, myinput, history = foo.PrepareDataAndVAE(checkpoint_name, DIFFERENT_YEARS=range(500))

print("X.shape = ", X.shape, " , np.max(X) = ", np.max(X), " , np.min (X) = ", np.min(X), " , np.mean(X[:,5,5,0]) = ", np.mean(X[:,5,5,0]), " , np.std(X[:,5,5,0]) = ", np.std(X[:,5,5,0]))

print("==loading the model: ", checkpoint_name)
vae = tf.keras.models.load_model(checkpoint_name, compile=False)


nb_zeros_c = 4-len(str(checkpoint))
checkpoint_i = '/cp-'+nb_zeros_c*'0'+str(checkpoint)+'.ckpt'

vae.load_weights(checkpoint_name+checkpoint_i)


        
example_images = X[:10]
import matplotlib.pyplot as plt
tff.plot_compare(vae,example_images)

from scipy.stats import norm
_,_,z_test = vae.encoder.predict(X[:200])
print("z_test.shape = ", z_test.shape)

Z_DIM = z_test.shape[1] #200 # Dimension of the latent vector (z)
x = np.linspace(-3, 3, 300)

fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(np.min([50, Z_DIM])):
    ax = fig.add_subplot(5, 10, i+1)
    ax.hist(z_test[:,i], density=True, bins = 20)
    ax.axis('off')
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
    ax.plot(x,norm.pdf(x))




        
tff.vae_generate_images(vae,Z_DIM,n_to_show=10)

plt.show()

