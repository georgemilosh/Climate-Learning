# George Miloshevich 2022
# This routine is written for two parameters: input folder for VAE weights and the given epoch. It shows us how good the reconstruction of the VAE works. 
# This version plots only geopotential and using North Atlantic Europe view
import os, sys
import shutil
from pathlib import Path
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # https://stackoverflow.com/questions/65907365/tensorflow-not-creating-xla-devices-tf-xla-enable-xla-devices-not-set
import logging
from colorama import Fore # support colored output in terminal
from colorama import Style
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO

fold_folder = Path(sys.argv[1])  # The name of the folder where the weights have been stored
checkpoint = sys.argv[2]       # The checkpoint at which the weights have been stored

import importlib.util
def module_from_file(module_name, file_path): #The code that imports the file which originated the training with all the instructions
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
logger.info(f"{Fore.BLUE}") #  indicates we are inside the routine       
logger.info(f"{fold_folder = }")
logger.info(f"loading module from  {fold_folder.parent}/Funs.py")
from importlib import import_module
#foo = import_module(fold_folder+'/Funs.py', package=None)
foo = module_from_file("foo", f'{fold_folder.parent}/Funs.py')
ef = foo.ef # Inherit ERA_Fields_New from the file we are calling
ln = foo.ln
ut = foo.ut

run_vae_kwargs = ut.json2dict(f"{fold_folder.parent}/config.json")

logger.info("==Importing tensorflow packages===")
import random as rd  
from scipy.stats import norm
import numpy as np


if len(sys.argv) > 3:
    rd.seed(a=int(sys.argv[3]))
else:
    rd.seed(a=None) # None = system time

tff = foo.tff # tensorflow routines 
ut = foo.ut # utilities
logger.info("==Checking GPU==")
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

logger.info("==Checking CUDA==")
tf.test.is_built_with_cuda()

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.mpl.geoaxes

import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeat
data_proj = ccrs.PlateCarree()

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.insert(1, '../ERA')
import cartopy_plots as cplt


logger.info("==Reading data==")

year_permutation = np.load(f'{fold_folder.parent}/year_permutation.npy')
i = int(np.load(f'{fold_folder}/fold_num.npy'))

time_start = ut.extract_nested(run_vae_kwargs, 'time_start')
time_end = ut.extract_nested(run_vae_kwargs, 'time_end')
T = ut.extract_nested(run_vae_kwargs, 'T')
if (ut.keys_exists(run_vae_kwargs, 'label_period_start') and ut.keys_exists(run_vae_kwargs, 'label_period_end')):
    label_period_start = ut.extract_nested(run_vae_kwargs, 'label_period_start')
    label_period_end = ut.extract_nested(run_vae_kwargs, 'label_period_end')
    if (label_period_start is not None) and (label_period_end is not None):
        summer_days = label_period_end - label_period_start - T + 1
    elif (label_period_start is None) and (label_period_end is not None):
        summer_days = label_period_end - time_start - T + 1
    elif (label_period_start is not None) and (label_period_end is None):
        summer_days = time_end - label_period_start - T + 1
    else:
        summer_days = time_end - time_start - T + 1
else:
    summer_days = time_end - time_start - T + 1
if ut.keys_exists(run_vae_kwargs, 'normalization_mode'):
    normalization_mode = ut.extract_nested(run_vae_kwargs, 'normalization_mode')
else:
    normalization_mode = None

if ut.keys_exists(run_vae_kwargs, 'keep_dims'):
    keep_dims = ut.extract_nested(run_vae_kwargs, 'keep_dims')
else:
    keep_dims = None
#X, lat, lon, vae, Z_DIM, N_EPOCHS, INITIAL_EPOCH, BATCH_SIZE, LEARNING_RATE, checkpoint_path, fold_folder, myinput, history = foo.PrepareDataAndVAE(fold_folder, DIFFERENT_YEARS=year_permutation[:800])
year_permutation_va = np.load(f'{fold_folder}/year_permutation_va.npy')
# Select times we want to show for reconstruction
if True: # select at random 10 years out of the validation set 
    year_permutation = list(year_permutation_va[rd.sample(range(len(year_permutation_va)), 10)])  
    day_permutation = rd.sample(range(summer_days*len(year_permutation)), 5) 
else: # avoid random permutation, just select minimum number of years allowed in fold
    year_permutation = [year_permutation_va[0]]
    day_permutation = range(23,23+5)##range(12,12+5)#[4,5,6,7,8] # the length has to be 5 for plotting purposes
    
#TODO: convert the day permuation to the appropriate day and year
logger.info(f"{year_permutation = },{day_permutation = }")

logger.info(f"{Style.RESET_ALL}")
run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'myinput' : 'N', 'year_permutation' :year_permutation})
if not os.path.exists(ut.extract_nested(run_vae_kwargs, 'mylocal')): # we are assuming that training was not run on R740server5
    run_vae_kwargs = ut.set_values_recursive(run_vae_kwargs, {'mylocal' : '/ClimateDynamics/MediumSpace/ClimateLearningFR/gmiloshe/PLASIM/'})
logger.info(f"{run_vae_kwargs = }")

history, N_EPOCHS, INITIAL_EPOCH, checkpoint_path, LAT, LON, vae, X_va, Y_va, X_tr, Y_tr, _ = foo.run_vae(fold_folder, **run_vae_kwargs)


logger.info(f"{Fore.BLUE}")
logger.info(f"{Y_va[day_permutation] = }")
logger.info(f"{X_va.shape = }, {np.max(X_va) = }, {np.min(X_va) = }, {np.mean(X_va[:,5,5,0]) = }, {np.std(X_va[:,5,5,0]) = }")
logger.info(f"==loading the model: {fold_folder}")
vae = tf.keras.models.load_model(fold_folder, compile=False)

nb_zeros_c = 4-len(str(checkpoint))
checkpoint_i = '/cp_vae-'+nb_zeros_c*'0'+str(checkpoint)+'.ckpt' # TODO: convert to f-strings

logger.info(f'load weights from {fold_folder}/{checkpoint_i}')
vae.load_weights(f'{fold_folder}/{checkpoint_i}')
      
example_images = X_va[day_permutation] # random sample of 5 images from X's 0 axis

num_samples = np.min([X_va.shape[0],200])
day_permutation2 = rd.sample(range(X_va.shape[0]), num_samples) # this one is just to show that we map to normal distribution so we have to take more points
logger.info(f'{len(day_permutation2) = }') 
_,_,z_test = vae.encoder.predict(X_va[day_permutation2])
logger.info(f"{z_test.shape = }")

Z_DIM = z_test.shape[1] #200 # Dimension of the latent vector (z)
x = np.linspace(-3, 3, 300)

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(np.min([50, Z_DIM])):
    ax = fig.add_subplot(5, 10, i+1)
    ax.hist(z_test[:,i], density=True, bins = 20)
    ax.axis('off')
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
    ax.plot(x,norm.pdf(x))

fig.tight_layout()
fig.savefig(f"Images/mapping{checkpoint}.png", bbox_inches='tight', dpi=200)
logger.info(f" saved to Images/mapping{checkpoint}.png")

def vae_generate_images(vae,Z_DIM,n_to_show=10):
    # Plot images generated by the autoencoder
    reconst_images = vae.decoder.predict(np.random.normal(0,1,size=(n_to_show,Z_DIM)))
    
    # prerolling has already occured so
    if keep_dims is None:
        reconst_images2 = reconst_images[...,2] # remove extra fields 
        reconst_images1 = reconst_images[...,1] # remove extra fields 
    else:# Here we are assuming is only one dimension to the last axis of X
        reconst_images2 = reconst_images[...,0] # remove extra fields 
        reconst_images1 = reconst_images[...,0] # remove extra fields 
    reconst_images0 = reconst_images[...,0] # remove extra fields 
    logger.info(f"{reconst_images.shape = }")
    
    if normalization_mode == 'global_logit':
        levels = np.linspace(-2,2,64)
    else:
        levels = np.linspace(0, 1, 64)
    logger.info(f"{levels = }"
               )
    fig2 = plt.figure(figsize=(25, 10))
    spec2 = gridspec.GridSpec(ncols=5, nrows=2, figure=fig2)
    iterate = 0
    jterate = 0
    ax = []
    axins= []
    for i in range(n_to_show):
        m = fig2.add_subplot(spec2[jterate,iterate], projection=ccrs.Orthographic(central_latitude=90))
        ax.append(m)
        img2 = reconst_images2[i].squeeze() 
        img1 = reconst_images1[i].squeeze()   
        img0 = reconst_images0[i].squeeze()   
        logger.info(f"{LON.shape = } ,{LAT.shape = } ,{img0.shape = }, {img1.shape = }")
        cplt.multiple_field_plot(LON, LAT, img0[...,np.newaxis],
                         projections=[
                             ccrs.Orthographic(10, 55)
                         ],
                         fig_num=8, figure=fig2, axes=m,
                         put_colorbar=False,
                         extents=[None, None, (-5, 10, 39, 60)],
                         mode='pcolormesh', levs=levels,
                         titles=f" generated",
                         draw_labels=False,
                         draw_gridlines=False,
                        )
        iterate += 1
        if iterate > 4:
            iterate = 0
            jterate = 1
    fig2.tight_layout()
    fig2.savefig(f"Images/generation{checkpoint}.png", bbox_inches='tight', dpi=200)  
    logger.info(f" saved to Images/generation{checkpoint}.png")
vae_generate_images(vae,Z_DIM,n_to_show=10)


def plot_compare(model, images=None): 
    # Plot images as well as their reconstruction
    model.encoder(images)
    mean, logvar, z_sample = model.encoder(images)
    reconst_images = model.decoder(z_sample).numpy()
    
    if keep_dims is None:
        reconst_images2 = reconst_images[...,2] # remove extra fields 
        reconst_images1 = reconst_images[...,1] # remove extra fields 
    else: # Here we are assuming is only one dimension to the last axis of X
        reconst_images2 = reconst_images[...,0] # remove extra fields 
        reconst_images1 = reconst_images[...,0] # remove extra fields 
    reconst_images0 = reconst_images[...,0] # remove extra fields 
    logger.info(f"{reconst_images.shape = }")
    if keep_dims is None:
        images2 = images[...,2]
        images1 = images[...,1]
    else:
        images2 = images[...,0]
        images1 = images[...,0]
    images0 = images[...,0]
    logger.info(f"{images0.shape = }")
    
    n_to_show = 2*images0.shape[0]
    
    if normalization_mode == 'global_logit':
        levels = np.linspace(-2,2,64)
    else:
        levels = np.linspace(0, 1, 64)
    logger.info(f"{levels = }")
    fig2 = plt.figure(figsize=(25, 10))
    spec2 = gridspec.GridSpec(ncols=5, nrows=2, figure=fig2)
    iterate = 0
    jterate = 0
    ax = []
    axins = []
    for i in range(n_to_show):
        m = fig2.add_subplot(spec2[jterate,iterate], projection=ccrs.Orthographic(central_latitude=90))
        ax.append(m)
        if jterate == 0:
            img2 = images2[i].squeeze()
            img1 = images1[i].squeeze() 
            img0 = images0[i].squeeze() 
        else:
            img2 = reconst_images2[i-5].squeeze() 
            img1 = reconst_images1[i-5].squeeze()  
            img0 = reconst_images0[i-5].squeeze()  
        logger.info(f"{iterate = }, {jterate = },{img0.shape = },{img0.min() = }, {img0.max() = }")
        
        logger.info(f"{LON.shape = } ,{LAT.shape = } ,{img0.shape = }, {img1.shape = }")
        if jterate == 0:
             cplt.multiple_field_plot(LON, LAT, img1[...,np.newaxis],
                         projections=[
                             ccrs.Orthographic(10, 55)
                         ],
                         fig_num=8, figure=fig2, axes=m,
                         put_colorbar=False,
                         extents=[None, None, (-5, 10, 39, 60)],
                         mode='pcolormesh', levs=levels,
                         titles=f" actual",
                         draw_labels=False,
                         draw_gridlines=False,
                        )
        else:
            cplt.multiple_field_plot(LON, LAT, img1[...,np.newaxis],
                         projections=[
                             ccrs.Orthographic(10, 55)
                         ],
                         fig_num=8, figure=fig2, axes=m,
                         put_colorbar=False,
                         extents=[None, None, (-5, 10, 39, 60)],
                         mode='pcolormesh', levs=levels,
                         titles=f" reconstructed",
                         draw_labels=False,
                         draw_gridlines=False,
                        )
            
            
        iterate += 1
        if iterate > 4:
            iterate = 0
            jterate = 1
    fig2.tight_layout()
    fig2.savefig(f"Images/reconstruction{checkpoint}.png", bbox_inches='tight', dpi=200)
    logger.info(f" saved to Images/generation{checkpoint}.png")
plot_compare(vae,example_images)
logger.info(f"{Style.RESET_ALL}")

plt.show()

