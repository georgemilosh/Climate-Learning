# Stochastic Weather Generator

This folder was originally conceived to address training Variational Autoencoder (`VAE`). It later evolved in a general folder for Stochastic Weather Generator (`SWG`).

## Folder structure

- [vae_learn2.py](vae_learn2.py) is the main script which is responsible for training the `VAE`. It includes a range of different routines for training, testing, and analyzing the `VAE`. It overloads many of the classes and methods defined in [Learn2_new.py](../PLASIM/Learn2_new.py), [ERA_Fields_New.py](../ERA/ERA_Fields_New.py) and [TF_Fields.py](../ERA/TF_Fields.py). **Note**, you may use [vae_learn2.py](vae_learn2.py) to also train `PCA` autoencoder and also generate folder structure without any dimensionality reduction necessary to work with the routines we will discuss below. For detailed description of use consult the doc file provided in [vae_learn2.py](vae_learn2.py)
- [history.py](history.py) and [history_training.py](history_training.py) are routines which allow plotting the history of the training of the `VAE`. It is used for the outputs of [vae_learn2.py](vae_learn2.py)
-  [reconstruction.py](reconstruction.py) and [reconstruction_zg.py](reconstruction_zg.py) are routines which allow to reconstruct the input data from the latent space of the `VAE`. They are used for the outputs of [vae_learn2.py](vae_learn2.py) and can be used to visualize the method.
- [analogue_dario.py](analogue_dario.py) routine finds the analogs for `SWG`. It should be applied to the output of [vae_learn2.py](vae_learn2.py) **without** the use of `VAE` or `PCA` autoencoder. 
- [analogue_george.py](analogue_george.py) routine finds the analogs for SWG. It should be applied to the output of [vae_learn2.py](vae_learn2.py) **with** the use of `VAE` or `PCA` autoencoder.
- [committor_analogue.py](committor_analogue.py) computes committor using the Matrix of analogs created by [analogue_dario.py](analogue_dario.py) or [analogue_george.py](analogue_george.py).
- [trajectory_analogue.ipynb](trajectory_analogue.ipynb) samples synthetic trajectories using the Matrix of analogs created by [analogue_dario.py](analogue_dario.py) or [analogue_george.py](analogue_george.py).
- [test_committor_dario.ipynb](test_committor_dario.ipynb) defines and uses a routine which compares the skill of the [Learn2_new.py](../PLASIM/Learn2_new.py) trained `CNN` with of `SWG` based on the output of [committor_analogue.py](committor_analogue.py).

## Usage:

If you would like to work with `SWG` you must run [vae_learn2.py](vae_learn2.py) even if you do **not** intend to train `VAE` because [vae_learn2.py](vae_learn2.py) generates the necessary folder structure consistent with k-fold cross validation. The first step is to call:
```
python vae_learn2.py <folder_name>
```
*Don't forget to set up the correct **kwargs*

After the routine has been called you may inspect the quality of reconstruction (if `VAE` was used) via the following script
```
python reconstruction.py <folder_name> <checkpoint_number> <random_seed>
```

To inspect the loss during training use may use
```
python history.py <folder_name> <number_of_folds>
python histroy_training.py <folder_name> <number_of_folds>
```
### Direct input
If you are don't want to use dimensinoality reduction, to compute the matrix of analogs for the SWG you may use
```
python analogue_dario.py <folder> <coefficients> <NN>
```
where typical usage is to set `<coefficients>=1,5,10,50,100,500` and `<NN>=100`

### Dimansionality reduction
In this case you run the following
```
analogue_george.py <folder> <coefficients> <NN>
```

To use the matrix of analogs and compute the committor function (independent of whether or not you use dimesnionality reduction) you should use
```
python committor_analogue.py <committor_file>
```
To investigate the resulting skill you should look at the notebook [test_committor_dario.ipynb](test_committor_dario.ipynb)

To use the analogs and compute the long trajectories you may inspect the notebook [trajectory_analogue.ipynb](trajectory_analogue.ipynb)
