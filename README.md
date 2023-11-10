# [Climate-Learning](https://github.com/georgemilosh/Climate-Learning) repo

## Extreme events in climate

This repository includes various routines used to analyze extreme events in climate models and reanalysis.

Below we show a composite conditioned on heatwaves in Scandinavia modelled by CESM (1000 years of data):

![Heat waves in Scandinavia modelled by CESM](/CESM/Images/Scandinavia_3.5.png)

## Predicting/estimating rare events: 
We are interested in predicting rare events such as heatwaves or cold spells etc. 

## Machine Learning
We use neural networks to compute *committor functions*, conditional probability of occurrence of such events. Computations are performed on the cluster [Centre Blaise Pascal](https://www.cbp.ens-lyon.fr/doku.php) at ENS de Lyon


<!-- ## Rare event algorithm
Because the events are rare we pursue importance sampling that can be achieved by geneological algorithms or other types of algorithms -->

## User guide

- To install the repo to your local space you need to execute:
```
    git clone --recursive git@github.com:georgemilosh/Climate-Learning.git
```
(`recursive` deals with the submodule contained in this repo)

<!-- - This repo links to a submodule repo which can be installed by commands like
```
    git submodule init
    git submodule update
``` -->

- To install the relevant packages run `setup.sh` that is included
- To see how to work with our routines (such as working with data and training neural networks) consult `Plasim/tutorial.ipynb`
- Another similar tutorial can be found in `CESM/CESM_tuto.ipynb`

<!-- ```
import tensorflow as tf
``` -->

## Data

Generally the data we used in this project is quite large. However, we were able to make a portion of data available through [Zenodo](https://zenodo.org/records/10102506) which contains 500 years for anomalies of

- `tas.nc`: 2 meter temperature
- `zg500.nc`: 500 hPa geopotential height
- `mrso.nc`: soil moisture
- `lsmask.nc`: land sea mask
- `gparea.nc`: cell area

For understanding our data it helps to look at the tutorial we created for [Critical Earth ESR Workshop 2](https://github.com/AlessandroLovo/EW2-heatwaves) held in April 2022 in Nijmegen, The Netherlands.

### Folder structure:

Where we store `*.py`, `*.ipynb` scripts related to the following models and methods:

- [PLASIM](https://georgemilosh.github.io/Climate-Learning/PLASIM/): Intermediate complexity climate model. That's where most of our scripts including `Learn2_new.py` (responsible for training `CNN`) are located. Also, this folder contains `hyperparameter_optimization.py`, a very useful Bayesian hyperparameter optimizer based on `optuna` library. 
- [CESM](https://georgemilosh.github.io/Climate-Learning/CESM/): High fidelity climate model
- [ERA5](https://georgemilosh.github.io/Climate-Learning/ERA/): ECMWF reanalysis
- [SWG](https://georgemilosh.github.io/Climate-Learning/VAE/) We store Stochastic Weather Generator `SWG` related routines in the folder called `VAE` which stands for `Variational Autoencoder` experiments. Importantly this folder also contains the `SWG` without the use of `VAE`.


### Customization

One of the big advantages of this repository is that it easily supports customization.

The simplest way is to `import Learn2_new as ln` and then simply use the features that you need. But this is hardly customization.

The second option is to leverage the full potential of the code by changing only some of its functions. Examples of this are [gaussian_approx](PLASIM/gaussian_approx.py), [committor_projection_NN](PLASIM/committor_projection_NN.py) or [hyperparameter_optimization](PLASIM/hyperparameter_optimization.py).
These modules _inherit_ from `Learn2_new`.

A template for how to properly implement this inheritance is available [here](PLASIM/inheritance_template.py)

### Publications

**Citation**:

    @article{PhysRevFluids.8.040501,
        title = {Probabilistic forecasts of extreme heatwaves using convolutional neural networks in a regime of lack of data},
        author = {Miloshevich, George and Cozian, Bastien and Abry, Patrice and Borgnat, Pierre and Bouchet, Freddy},
        journal = {Phys. Rev. Fluids},
        volume = {8},
        issue = {4},
        pages = {040501},
        numpages = {40},
        year = {2023},
        month = {Apr},
        publisher = {American Physical Society},
        doi = {10.1103/PhysRevFluids.8.040501},
        url = {https://link.aps.org/doi/10.1103/PhysRevFluids.8.040501}
    }

#### Media coverage

[CNRS press](https://www.cnrs.fr/fr/changements-climatiques-une-meilleure-prediction-des-canicules-grace-lia)
