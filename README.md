# Climate-Learning

## Data Analysis

This repository includes various routines used to Analyze various climate models which a priori include
- Plasim (Intermediate complexity)
- CESM (High complexity)
- ERA Reanalysis (observational)
The functions and classes we define can be used for estimation of heat wave occurances 

![Heat waves in Scandinavia modelled by CESM](/CESM/Images/Scandinavia_3.5.png)

## Rare events
We are interested in predicting rare events such as heat waves or cold spells etc. We use climate models because the data is scarce and we are intersted in large scale long duration events

## Machine Learning
We use neural networks to compute committor functions. Computations are performed on the cluster [Centre Blaise Pascal](https://www.cbp.ens-lyon.fr/doku.php) at ENS de Lyon


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
- tbd: describe the datasets

<!-- ```
import tensorflow as tf
``` -->


### Customization

One of the big advantages of this repository is that it easily supports customization.

The simplest way is to `import Learn2_new as ln` and then simply use the features that you need. But this is hardly customization.

The second option is to leverage the full potential of the code by changing only some of its functions. Examples of this are [gaussian_approx](PLASIM/gaussian_approx.py), [committor_projection_NN](PLASIM/committor_projection_NN.py) or [hyperparameter_optimization](PLASIM/hyperparameter_optimization.py).
These modules _inherit_ from `Learn2_new`.

A template for how to properly implement this inheritance is available [here](PLASIM/inheritance_template.py)
