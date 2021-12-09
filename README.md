# Climate-Learning

## Data Analysis

This repository includes various routines used to Analyze various climate models which a priori include
- Plasim (Intermediate complexity)
- CESM (High complexity)
- ERA Reanalysis (observational)
The functions and classes we define can be used for estimation of heat wave occurances 
![Heat waves in Scandinavia modelled by CESM](CESM/IMages/Scandinavia_3.5.png)
## Rare events
We are interested in predicting rare eevents such as heat waves or cold spells etc. We use climate models because the data is scarce and we are intersted in large scale long duration events

## Machine Learning
We use neural networks to compute committor functions. Computations are performed on the cluster [Centre Blaise Pascal](https://www.cbp.ens-lyon.fr/doku.php) at ENS de Lyon


## Rare event algorithm
Because the events are rare we pursue importance sampling that can be achieved by geneological algorithms or other types of algorithms

## User guide

```
import tensorflow as tf
```
