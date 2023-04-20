# Plasim routines

Intermediate complexity climate model. Ideal for method development: allows very long simulations

## Folder structure
- Current setup (all of the functions relate to `Learn2_new.py`):
    - `Learn2_new.py` is responsible for training the `CNN`. It includes a range of different routines for training, testing, and analyzing the `CNN`. Please consult the `tutorial.ipynb` for use cases and the documentation provided in `Learn2_new.py`. It helps to look at `example_config.json` to understand the structure of the routines and the **kwargs used.
    - `hyperparameters.py` contains the routine which optimizes hyperparameters for the `CNN` using `optuna` package. It is all automated via overloading the `trainer` class of `Learn2_new.py`.
    - `network_inspection_tools.py` contains a set of routines which allow to inspect the `CNN` and its performance. 
    - `resampling-sensitivity.py` is a routine which allows to test the sensitivity of the `CNN` to the resampling of the data. 
    - `inheritence-template.py` which shows how to efficiently inherit from the `CNN` class. 
    - `import_config.py` is a routine which allows to import the `config.json` file and overwrite the **kwargs of the `CNN` class. 
- Plasim data analysis:
    - `Plasim_France.ipynb` and `Plasim_Scandinavia.ipynb` are notebooks which contain the analysis of the Plasim data for France and Scandinavia heatwaves. It is a good example of how to work with the data, and how to use the `Plasim` class. It contains statistical analysis of the temperature time series and geopotential teleconnection patterns.
- Older routines (support discontinued):
    - `Learn2.py` is the older version of `Learn2_new.py`. 
    - `History.py` allows plotting the history of the training of the `CNN`. It is used for the outputs of `Learn2.py`.
    - `Recalc_Tau_Metrics.py`/ `Plot_Tau_Metrics.py` are routines which allow to recalculate the metrics of the `CNN` for a given `tau` value. It is used for the outputs of `Learn2.py`.
    - `Recalc_Comm_Faster.ipynb` is a notebook which plots the committor function of the `CNN` for a given `tau` value. It is used for the outputs of `Learn2.py`.
    - `Committor_LONG-smoothness.ipynb` is a notebook which investigates the smoothness of a committor learned by `CNN` as a function of `tau`. It is used for the outputs of `Learn2.py`.