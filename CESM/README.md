# CESM routines

State-of-the-art climate model.

## Folder structure
- [CESM_tuto.ipynb](CESM/CESM_tuto.ipynb) notebook that shows how [Learn2_new.py](PLASIM/Learn2_new.py) can be used to train a `CNN` on CESM data. It is a good starting point to understand how to use the `CNN` class and overwrite **kwargs controlling `config.json` files appropriate for the CESM data compared to Plasim data.
- CESM data analysis:
    - [CESM_France.ipynb](CESM/CESM_France.ipynb) and [CESM_Scandinavia.ipynb](CESM/CESM_Scandinavia.ipynb) are notebooks which contain the analysis of the CESM data for France and Scandinavia heatwaves. It is a good example of how to work with the data, and how to use the `Plasim` class. It contains statistical analysis of the temperature time series and geopotential teleconnection patterns.
    - [Hayashi.py](CESM/Hayashi.py) and [run_Hayashi.sh](CESM/run_Hayashi.sh) scripts which are used to compute the Rossby wave spectrum and split it into standing/eastward and westward propagating Rossby waves. The plotting of the spectra occurs with the routine [plotta_Hayashi.py](CESM/plotta_Hayashi.py). 
    - Some bash scripts that use `cdo` to concatenate and rename the data. 
