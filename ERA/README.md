# ERA5 and other routines

This folder was originally conceived for ERA5 reanalysis. For historical reasons the development of many routines in folders such as `PLASIM`, `CESM`, `VAE` depend on the routines in this folder, most notably in the files `ERA_Fields_New.py` and `TF_Fields.py`.  

## Folder structure

- [ERA_Fields_New.py](ERA_Fields_New.py) scripts for routines and classes which read the data and prepare it for the training of the `CNN` in [Learn2_new.py](../PLASIM/Learn2_new.py) and other downstream tasks.
- [TF_Fields.py](TF_Fields.py) scripts for routines and classes which serve `tensorflow` routines.
- [ERA_France.ipynb](ERA_France.ipynb) and [ERA_Scandinavia.ipynb](ERA_Scandinavia.ipynb) notebooks which show how to use the routines in [ERA_Fields_New.py](ERA_Fields_New.py). They read ERA5 reanalysis and show statistics of time series and geopotential teleconnection patterns for heatwaves occurring in France and Scandinavia. 
- [Hayashi.py](Hayashi.py) and [run_Hayashi.sh](run_Hayashi.sh) scripts which are used to compute the Rossby wave spectrum and split it into standing/eastward and westward propagating Rossby waves.
- [ERA_Filds.py](ERA_Filds.py) old version of [ERA_Fields_New.py](ERA_Fields_New.py) which is not used anymore (useful for [Learn2.py](../PLASIM/Learn2.py))
