#!/bin/bash

# This script installs all the dependecies to successfully run the code of this repository
# We recommend working in a dedicated conda environment

echo "This script will intall packages and alter your current conda environment. We highly recommend running it inside a virgin new conda environment."
echo "To create a new environment run"
echo "    conda create -n <env name>"

proceed=false
read -p "Proceed? (Y/n)" -n 1 -r
# read -n 1 REPLY
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Y]$ ]] ; then
    proceed=true
fi

if ! $proceed ; then
    echo "Aborting"
    return 0
    exit 0
fi

kernel_name="clk"

echo "Installing jupyter kernel"

# create jupyter kernel
conda install ipykernel
ipython kernel install --user --name="$kernel_name"

# install jupyter nbextensions
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# install jupyter widgets
conda install -c conda-forge ipywidgets ipympl
jupyter nbextension enable --py widgetsnbextension

# install jupyter lab
conda install -c conda-forge jupyterlab

# install jupyther themes
conda install -c conda-forge jupyterthemes

echo "Installing packages"

echo "Geoscience packages"
conda install -c conda-forge xarray netcdf4 scipy cartopy cmocean

echo "Machine Learning packages"
conda install -c conda-forge tensorflow tensorflow-gpu scikit-learn scikit-image imbalanced-learn

echo "Additional packages"
conda install -c conda-forge plotly tqdm uncertainties #lmfit optuna