#!/bin/bash

# This script installs all the dependecies to successfully run the code of this repository
# We recommend working in a dedicated conda environment, see the guide @: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

echo "This script will intall packages and alter your current conda environment. We highly recommend running it inside a virgin new conda environment."
echo "To create a new environment run"
echo "    conda create -n <env name>"

proceed=false
read -p "Proceed? (Y/n) " -n 1 -r
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
conda install -y ipykernel
ipython kernel install --user --name="$kernel_name"

# ==== If you would like to have fancy widgets uncomment the lines below ===
# install jupyter nbextensions
#conda install -c conda-forge -y jupyter_contrib_nbextensions
#jupyter contrib nbextension install --user

# install jupyter widgets
#conda install -c conda-forge -y ipywidgets ipympl
#jupyter nbextension enable --py widgetsnbextension

# install jupyter lab
#conda install -c conda-forge -y jupyterlab

# install jupyther themes
#conda install -c conda-forge -y jupyterthemes
# ======== up to here ======================================================

echo "Installing necessary packages"

echo "Geoscience packages"
conda install -c conda-forge -y xarray netcdf4 scipy cartopy cmocean

echo "Machine Learning packages"
# if you plain to run this on cpu uncomment:
# conda install -c conda-forge -y tensorflow
# and comment out the following:
conda install -c conda-forge -y tensorflow-gpu=2.6 # at the moment our scripts work only up to this version
# the rest is ok

conda install -c conda-forge -y scikit-learn scikit-image imbalanced-learn

echo "Additional packages"
conda install -c conda-forge -y plotly tqdm uncertainties #lmfit optuna
