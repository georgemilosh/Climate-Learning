# '''
# Created in March 2022

# @author: Alessandro Lovo
# '''
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Learn2_new as ln
ut = ln.ut


def make_groups(runs: dict, variable: str = 'tau', config_dict_flat: dict = None) -> list[dict]:
    '''
    Divides the runs into groups according to one variable. Basically is ln.group_by_varying with some extra steps.

    Parameters
    ----------
    runs : dict
        dictionary with the runs of interest
    variable : str, optional
        variable that will be the only difference between members of the same group, by default 'tau'
    config_dict_flat : dict, optional
        values of default parameters. If provided the only member that matters is `variable`, by default None

    Returns
    -------
    list[dict]
        list of groups. Every group is a dictionary with the following structure:
        {
            'args': dict of arguments shared by the group members,
            'runs': list of dict with the runs of the members of the group
            f'{variable}': list of the values of `variable` assumed by every member
        }
    '''
    run_args = {k:v['args'] for k,v in runs.items()}
    groups = ln.group_by_varying(run_args, variable=variable, config_dict_flat=config_dict_flat)
    for g in groups:
        g['runs'] = [runs[k] for k in g['runs']]
    return groups

################################
# Retrieve network predictions #
################################

def get_q(root_folder: str, run: dict, nfolds: int = 10, fold_subfolder=None, flatten: bool = True) -> np.ndarray:
    '''
    Loads the predicted committor on the validation set for a given run over all the folds

    Parameters
    ----------
    root_folder : str
        folder where the runs.json file is located
    run : dict
        dictionary with the informations of the run, as extracted from runs.json
    nfolds : int, optional
        number of folds, by default 10
    fold_subfolder : str, list[str] or None, optional
        subfolder inside every fold folder where the checkpoints are located, by default None
        If list the names of the fold subfolder for every fold
    concatenate : bool, optional
        Whether to keep the fold structure in the committor (flatten=False) or concatenate the folds (flatten=True). By default True

    Returns
    -------
    np.ndarray
        committor of shape (nfolds*data_per_fold,) if `flatten` else (nfolds, data_per_fold)
    '''
    if not isinstance(fold_subfolder, list):
        fold_subfolder = [fold_subfolder.rstrip('/') if fold_subfolder else '']*nfolds
    qs = []
    for fold in range(nfolds):
        fold_folder = f"{root_folder}/{run['name']}/fold_{fold}{fold_subfolder[fold]}"
        try:
            Y_pred = np.load(f'{fold_folder}/Y_pred_unbiased.npy')
            qs.append(Y_pred[:,1])
        except FileNotFoundError:
            try:
                qs.append(np.load(f'{fold_folder}/q_va.npy'))
            except FileNotFoundError:
                raise FileNotFoundError(f'Unable to find committor predictions in {fold_folder}')
    if flatten:
        q = np.concatenate(qs)
    else:
        q = np.vstack(qs)
    return q

def get_Y(root_folder: str, run: dict, nfolds: int = 10, fold_subfolder=None, flatten: bool = True) -> np.ndarray:
    '''
    Loads the labels of the validation set for a given run over all the folds

    Parameters
    ----------
    root_folder : str
        folder where the runs.json file is located
    run : dict
        dictionary with the informations of the run, as extracted from runs.json
    nfolds : int, optional
        number of folds, by default 10
    fold_subfolder : str, list[str] or None, optional
        subfolder inside every fold folder where the checkpoints are located, by default None
        If list the names of the fold subfolder for every fold
    concatenate : bool, optional
        Whether to keep the fold structure in the labels (flatten=False) or concatenate the folds (flatten=True). By default True

    Returns
    -------
    np.ndarray
        labels of shape (nfolds*data_per_fold,) if `flatten` else (nfolds, data_per_fold)
    '''
    if not isinstance(fold_subfolder, list):
        fold_subfolder = [fold_subfolder.rstrip('/') if fold_subfolder else '']*nfolds
    Y_vas = []
    for fold in range(nfolds):
        fold_folder = f"{root_folder}/{run['name']}/fold_{fold}{fold_subfolder[fold]}"
        Y_vas.append(np.load(f'{fold_folder}/Y_va.npy'))
    if flatten:
        Y_va = np.concatenate(Y_vas)
    else:
        Y_va = np.vstack(Y_vas)
    return Y_va


############
# Plotting #
############

def committor_histogram(q: np.ndarray, nbins: int = 50, weights: np.ndarray = None, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Creates an histogram of the committor (no plot)

    Parameters
    ----------
    q : np.ndarray
        committor
    nbins : int, optional
        number of bins of the histogram, by default 50
    weights : np.ndarray, optional
        weights for every committor datapoint, by default None
    normalize : bool, optional
        If true the sum of the values of each bin will be one (beware: the sum of the values, not the integral of the histogram, which means we ignore bin width), by default True

    Returns
    -------
    x : np.ndarray
        bin centers
    y : np.ndarray
        bin values
    '''
    y, x = np.histogram(q, bins=np.linspace(0,1, nbins+1), density=False, weights=weights)
    x = 0.5*(x[1:] + x[:-1])
    if normalize:
        y /= len(q)
    return x,y

def categorical_committor_histogram(q: np.ndarray, Y: np.ndarray, nbins: int = 50, weight: str = 'loss', normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Creates 3 histograms of the committor (no plot): [Y = 0, Y = 1, total]

    Parameters
    ----------
    q : np.ndarray
        committor
    Y : np.ndarray
        labels
    nbins : int, optional
        number of bins of the histogram, by default 50
    weight : 'loss' or None, optional
        if 'loss' the weight will be the contribution to the cross entropy loss, by default 'loss'
    normalize : bool, optional
        If true the sum of the values of each bin will be one (beware: the sum of the values, not the integral of the histogram, which means we ignore bin width), by default True

    Returns
    -------
    x : np.ndarray
        bin centers
    y0 : np.ndarray
        bin values for Y = 0 histogram
    y1 : np.ndarray
        bin values for Y = 1 histogram
    y : np.ndarray
        bin values for total histogram
    '''
    q0 = q[Y==0]
    w0 = -np.log(1 - q0) if weight == 'loss' else None
    x, y0 = committor_histogram(q0, nbins=nbins, weights=w0, normalize=False)

    q1 = q[Y==1]
    w1 = -np.log(q1) if weight == 'loss' else None
    x, y1 = committor_histogram(q1, nbins=nbins, weights=w1, normalize=False)

    y = y0 + y1
    
    if normalize:
        y0 /= len(q0)
        y1 /= len(q1)
        y /= len(q0) + len(q1)

    return x, y0, y1, y
