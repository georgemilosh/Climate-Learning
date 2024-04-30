# '''
# Created on 30 Apr 2024

# @author: Alessandro Lovo
# '''
'''
description
-----------

Compute more metrics for the gaussian approximation
'''


import probabilistic_regression as pr # for regression metrics
pr.enable()
import gaussian_approx as ga

ln = pr.ln

logger = ln.logger
ut = ln.ut
tf = ln.tf

# log to stdout
import logging
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

trainer = None # module level trainer

metrics = {
    'CRPS': pr.CRPS(),
    'NLL': pr.ProbRegLoss(),
}

def get_run(path):
    path = Path(path)

    # we assume that path points to a run folder, if not we will throw an error
    run_name = path.name
    path = path.parent
    runs = ut.json2dict(f'{path}/runs.json')

    run_id = int(run_name.split('--',1)[0]) # if the run is not completed there will be a letter inside run ID and so it will throw an error
    run =  runs[str(run_id)]

    run['folder'] = str(path)

    return run

def get_run_kwargs(run) -> dict:
    config_dict = ut.json2dict(f'{run["folder"]}/config.json')
    year_permutation = np.load(f'{run["folder"]}/{run["name"]}/year_permutation.npy')
    mylocal = ut.extract_nested(config_dict, 'mylocal')
    shared_local = '/ClimateDynamics/MediumSpace/ClimateLearningFR/gmiloshe/PLASIM'
    if isinstance(mylocal, str) and mylocal != shared_local:
        mylocal = [mylocal, shared_local]
    elif isinstance(mylocal, list) and shared_local not in mylocal:
        mylocal.append(shared_local)
    return ut.set_values_recursive(config_dict['run_kwargs'], {**run['args'], 'year_permutation': year_permutation, 'mylocal': mylocal})

def get_arg(run, key, config_dict):
    return run['args'].get(key, ut.extract_nested(config_dict, key))

def get_completed(folder):
    return {k:v for k,v in ut.json2dict(f'{folder}/runs.json').items() if v['status'] == 'COMPLETED'}

def get_kwarg(run, key):
    config_dict = ut.json2dict(f'{run["folder"]}/config.json')
    return get_arg(run, key, config_dict)

def update_data(run):
    global trainer
    if trainer is None:
        trainer = pr.Trainer(config=ut.json2dict(f'{run["folder"]}/config.json'))

    run_kwargs = get_run_kwargs(run)
    load_data_kwargs = ut.extract_nested(run_kwargs, 'load_data_kwargs')
    trainer.load_data(**load_data_kwargs)
    prepare_XY_kwargs = ut.extract_nested(run_kwargs, 'prepare_XY_kwargs')
    trainer.prepare_XY(trainer.fields, **prepare_XY_kwargs)

def get_threshold(run):
    threshold_path = f"{run['folder']}/{run['name']}/threshold.npy"
    if os.path.exists(threshold_path):
        threshold = np.load(threshold_path).item()
        return threshold
    logger.info(f'Computing threshold for {threshold_path}')
    nfolds = get_kwarg(run, 'nfolds')

    threshold = np.infty
    for fold in range(nfolds):
        fold_subfolder = f"{run['folder']}/{run['name']}/fold_{fold}"
        A = np.load(f'{fold_subfolder}/A_va.npy')
        Y = np.load(f'{fold_subfolder}/Y_va.npy')
        threshold = min(np.min(A[Y==1]), threshold) # a needs to be shared between the folds as this was the a used during training

    np.save(threshold_path, threshold)
    return threshold

def get_msigma(run, fold):
    fold_subfolder = f"{run['folder']}/{run['name']}/fold_{fold}"
    msigma_path = f"{fold_subfolder}/msigma.npy"
    try:
        return np.load(msigma_path)
    except FileNotFoundError:
        logger.info(f'Computing msigma for {fold_subfolder}')

        a, b = np.load(f'{fold_subfolder}/ab.npy')
        threshold = get_threshold(run)
        m, sigma = ga.ab2msigma(a,b,threshold)
        np.save(msigma_path, np.array([m,sigma]))
        return m, sigma

def compute_metrics(run, recompute_f_va=False, compute_training_metrics=False):
    folder = run['folder']
    nfolds = get_kwarg(run, 'nfolds')
    logger.info(f"Computing metrics for {folder}/{run['name']}")

    threshold = get_threshold(run)
    _metrics = {k:v for k,v in metrics.items()}
    _metrics['BCE'] = pr.ParametricCrossEntropyLoss(threshold)

    for fold in range(nfolds):
        print(f'{fold = }')
        fold_subfolder = f"{folder}/{run['name']}/fold_{fold}"
        m, sigma = get_msigma(run, fold)
        A_va_ = np.load(f"{fold_subfolder}/A_va.npy")
        try:
            if recompute_f_va:
                raise FileNotFoundError
            f_va = np.load(f"{fold_subfolder}/f_va.npy")
        except FileNotFoundError:
            print('Computing f_va')
            proj = np.load(f"{fold_subfolder}/proj.npy")
            geosep = ut.Reshaper(proj != 0)
            update_data(run)
            X_tr, A_tr, X_va, A_va = ln.k_fold_cross_val_split(fold, trainer.X, trainer.Y, nfolds=nfolds)
            assert (A_va == A_va_).all(), 'computed and loaded A_va differ'

            # reshape data
            X_va = geosep.reshape(X_va)
            proj = geosep.reshape(proj)

            # normalize
            X_mean = np.load(f'{fold_subfolder}/X_mean.npy')
            X_std = np.load(f'{fold_subfolder}/X_std.npy')
            X_va = (X_va - X_mean)/X_std

            # project the data
            f_va = X_va @ proj
            assert f_va.shape == X_va.shape[:1]
            np.save(f"{fold_subfolder}/f_va.npy", f_va)

            if compute_training_metrics:
                X_tr = geosep.reshape(X_tr)
                X_tr = (X_tr - X_mean)/X_std
                f_tr = X_tr @ proj
                assert f_tr.shape == X_tr.shape[:1]
                np.save(f"{fold_subfolder}/f_tr.npy", f_tr)

        logger.debug('Computing metrics')
        # compute mu and sigma
        mu = m*f_va
        A_pred = np.stack([mu, sigma*np.ones_like(mu)], axis=-1)
        # convert to tensor
        A_va = tf.convert_to_tensor(A_va.reshape((-1,1)), dtype=tf.float32)
        A_pred = tf.convert_to_tensor(A_pred, dtype=tf.float32)

        mtrcs = {f'val_{me}': loss_fn(A_va, A_pred).numpy() for me,loss_fn in metrics.items()}

        if compute_training_metrics:
            mu = m*f_tr
            A_pred = np.stack([mu, sigma*np.ones_like(mu)], axis=-1)
            A_tr = tf.convert_to_tensor(A_tr.reshape((-1,1)), dtype=tf.float32)
            A_pred = tf.convert_to_tensor(A_pred, dtype=tf.float32)
            mtrcs.update({me: loss_fn(A_tr, A_pred).numpy() for me,loss_fn in metrics.items()})

        # get metrics from history
        history = pd.read_csv(f"{fold_subfolder}/history.csv").reset_index()

        # add the metrics we computed
        for me,l in mtrcs.items():
            history[me] = l

        logger.debug(history)
        
        # check that val_CrossEntropyLoss and val_ParametricCrossEntropyLoss are the same
        vCEL = history['val_CrossEntropyLoss'].item()
        vPCEL = history['val_BCE'].item()
        if np.abs(vCEL - vPCEL) > 1e-5:
            logger.warning(f'WARNING!!!: {vCEL = }, {vPCEL = }')

        history.to_csv(f"{fold_subfolder}/metrics.csv", index=False)
