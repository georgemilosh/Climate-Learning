# '''
# Created on 25 January 2022

# @author: Alessandro Lovo
# '''
'''
Usage
-----

This modules computes the metrics of an already trained network and saves them in a .csv file

To run it from terminal
    python Recalc_Metrics.py <folder> <options>

where <folder> is either the folder of a run or its parent, in which case the metrics will be computed for every successful run.
options can be provided either like
    metric val_MCC direction maximize
or
    metric=val_MCC direction=maximize

possible options are
metric: the metric to use for finding the optimal checkpoint, default val_CustomLoss
direction: either maximize or minimize, default minimize
first_epoch: the number of the first epoch, default 1
'''

import numpy as np
import sys
import os
from tensorflow import keras
import pandas as pd
import ast
from pathlib import Path
import logging

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.handlers = [logging.StreamHandler(sys.stdout)]
else:
    logger = logging.getLogger(__name__)
logger.level = logging.INFO

this_module = sys.modules[__name__]

# ensure the proper namescapes
path_to_PLASIM = str(Path(__file__).resolve().parent)
if not path_to_PLASIM in sys.path:
    sys.path.insert(1,path_to_PLASIM)
path_to_project_root = str(Path(__file__).resolve().parent.parent)
if not path_to_project_root in sys.path:
    sys.path.insert(1, path_to_project_root)

import ERA.ERA_Fields_New as ef
import ERA.utilities as ut
import Learn2_new as ln

def usage(): 
    '''
    Returns the documentation of this module that explains how to use it.
    '''
    return this_module.__doc__

def compute_metrics(Y_test, Y_pred_prob, percent, u=1, assignment_threshold=None):
    '''
    Computes several metrics from labels and predicted prbabilities

    Parameters
    ----------
    Y_test : np.ndarray of shape (n,)
        Has values in {0 (no heatwave), 1 (heatwave)}
    Y_pred_prob : np.ndarray of shape (n, 2)
        Probability that the event is or not a heatwave
    percent : float between 0 and 100
        Percentage associated to how rare the events are
    u : float >= 1, optional
        undersampling factor, used to unbias the probabilities, by default 1
    assignment_threshold : float in [0,1], optional
        If provided events are considered heatwaaves if their probability is higher than `assignment_threshold`, by default None

    Returns
    -------
    dict
        dictionary of metrics
    '''
    Y_pred_unbiased = ut.unbias_probabilities(Y_pred_prob, u=u)
    perc = percent/100.
    if assignment_threshold is None:
        label_assignment = np.argmax(Y_pred_unbiased, axis=1)
    else:
        label_assignment = np.array(Y_pred_unbiased[:,1] > assignment_threshold, dtype=int)
    
    climatological_entropy = ut.entropy(perc) # this is the entropy associated with just knowing that the heatwaves cover `percent` of the data

    metrics = {}
    
    metrics['entropy'] = np.mean(ut.entropy(1 - Y_test, Y_pred_unbiased[:,0]))
    metrics['norm_entropy_skill'] = 1 - metrics['entropy']/climatological_entropy # max value is 1, if = 0 your model didn't learn any conditional probabilities, if < 0 your model really sucks!
    metrics['brier_score'] = np.mean((Y_test - Y_pred_unbiased[:,1])**2)

    metrics['MCC'] = ef.ComputeMCC(Y_test, label_assignment)[-1]
    metrics['frequency'] = np.sum(label_assignment)/len(Y_test)

    return metrics


def get_run_arguments(run_folder):
    '''
    Retrieves the values of the parameters of a run

    Parameters
    ----------
    run_folder : str
        folder where the run is located, with subfolders containing the folds

    Returns
    -------
    dict
        nested dictionary with the arguments of the run
    '''
    run_folder = run_folder.rstrip('/')
    root_folder, run_name = run_folder.rsplit('/', 1)
    run_id = run_name.split('--',1)[0]
    runs = ut.json2dict(f'{root_folder}/runs.json')
    try:
        run_id = int(run_id)
        run = runs[str(run_id)]
    except (ValueError, KeyError):
        logger.error(f'{run_name} is not a successful run')
        raise

    config_dict = ut.json2dict(f'{root_folder}/config.json')

    run_config_dict = ut.set_values_recursive(config_dict, run['args'])

    return run_config_dict


class MetricComputer():
    def __init__(self):
        self.load_data_kwargs = None
        self.prepare_XY_kwargs = None

        self.fields = None
        self.X = None
        self.Y = None

    @ut.execution_time
    @ut.indent_logger(logger)
    def prepare_data(self, run_folder, run_config_dict=None, ignore_year_permutation=False):
        '''
        Prepares the data as they were for training

        Parameters
        ----------
        run_folder : str
            folder where the run is located
        run_config_dict : dict, optional
            dictionary of the arguments for training, by default None, in which case it is computed.
        ignore_year_permutation : bool, optional
            if True, then years are loaded and mixed instead of loading them according to the 'year_permutation.npy' file

        Returns
        -------
        X : np.ndarray
            data
        Y : np.ndarray
            labels
        '''
        if run_config_dict is None:
            run_config_dict = get_run_arguments(run_folder)
        run_config_dict = ut.set_values_recursive(run_config_dict, {'flatten_time_axis': True})

        if not ignore_year_permutation:
            path_to_ylist = f'{run_folder}/year_permutation.npy'
            if os.path.exists(path_to_ylist):
                year_permutation = list(np.load(path_to_ylist, allow_pickle=True))
                run_config_dict = ut.set_values_recursive(run_config_dict, {'year_permutation': year_permutation})

        load_data_kwargs = ut.extract_nested(run_config_dict, 'load_data_kwargs')
        if self.load_data_kwargs != load_data_kwargs:
            self.load_data_kwargs = load_data_kwargs
            self.prepare_XY_kwargs = None # force the computation of prepare_XY
            self.fields = ln.load_data(**load_data_kwargs)

        prepare_XY_kwargs = ut.extract_nested(run_config_dict, 'prepare_XY_kwargs')
        if self.prepare_XY_kwargs != prepare_XY_kwargs:
            self.prepare_XY_kwargs = prepare_XY_kwargs
            self.X, self.Y, _ = ln.prepare_XY(self.fields, **prepare_XY_kwargs)

        return self.X, self.Y

    def recalc_metrics(self, run_folder, optimal_checkpoint_kwargs, save=True):
        run_folder = run_folder.rstrip('/')
        if os.path.exists(f'{run_folder}/metrics.csv'):
            run_name = run_folder.rsplit('/',1)[-1]
            logger.warning(f'Skipping {run_name}')
            metrics = pd.read_csv(f'{run_folder}/metrics.csv')
            return metrics

        run_config_dict = get_run_arguments(run_folder)

        self.prepare_data(run_folder, run_config_dict=run_config_dict) # computes self.X, self.Y

        nfolds = ut.extract_nested(run_config_dict, 'nfolds')
        val_folds = ut.extract_nested(run_config_dict, 'val_folds')
        u = ut.extract_nested(run_config_dict, 'u')
        percent = ut.extract_nested(run_config_dict, 'percent')

        # if threshold is provided, percent must be computed because threshold overrides percent
        threshold = ut.extract_nested(run_config_dict, 'threshold')
        if threshold is not None:
            percent = 100*np.sum(self.Y)/len(self.Y)

        opt_checkpoint = ln.optimal_checkpoint(run_folder,nfolds, **optimal_checkpoint_kwargs)

        metrics = {}
        # compute the metrics for each fold
        for i in range(nfolds):
            logger.info('========')
            logger.log(35, f'fold_{i} ({i+1}/{nfolds})')
            logger.info('========')
            fold_folder = f'{run_folder}/fold_{i}'
            # get the validation set
            X_tr, Y_tr, X_va, Y_va = ln.k_fold_cross_val_split(i, self.X, self.Y, nfolds=nfolds, val_folds=val_folds)

            # normalize data
            X_mean = np.load(f'{fold_folder}/X_mean.npy')
            X_std = np.load(f'{fold_folder}/X_std.npy')
            X_va = (X_va - X_mean)/X_std

            # load the model
            model = keras.models.load_model(f'{fold_folder}', compile=False)
            model.load_weights(f'{fold_folder}/cp-{opt_checkpoint:04d}.ckpt')

            # get predicted labels
            Y_pred = model.predict(X_va) # now these are logits, so we apply a softmax layer
            logger.debug(f'{Y_pred[0] = }')
            Y_pred_prob = keras.layers.Softmax()(Y_pred) # these are the probabilities
            logger.debug(f'{Y_pred_prob[0] = }')

            # compute metrics
            metrics[f'fold_{i}'] = compute_metrics(Y_va,Y_pred_prob, percent=percent, u=u)

        # create a pandas dataframe
        metrics = pd.DataFrame(metrics).T # transpose so the rows are the folds and the columns are the metrics

        # compute mean and std
        metrics.loc['mean'] = metrics.mean()
        metrics.loc['std'] = metrics.std()

        if save:
            metrics.to_csv(f'{run_folder}/metrics.csv')

        return metrics   


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(usage())
        sys.exit(0)
    
    folder = sys.argv[1].rstrip('/')

    cl_args = sys.argv[2:]
    i = 0
    arg_dict = {}
    while(i < len(cl_args)):
        key = cl_args[i]
        if '=' in key:
            key, value = key.split('=')
            i += 1
        else:
            value = cl_args[i+1]
            i += 2
        # `value` is a string. Here we try to cast it to the correct type
        try:
            value = ast.literal_eval(value)
        except:
            print(f'Could not evaluate {value}. Keeping string type')
        arg_dict[key] = value

    print(f'{arg_dict = }')

    mc = MetricComputer()

    if os.path.exists(f'{folder}/runs.json'):
        logger.info('Calculating metrics for every run')

        runs = ut.json2dict(f'{folder}/runs.json')
        runs = [r for r in runs.values() if r['status'] == 'COMPLETED'] # restrict to successfull runs

        # sort the runs such as to load data efficiently TODO

        for i,r in enumerate(runs):
            logger.log(35,f"\n\n\nComputing metrics for {r['name']} ({i+1}/{len(runs)})\n")
            metrics = mc.recalc_metrics(f"{folder}/{r['name']}", arg_dict, save=True)
            logger.log(35,metrics)

    else:
        metrics = mc.recalc_metrics(f'{folder}', arg_dict, save=True)
