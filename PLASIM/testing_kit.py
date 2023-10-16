
"""
This module contains functions for analyzing and visualizing the results of ML training. 
Specifically, it includes functions for importing input data, plotting cross-validation results, 
and computing model scores.

Functions:
- import_module(full_path_to_module): dynamically imports a module given its absolute path
- get_run_parameters(global_folder, group_label = 'fields'): loads run parameters from a specified folder
- CV_plot(runs, maxskill, config_dict, ln, ut, xaxis = 'lr'): plots cross-validation results for a set of runs
- optimal_run(runs, conditions, config_dict, global_folder, ln, ut): finds the id of the run given a set 
of conditions corresponding to this run
- load_XY(folder,load_data_kwargs,prepare_XY_kwargs,k_fold_cross_val_kwargs, ln, ut): loads all relevant data for validation/testing purposes
- compute_score(X, Y, run, folder,nfolds,val_folds, maxskill, load_from, k_fold_cross_val_kwargs,run_kwargs, ln, ut, CV_when_testing=True, filename='scores.npz'): computes the score for a PLASIM run
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import os.path
matplotlib.rc('font', size=18)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def import_module(full_path_to_module):
    """
    Dynamically imports a module given the full path to the module file.

    Args:
        full_path_to_module (str): The full path to the module file.

    Returns:
        module_obj: The imported module object.

    Raises:
        ImportError: If the module cannot be imported.
    """
    try:
        import os
        module_dir, module_file = os.path.split(full_path_to_module)
        module_name, _ = os.path.splitext(module_file)
        save_cwd = os.getcwd()
        os.chdir(module_dir)
        module_obj = __import__(module_name)
        module_obj.__file__ = full_path_to_module
        globals()[module_name] = module_obj
        os.chdir(save_cwd)
    except Exception as e:
        raise ImportError(e)
    return module_obj

def get_run_parameters(global_folder, group_label = 'fields', ignore=None):
    ln = import_module(f'{global_folder}/Learn2_new.py')
    ut = import_module(f'{global_folder}/ERA/utilities.py')
    ef = import_module(f'{global_folder}/ERA/ERA_Fields_New.py')
    tf = import_module(f'{global_folder}/ERA/TF_Fields.py')
    keras = ln.keras

    runs_global = ut.json2dict(f'{global_folder}/runs.json')
    runs_global = {k:v for k,v in runs_global.items() if v['status'] == 'COMPLETED'}
    config_dict = ut.json2dict(f'{global_folder}/config.json')
    percent =  ut.extract_nested(config_dict, 'percent')
    maxskill = -(percent/100.)*np.log(percent/100.)-(1-percent/100.)*np.log(1-percent/100.)
    g0 = ln.make_groups(runs_global, variable = group_label, config_dict_flat={group_label: ut.extract_nested(config_dict,group_label)}, ignore=ignore)
    print(f'{len(runs_global)} runs in groups: {g0[0][group_label]}')
    return ln, ut, ef, tf, keras, runs_global, config_dict, percent, maxskill, g0

def CV_plot(runs, maxskill, config_dict, ln, ut, xaxis = 'lr'):

    plt.figure()

    g1 = ln.make_groups(runs, variable = xaxis, config_dict_flat={xaxis: ut.extract_nested(config_dict,xaxis)})
    for g in g1:
        mean_skills = [(maxskill - run['scores']['mean'])/maxskill for run in g['runs']]
        std_skills = [-run['scores']['std']/maxskill  for run in g['runs']]
        plt.errorbar(g[xaxis], mean_skills, yerr=std_skills, label=g['args'], capsize=5, elinewidth=2, markeredgewidth=2, marker='o')
        plt.xscale('log')
        max_skill = max(mean_skills)
        max_skill_index = mean_skills.index(max_skill)
        print(g['args'], g[xaxis], f"max skill = {max_skill} at {xaxis} = { g[xaxis][max_skill_index]}" )

    plt.xlabel('lr')
    plt.ylabel('score')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend(prop={'size': 12}) # increased fontsize to 12
    plt.ylim(bottom=0)
    plt.show()

def optimal_run(runs, conditions, config_dict, global_folder, ln, ut):
    if isinstance(conditions, dict):
        run_id = list(ln.get_subset(runs, conditions=conditions, config_dict=config_dict).keys())[0]
    elif isinstance(conditions, int):
        run_id = str(conditions)
    else:
        raise ValueError('`conditions` must be dict or int')
    run = runs[run_id]
    run_kwargs = ut.set_values_recursive(config_dict, run['args']) # this ensures that the run_kwargs are consistent with the specific chosen file
    ln.check_config_dict(run_kwargs) 
    subfolder=run['name']
    folder = f'{global_folder}/{subfolder}'
    prepare_XY_kwargs = ut.extract_nested(run_kwargs, 'prepare_XY_kwargs')
    load_data_kwargs = ut.extract_nested(run_kwargs, 'load_data_kwargs')
    k_fold_cross_val_kwargs = ut.extract_nested(run_kwargs, 'k_fold_cross_val_kwargs')
    print(run['name'])
    return run_id, run, run_kwargs, subfolder, folder, prepare_XY_kwargs, load_data_kwargs, k_fold_cross_val_kwargs

def load_XY(load_data_kwargs,prepare_XY_kwargs,k_fold_cross_val_kwargs, ln, ut):
    X, Y, yp, lat, lon = ln.prepare_data(load_data_kwargs=load_data_kwargs, prepare_XY_kwargs=prepare_XY_kwargs)
    nfolds = ut.extract_nested(k_fold_cross_val_kwargs, 'nfolds')
    val_folds = ut.extract_nested(k_fold_cross_val_kwargs, 'val_folds')
    #i = 0
    #fold_folder = f'{folder}/fold_{i}'
    #_, _, X_va, Y_va = ln.k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)
    #X_va, _, _ = ln.normalize_X(X_va, fold_folder) 
    return X, Y, yp, lat, lon, nfolds, val_folds #, i, fold_folder, X_va, Y_va

def compute_score(X, Y, run, folder,nfolds,val_folds, maxskill, load_from, k_fold_cross_val_kwargs,run_kwargs, ln, ut, CV_when_testing=True, filename='scores.csv', years=0):
    score = []
    for i in range(nfolds):
        fold_folder = f'{folder}/fold_{i}'
        if CV_when_testing:
            _, _, X_va, Y_va = ln.k_fold_cross_val_split(i, X, Y, nfolds=nfolds, val_folds=val_folds)
        else:
            X_va, Y_va = X, Y
        X_va, _, _ = ln.normalize_X(X_va, fold_folder) 
        model = ln.load_model(load_from[i], compile=False)

        k_fold_cross_val_kwargs = ut.extract_nested(run_kwargs, 'k_fold_cross_val_kwargs')
        train_model_kwargs = ut.extract_nested(k_fold_cross_val_kwargs, 'train_model_kwargs')
        fullmetrics =   ut.extract_nested(k_fold_cross_val_kwargs, 'fullmetrics')
        u = ut.extract_nested(k_fold_cross_val_kwargs, 'u')
        loss = k_fold_cross_val_kwargs.pop('loss', None)
        metrics = train_model_kwargs.pop('metrics', None)
        if metrics is None:
            metrics = ln.get_default_metrics(fullmetrics, u=u)

        # optimizer
        optimizer = train_model_kwargs.pop('optimizer',ln.keras.optimizers.Adam()) # if optimizer is not provided in train_model_kwargs use Adam
        # loss function
        loss_fn = train_model_kwargs.pop('loss',None)
        if loss_fn is None:
            loss_fn = ln.get_loss_function(loss, u=u)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        score.append(model.evaluate(X_va, Y_va))
    mean_score = np.mean([s[-1] for s in score])
    std_score = np.std([s[-1] for s in score])
    mean_skill = np.mean((maxskill-[s[-1] for s in score])/maxskill)
    std_skill = np.std((maxskill-[s[-1] for s in score])/maxskill)
    print(f'score = {mean_score} +- {std_score}, skill = {mean_skill} +- {std_skill}, saving the files to {filename}')
    print(f"valid = {run['scores']['mean']} +- {run['scores']['std']}")

    if os.path.isfile(filename):
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) > 1 and rows[1][0] == str(years):
                rows[1] = [str(years), str(mean_score), str(std_score), str(mean_skill), str(std_skill)]
            else:
                rows.append([str(years), str(mean_score), str(std_score), str(mean_skill), str(std_skill)])
        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
    else:
        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['years', 'mean_score', 'std_score', 'mean_skill', 'std_skill'])
            writer.writerow([years, mean_score, std_score, mean_skill, std_skill])
    #np.savez(f'{filename}', mean_score=mean_score, std_score=std_score, mean_skill=mean_skill, std_skill=std_skill)
    return model, mean_score, std_score, mean_skill, std_skill