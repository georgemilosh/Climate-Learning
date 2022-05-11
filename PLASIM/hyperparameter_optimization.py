# '''
# Created on 12 May 2022

# @author: Alessandro Lovo
# '''
from ast import arg
from sqlalchemy import false
import Learn2_new as ln
logger = ln.logger
ut = ln.ut
np = ln.np
keras = ln.keras
pd = ln.pd

import optuna

# log to stdout
import logging
import sys
import os
logging.getLogger().level = logging.INFO
logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]


class ScoreOptimizer():
    def __init__(self, trainer, study_name='', common_kwargs=None):
        self.trainer = trainer
        self.common_kwargs = common_kwargs or {}
        name = ln.make_run_name(study_name, **common_kwargs)
        self.study = optuna.create_study(study_name=name, storage=f'sqlite:///{name}.db', load_if_exists=True)
        

    def objective(self, trial):
        #### select hyperparameters ####

        hyp = {}
        hyp['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True) # learning rate
        hyp['batch_size'] = trial.suggest_int('batch_size', 128, 2048, log=True)

        # convolutional layers
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 4)
        hyp['conv_channels'] = []
        hyp['kernel_sizes'] = []
        hyp['strides'] = []
        hyp['batch_normalizations'] = []
        hyp['conv_dropouts'] = []
        hyp['max_pool_sizes'] = []
        for i in range(n_conv_layers):
            hyp['conv_channels'].append(trial.suggest_int(f'conv_channels_{i+1}', 8, 128))
            hyp['kernel_sizes'].append(trial.suggest_int(f'kernel_sizes_{i+1}', 2, 10))
            hyp['strides'].append(trial.suggest_int(f'strides_{i+1}', 1, hyp['kernel_sizes'][-1]))
            hyp['batch_normalizations'].append(trial.suggest_categorical(f'batch_normalizations_{i+1}', [True, False]))
            hyp['conv_dropouts'].append(trial.suggest_float(f'conv_dropouts_{i+1}', 0, 0.8))
            hyp['max_pool_sizes'].append(trial.suggest_int(f'max_pool_sizes_{i+1}', 1, 4))

        # fully connected layers
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 4)
        hyp['dense_units'] = []
        hyp['dense_activations'] = ['relu']*(n_dense_layers - 1) + [False]
        hyp['dense_dropouts'] = []
        for i in range(n_dense_layers - 1):
            hyp['dense_units'].append(trial.suggest_int(f'dense_units_{i+1}', 8, 128))
            hyp['dense_dropouts'].append(trial.suggest_float(f'dense_dropouts_{i+1}', 0, 0.8))
        hyp['dense_units'].append(2)
        hyp['dense_dropouts'].append(False)

        # remove arguments that remained empty lists (this facilitates commenting lines to remove kwargs to optimize)
        kw_to_remove = []
        for k,v in hyp:
            if isinstance(v, list) and len(v) == 0:
                kw_to_remove.append(k)
            if k in self.common_kwargs:
                raise KeyError(f'{k} appears both in common kwargs and hyp')
        for k in kw_to_remove:
            logger.warning(f'removing hyperparameter {k} from the optimization list')
            hyp.pop(k)

        #### run
        run_args = {**self.common_kwargs, **hyp}

        try:
            score, info = self.trainer._run(**run_args)

            if info['status'] in ['FAILED', 'PRUNED']:
                raise optuna.TrialPruned(f"Run {info['status']}: pruning.")

        except KeyboardInterrupt:
            raise
        except optuna.TrialPruned:
            raise
        except Exception as e:
            raise RuntimeError('Something very bad happened if we reached this block')

        return score


    def optimize(self, **kwargs):
        self.study.optimize(self.objective, **kwargs)



def main():
    if ln.deal_with_lock():
        folder = sys.argv[1]
        print(f'moving code to {folder = }')
        # copy this file
        path_to_here = ln.Path(__file__).resolve() # path to this file
        ln.shutil.copy(path_to_here, folder)
        return

    arg_dict = ln.parse_command_line()

    trainer_kwargs = ln.get_default_params(ln.Trainer)
    trainer_kwargs.pop('config')
    trainer_kwargs.pop('root_folder') # this two parameters cannot be changed
    trainer_kwargs['upon_failed_run'] = 'continue'
    for k in arg_dict:
        if k in trainer_kwargs:
            trainer_kwargs[k] = arg_dict.pop(k)

    # create trainer
    trainer = ln.Trainer(config='./config.json', **trainer_kwargs)

    # deal with telegram kwargs
    for k in trainer.telegram_kwargs:
        if k in arg_dict:
            trainer.telegram_kwargs[k] = arg_dict.pop(k)

    # check conditions
    if trainer.config_dict_flat['load_from'] is not None:
        raise ValueError('load_from is not None!')

    study_name = arg_dict.pop('study_name', 'study')
    n_trials = arg_dict.pop('n_trials', None)
    if not n_trials:
        raise ValueError('You must provide a valid number of trials with n_trials=<number of trials>')

    # create a ScoreOptimizer
    so = ScoreOptimizer(trainer=trainer, study_name=study_name, common_kwargs=arg_dict)

    # run
    so.optimize(n_trials=n_trials)



if __name__ == '__main__':
    main()
