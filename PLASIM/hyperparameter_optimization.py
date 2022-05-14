# '''
# Created on 12 May 2022

# @author: Alessandro Lovo
# '''
from ast import arg, literal_eval
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
        name_kwargs = {k:v for k,v in self.common_kwargs.items() if not k.startswith('prune')} # ignore kwargs related to pruning in the name of the study
        name = ln.make_run_name(study_name, **name_kwargs)
        self.study = optuna.create_study(study_name=name, storage=f'sqlite:///{name}.db', load_if_exists=True)

        self._pruned_trials = 0 # number of pruned trials in the last optimize run
        

    def objective(self, trial):
        #### select hyperparameters ####

        hyp = {}

        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True) # learning rate
        lr = literal_eval(f'{lr:.7f}') # limit the resolution of the learning rate
        hyp['lr'] = lr
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
            hyp['conv_dropouts'].append(literal_eval(f"{trial.suggest_float(f'conv_dropouts_{i+1}', 0, 0.8, step=0.01):.2f}"))
            hyp['max_pool_sizes'].append(trial.suggest_int(f'max_pool_sizes_{i+1}', 1, 4))

        # fully connected layers
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 4)
        hyp['dense_units'] = []
        hyp['dense_activations'] = ['relu']*(n_dense_layers - 1) + [None]
        hyp['dense_dropouts'] = []
        for i in range(n_dense_layers - 1):
            hyp['dense_units'].append(trial.suggest_int(f'dense_units_{i+1}', 8, 128))
            hyp['dense_dropouts'].append(literal_eval(f"{trial.suggest_float(f'dense_dropouts_{i+1}', 0, 0.8, step=0.01):.2f}"))
        hyp['dense_units'].append(2)
        hyp['dense_dropouts'].append(False)

        # remove arguments that remained empty lists (this facilitates commenting lines to remove kwargs to optimize)
        kw_to_remove = []
        for k,v in hyp.items():
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

            if info['status'] == 'FAILED': # most likely due to invalid network architecture
                self._pruned_trials += 1
                raise optuna.TrialPruned(f"Run failed: pruning.") # we prune the trial

            ## we could prune also PRUNED runs, but since we have access to a partial score on the few first folds we can keep them to instruct optuna

        except KeyboardInterrupt:
            raise
        except optuna.TrialPruned:
            raise
        except Exception as e:
            # we get an exception that is not handled by Trainer._run
            raise RuntimeError("If upon_failed_run was set to 'continue', something very bad happened if we reached this block") from e

        return score


    def optimize(self, n_trials=20, count_pruned=True, **kwargs):
        # add telegram logger
        th = self.trainer.telegram(**self.trainer.telegram_kwargs)
        logger.log(45, f"Starting {n_trials} runs")

        _n_trials = n_trials
        try:
            while _n_trials:
                self._pruned_trials = 0
                self.study.optimize(self.objective, n_trials=_n_trials, **kwargs)
                logger.log(45, f'Completed {_n_trials} runs, {self._pruned_trials} of which failed due to invalid network architecture')
                if count_pruned:
                    _n_trials = 0 # if we consider also the pruned runs, there is no second round
                else:
                    _n_trials = self._pruned_trials # new number of trials for the next round
                    logger.log(45, f'Starting another {_n_trials} runs')
            logger.log(45, '\n\nAll runs completed!')

        finally:
            # remove telegram logger
            if th is not None:
                logger.handlers.remove(th)
                logger.log(45, 'Removed telegram logger')




def main():
    if ln.deal_with_lock(additional_files=[ln.Path(__file__).resolve()]):
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
    count_pruned = arg_dict.pop('count_pruned', True)
    if not n_trials:
        raise ValueError('You must provide a valid number of trials with n_trials=<number of trials>')

    # create a ScoreOptimizer
    so = ScoreOptimizer(trainer=trainer, study_name=study_name, common_kwargs=arg_dict)

    # run
    so.optimize(n_trials=n_trials, count_pruned=count_pruned)



if __name__ == '__main__':
    main()
