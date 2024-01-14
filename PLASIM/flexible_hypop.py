# '''
# Created on 23 Nov 2023

# @author: Alessandro Lovo
# '''

'''
description
-----------

This module is meant to help optimize the hyperparameters of neural networks, based on the package `optuna`.
Contrary to `hyperparameter_optimization.py`, which is a wrapper around `Learn2_new.py`, this module is meant to be more flexible and work with any code that uses the `Trainer` object.

usage
-----

Copy this template into a folder where you want to perform some runs. Edit the python file to your needs in the sections delimited by `>> EDITABLE`.

Every time you run the code, a copy of it will be saved with the name of the corresponding optuna study. This means you can keep modifying the code and run different studies in the same folder.
'''

import os
from stat import S_IREAD, S_IROTH, S_IRGRP
from ast import literal_eval
import numpy as np
import optuna

import logging
import sys

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]


# >> EDITABLE

# import the module you want to optimize with alias `core`
# e.g. import Learn2_new as core

import committor_projection_NN as core

# << EDITABLE

try:
    ln = core.ln
except AttributeError:
    ln = core
assert ln.__name__ == 'Learn2_new', 'core must import Learn2_new as ln or be Learn2_new itself'


Trainer = ln.Trainer

logger = core.logger

def new_study_name(study_name):
    if study_name is None:
        study_name = ''
    previous_studies = [f.rsplit('.',1)[0] for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.db') and f.startswith(study_name)]
    if not study_name:
        study_name = 'study'
    if len(previous_studies) == 0:
        return study_name
    
    c = 1
    while True:
        new_name = f'{study_name}_{c}'
        if new_name not in previous_studies:
            return new_name
        c += 1

class ScoreOptimizer():
    """
    This class is used to optimize the hyperparameters of the machine learning model. It uses the Optuna library. 
    The class takes in a trainer object, a study_name, and a dictionary of common_kwargs as arguments. 
    The trainer object is used to train the machine learning model and evaluate its performance. 
    The study_name is used to name the Optuna study, which stores the results of the optimization 
    process. The common_kwargs are additional arguments that are passed to the trainer object when training the model.
    """
    def __init__(self, trainer, study_name='study', common_kwargs=None, repetitions=1, load_if_exists=True):
        self.trainer = trainer
        self.common_kwargs = common_kwargs or {}
        name_kwargs = {k:v for k,v in self.common_kwargs.items() if 'prun' not in k} # ignore kwargs related to pruning in the name of the study
        self.name = ln.make_run_name(study_name, **name_kwargs)
        if not load_if_exists:
            self.name = new_study_name(self.name)
        self.study = optuna.create_study(study_name=self.name, storage=f'sqlite:///{self.name}.db', load_if_exists=load_if_exists)

        self._pruned_trials = 0 # number of pruned trials in the last optimize run

        self.repetitions = repetitions
        if self.repetitions > 1:
            self.trainer.skip_existing_run = False

    def save_script(self):
        script_root = os.path.basename(__file__).rsplit('.',1)[0] + f'___{self.name}'
        c = 1
        script = f'{script_root}___{c}.py'
        while os.path.exists(script):
            c += 1
            script = f'{script_root}___{c}.py'
        if not os.path.exists(script):
            with open(script, 'w') as f:
                f.write(f'{"#"*20}\n# This script was run for study {self.name} on {ln.ut.now()}.\n# It was the {c}th script to be run for this study.\n\n# This script was saved for logging purposes. Do not attempt to run it again\n{"#"*20}\n\n')
                f.write(open(__file__).read())
            os.chmod(script, S_IREAD | S_IROTH | S_IRGRP)
        

    def objective(self, trial):
        #### select hyperparameters ####
        """
        The ScoreOptimizer class has an objective method that defines the objective function for the Optuna study. 
        This method takes in a trial object from Optuna and uses it to suggest hyperparameters for the machine learning model. 
        These hyperparameters are then passed to the trainer object to train the model and evaluate its performance. 
        The performance score is returned as the result of the objective function.
        """
        hyp = {}

        # >> EDITABLE

        # Here you can add the hyperparameters to optimize using optuna suggestions. For example the following code optimizes learning rate, batch size and regularization
        

        # optimizing learning rate, batch size and regularization:
        
        ## learning rate
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True) # learning rate
        lr = literal_eval(f'{lr:.7f}') # limit the resolution of the learning rate
        hyp['lr'] = lr

        ## extra lr prameters
        # lr_min = trial.suggest_float('lr_min', 1e-7, lr, log=True) 
        # lr_min = literal_eval(f'{lr_min:.7f}') 
        # hyp['lr_min'] = lr_min
        # hyp['epoch_tol'] = trial.suggest_int('epoch_tol', 1, 5)
        # hyp['decay'] = literal_eval(f"{trial.suggest_float(f'decay', 0.05, 0.5, log=True):.05f}")
        # hyp['warmup'] = trial.suggest_categorical(f'warmup', [True, False])
        
        ## batch size
        hyp['batch_size'] = trial.suggest_categorical(f'batch_size', [32, 64, 128, 256, 512])

        ## regularization
        hyp['dense_l2coef'] = literal_eval(f"{trial.suggest_float(f'dense_l2coef', 1e-5, 1, log=True):.05f}")

        # Add more hyperparameters here

        # << EDITABLE
        


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
        logger.info(f'{run_args = }')
        scores = []
        run_ids = []
        run_names = []
        for rep in range(self.repetitions):
            try:
                score, info = self.trainer._run(**run_args)
                scores.append(score)

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
            
            finally:
                runs = ln.ut.json2dict('runs.json')
                run_ids.append(list(runs.keys())[-1])
                run_names.append(list(runs.values())[-1]['name'])

        trial.set_user_attr('run_ids', literal_eval(str(run_ids))) # we save the run ids to be able to easily connect the trial with its runs
        trial.set_user_attr('run_names', literal_eval(str(run_names))) # we save also the run names for redundancy in case numeration changes

        return np.mean(scores)


    def optimize(self, n_trials=20, count_pruned=True, **kwargs):
        """
        This method is used to run the optimization process. It takes in the number of trials to run and a dictionary of
        additional arguments for the Optuna study. The method runs the optimization process and prints the results to the
        console. The results are also stored in the Optuna study.
        """
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



if __name__ == '__main__':
    # >> EDITABLE
    
    # specify the name of the study
    study_name = 'study'
    # specify wheter to load an existing compatible study
    load_if_exists = True
    # specify the number of trials
    ntrials = 20
    # specify whether to count pruned trials
    count_pruned = True
    # specify the number of repetitions
    repetitions = 1
    # specify what to do if a run fails ('raise' or 'continue')
    upon_failed_run='raise'

    # specify the kwargs common to all runs
    common_kwargs = {

    }
    # << EDITABLE

    # create a trainer
    trainer = Trainer(config = './config.json', upon_failed_run=upon_failed_run)

    # create a ScoreOptimizer
    so = ScoreOptimizer(trainer=trainer, study_name=study_name, common_kwargs=common_kwargs, repetitions=repetitions, load_if_exists=load_if_exists)
    # save a copy of this file
    so.save_script()
    # run
    so.optimize(n_trials=ntrials, count_pruned=count_pruned)