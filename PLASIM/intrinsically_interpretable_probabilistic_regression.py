# '''
# Created on January 11, 2024

# @author: Alessandro Lovo
# '''
import committor_projection_NN as iinn
import probabilistic_regression as pr
# we need to import iinn first as both pr and iinn modify the create_model function. And pr adds the activation function to the core model, so it needs to come last.

ln = pr.ln

logger = ln.logger
ut = ln.ut

# log to stdout
import logging
import sys

ln.add_mod(__file__)

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

# pr.Trainer overwrote the prepate_XY mod of iin that saved the lat variable. We need to restore it
class Trainer(ln.Trainer):
    def prepare_XY(self, fields, **prepare_XY_kwargs):
        res =  super().prepare_XY(fields, **prepare_XY_kwargs)
        logger.info('Saving latitude as module level variable')
        ln.lat = self.lat
        return res
    
ln.Trainer = Trainer
ln.CONFIG_DICT = ln.build_config_dict([ln.Trainer.run, ln.Trainer.telegram]) # module level config dictionary
# now that we rebuilt the config dictionary, we have to set the default values of pr again
# TODO: improve this

ut.set_values_recursive(ln.CONFIG_DICT,
                        {
                            'return_threshold': True,
                            'loss': 'crps',
                            'return_metric': 'val_ParametricCrossEntropyLoss',
                            'monitor' : 'val_ParametricCrossEntropyLoss',
                            'metric' : 'val_ParametricCrossEntropyLoss',
                        },
                        inplace=True) 

if __name__ == '__main__':
    ln.main()
