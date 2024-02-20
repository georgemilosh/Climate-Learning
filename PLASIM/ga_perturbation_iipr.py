# '''
# Created on January 11, 2024

# @author: Alessandro Lovo
# '''
description = """Applies gap to simple probabilistic regression"""
dependencies = None # we don't have any dependencies intrinsic to this module.

import committor_projection_NN as iinn
iinn.enable()
import ga_perturbation as gap
iinn.disable()

ln = gap.ln

logger = ln.logger
ut = ln.ut

# log to stdout
import logging
import sys

if __name__ == '__main__':
    logging.getLogger().level = logging.INFO
    logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

## we don't have to add anything as pr and iinn are already imported and will combine well
def enable():
    # we need to enable iinn first as both pr and iinn modify the create_model function. And pr adds the activation function to the core model, so it needs to come last.
    iinn.enable()
    gap.enable()
    ln.add_mod(__file__, description, dependencies)

def disable():
    # when disabling we go in reverse order: pr and then iinn
    ln.remove_mod(__file__)
    gap.disable()
    iinn.disable()

if __name__ == '__main__':
    enable()
    ln.main()
