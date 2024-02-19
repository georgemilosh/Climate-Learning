# '''
# Created on January 11, 2024

# @author: Alessandro Lovo
# '''
description = """Applies gap to simple probabilistic regression"""
dependencies = None # we don't have any dependencies intrinsic to this module.

import ga_perturbation as gap
gap.enable() # we have to activate gap first, before even importing pr, as both pr and gap modify the create_model function. This way pr will see the original ln as the one modified by gap.
import probabilistic_regression as pr
gap.disable()

ln = pr.ln

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
    gap.enable()
    pr.enable()
    ln.add_mod(__file__, description, dependencies)

def disable():
    # when disabling we go in reverse order: pr and then iinn
    ln.remove_mod(__file__)
    pr.disable()
    gap.disable()

if __name__ == '__main__':
    enable()
    ln.main()
