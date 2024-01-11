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

## we don't have to add anything as pr and iinn are already imported and will combine well

if __name__ == '__main__':
    ln.main()
