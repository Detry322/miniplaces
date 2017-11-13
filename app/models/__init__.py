INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 100

import importlib
import logging

def show_info(args):
    logging.info("Showing {} model info...".format(args.model_type))
    model_module = importlib.import_module('app.models.{}'.format(args.model_type))
    model = model_module.create_model()
    model.summary()
