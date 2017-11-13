import importlib
import os
import numpy as np
import logging

from keras.callbacks import ModelCheckpoint

from app.data import DataLoaderDisk

DATA_TRAIN_CONFIG = {
  'data_root': 'data/images/',
  'data_list': 'data/train.txt',
  'load_size': 256,
  'fine_size': 224,
  'data_mean': np.asarray([0.45834960097,0.44674252445,0.41352266842]),
  'randomize': True
}

DATA_VAL_CONFIG = {
  'data_root': 'data/images/',
  'data_list': 'data/val.txt',
  'load_size': 256,
  'fine_size': 224,
  'data_mean': np.asarray([0.45834960097,0.44674252445,0.41352266842]),
  'randomize': False
}

def create_generator(loader, batch_size):
    while True:
        yield loader.next_batch(batch_size)

def train_model(args):
    model_type = args.model_type
    model_file = args.model_file

    train_loader = DataLoaderDisk(**DATA_TRAIN_CONFIG)
    val_loader = DataLoaderDisk(**DATA_VAL_CONFIG)

    assert val_loader.num % args.batch_size == 0, "Batch size must be a divisor of {}".format(val_loader.num)
    steps_per_epoch = train_loader.num / args.batch_size
    validation_steps = val_loader.num / args.batch_size

    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Epochs: {}".format(args.epochs))
    logging.info("Batches per epoch: {}".format(steps_per_epoch))

    model_module = importlib.import_module('app.models.{}'.format(model_type))
    if os.path.isfile(model_file):
        logging.info("Loading model from {}...".format(model_file))
        model = model_module.load_model(model_file)
    else:
        logging.info("Creating new {} model...".format(model_type))
        model = model_module.create_model()
        model_module.compile_model(model)

    logging.info("Training {} model...".format(model_type))
    checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(
        generator=create_generator(train_loader, args.batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        callbacks=callbacks_list,
        validation_data=create_generator(val_loader, args.batch_size),
        validation_steps=validation_steps
    )
