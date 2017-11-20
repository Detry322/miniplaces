from app.data import DataLoaderDisk, TestDataLoader

import numpy as np
import heapq
import importlib
import logging

TEST_DATA_DIR = 'data/images/test/'

def get_config(args):
    DATA_TEST_CONFIG = {
      'data_root': 'data/images',
      'data_folder': 'test',
      'load_size': args.full_size,
      'fine_size': args.crop_size,
      'data_mean': np.asarray([0.45834960097,0.44674252445,0.41352266842])
    }

    DATA_VAL_CONFIG = {
      'data_root': 'data/images/',
      'data_list': 'data/val.txt',
      'load_size': args.full_size,
      'fine_size': args.crop_size,
      'num_categories': 100,
      'data_mean': np.asarray([0.45834960097,0.44674252445,0.41352266842]),
      'randomize': False
    }
    return DATA_TEST_CONFIG, DATA_VAL_CONFIG

def create_generator(loader, batch_size):
    while True:
        yield loader.next_batch(batch_size)

def get_top_k(result, k=5):
    results = [(-r, i) for r, i in zip(result, range(100))]
    heapq.heapify(results)
    return [heapq.heappop(results)[1] for i in range(k)]

def evaluate_model(args):
    DATA_TEST_CONFIG, DATA_VAL_CONFIG = get_config(args)
    model_type = args.model_type
    model_file = args.model_file
    model_module = importlib.import_module('app.models.{}'.format(model_type))
    logging.info("Evaluating {} model...".format(model_type))

    logging.info("Loading model from {}...".format(model_file))
    model = model_module.load_model(model_file)

    val_loader = DataLoaderDisk(**DATA_VAL_CONFIG)
    test_loader = TestDataLoader(**DATA_TEST_CONFIG)

    logging.info("Evaluating on val set...")
    val_results = model.evaluate_generator(
        generator=create_generator(val_loader, args.batch_size),
        steps = (val_loader.num / args.batch_size)
    )

    logging.info("Validation results -- {}".format(', '.join('{}: {}'.format(name, result) for name, result in zip(model.metrics_names, val_results))))
    logging.info("Running on test set...")

    results = model.predict_generator(
        generator=create_generator(test_loader, test_loader.batch_size),
        steps=(test_loader.size() / args.batch_size),
        verbose=1
    )

    filenames = test_loader.filenames()

    predictions = []
    for i in range(test_loader.size()):
        prediction = results[i]
        top = get_top_k(prediction)
        image_name = filenames[i]
        predictions.append((image_name, top))

    predictions.sort()

    logging.info("Creating {}...".format(args.output))
    with open(args.output, 'w') as f:
        for filename, topk in predictions:
            f.write("{} {}\n".format(filename, ' '.join(str(i) for i in topk)))


