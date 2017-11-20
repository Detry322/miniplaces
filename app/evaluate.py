from app.data import DataLoaderDisk, TestDataLoader

import numpy as np
import heapq
import importlib
import logging

TEST_DATA_DIR = 'data/images/test/'

DATA_TEST_CONFIG = {
  'data_root': 'data/images',
  'data_folder': 'test',
  'load_size': 256,
  'fine_size': 224,
  'data_mean': np.asarray([0.45834960097,0.44674252445,0.41352266842]),
  'copy_count': 1
}

DATA_VAL_CONFIG = {
  'data_root': 'data/images/',
  'data_list': 'data/val.txt',
  'load_size': 256,
  'fine_size': 224,
  'num_categories': 100,
  'data_mean': np.asarray([0.45834960097,0.44674252445,0.41352266842]),
  'randomize': False
}

def create_generator(loader, batch_size):
    while True:
        yield loader.next_batch(batch_size)

def get_top_k(result, k=5):
    results = [(-r, i) for r, i in zip(result, range(100))]
    heapq.heapify(results)
    return [heapq.heappop(results)[1] for i in range(k)]

def evaluate_model(args):
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
        steps = (val_loader.num / args.batch_size),
    )

    logging.info("Validation results -- {}".format(', '.join('{}: {}'.format(name, result) for name, result in zip(model.metrics_names, val_results))))
    logging.info("Running on test set with {} redundancy...".format(test_loader.copy_count))

    results = []
    i = 0
    for image_name, image_data in test_loader:
        i += 1
        if i % 100 == 0:
            logging.info("Predicting {}...".format(image_name))
        result = model.predict_on_batch(image_data)
        prediction = np.mean(result, axis=0)
        top = get_top_k(prediction)
        results.append((image_name, top))

    results.sort()

    logging.info("Creating {}...".format(args.output))
    with open(args.output, 'w') as f:
        for filename, topk in results:
            f.write("{} {}\n".format(filename, ' '.join(str(i) for i in topk)))


