import argparse
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
np.random.seed(234421)

from app.download import download_data
from app.train import train_model
from app.evaluate import evaluate_model
from app.models import show_info

def get_args():
    parser = argparse.ArgumentParser(description='Miniplaces project to classify images')
    parser.add_argument('--download', help='Download dataset', action='store_true')
    parser.add_argument('--train', help='Train model', action='store_true')
    parser.add_argument('--evaluate', help='Evaluate model', action='store_true')
    parser.add_argument('--output', help='Where to store evaluation results', type=str)
    parser.add_argument('--info', help='Print information about a model', action='store_true')
    parser.add_argument('--force', help='Force an action', action='store_true')
    parser.add_argument('--model_type', help='The model to train with', type=str, default='basic_model')
    parser.add_argument('--model_file', help='The h5 model file to input/output.', type=str)
    parser.add_argument('--batch_size', help='The batch size to train on.', type=int, default=25)
    parser.add_argument('--epochs', help='The batch size to train on.', type=int, default=40)
    args = parser.parse_args()
    if not args.model_file:
        args.model_file = 'models/{}.h5'.format(args.model_type)
    if not args.output:
        args.output = 'results/{}.txt'.format(args.model_type)
    return args

def main():
    args = get_args()
    if args.download:
        download_data(force=args.force)
    elif args.train:
        train_model(args)
    elif args.evaluate:
        evaluate_model(args)
    elif args.info:
        show_info(args)


if __name__ == '__main__':
    main()
