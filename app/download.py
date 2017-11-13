import os
import logging
import requests
import shutil
import tarfile
from clint.textui import progress

DATASET_LOCATION = 'http://6.869.csail.mit.edu/fa17/miniplaces/data.tar.gz'
CATEGORIES_LOCATION = 'https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/categories.txt'
OBJECT_CATEGORIES_LOCATION = 'https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/object_categories.txt'
TRAIN_LABELS_LOCATION = 'https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/train.txt'
VALIDATE_LABELS_LOCATION = 'https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/val.txt'

def download_file(output_filename, url):
    logging.info("Downloading {}...".format(output_filename))
    r = requests.get(url, stream=True)
    with open(output_filename, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()

def download_dataset():
    datafile = 'data.tar.gz'
    download_file(datafile, DATASET_LOCATION)
    logging.info('Extracting {}...'.format(datafile))
    with tarfile.open(datafile) as f:
        f.extractall(path='data')
    os.unlink(datafile)

def download_data(force=False):
    logging.info("Downloading dataset...")
    if not force and os.path.isdir('data'):
        logging.error('data/ directory already downloaded. --force flag not specified. Aborting.')
        return
    if os.path.isdir('data'):
        shutil.rmtree('data')

    download_dataset()
    download_file('data/categories.txt', CATEGORIES_LOCATION)
    download_file('data/object_categories.txt', OBJECT_CATEGORIES_LOCATION)
    download_file('data/train.txt', TRAIN_LABELS_LOCATION)
    download_file('data/val.txt', VALIDATE_LABELS_LOCATION)
