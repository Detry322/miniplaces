from app.models import INPUT_SHAPE, NUM_CLASSES
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential, Model 
from keras.optimizers import SGD
from keras.models import load_model as keras_load_model
from keras.applications.resnet50 import ResNet50
# from kerasmodelzoo.utils.data import download_file, load_np_data

# _VGG_19_WEIGHTS_URL = 'http://files.heuritech.com/weights/vgg19_weights.h5'

def create_model(weights=False, summary=True):

    shape, classes = INPUT_SHAPE, NUM_CLASSES
    x = Input(shape)
    model = ResNet50(include_top=True, 
                   weights=None, 
                   input_tensor=x,  
                   pooling='max', 
                   classes=100)

    return model

    if summary:
        print(resnet_model.summary())

def compile_model(model):
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', \
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

def load_model(model_filename):
    return keras_load_model(model_filename)


