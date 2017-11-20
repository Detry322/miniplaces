from app.models import NUM_CLASSES
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam
from keras.models import load_model as keras_load_model
from keras.applications.resnet50 import ResNet50
# from kerasmodelzoo.utils.data import download_file, load_np_data

# _VGG_19_WEIGHTS_URL = 'http://files.heuritech.com/weights/vgg19_weights.h5'

def create_model(input_size, weights=False, summary=True):

    shape, classes = input_size, NUM_CLASSES
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
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', \
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

def load_model(model_filename):
    return keras_load_model(model_filename)


