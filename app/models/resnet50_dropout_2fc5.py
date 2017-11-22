from app.models import NUM_CLASSES
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, \
                         BatchNormalization, Activation
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import load_model as keras_load_model
from keras.applications.resnet50 import ResNet50

def create_model(input_size, weights=True, summary=True):

    shape, classes = input_size, NUM_CLASSES
    res_conv = ResNet50(include_top=False, 
                   weights=None,
                   pooling='max', 
                   classes=100)

    res_conv.summary()
    res_in = Input(shape)
    res_out = res_conv(res_in)

    x = Dense(500, activation='relu', name='fc2')(res_out)
    x = Dropout(0.2)(x)
    x = Dense(500, activation='relu', name='fc3')(x)
    x = Dropout(0.2)(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(input=res_in, output=x)    
    weights = True
    if weights:
        model.load_weights('/home/nick/miniplaces/models/resnet50_dropout_2fc4.h5', by_name=True)
    return model

def compile_model(model):
    sgd = SGD(lr=0.00001, decay=1e-7, momentum=0.5, nesterov=True)
    # sgd = SGD(lr=5e-7, decay=5e-8, momentum=0.5, nesterov=False)
    # rmspop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', \
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

def load_model(model_filename):
    return keras_load_model(model_filename)