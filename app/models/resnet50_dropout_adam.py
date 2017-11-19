from app.models import INPUT_SHAPE, NUM_CLASSES
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam
from keras.models import load_model as keras_load_model
from keras.applications.resnet50 import ResNet50

def create_model(weights=False, summary=True):

    shape, classes = INPUT_SHAPE, NUM_CLASSES

    res_conv = ResNet50(include_top=False, 
                   weights=None,
                   # input_shape=shape,   
                   pooling='max', 
                   classes=100)
    # res_conv.layers.pop() # Get rid of the classification layer
    # res_conv.outputs = [res_conv.layers[-1].output]
    # res_conv.layers[-1].outbound_nodes = []
    res_conv.summary()

    res_in = Input(shape)
    res_out = res_conv(res_in)

    # x = Flatten(name='flatten')(res_out)
    x = Dense(500, activation='relu', name='fc2')(res_out)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(input=res_in, output=x)
    return model

def compile_model(model):
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', \
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

def load_model(model_filename):
    return keras_load_model(model_filename)