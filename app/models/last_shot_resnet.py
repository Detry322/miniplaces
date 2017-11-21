from app.models import NUM_CLASSES
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, \
                         BatchNormalization, Activation, GlobalMaxPooling2D, AveragePooling2D
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import load_model as keras_load_model
from keras.applications.resnet50 import ResNet50, identity_block, conv_block

def create_model(input_size, weights=False, summary=True):
    assert input_size == (112, 112, 3)

    res_in = Input(input_size)

    x = Conv2D(64, (7, 7), padding='same', name='conv1')(res_in)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='d')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='e')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)

    x = Dense(500, activation='relu', name='fc3')(x)
    x = Dropout(0.25)(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(input=res_in, output=x)    
    model.load_weights('models/resnet50_dropout_sgd_cont_declrd.h5', by_name=True)
    return model

def compile_model(model):
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=5e-7, decay=5e-8, momentum=0.5, nesterov=False)
    # rmspop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', \
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

def load_model(model_filename):
    return keras_load_model(model_filename)
