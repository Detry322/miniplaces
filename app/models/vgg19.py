from app.models import INPUT_SHAPE, NUM_CLASSES
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Sequential, Model 
from keras.optimizers import SGD
from keras.models import load_model as keras_load_model
# from kerasmodelzoo.utils.data import download_file, load_np_data

# _VGG_19_WEIGHTS_URL = 'http://files.heuritech.com/weights/vgg19_weights.h5'

def create_model(weights=False, summary=True):

    # vgg19_model = Sequential()
    input_ = Input(shape=INPUT_SHAPE)
    x = input_

    # x =ZeroPadding2D((1, 1),input_shape=(3, 224, 224)))
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='conv3_4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv4_4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv5_4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classification layer
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='dense_2')(x)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
    # x = Activation("softmax")(x)
    return Model(inputs=input_, outputs=x)

    if weights:
        filepath = download_file('vgg19_weights.h5',
            _VGG_19_WEIGHTS_URL)
        vgg19_model.load_weights(filepath)

    if summary:
        print(vgg19_model.summary())

    # return vgg19_model
    # mean = load_np_data('vgg_mean.npy')

def compile_model(model):
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', \
                  metrics=['categorical_accuracy'])

def load_model(model_filename):
    return keras_load_model(model_filename)


