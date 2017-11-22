from keras.utils import plot_model


from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, \
                         BatchNormalization, Activation
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam
from keras.models import load_model as keras_load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

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

def create_model(weights=False, summary=True):

    assert input_size == (112, 112, 3)

    res_in = Input(input_size)

    x = Conv2D(64, (7, 7), padding='same', name='conv1nmnmmn')(res_in)
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

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = GlobalMaxPooling2D()(x)

    x = Dense(500, activation='relu', name='fc2')(x)
    x = Dropout(0.25)(x)
    x = Dense(500, activation='relu', name='fc3')(x)
    x = Dropout(0.25)(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(input=res_in, output=x)    
    model.load_weights('models/last_shot_resnet4.h5', by_name=True)
    return model
    # return res_conv

    if summary:
        print(inceptionresnet_model.summary())

def compile_model(model):
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', \
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

def load_model(model_filename):
    return keras_load_model(model_filename)

model = create_model((112,112,3))
plot_model(model, to_file='last_shot.png')