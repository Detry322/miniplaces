from keras.utils import plot_model


from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, \
                         BatchNormalization, Activation
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam
from keras.models import load_model as keras_load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

def create_model(weights=False, summary=True):

    
    # model = InceptionV3(include_top=True, 
    #                weights=None, 
    #                input_tensor=x,  
    #                pooling='max', 
    #                classes=100)
    res_conv = ResNet50(include_top=False, 
                   weights=None,   
                   pooling='max', 
                   classes=500)
    res_conv.layers.pop() # Get rid of the classification layer
    # res_conv.layers.pop() # Get rid of the dropout layer
    res_conv.outputs = [res_conv.layers[-1].output]
    res_conv.layers[-1].outbound_nodes = []
    res_conv.summary()

    shape, classes = (224,224,3), 100
    res_in = Input(shape)
    res_out = res_conv(res_in)

    x = Flatten(name='flatten')(res_out)
    x = Dense(500, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = x = Dense(100, activation='softmax', name='predictions')(x)

    model = Model(input=res_in, output=x)
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

model = create_model()
plot_model(model, to_file='ResNet50_noTop.png')