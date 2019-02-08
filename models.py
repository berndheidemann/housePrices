# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
import keras
def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(4, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))

    # return our model
    return model


def create_conv_regress(dim):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=(
        dim[0], dim[1], dim[2]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (5, 5), strides=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))



    model.add(Dense(1, activation="linear"))
    #plot_model(model, to_file='model.png')   # graphviz muss im Path sein https://graphviz.gitlab.io
    print(model.summary())
    return model

def create_cnn(inputShape, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
    chanDim = -1

    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


def createCombined_cnn_mlp(inputShapeCnn, inputShapeMlp, filters=(16, 32, 64)):
        # initialize the input shape and channel dimension, assuming
        # TensorFlow/channels-last ordering
        chanDim = -1

        inputs = Input(shape=inputShapeCnn)

        # loop over the number of filters
        for (i, f) in enumerate(filters):
            # if this is the first CONV layer then set the input
            # appropriately
            if i == 0:
                x = inputs

            # CONV => RELU => BN => POOL
            x = Conv2D(f, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
       # x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.4)(x)
        x = Dense(10, activation='relu')(x)
        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP

        inputMlp = Input(shape=inputShapeMlp)
        xMlp = inputMlp
        xMlp = Dense(8, activation="relu")(xMlp)
        xMlp = Dense(32, activation='relu')(xMlp)
        xMlp = Dense(10, activation="relu")(xMlp)

        xCombined = keras.layers.concatenate([x, xMlp])  # verknüpfen des Conv-Nets mit dem MLP
       # xCombined = Dense(20, activation='relu')(xCombined)
       # xCombined = Dropout(0.3)(xCombined)
       # xCombined = Dense(8, activation='relu' )(xCombined)
        xCombined = Dense(10, activation="relu")(xCombined)
        xCombined = Dense(1, activation="linear")(xCombined)

        # construct the CNN/MLP
        model = Model([inputs, inputMlp], xCombined)

        # return the CNN
        return model
