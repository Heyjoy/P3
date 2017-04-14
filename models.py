#import math
import datafiled as df
import tensorflow as tf

from utils import *
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
def end2endNiv():
    model = Sequential()
    model.add(Cropping2D(cropping=((df.cropTop, df.cropBottom), (0, 0)), input_shape=df.ImgShape, name='input'))
    #model.add(GaussianNoise(df.GaussianNoiseStddev))
    model.add(Lambda(resize))
    model.add(Lambda(normalize))

    # In: 64x64
    #Convo. layer
    model.add(Conv2D(24,5,5,activation = 'elu'))
    model.add(MaxPooling2D((2,2)))

    #Convo. layer
    model.add(Conv2D(36,5,5,activation = 'elu'))
    model.add(MaxPooling2D((2,2)))

    #Convo. layer
    model.add(Conv2D(48,5,5,activation = 'elu'))
    #Convo. layer
    model.add(Conv2D(64,3,3,activation = 'elu'))
    model.add(Conv2D(64,3,3,activation = 'elu'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model
