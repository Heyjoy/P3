import math
import datafiled as df
import tensorflow as tf

from utils import *
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def end2endNiv():
    model = Sequential()
    crop_bottom = math.floor(df.imgShape[0]/6)
    crop_top = crop_bottom * 2

    model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=df.imgShape, name='input'))
    model.add(Lambda(resize))
    model.add(Lambda(normalize))

    # In: 64x64
    #Convo. layer
    model.add(Conv2D(24,(5,5),activation = 'elu') )
    model.add(MaxPooling2D((2,2)))

    #Convo. layer
    model.add(Conv2D(36,(5,5),activation = 'elu'))
    model.add(MaxPooling2D((2,2)))

    #Convo. layer
    model.add(Conv2D(48,(5,5),activation = 'elu'))
    model.add(MaxPooling2D((2,2)))

    #Convo. layer
    model.add(Conv2D(64,(3,3),activation = 'elu'))
    model.add(MaxPooling2D((2,2)))

    #Convo. layer
    #model.add(Conv2D(128, (3, 3),activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model
