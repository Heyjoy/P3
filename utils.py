import csv
import cv2
import os
import sklearn

import numpy as np
import datafiled as df
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def getTrainDate():
    samples = []
    count = 0
    with open(df.CSVPath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if (line[0] != 'center' ):
                if(float(line[3]) == 0 and count!=df.zeroSteeringCount):
                    count += 1
                elif(float(line[3]) == 0 and count>=df.zeroSteeringCount):
                    count = 0
                    samples.append(line)
                else:
                    samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=df.TrainTestSplitSize)
    return train_samples, validation_samples

### plot the training and validation loss for each epoch
def plotHistroy(history_object):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# generator samples for training.
def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image,angle = getImage(batch_sample)
                images.extend([image])
                angles.extend([angle])
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def normalize(image):
    return image / 127.5 - 1

def resize(image):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(image, [64, 64])
def getImage(batch_sample):
    path = df.IMGPath

    #image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    # random choose Left Center Right, each 30%
    diceLCR = np.random.randint(0, 3)
    # Center
    if diceLCR == 0 :
        image = cv2.imread(path+os.path.split(batch_sample[0])[-1])
        angle = float(batch_sample[3])
    # Left
    elif diceLCR == 1 :
        image = cv2.imread(path+os.path.split(batch_sample[1])[-1])
        angle = float(batch_sample[3])+df.AngleOffset
    # Right
    elif diceLCR == 2:
        image = cv2.imread(path+os.path.split(batch_sample[2])[-1])
        angle = float(batch_sample[3])-df.AngleOffset
    else:
        print("Error happend.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, angle = process_image(image, angle)
    return image,angle
#implement some extra random conditions
#https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1zm9bk6xi
def process_image(image, angle):
    # for ru
    image = random_bright(image)
    image, angle = random_translate(image,angle)
    image, angle = random_flip(image, angle)
    return image,angle

def random_translate(image, angle):
    tr_x = df.x_tr_range*np.random.uniform()-df.x_tr_range/2
    angle_tr = angle + tr_x/df.x_tr_range*df.trShiftAngle

    tr_y = df.y_tr_range* np.random.uniform() - df.y_tr_range/2
    Tr_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]]) # translate Martix.
    image_tr = cv2.warpAffine(image,Tr_M,(df.ImgShape[1],df.ImgShape[0]))

    return image_tr, angle_tr


def random_bright(image):
     res = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
     random_brightness =np.random.uniform(df.RandomBrightOffset,1+df.RandomBrightOffset)
     res[:,:,2] = np.minimum(res[:,:,2]*random_brightness, 255)
     res = cv2.cvtColor(res,cv2.COLOR_HSV2RGB)
     return res
def random_flip(image, angle):
    if (np.random.rand() < df.FilpProb):
        image = cv2.flip(image,1)
        angle = angle*-1.0
    return image,angle
