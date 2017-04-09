import csv
import cv2
import os
import sklearn

import numpy as np
import datafiled as df
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def getTrainDate():
    samples = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if (line[0] != 'center'):
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
                path = '../data/IMG/'
                center_image = cv2.imread(path+os.path.split(batch_sample[0])[-1])
                left_image = cv2.imread(path+os.path.split(batch_sample[1])[-1])
                right_image = cv2.imread(path+os.path.split(batch_sample[2])[-1])

                # if have 1 or -1 means -25 or 25, no big turn.
                center_angle = float(batch_sample[3])
                #if (center_angle == 1) or(center_angle == -1):
                #   center_angle = center_angle*0.9
                left_angle = center_angle*1.1
                if (left_angle > 1):
                    left_angle = 1
                right_angle = center_angle*0.9
                if (right_angle<-1):
                    right_angle = -1

                #images.extend([center_image,left_image,right_image])
                #angles.extend([center_angle,left_angle,right_angle] )
                images.extend([center_image])
                angles.extend([center_angle])
                #print(path+batch_sample[0].split('\\')[-1])
                #print(center_image)

            augmented_images, augmented_angles = [],[]
            for image,angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)
            # trim image to only see section with road
            #X_train = np.array(augmented_images)
            #y_train = np.array(augmented_angles)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def normalize(image):
    return image / 127.5 - 1

def resize(image):
    return tf.image.resize_images(image, [64, 64])
