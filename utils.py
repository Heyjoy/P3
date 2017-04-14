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

def flipImage(images,angles):
    augmented_images, augmented_angles = [],[]
    for image,angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image,1))
        augmented_angles.append(angle*-1.0)
    return augmented_images,augmented_angles
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
                path = df.IMGPath
                center_image = cv2.imread(path+os.path.split(batch_sample[0])[-1])
                center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2YUV)

                left_image = cv2.imread(path+os.path.split(batch_sample[1])[-1])
                right_image = cv2.imread(path+os.path.split(batch_sample[2])[-1])

                # if have 1 or -1 means -25 or 25, no big turn.
                center_angle = float(batch_sample[3])
                #if (center_angle == 1) or(center_angle == -1):
                #   center_angle = center_angle*0.9
                left_angle = center_angle+df.AngleOffset
                if left_angle>1:
                    left_angle = 1
                right_angle = center_angle-df.AngleOffset
                if right_angle < -1:
                    right_angle = -1

                #images.extend([center_image,left_image,right_image])
                #angles.extend([center_angle,left_angle,right_angle])
                images.extend([center_image])
                angles.extend([center_angle])
                #print(path+batch_sample[0].split('\\')[-1])
                #print(center_image)

            images,angles = flipImage(images,angles)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def normalize(image):
    return image / 127.5 - 1

def resize(image):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(image, [64, 64])

'''def plotHistogram(samples):
    from collections import Counter
    plt.figure(figsize=(16,5))
    D = len(samples)
    classes = [int(i) for i in range(n_classes)]
    y_pos = np.arange(samples)
    count = [D[i] for i in range(n_classes)]

    plt.bar(y_pos, count, align='center', alpha=0.5)
    plt.xticks(y_pos, classes)
    plt.xlabel('Label number')
    plt.ylabel('Samples number')
    plt.title('Training Set Class Distribution')
    plt.savefig("./Report_image/class_distribution.png")
    plt.show()
    print("end")'''
