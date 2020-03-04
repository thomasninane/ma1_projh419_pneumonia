##############################################################################
#IMPORTS
##############################################################################

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#matplotlib inline

##############################################################################
#PARAMETERS
##############################################################################

test_img_dir = '../MA1_PROJH419_pneumonia_data/test/'
train_img_dir = '../MA1_PROJH419_pneumonia_data/train/'
val_img_dir = '../MA1_PROJH419_pneumonia_data/val/'

nb_test_samples = 624
nb_train_samples = 5216
nb_val_samples = 16

EPOCHS = 20
BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

##############################################################################
#FUNCTIONS
##############################################################################

def createModel(inputShape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy']
                  )
    return model

def inputShape(WIDTH, HEIGHT):
    if K.image_data_format() == 'channels_first':
        inputshape = (3, WIDTH, HEIGHT)
    else:
        inputShape = (WIDTH, HEIGHT, 3)
    return inputShape
    

##############################################################################
#
##############################################################################
    
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_img_dir,
                                                    target_size = (WIDTH, HEIGHT),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'binary'
                                                    )

val_generator = test_datagen.flow_from_directory(val_img_dir,
                                                 target_size = (WIDTH, HEIGHT),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'binary'
                                                 )

test_generator = test_datagen.flow_from_directory(test_img_dir,
                                                  target_size = (WIDTH, HEIGHT),
                                                  batch_size = BATCH_SIZE,
                                                  class_mode = 'binary'
                                                  )

inputShape = inputShape(WIDTH, HEIGHT)
model = createModel(inputShape)

model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = val_generator,
                    validation_steps = nb_val_samples // BATCH_SIZE)