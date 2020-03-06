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
pd.set_option('display.expand_frame_repr', False)

img_dir = '../MA1_PROJH419_pneumonia_data/flow_from_dir/'
test_img_dir = img_dir + 'test/'
train_img_dir = img_dir + 'train/'
val_img_dir = img_dir + 'val/'

csv_dir = 'csv/'

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

def undersample(df0, df1):
    minimum = min(df0.shape[0], df1.shape[0])

    df0_under = df0.sample(minimum)
    df1_under = df1.sample(minimum)
    
    return df0_under, df1_under

def oversample(df0, df1):
    maximum = max(df0.shape[0], df1.shape[0])
    
    df0_over = df0.sample(maximum, replace=True)
    df1_over = df1.sample(maximum, replace=True)
    
    #We do not want to have duplicates in the biggest class
    if maximum==len(df0):
        df0_over=df0
    else:
        df1_over=df1
    
    return df0_over, df1_over

def regroup_and_shuffle(df0, df1):
    df = df0
    df = df.append(df1, ignore_index=True)
    
    df = df.sample(frac=1)  #Shuffle
    df = df.reset_index(drop="True")
    return df

def oversample_or_undersample(string, df0, df1):
    print("Len of train_normal (unbalanced): " , df0.shape[0])
    print("Len of train_pneumonia (unbalanced): " , df1.shape[0])
    
    if string=="under":
        df0, df1 = undersample(df0, df1)
    else:
        df0, df1 = oversample(df0, df1)
    
    print("Len of train_normal (balanced): ", df0.shape[0])
    print("Len of train_pneumonia (balanced): ", df1.shape[0])
    
    df = regroup_and_shuffle(df0, df1)
    return df
    

##############################################################################
#DATAFRAME
##############################################################################

df_train_n = pd.read_csv(csv_dir + 'train_normal.csv')
df_train_n = df_train_n[['name', 'set_name', 'normal/pneumonia']]
print(df_train_n.head())

df_train_p = pd.read_csv(csv_dir + 'train_pneumonia.csv')
df_train_p = df_train_p[['name', 'set_name', 'normal/pneumonia']]
print(df_train_p.head())


df_train_balanced = oversample_or_undersample('over', df_train_n, df_train_p)


##############################################################################
#
##############################################################################

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe = df_train_balanced,
                                                 directory = '../MA1_PROJH419_pneumonia_data/flow_from_df/train/',
                                                 x_col = 'name',
                                                 y_col = 'normal/pneumonia',
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

'''Oversampling/Undersampling'''
model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // BATCH_SIZE,
                    epochs = EPOCHS,
                    validation_data = val_generator,
                    validation_steps = nb_val_samples // BATCH_SIZE
                    )