##############################################################################
# IMPORTS
##############################################################################


import gc
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from contextlib import redirect_stdout

##############################################################################
# PARAMETERS
##############################################################################


pd.set_option('display.expand_frame_repr', False)

IMG_DIR = '../../OneDrive/Temp/projh419_data/flow_from_dir/'
IMG_DIR_DF = '../../OneDrive/Temp/projh419_data/flow_from_df/'

CSV_DIR = '../../OneDrive/Temp/projh419_data/csv/'
# PLOT_DIR = '../../OneDrive/Temp/projh419_data/plots/'
# MODEL_DIR = '../../OneDrive/Temp/projh419_data/models/'
SUMMARY_DIR = '../../OneDrive/Temp/projh419_data/loop/'
LOG_DIR = '..\\..\\OneDrive\\Temp\\projh419_data\\loop\\logs\\'

EPOCHS = 20
BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150
# K=5

BALANCE_TYPE = 'no'  # no, weights, over, under

date = datetime.today().strftime('%Y-%m-%d_%H-%M')
# NAME = date + '_' + BALANCE_TYPE + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_CV'


##############################################################################
# FUNCTIONS
##############################################################################

def create_model(img_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=img_shape))
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


def input_shape():
    if tf.keras.backend.image_data_format() == 'channels_first':
        res = (3, WIDTH, HEIGHT)
    else:
        res = (WIDTH, HEIGHT, 3)
    return res


def balance_classes(balance_type, df0, df1, subset_number):
    print("\n", "SUBSET NUMBER:", subset_number)

    print("Len of train_normal (unbalanced): ", df0.shape[0])
    print("Len of train_pneumonia (unbalanced): ", df1.shape[0])

    if (balance_type == 'no') or (balance_type == 'weights'):
        df = merge_and_shuffle(df0, df1)
        return df

    else:
        if balance_type == 'under':
            df0, df1 = undersample(df0, df1)
        elif balance_type == 'over':
            df0, df1 = oversample(df0, df1)
        else:
            print('Class balancing error')
            return 0

        print("Len of train_normal ( after balance_classes() ): ", df0.shape[0])
        print("Len of train_pneumonia ( after balance_classes() ): ", df1.shape[0])

        df = merge_and_shuffle(df0, df1)

        return df


def merge_and_shuffle(df0, df1):
    """merges two dataframes and shuffles the merged dataframe"""
    df = df0
    df = df.append(df1, ignore_index=True)

    df = df.sample(frac=1)  # Shuffle
    df = df.reset_index(drop="True")

    return df


def undersample(df0, df1):
    """undersamples the data in the biggest class"""
    minimum = min(df0.shape[0], df1.shape[0])

    df0_under = df0.sample(minimum)
    df1_under = df1.sample(minimum)

    return df0_under, df1_under


def oversample(df0, df1):
    """oversamples the data in the smallest class (by duplicating items)"""
    maximum = max(df0.shape[0], df1.shape[0])

    df0_over = df0.sample(maximum, replace=True)
    df1_over = df1.sample(maximum, replace=True)

    # We do not want to have duplicates in the biggest class
    if maximum == len(df0):
        df0_over = df0
    else:
        df1_over = df1

    return df0_over, df1_over


##############################################################################
# DATAFRAME
##############################################################################

df_train_n = pd.read_csv(CSV_DIR + 'train_normal.csv')
df_train_n = df_train_n[['filename', 'normal/pneumonia']]
print(df_train_n.head(), "\n")

df_train_p = pd.read_csv(CSV_DIR + 'train_pneumonia.csv')
df_train_p = df_train_p[['filename', 'normal/pneumonia']]
print(df_train_p.head(), "\n")

'''Train Test Split'''
df_train_n, df_val_n = train_test_split(df_train_n, test_size=0.2)
df_train_p, df_val_p = train_test_split(df_train_p, test_size=0.2)

df_val = merge_and_shuffle(df_val_n, df_val_p)

'''Oversampling/undersampling the train dataframe (unbalanced, classWeights, undersample, oversample)'''
df_train = merge_and_shuffle(df_train_n, df_train_p)

##############################################################################
#
##############################################################################

input_shape = input_shape()

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                   )

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory=IMG_DIR_DF + 'train/',
                                                    x_col='filename',
                                                    y_col='normal/pneumonia',
                                                    target_size=(WIDTH, HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary'
                                                    )

val_generator = val_datagen.flow_from_dataframe(dataframe=df_val,
                                                directory=IMG_DIR_DF + 'train/',
                                                x_col='filename',
                                                y_col='normal/pneumonia',
                                                target_size=(WIDTH, HEIGHT),
                                                batch_size=BATCH_SIZE,
                                                class_mode='binary',
                                                )

##############################################################################

dense_layers = [0, 1, 2, 3]
layer_sizes = [16, 32, 64, 128]
conv_layers = [1, 2, 3]
dropouts = [0.2, 0.3, 0.45]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for dropout in dropouts:
                # NAME = f"{dropout}-drop-{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-" + date
                NAME = f"{dense_layer}-dense-{layer_size}-filters-{conv_layer}-conv-{dropout}-drop-" + date
                print(NAME)

                tensorboard = TensorBoard(log_dir=LOG_DIR + NAME)

                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), input_shape=input_shape))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                i = 2
                for l in range(conv_layer - 1):
                    model.add(Conv2D(layer_size * i, (3, 3), input_shape=input_shape))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    i += 1

                model.add(Flatten())
                for l in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dropout(dropout))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                model.compile(loss='binary_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy']
                              )

                with open(SUMMARY_DIR + NAME + '.txt', 'w') as f:
                    with redirect_stdout(f):
                        model.summary()

                H = model.fit(train_generator,
                              steps_per_epoch=train_generator.samples // BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=val_generator,
                              validation_steps=val_generator.samples // BATCH_SIZE,
                              callbacks=[tensorboard]
                              )

                del model
                tf.keras.backend.clear_session()
                gc.collect()
