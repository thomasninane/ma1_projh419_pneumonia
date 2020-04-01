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

from tensorflow.keras import backend as kB
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

##############################################################################
# PARAMETERS
##############################################################################

pd.set_option('display.expand_frame_repr', False)

IMG_DIR_DF = '../../OneDrive/Temp/projh419_data/flow_from_df/'
CSV_DIR = '../../OneDrive/Temp/projh419_data/csv/'

NAME = '2020-04-01_13-01_weights_w150_h150_e25_da'
RUN = 'r5'

BATCH_SIZE = 16
WIDTH = 150
HEIGHT = 150

NAME_DIR = '..\\..\\OneDrive\\Temp\\projh419_data\\trainings\\' + NAME + '\\'
DATA_DIR = NAME_DIR + 'data\\'
MODEL_DIR = NAME_DIR + 'models\\'


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


def predictions(ls):
    res_binary = []
    res_string = []

    for element in ls:
        if element >= 0.5:
            res_binary.append(1)
            res_string.append('PNEUMONIA')
        else:
            res_binary.append(0)
            res_string.append('NORMAL')

    return res_binary, res_string


def get_checkpoint_path(word):
    if word in NAME:
        res = MODEL_DIR + RUN + "\\cp.ckpt"
    else:
        res = MODEL_DIR + "cp.ckpt"
    print('Checkpoint Path: ', res, "\n")
    return res


def merge_and_shuffle(df0, df1):
    """merges two dataframes and shuffles the merged dataframe"""
    df = df0
    df = df.append(df1, ignore_index=True)

    df = df.sample(frac=1)  # Shuffle
    df = df.reset_index(drop="True")

    return df


##############################################################################
# DATAFRAME HANDLING
##############################################################################

df_test_n = pd.read_csv(CSV_DIR + 'test_normal.csv')
df_test_n = df_test_n[['filename', 'normal/pneumonia']]
print('Test Normal DF')
print(df_test_n.head(), "\n")

df_test_p = pd.read_csv(CSV_DIR + 'test_pneumonia.csv')
df_test_p = df_test_p[['filename', 'normal/pneumonia']]
print('Test Pneumonia DF')
print(df_test_p.head(), "\n")

# df = pd.concat([df_test_n, df_test_p], ignore_index=True)
df = merge_and_shuffle(df_test_n, df_test_p)
print('Test Combined DF')
print(df.head(), "\n")

##############################################################################
#
##############################################################################

CHECKPOINT_PATH = get_checkpoint_path('CV')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_dataframe(dataframe=df,
                                                  directory=IMG_DIR_DF + 'test/',
                                                  x_col='filename',
                                                  y_col='normal/pneumonia',
                                                  class_mode='binary',
                                                  batch_size=BATCH_SIZE,
                                                  target_size=(WIDTH, HEIGHT),
                                                  shuffle=False
                                                  )

input_shape = input_shape()
model = create_model(input_shape)
model.load_weights(CHECKPOINT_PATH)

##############################################################################
#
##############################################################################

scores = model.evaluate(test_generator)
print('Scores: ' + str(scores[1] * 100))

Y_pred = model.predict(test_generator)
df['predicted (probability)'] = Y_pred

y_pred, y_pred_str = predictions(Y_pred)
df['predicted (string)'] = y_pred_str
print(df.head())

# CONFUSION MATRIX
print('CONFUSION MATRIX')
cm = confusion_matrix(test_generator.classes, y_pred)
print(cm)

# CLASSIFICATION REPORT
print('CLASSIFICATION REPORT')
TARGET_NAMES = ['Normal', 'Pneumonia']  # 0: normal 1: pneumonia
cr = classification_report(test_generator.classes, y_pred, target_names=TARGET_NAMES)
print(cr)
