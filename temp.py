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

EPOCHS = 15
BATCH_SIZE = 32

WIDTH = 150
HEIGHT = 150
# K=5

BALANCE_TYPE = 'under'  # no, weights, over, under

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


model = create_model(input_shape())

with open('C:\\Thomas_Data\\OneDrive\\Temp\\projh419_data\\model.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()