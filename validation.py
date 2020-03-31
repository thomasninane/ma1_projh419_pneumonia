##############################################################################
# IMPORTS
##############################################################################

import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##############################################################################
# PARAMETERS
##############################################################################

pd.set_option('display.expand_frame_repr', False)

IMG_DIR_DF = '../../OneDrive/Temp/projh419_data/flow_from_df/'

NAME = '2020-03-30_18-03_no_w150_h150_e25_da_CV'
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


def get_checkpoint_and_csv_path(word):
    if word in NAME:
        checkpoint_path = MODEL_DIR + RUN + "\\cp.ckpt"
        csv_path = DATA_DIR + RUN + '\\'
    else:
        checkpoint_path = MODEL_DIR + "cp.ckpt"
        csv_path = DATA_DIR
    print('Checkpoint Path: ', checkpoint_path, "\n")
    print('CSV Path: ', csv_path, "\n")
    return checkpoint_path, csv_path


##############################################################################
#
##############################################################################

CHECKPOINT_PATH, CSV_PATH = get_checkpoint_and_csv_path('CV')

df = pd.read_csv(CSV_PATH + 'val.csv')

val_datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = val_datagen.flow_from_dataframe(dataframe=df,
                                                directory=IMG_DIR_DF + 'train\\',
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

scores = model.evaluate(val_generator)
print('Scores: ' + str(scores[1] * 100))

Y_pred = model.predict(val_generator)
df['predicted (probability)'] = Y_pred

y_pred, y_pred_str = predictions(Y_pred)
df['predicted (string)'] = y_pred_str
print(df.head())

# CONFUSION MATRIX
print('CONFUSION MATRIX')
cm = confusion_matrix(val_generator.classes, y_pred)
print(cm)

# CLASSIFICATION REPORT
print('CLASSIFICATION REPORT')
TARGET_NAMES = ['Normal', 'Pneumonia']  # 0: normal 1: pneumonia
cr = classification_report(val_generator.classes, y_pred, target_names=TARGET_NAMES)
print(cr)
