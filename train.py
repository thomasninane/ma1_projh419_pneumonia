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

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from metrics import *

##############################################################################
# PARAMETERS
##############################################################################


pd.set_option('display.expand_frame_repr', False)

IMG_DIR = '../../OneDrive/Temp/projh419_data/flow_from_dir/'
IMG_DIR_DF = '../../OneDrive/Temp/projh419_data/flow_from_df/'
CSV_DIR = '../../OneDrive/Temp/projh419_data/csv/'

EPOCHS = 5
BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

BALANCE_TYPE = 'no'  # no, weights, over, under
da = True
K = 5

date = datetime.today().strftime('%Y-%m-%d_%H-%M')
if da:
    NAME = date + '_' + BALANCE_TYPE + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_da'
else:
    NAME = date + '_' + BALANCE_TYPE + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS)

NAME_DIR = '..\\..\\OneDrive\\Temp\\projh419_data\\trainings\\' + NAME + '\\'
DATA_DIR = NAME_DIR + 'data\\'
LOG_DIR = NAME_DIR + 'logs\\'
MODEL_DIR = NAME_DIR + 'models\\'
PLOT_DIR = NAME_DIR + 'plots\\'


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

    print("LR: ", model.optimizer.lr)

    # model.optimizer.lr = 1e-06
    # print("LR: ", model.optimizer.lr)

    return model


def input_shape():
    if tf.keras.backend.image_data_format() == 'channels_first':
        res = (3, WIDTH, HEIGHT)
    else:
        res = (WIDTH, HEIGHT, 3)
    return res


def balance_classes(df0, df1):
    print("Len of train_normal (unbalanced): ", df0.shape[0])
    print("Len of train_pneumonia (unbalanced): ", df1.shape[0])

    if (BALANCE_TYPE == 'no') or (BALANCE_TYPE == 'weights'):
        df = merge_and_shuffle(df0, df1)
        return df

    else:
        if BALANCE_TYPE == 'under':
            df0, df1 = undersample(df0, df1)
        elif BALANCE_TYPE == 'over':
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


def train_model(img_shape, val_df, train_df):
    model = create_model(img_shape)

    '''Saving train df and val df'''
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    train_df.to_csv(DATA_DIR + "train.csv")
    val_df.to_csv(DATA_DIR + "val.csv")

    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                        directory=IMG_DIR_DF + 'train/',
                                                        x_col='filename',
                                                        y_col='normal/pneumonia',
                                                        target_size=(WIDTH, HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary'
                                                        )

    val_generator = test_datagen.flow_from_dataframe(dataframe=val_df,
                                                     directory=IMG_DIR_DF + 'train/',
                                                     x_col='filename',
                                                     y_col='normal/pneumonia',
                                                     target_size=(WIDTH, HEIGHT),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='binary',
                                                     )

    '''CALLBACKS'''
    tensorboard = TensorBoard(log_dir=LOG_DIR)

    checkpoint_path = MODEL_DIR + "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=4, min_lr=1e-7)

    # metrics_val = PrecisionRecallF1scoreMetrics(val_generator, model)

    if BALANCE_TYPE == 'weights':
        class_weights = class_weight.compute_class_weight("balanced",
                                                          np.unique(train_generator.classes),
                                                          train_generator.classes)
        print("Class weights:", class_weights)

        H = model.fit(train_generator,
                      steps_per_epoch=train_generator.samples // BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=val_generator,
                      validation_steps=val_generator.samples // BATCH_SIZE,
                      callbacks=[tensorboard, cp_callback],  # , reduce_lr, metrics_val
                      class_weight=class_weights
                      )

    else:
        H = model.fit(train_generator,
                      steps_per_epoch=train_generator.samples // BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=val_generator,
                      validation_steps=val_generator.samples // BATCH_SIZE,
                      callbacks=[tensorboard, cp_callback]  # , reduce_lr, metrics_val
                      )

    hist_df = pd.DataFrame(H.history)
    hist_df.to_csv(DATA_DIR + "history.csv")
    # export_metrics(metrics_val, DATA_DIR)
    metrics_val = 0

    return H, metrics_val


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
df_train = balance_classes(df_train_n, df_train_p)

##############################################################################
#
##############################################################################


input_shape = input_shape()

if da:
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       brightness_range=[0.8, 1.2],
                                       rotation_range=10,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=False,
                                       vertical_flip=False,
                                       )
else:
    train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

H, METRICS = train_model(input_shape, df_val, df_train)

###############################################################################
# GENERATING PLOTS
###############################################################################

matplotlib.use("Agg")
plt.style.use("ggplot")
X = np.arange(0, EPOCHS)
Y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def plot(title, save_name, curve1, label1, curve2, label2):
    plt.figure()
    plt.plot(X, H.history[curve1], label=label1)
    plt.plot(X, H.history[curve2], label=label2)

    plt.xticks(X)
    plt.yticks(Y)
    plt.grid(True)
    plt.title(title + "\n on pneumonia detection")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(PLOT_DIR + save_name + ".png")
    return


print("GENERATING PLOTS")

'''Making plot directory'''
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Plot loss, val_loss
plot("Training/Validation Loss", "loss", "loss", "train_loss", "val_loss", "val_loss")

# Plot acc, val_acc
plot("Training/Validation Accuracy", "acc", "accuracy", "train_acc", "val_accuracy", "val_acc")

# Plot acc, loss, val_acc, val_loss
plt.figure()
plt.plot(X, H.history["loss"], label="train_loss")
plt.plot(X, H.history["val_loss"], label="val_loss")
plt.plot(X, H.history["accuracy"], label="train_acc")
plt.plot(X, H.history["val_accuracy"], label="val_acc")

plt.xticks(X)
plt.yticks(Y)
plt.grid(True)
plt.title("Training/Validation Accuracy and Loss \n on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + "all.png")

# Plot precision, recall and F1-score for each class
# plot_metrics(METRICS, X, PLOT_DIR, 0)
# plot_metrics(METRICS, X, PLOT_DIR, 1)

print("FINISHED")
