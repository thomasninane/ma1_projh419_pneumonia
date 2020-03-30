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

EPOCHS = 25
BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

BALANCE_TYPE = 'no'  # no, weights, over, under
da = True
K = 5

date = datetime.today().strftime('%Y-%m-%d_%H-%M')
if da:
    NAME = date + '_' + BALANCE_TYPE + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_da_CV'
else:
    NAME = date + '_' + BALANCE_TYPE + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_CV'

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


def balance_train_set(ls):
    """
    Balances the train subsets.
    Before applying balance_classes(), we must separate the train subset in two dataframes (normal and pneumonia)
    """
    res = dict()

    print("BALANCE_TYPE:", BALANCE_TYPE)
    for i in range(len(ls)):
        df = ls[i]
        df_n = df[df['normal/pneumonia'] == 'NORMAL']
        df_p = df[df['normal/pneumonia'] == 'PNEUMONIA']

        '''we can now apply balance_classes() to the two dataframes'''
        train_subset_balanced = balance_classes(df_n, df_p, i + 1)
        res[i] = train_subset_balanced

    return res


def balance_classes(df0, df1, subset_number):
    print("\n", "SUBSET NUMBER:", subset_number)

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


def shuffle_and_divide(df):
    """returns an array of k dataframes (they all have the same size)"""
    df = df.sample(frac=1)
    df = df.reset_index(drop="True")
    res = np.array_split(df, K)

    return res


def merge_pneumonia_and_normal_dataframes(ls_of_df_1, ls_of_df_2):
    """merges the normal df and the pneumonia df into one subset of a dictionary (with class balancing and shuffling)"""
    subsets = dict()

    for i in range(K):
        subsets[i] = merge_and_shuffle(ls_of_df_1[i], ls_of_df_2[i])

    return subsets


def set_train_and_val_set(subsets):
    """returns one dictionary with the validation sets and another dictionary with the k-1 combined training sets"""
    val_dict = dict()
    train_dict = dict()

    for i in range(K):
        val_dict[i] = subsets[i]

    '''combine the other subsets'''
    for i in range(K):
        if i == 0:
            train_dict[i] = subsets[1]
            for j in range(2, K):
                train_dict[i] = train_dict[i].append(subsets[j], ignore_index=True)
        elif i == K - 1:
            train_dict[i] = subsets[0]
            for j in range(1, K - 1):
                train_dict[i] = train_dict[i].append(subsets[j], ignore_index=True)
        else:
            train_dict[i] = subsets[0]
            for j in range(1, i):
                train_dict[i] = train_dict[i].append(subsets[j], ignore_index=True)
            for j in range(i + 1, K):
                train_dict[i] = train_dict[i].append(subsets[j], ignore_index=True)

    return val_dict, train_dict


def train_model(img_shape, val_dict, train_dict):
    """trains the model for k folds and returns a dictionary containing the history of each model"""

    history = dict()
    metrics = dict()

    for i in range(K):
        print("\n", "\n", 'RUN: ' + str(i + 1))
        run = "r" + str(i + 1) + "\\"

        model = create_model(img_shape)

        '''Saving train df and val df'''
        if not os.path.exists(DATA_DIR + run):
            os.makedirs(DATA_DIR + run)
        train_dict[i].to_csv(DATA_DIR + run + "train.csv")
        val_dict[i].to_csv(DATA_DIR + run + "val.csv")

        train_generator = train_datagen.flow_from_dataframe(dataframe=train_dict[i],
                                                            directory=IMG_DIR_DF + 'train/',
                                                            x_col='filename',
                                                            y_col='normal/pneumonia',
                                                            target_size=(WIDTH, HEIGHT),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='binary'
                                                            )

        val_generator = test_datagen.flow_from_dataframe(dataframe=val_dict[i],
                                                         directory=IMG_DIR_DF + 'train/',
                                                         x_col='filename',
                                                         y_col='normal/pneumonia',
                                                         target_size=(WIDTH, HEIGHT),
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary',
                                                         )

        '''CALLBACKS'''
        tensorboard = TensorBoard(log_dir=LOG_DIR + run)

        checkpoint_path = MODEL_DIR + run + "cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=4, min_lr=1e-7)

        metrics_train = PrecisionRecallF1scoreMetrics(train_generator, model)
        metrics_val = PrecisionRecallF1scoreMetrics(val_generator, model)

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
                          callbacks=[tensorboard, cp_callback, metrics_val],  # , reduce_lr
                          class_weight=class_weights
                          )

        else:
            H = model.fit(train_generator,
                          steps_per_epoch=train_generator.samples // BATCH_SIZE,
                          epochs=EPOCHS,
                          validation_data=val_generator,
                          validation_steps=val_generator.samples // BATCH_SIZE,
                          callbacks=[tensorboard, cp_callback, metrics_val]  # , reduce_lr
                          )

        hist_df = pd.DataFrame(H.history)
        hist_df.to_csv(DATA_DIR + run + "history.csv")
        export_metrics(metrics_val, DATA_DIR + run)

        history[i] = H
        metrics[i] = metrics_val
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    return history, metrics


##############################################################################
# DATAFRAME
##############################################################################


df_train_n = pd.read_csv(CSV_DIR + 'train_normal.csv')
df_train_n = df_train_n[['filename', 'normal/pneumonia']]
print(df_train_n.head(), "\n")

df_train_p = pd.read_csv(CSV_DIR + 'train_pneumonia.csv')
df_train_p = df_train_p[['filename', 'normal/pneumonia']]
print(df_train_p.head(), "\n")

'''Divide the normal and pneumonia dataframes into k subsets'''
ls_train_n = shuffle_and_divide(df_train_n)
ls_train_p = shuffle_and_divide(df_train_p)

'''Merge the normal and pneumonia subsets'''
data_sets = merge_pneumonia_and_normal_dataframes(ls_train_n, ls_train_p)

val_sets, train_sets = set_train_and_val_set(data_sets)
train_sets = balance_train_set(train_sets)

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

H, METRICS = train_model(input_shape, val_sets, train_sets)

###############################################################################
# GENERATING PLOTS
###############################################################################

matplotlib.use("Agg")
plt.style.use("ggplot")
X = np.arange(0, EPOCHS)
Y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def plot_k_curves(title, y_label, save_name, curve, label):
    plt.figure()
    for i in range(len(H)):
        plt.plot(X, H[i].history[curve], label=label + " for run " + str(i))

    plt.xticks(X)
    plt.yticks(Y)
    plt.grid(True)
    plt.title(title + "\n on pneumonia detection")
    plt.xlabel("Epoch #")
    plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.savefig(PLOT_DIR + save_name + ".png")
    return


print("GENERATING PLOTS")

'''Making directory'''
if not os.path.exists(PLOT_DIR + NAME):
    os.makedirs(PLOT_DIR + NAME)

# Plot loss
plot_k_curves("Training Loss", "Loss", "train_loss", "loss", "train_loss")

# Plot val_loss
plot_k_curves("Validation Loss", "Loss", "val_loss", "val_loss", "val_loss")

# Plot acc
plot_k_curves("Training Accuracy", "Accuracy", "train_acc", "accuracy", "train_acc")

# Plot val_acc
plot_k_curves("Validation Accuracy", "Accuracy", "val_acc", "val_accuracy", "val_acc")


###############################################################################
# GENERATING PLOTS (MEAN)
###############################################################################

def calculate_mean_history(h, string):
    res = np.zeros(EPOCHS)
    temp = dict()
    for i in range(len(h)):
        temp[i] = h[i].history[string]

    for i in range(K):
        for j in range(EPOCHS):
            res[j] = res[j] + temp[i][j]
    res = res / K

    return res


def plot_mean(title, y_label, save_name, data_1, data_2, label_1, label_2):
    plt.figure()
    plt.plot(X, data_1, label=label_1)
    plt.plot(X, data_2, label=label_2)

    plt.xticks(X)
    plt.yticks(Y)
    plt.grid(True)
    plt.title("Training/Validation " + title + " \n on pneumonia detection")
    plt.xlabel("Epoch #")
    plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.savefig(PLOT_DIR + save_name + ".png")
    return


train_loss_mean = calculate_mean_history(H, "loss")
val_loss_mean = calculate_mean_history(H, "val_loss")
train_acc_mean = calculate_mean_history(H, "accuracy")
val_acc_mean = calculate_mean_history(H, "val_accuracy")

# Plot train_loss_mean and val_loss_mean
plot_mean("Loss", "Loss", "loss_mean", train_loss_mean, val_loss_mean, "train_loss_mean", "val_loss_mean")

# Plot train_acc_mean and val_acc_mean
plot_mean("Accuracy", "Accuracy", "acc_mean", train_acc_mean, val_acc_mean, "train_acc_mean", "val_acc_mean")

# Plot all on the same curve
plt.figure()
plt.plot(X, train_loss_mean, label="train_loss_mean")
plt.plot(X, val_loss_mean, label="val_loss_mean")
plt.plot(X, train_acc_mean, label="train_acc_mean")
plt.plot(X, val_acc_mean, label="val_acc_mean")

plt.xticks(X)
plt.yticks(Y)
plt.grid(True)
plt.title("Training/Validation Accuracy and Loss \n on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + "all_mean.png")

###############################################################################
# SAVE MEAN VALUES AS CSV
###############################################################################

df_mean = pd.DataFrame()
df_mean['train_loss_mean'] = train_loss_mean
df_mean['val_loss_mean'] = val_loss_mean
df_mean['train_acc_mean'] = train_acc_mean
df_mean['val_acc_mean'] = val_acc_mean
export_csv = df_mean.to_csv(DATA_DIR + 'mean_df.csv')


###############################################################################
# GENERATING PLOTS (METRICS)
###############################################################################

def calculate_mean_metric(metric_name, class_number):
    res = np.zeros(EPOCHS)
    df_dict = dict()

    for i in range(K):
        df = pd.read_csv(DATA_DIR + "r" + str(i + 1) + "/metrics.csv")
        df_dict[i] = df[metric_name + "_" + str(class_number)].to_numpy()

    for j in range(len(df_dict)):
        for k in range(len(df_dict[j])):
            res[k] = res[k] + df_dict[j][k]

    res = res / K
    return res


recall_0 = calculate_mean_metric("recall", 0)
precision_0 = calculate_mean_metric("precision", 0)
f1_0 = calculate_mean_metric("f1", 0)
plot_metrics_2(recall_0, precision_0, f1_0, X, PLOT_DIR, 0)

recall_1 = calculate_mean_metric("recall", 1)
precision_1 = calculate_mean_metric("precision", 1)
f1_1 = calculate_mean_metric("f1", 1)
plot_metrics_2(recall_1, precision_1, f1_1, X, PLOT_DIR, 1)

print("FINISHED")
