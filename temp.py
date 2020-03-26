##############################################################################
#IMPORTS
##############################################################################


import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

##############################################################################
#PARAMETERS
##############################################################################


pd.set_option('display.expand_frame_repr', False)

IMG_DIR = '../../OneDrive/Temp/projh419_data/flow_from_dir/'
IMG_DIR_DF = '../../OneDrive/Temp/projh419_data/flow_from_df/'

CSV_DIR = '../../OneDrive/Temp/projh419_data/csv/'
PLOT_DIR = '../../OneDrive/Temp/projh419_data/plots/'
MODEL_DIR = '../../OneDrive/Temp/projh419_data/models/'
LOG_DIR = '..\\..\\OneDrive\\Temp\\projh419_data\\logs\\'
DATAVIZ_DIR = '../../OneDrive/Temp/projh419_data/dataviz/'


EPOCHS = 20
BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

BALANCE_TYPE = 'no'         #no, weights, over, under
K = 5

NAME = '_' + BALANCE_TYPE + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_CV'


##############################################################################
#DATAFRAME
##############################################################################
MODEL_NAME = '2020-03-26_09-00_weights_w150_h150_e20_CV'

matplotlib.use("Agg")
plt.style.use("ggplot")

df = pd.read_csv(PLOT_DIR + MODEL_NAME + '/mean_df.csv')
print(df.head(10))

N = EPOCHS
x = np.arange(0, N)

train_loss_mean = df['train_loss_mean']
val_loss_mean = df['val_loss_mean']
train_acc_mean = df['train_acc_mean']
val_acc_mean = df['val_acc_mean']

plt.figure()
plt.plot(x, train_loss_mean, label="train_loss_mean")
plt.plot(x, val_loss_mean, label="val_loss_mean")

plt.xticks(x)
plt.grid(True)
plt.title("Training/Validation Loss on pneumonia dataction")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + MODEL_NAME + "/train_val_loss_mean.png")

# PLOT MEAN ACCURACY

plt.figure()
plt.plot(x, train_acc_mean, label="train_acc_mean")
plt.plot(x, val_acc_mean, label="val_acc_mean")

plt.xticks(x)
plt.grid(True)
plt.title("Training/Validation Accuracy on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + MODEL_NAME + "/train_val_acc_mean.png")