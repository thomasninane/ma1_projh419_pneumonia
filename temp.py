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

def plotHistogram(x, y, title, xlabel, ylabel, width):
    plt.figure()
    plt.bar(x, y, width)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(DATAVIZ_DIR + title + ".png")
    return 0

# categories = ("NORMAL", "PNEUMONIA", "NORMAL (over)", "PNEUMONIA (over)", "NORMAL (under)", "PNEUMONIA (under)")
# ylabel = "Number of observations"
# y = [1073, 3100, 3100, 3100, 1073, 1073]
# plt.figure()
# plt.bar(categories, y)
# plt.savefig(DATAVIZ_DIR + 'test.png')


plt.figure()
plt.title("Number of images used for training the model \n (5-fold cross validation)")
plt.ylabel("Number of observations")

x1 = ('normal', 'pneumonia')
y1 = [1073, 3100]
plt.bar(x1, y1)

x2 = ('normal \n (under)', 'pneumonia \n (under)')
y2 = [1073, 1073]
plt.bar(x2, y2)

x3 = ('normal \n (over)', 'pneumonia \n (over)')
y3 = [3100, 3100]
plt.bar(x3, y3)

plt.savefig(DATAVIZ_DIR + 'all_counts.png')
