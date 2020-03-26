##############################################################################
#IMPORTS
##############################################################################


import os
import numpy as np
import pandas as pd

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

MODEL_NAME = '2020-03-26_06-59_no_w150_h150_e20_CV'


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

import sys, os
print(os.listdir())

df = pd.read_csv(PLOT_DIR + 'dataviz.csv')
print(df.head(10))
