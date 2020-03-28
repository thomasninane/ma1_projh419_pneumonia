##############################################################################
#IMPORTS
##############################################################################


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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

matplotlib.use("Agg")
plt.style.use("ggplot")

N = 25
x = np.arange(0, N)

NAME = '2020-03-28_17-04_weights_w150_h150_e25_da_CV'
df = pd.read_csv(PLOT_DIR + NAME + '/mean_df.csv')

train_loss_mean = df['train_loss_mean']
val_loss_mean = df['val_loss_mean']
train_acc_mean = df['train_acc_mean']
val_acc_mean = df['val_acc_mean']

plt.figure()
plt.plot(x, train_loss_mean, label="train_loss_mean")
plt.plot(x, val_loss_mean, label="val_loss_mean")
plt.plot(x, train_acc_mean, label="train_acc_mean")
plt.plot(x, val_acc_mean, label="val_acc_mean")

plt.xticks(x)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid(True)
plt.title("Training/Validation Accuracy and Loss \n on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + NAME + "/all_mean.png")

# df = pd.read_csv(DATAVIZ_DIR + 'dataviz.csv')
# df = df.iloc[:, 1:]
# print(df.head(10))
#
# def append(column):
#     res = column.to_numpy()
#     res = res[1:]
#     return res
#
# res = pd.DataFrame()
# res['Set'] = ['Test', 'Test', 'Train', 'Train', 'Val', 'Val']
# res['Disease'] = ['Normal', 'Pneumonia', 'Normal', 'Pneumonia', 'Normal', 'Pneumonia']
# res['# of samples'] = append(df['samples #'])
# res['Max WIDTH'] = append(df['max_width'])
# res['Min WIDTH'] = append(df['min_width'])
# res['Max HEIGHT'] = append(df['max_height'])
# res['Min HEIGHT'] = append(df['min_height'])
#
#
# print(res.head())
