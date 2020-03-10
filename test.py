##############################################################################
#IMPORTS
##############################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

##############################################################################
#PARAMETERS
##############################################################################
pd.set_option('display.expand_frame_repr', False)

img_dir = '../MA1_PROJH419_pneumonia_data/flow_from_dir/'
test_img_dir = img_dir + 'test/'
train_img_dir = img_dir + 'train/'
val_img_dir = img_dir + 'val/'

csv_dir = 'csv/'
model_dir = '../MA1_PROJH419_pneumonia_data/models/'
model_name = 'flow_from_dir_unbalanced_w150_h150_e20'

BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

##############################################################################
#DATAFRAME HANDLING
##############################################################################

df_test_n = pd.read_csv(csv_dir + 'test_normal.csv')
df_test_n = df_test_n[['name', 'set_name', 'normal/pneumonia']]
print('Test Normal DF')
print(df_test_n.head())

df_test_p = pd.read_csv(csv_dir + 'test_pneumonia.csv')
df_test_p = df_test_p[['name', 'set_name', 'normal/pneumonia']]
print('Test Pneumonia DF')
print(df_test_p.head())

def regroup_and_shuffle(df0, df1):
    df = df0
    df = df.append(df1, ignore_index=True)
    
    df = df.sample(frac=1)  #Shuffle
    df = df.reset_index(drop="True")
    return df

df = pd.concat([df_test_n, df_test_p], ignore_index=True)
#df = regroup_and_shuffle(df_test_n, df_test_p)
print('Test Combined DF')
print(df.head())

##############################################################################
#
##############################################################################

test_datagen = ImageDataGenerator(rescale = 1./255)

# test_generator = test_datagen.flow_from_directory(test_img_dir,
#                                                   target_size = (WIDTH, HEIGHT),
#                                                   batch_size = BATCH_SIZE,
#                                                   class_mode = 'binary'
#                                                   )

test_generator = test_datagen.flow_from_dataframe(dataframe = df,
                                                  directory = '../MA1_PROJH419_pneumonia_data/flow_from_df/test',
                                                  x_col = 'name',
                                                  y_col = 'normal/pneumonia',
                                                  class_mode = 'binary',
                                                  batch_size = BATCH_SIZE,
                                                  target_size = (WIDTH, HEIGHT),
                                                  shuffle=False
                                                  )

model = load_model(model_dir + model_name)


scores = model.evaluate_generator(test_generator)
print('Scores: ' + str(scores[1]*100))

Y_pred = model.predict(test_generator)
df['predicted (probability)'] = Y_pred

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

y_pred, y_pred_str = predictions(Y_pred)
df['predicted (string)'] = y_pred_str
print(df.head())


print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
TARGET_NAMES = ['Normal', 'Pneumonia']  #0: normal 1: pneumonia
print(classification_report(test_generator.classes, y_pred, target_names=TARGET_NAMES))