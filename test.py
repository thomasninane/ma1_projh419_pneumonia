##############################################################################
#IMPORTS
##############################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

##############################################################################
#PARAMETERS
##############################################################################
pd.set_option('display.expand_frame_repr', False)

img_dir = '../MA1_PROJH419_pneumonia_data/flow_from_dir/'
test_img_dir = img_dir + 'test/'
train_img_dir = img_dir + 'train/'
val_img_dir = img_dir + 'val/'

model_dir = '../MA1_PROJH419_pneumonia_data/models/'
model_name = 'flow_from_dir_unbalanced_w150_h150_e20'

BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

##############################################################################
#
##############################################################################

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(test_img_dir,
                                                  target_size = (WIDTH, HEIGHT),
                                                  batch_size = BATCH_SIZE,
                                                  class_mode = 'binary'
                                                  )

model = load_model(model_dir + model_name)

#Y_pred = model.predict(test_generator, test_generator.samples // BATCH_SIZE)
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
TARGET_NAMES = ['Normal', 'Pneumonia']
print(classification_report(test_generator.classes, y_pred, target_names=TARGET_NAMES))