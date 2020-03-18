##############################################################################
#IMPORTS
##############################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from datetime import datetime

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


##############################################################################
#PARAMETERS
##############################################################################
pd.set_option('display.expand_frame_repr', False)

img_dir = '../../OneDrive/Temp/MA1_PROJH419_pneumonia_data/flow_from_dir/'
img_dir_df = '../../OneDrive/Temp/MA1_PROJH419_pneumonia_data/flow_from_df/'

csv_dir = 'csv/'
plot_dir = '../../OneDrive/Temp/MA1_PROJH419_pneumonia_data/plots/'
#model_dir = '../../OneDrive/Temp/MA1_PROJH419_pneumonia_data/models/'
model_dir = '..\\..\\OneDrive\\Temp\\MA1_PROJH419_pneumonia_data\\models\\'
logDir = '..\\..\\OneDrive\\Temp\\MA1_PROJH419_pneumonia_data\\logs\\'

# nb_test_samples = 624
# nb_train_samples = 5216
nb_val_samples = 16

EPOCHS = 5
BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

#unbalanced, classWeights, oversample, undersample
class_balance = 'unbalanced'
split = 'False'

date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
NAME = 'df_' + class_balance + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_s=' + str(split) + '_' + date
print(NAME)

# plot_name = 'df_' + class_balance + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_s=' + str(split) + "-{}".format(int(time.time()))
# model_name = plot_name


##############################################################################
#FUNCTIONS
##############################################################################

def createModel(inputShape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=inputShape))
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

def inputShape(WIDTH, HEIGHT):
    if K.image_data_format() == 'channels_first':
        inputshape = (3, WIDTH, HEIGHT)
    else:
        inputShape = (WIDTH, HEIGHT, 3)
    return inputShape

def undersample(df0, df1):
    minimum = min(df0.shape[0], df1.shape[0])

    df0_under = df0.sample(minimum)
    df1_under = df1.sample(minimum)
    
    return df0_under, df1_under

def oversample(df0, df1):
    maximum = max(df0.shape[0], df1.shape[0])
    
    df0_over = df0.sample(maximum, replace=True)
    df1_over = df1.sample(maximum, replace=True)
    
    #We do not want to have duplicates in the biggest class
    if maximum==len(df0):
        df0_over=df0
    else:
        df1_over=df1
    
    return df0_over, df1_over

def regroup_and_shuffle(df0, df1):
    df = df0
    df = df.append(df1, ignore_index=True)
    
    df = df.sample(frac=1)  #Shuffle
    df = df.reset_index(drop="True")
    return df

def oversample_or_undersample(string, df0, df1):
    print("Len of train_normal (unbalanced): " , df0.shape[0])
    print("Len of train_pneumonia (unbalanced): " , df1.shape[0])
    
    if (string=='unbalanced') or (string=='classWeights'):
        df = regroup_and_shuffle(df0, df1)
        return df
    
    else:
        if string == 'undersample':
            df0, df1 = undersample(df0, df1)
        elif string == 'oversample':
            df0, df1 = oversample(df0, df1)
        else:
            print('oversample or undersample error')
            return 0
        
        print("Len of train_normal (balanced): ", df0.shape[0])
        print("Len of train_pneumonia (balanced): ", df1.shape[0])
        
        df = regroup_and_shuffle(df0, df1)
        return df
    

##############################################################################
#DATAFRAME
##############################################################################

df_train_n = pd.read_csv(csv_dir + 'train_normal.csv')
df_train_n = df_train_n[['name', 'set_name', 'normal/pneumonia']]
print(df_train_n.head())

df_train_p = pd.read_csv(csv_dir + 'train_pneumonia.csv')
df_train_p = df_train_p[['name', 'set_name', 'normal/pneumonia']]
print(df_train_p.head())



'''Train Test Split'''
if split=='True':
    df_train_n, df_val_n = train_test_split(df_train_n, test_size=0.2)
    df_train_p, df_val_p = train_test_split(df_train_p, test_size=0.2)
    
    df_val = regroup_and_shuffle(df_val_n, df_val_p)
    nb_val_samples = df_val.shape[0]


'''Oversampling/Undersampling the train dataframe (unbalanced, classWeights, undersample, oversample)'''
df_train = oversample_or_undersample(class_balance, df_train_n, df_train_p)


##############################################################################
#
##############################################################################

#tensorboard = TensorBoard(log_dir='../../OneDrive/Temp/MA1_PROJH419_pneumonia_data/logs/' + NAME)
tensorboard = TensorBoard(log_dir = logDir + NAME)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe = df_train,
                                                    directory = img_dir_df + 'train/',
                                                    x_col = 'name',
                                                    y_col = 'normal/pneumonia',
                                                    target_size = (WIDTH, HEIGHT),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'binary'
                                                    )

if split=='True':
    val_generator = test_datagen.flow_from_dataframe(dataframe = df_val,
                                                    directory = img_dir_df + 'train/',
                                                    x_col = 'name',
                                                    y_col = 'normal/pneumonia',
                                                    target_size = (WIDTH, HEIGHT),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'binary'
                                                    )
else:
    val_generator = test_datagen.flow_from_directory(img_dir + 'val/',
                                                      target_size = (WIDTH, HEIGHT),
                                                      batch_size = BATCH_SIZE,
                                                      class_mode = 'binary'
                                                      )

inputShape = inputShape(WIDTH, HEIGHT)
model = createModel(inputShape)


if class_balance == 'classWeights':
    CLASS_WEIGHTS = class_weight.compute_class_weight("balanced",
                                                      np.unique(train_generator.classes),
                                                      train_generator.classes)
    print("Class weights:", CLASS_WEIGHTS)
    
    H = model.fit_generator(train_generator,
                            steps_per_epoch = df_train.shape[0] // BATCH_SIZE,
                            epochs = EPOCHS,
                            validation_data = val_generator,
                            validation_steps = nb_val_samples // BATCH_SIZE,
                            class_weight = CLASS_WEIGHTS
                            )
else:
    H = model.fit_generator(train_generator,
                            steps_per_epoch = df_train.shape[0] // BATCH_SIZE,
                            epochs = EPOCHS,
                            validation_data = val_generator,
                            validation_steps = nb_val_samples // BATCH_SIZE,
                            callbacks = [tensorboard]
                            )


model.save(model_dir + NAME)

###############################################################################
#GENERATING PLOTS
###############################################################################

print("Generating plots")

matplotlib.use("Agg")
plt.style.use("ggplot")
plt.figure()

N = EPOCHS

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_dir + NAME + ".png")

print("Finished")