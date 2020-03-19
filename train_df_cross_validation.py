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

from tensorflow.keras import backend as kB
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
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

IMG_DIR = '../../OneDrive/Temp/projh419_data/flow_from_dir/'
IMG_DIR_DF = '../../OneDrive/Temp/projh419_data/flow_from_df/'

CSV_DIR = '../../OneDrive/Temp/projh419_data/csv/'
PLOT_DIR = '../../OneDrive/Temp/projh419_data/plots/'
MODEL_DIR = '../../OneDrive/Temp/projh419_data/models/'
LOG_DIR = '..\\..\\OneDrive\\Temp\\projh419_data\\logs\\'


EPOCHS = 20
BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

BALANCE_TYPE = 'no'         #no, weights, over, under
K = 5

date = datetime.today().strftime('%Y-%m-%d_%H-%M')
NAME = date + '_' + BALANCE_TYPE + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_CV'


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
    if kB.image_data_format() == 'channels_first':
        input_shape = (3, WIDTH, HEIGHT)
    else:
        input_shape = (WIDTH, HEIGHT, 3)
    return input_shape


def balanceClasses(balanceType, df0, df1):
    print("Len of train_normal (unbalanced): " , df0.shape[0])
    print("Len of train_pneumonia (unbalanced): " , df1.shape[0])
    
    if (balanceType=='no') or (balanceType=='weights'):
        df = mergeAndShuffle(df0, df1)
        return df
    
    else:
        if balanceType == 'under':
            df0, df1 = undersample(df0, df1)
        elif balanceType == 'over':
            df0, df1 = oversample(df0, df1)
        else:
            print('Class balancing error')
            return 0
        
        print("Len of train_normal ( after balanceClasses() ): ", df0.shape[0])
        print("Len of train_pneumonia ( after balanceClasses() ): ", df1.shape[0])
        
        df = mergeAndShuffle(df0, df1)
        
        return df


def mergeAndShuffle(df0, df1):
    '''merges two dataframes and shuffles the merged dataframe'''
    df = df0
    df = df.append(df1, ignore_index=True)
    
    df = df.sample(frac=1)  #Shuffle
    df = df.reset_index(drop="True")
    return df


def undersample(df0, df1):
    '''undersamples the data in the biggest class'''
    minimum = min(df0.shape[0], df1.shape[0])

    df0_under = df0.sample(minimum)
    df1_under = df1.sample(minimum)
    
    return df0_under, df1_under


def oversample(df0, df1):
    '''oversamples the data in the smallest class (by duplicating items)'''
    maximum = max(df0.shape[0], df1.shape[0])
    
    df0_over = df0.sample(maximum, replace=True)
    df1_over = df1.sample(maximum, replace=True)
    
    #We do not want to have duplicates in the biggest class
    if maximum==len(df0):
        df0_over=df0
    else:
        df1_over=df1
    
    return df0_over, df1_over


def shuffleAndDivide(df):
    '''returns an array of k dataframes (they all have the same size)'''
    df = df.sample(frac=1)
    df = df.reset_index(drop="True")
    res = np.array_split(df, K)
    
    return res  


def mergePneumoniaAndNormalDataframes(ls_train_n, ls_train_p):
    '''merges the normal df and the pneumonia df into one subset of a dictionary (with class balancing and shuffling)'''
    subsets = dict()
    
    for i in range(K):
        subsets[i]  = balanceClasses(BALANCE_TYPE, ls_train_n[i], ls_train_p[i])
    
    return subsets


def setTrainAndValidationSet(subsets):
    '''returns one dictionary with the validation sets and another dictionary with the k-1 combined training sets'''
    val_sets = dict()
    train_sets = dict()
    
    for i in range(K):
        val_sets[i] = subsets[i]
    
    '''combine the other subsets'''
    for i in range(K):
       if i==0:
           train_sets[i] = subsets[1]
           for j in range(2, K):
              train_sets[i] =  train_sets[i].append(subsets[j], ignore_index=True)
       elif i==K-1:
           train_sets[i] = subsets[0]
           for j in range(1, K-1):
               train_sets[i] = train_sets[i].append(subsets[j], ignore_index=True)
       else:
           train_sets[i] = subsets[0]
           for j in range(1, i):
               train_sets[i] = train_sets[i].append(subsets[j], ignore_index=True)
           for j in range(i+1, K):
               train_sets[i] = train_sets[i].append(subsets[j], ignore_index=True)
   
    return val_sets, train_sets


def trainModel(val_sets, train_sets):
    '''trains the model for k folds and returns a dictionary containing the history of each model'''
    
    History = dict()
    
    for i in range(K):
        print('RUN: ' + str(i+1))
        
        name = NAME + str(i+1)
        tensorboard = TensorBoard(log_dir = LOG_DIR + name)

        checkpoint_path = MODEL_DIR + name + "/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = ModelCheckpoint( filepath=checkpoint_path, save_weights_only=True, verbose=1 )
                
        train_generator = train_datagen.flow_from_dataframe(dataframe = train_sets[i],
                                                            directory = IMG_DIR_DF + 'train/',
                                                            x_col = 'filename',
                                                            y_col = 'normal/pneumonia',
                                                            target_size = (WIDTH, HEIGHT),
                                                            batch_size = BATCH_SIZE,
                                                            class_mode = 'binary'
                                                            )
    
        val_generator = test_datagen.flow_from_dataframe(dataframe = val_sets[i],
                                                         directory = IMG_DIR_DF + 'train/',
                                                         x_col = 'filename',
                                                         y_col = 'normal/pneumonia',
                                                         target_size = (WIDTH, HEIGHT),
                                                         batch_size = BATCH_SIZE,
                                                         class_mode = 'binary',
                                                         )
        
    
        if BALANCE_TYPE == 'classWeights':
            CLASS_WEIGHTS = class_weight.compute_class_weight("balanced",
                                                              np.unique(train_generator.classes),
                                                              train_generator.classes)
            print("Class weights:", CLASS_WEIGHTS)
            
            H = model.fit(train_generator,
                                    steps_per_epoch = train_generator.samples // BATCH_SIZE,
                                    epochs = EPOCHS,
                                    validation_data = val_generator,
                                    validation_steps = val_generator.samples // BATCH_SIZE,
                                    callbacks = [tensorboard, cp_callback],
                                    class_weight = CLASS_WEIGHTS
                                    )
        
        else:
            H = model.fit(train_generator,
                                    steps_per_epoch =  train_generator.samples // BATCH_SIZE,
                                    epochs = EPOCHS,
                                    validation_data = val_generator,
                                    validation_steps = val_generator.samples // BATCH_SIZE,
                                    callbacks = [tensorboard, cp_callback]
                                    )
            
        History[i] = H
        
    return History


##############################################################################
#DATAFRAME
##############################################################################


df_train_n = pd.read_csv(CSV_DIR + 'train_normal.csv')
df_train_n = df_train_n[['filename', 'normal/pneumonia']]
print(df_train_n.head())

df_train_p = pd.read_csv(CSV_DIR + 'train_pneumonia.csv')
df_train_p = df_train_p[['filename', 'normal/pneumonia']]
print(df_train_p.head())

'''Divide the normal and pneumonia dataframes into k subsets'''
ls_train_n = shuffleAndDivide(df_train_n)
ls_train_p = shuffleAndDivide(df_train_p)


data_sets = mergePneumoniaAndNormalDataframes(ls_train_n, ls_train_p)
val_sets, train_sets = setTrainAndValidationSet(data_sets)
    

##############################################################################
#
##############################################################################


input_shape = inputShape(WIDTH, HEIGHT)
model = createModel(input_shape)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True
                                    )

test_datagen = ImageDataGenerator(rescale = 1./255)


H = trainModel(val_sets, train_sets)


# ###############################################################################
# #GENERATING PLOTS
# ###############################################################################


print("Generating plots")

'''Making directory'''
if not os.path.exists(PLOT_DIR + NAME):
    os.makedirs(PLOT_DIR + NAME)


matplotlib.use("Agg")
plt.style.use("ggplot")

N = EPOCHS

plt.figure()
for i in range(len(H)):
    plt.plot(np.arange(0, N), H[i].history["loss"], label="train_loss for run " + str(i))
plt.title("Training Loss on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + NAME + "/train_loss.png")


plt.figure()
for i in range(len(H)):
    plt.plot(np.arange(0, N), H[i].history["val_loss"], label="val_loss for run " + str(i))
plt.title("Validation Loss on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + NAME + "/val_loss.png")


plt.figure()
for i in range(len(H)):
    plt.plot(np.arange(0, N), H[i].history["accuracy"], label="train_acc for run " + str(i))
plt.title("Training Accuracy on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + NAME + "/train_acc.png")


plt.figure()
for i in range(len(H)):
    plt.plot(np.arange(0, N), H[i].history["val_accuracy"], label="val_acc for run " + str(i))
plt.title("Validation Accuracy on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig(PLOT_DIR + NAME + "/val_acc.png")

print("FINISHED")