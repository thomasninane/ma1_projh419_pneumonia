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


EPOCHS = 3
BATCH_SIZE = 16

WIDTH = 150
HEIGHT = 150

balance_type = 'unbalanced'         #unbalanced, classWeights, oversample, undersample


date = datetime.today().strftime('%Y-%m-%d--%H-%M')
NAME = date + '__' + balance_type + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS) + '_CV'

k = 3


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


def balanceClasses(balanceType, df0, df1):
    print("Len of train_normal (unbalanced): " , df0.shape[0])
    print("Len of train_pneumonia (unbalanced): " , df1.shape[0])
    
    if (balanceType=='unbalanced') or (balanceType=='classWeights'):
        df = mergeAndShuffle(df0, df1)
        return df
    
    else:
        if balanceType == 'undersample':
            df0, df1 = undersample(df0, df1)
        elif balanceType == 'oversample':
            df0, df1 = oversample(df0, df1)
        else:
            print('oversample or undersample error')
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


def shuffleAndDivide(df, k):
    '''returns an array of k dataframes (they all have the same size)'''
    df = df.sample(frac=1)
    df = df.reset_index(drop="True")
    res = np.array_split(df, k)
    
    return res  


def mergePneumoniaAndNormalDataframes(k):
    '''merges the normal df and the pneumonia df into one subset of a dictionary'''
    subsets = dict()
    
    for i in range(k):
        subsets[i]  = balanceClasses(balance_type, ls_train_n[i], ls_train_p[i])
    
    return subsets


def setTrainAndValidationSet(k, data_sets):
    '''returns one dictionary with the validation sets and another dictionary with the k-1 combined training sets'''
    val_sets = dict()
    train_sets = dict()
    
    for i in range(k):
        val_sets[i] = data_sets[i]
    
    '''combine the other subsets'''
    for i in range(k):
       if i==0:
           train_sets[i] = data_sets[1]
           for j in range(2, k):
              train_sets[i] =  train_sets[i].append(data_sets[j], ignore_index=True)
       elif i==k-1:
           train_sets[i] = data_sets[0]
           for j in range(1, k-1):
               train_sets[i] = train_sets[i].append(data_sets[j], ignore_index=True)
       else:
           train_sets[i] = data_sets[0]
           for j in range(1, i):
               train_sets[i] = train_sets[i].append(data_sets[j], ignore_index=True)
           for j in range(i+1, k):
               train_sets[i] = train_sets[i].append(data_sets[j], ignore_index=True)
   
    return val_sets, train_sets


def trainModel(k, val_sets, train_sets, balance_type, name):
    '''trains the model for k folds and returns a dictionary containing the history of each model'''
    
    History = dict()
    
    for i in range(k):
        print('K: ' + str(i))
        
        NAME = name + '_run' + str(i)
        tensorboard = TensorBoard(log_dir = logDir + NAME)
                
        train_generator = train_datagen.flow_from_dataframe(dataframe = train_sets[i],
                                                            directory = img_dir_df + 'train/',
                                                            x_col = 'name',
                                                            y_col = 'normal/pneumonia',
                                                            target_size = (WIDTH, HEIGHT),
                                                            batch_size = BATCH_SIZE,
                                                            class_mode = 'binary'
                                                            )
    
        val_generator = test_datagen.flow_from_dataframe(dataframe = val_sets[i],
                                                         directory = img_dir_df + 'train/',
                                                         x_col = 'name',
                                                         y_col = 'normal/pneumonia',
                                                         target_size = (WIDTH, HEIGHT),
                                                         batch_size = BATCH_SIZE,
                                                         class_mode = 'binary'
                                                         )
        
    
        if balance_type == 'classWeights':
            CLASS_WEIGHTS = class_weight.compute_class_weight("balanced",
                                                              np.unique(train_generator.classes),
                                                              train_generator.classes)
            print("Class weights:", CLASS_WEIGHTS)
            
            H = model.fit_generator(train_generator,
                                    steps_per_epoch = train_generator.samples // BATCH_SIZE,
                                    epochs = EPOCHS,
                                    validation_data = val_generator,
                                    validation_steps = val_generator.samples // BATCH_SIZE,
                                    callbacks = [tensorboard],
                                    class_weight = CLASS_WEIGHTS
                                    )
        
        else:
            # H = model.fit_generator(train_generator,
            H = model.fit(train_generator,
                                    steps_per_epoch =  train_generator.samples // BATCH_SIZE,
                                    epochs = EPOCHS,
                                    validation_data = val_generator,
                                    validation_steps = val_generator.samples // BATCH_SIZE,
                                    callbacks = [tensorboard]
                                    )
            
        History[i] = H
        #model.save(model_dir + name)
        
    #print('FINISHED')
    return History


##############################################################################
#DATAFRAME
##############################################################################

df_train_n = pd.read_csv(csv_dir + 'train_normal.csv')
df_train_n = df_train_n[['name', 'set_name', 'normal/pneumonia']]
print(df_train_n.head())

df_train_p = pd.read_csv(csv_dir + 'train_pneumonia.csv')
df_train_p = df_train_p[['name', 'set_name', 'normal/pneumonia']]
print(df_train_p.head())


ls_train_n = shuffleAndDivide(df_train_n, k)
ls_train_p = shuffleAndDivide(df_train_p, k)

data_sets = mergePneumoniaAndNormalDataframes(k)
val_sets, train_sets = setTrainAndValidationSet(k, data_sets)
    


##############################################################################
#
##############################################################################


inputShape = inputShape(WIDTH, HEIGHT)
model = createModel(inputShape)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True
                                    )

test_datagen = ImageDataGenerator(rescale = 1./255)


H = trainModel(k, val_sets, train_sets, balance_type, NAME)
    



# model.save(model_dir + NAME)

# ###############################################################################
# #GENERATING PLOTS
# ###############################################################################

print("Generating plots")

'''Making directory'''
if not os.path.exists(plot_dir + NAME):
    os.makedirs(plot_dir + NAME)


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
plt.savefig(plot_dir + NAME + "/train_loss.png")


plt.figure()
for i in range(len(H)):
    plt.plot(np.arange(0, N), H[i].history["val_loss"], label="val_loss for run " + str(i))
plt.title("Validation Loss on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig(plot_dir + NAME + "/val_loss.png")


plt.figure()
for i in range(len(H)):
    plt.plot(np.arange(0, N), H[i].history["accuracy"], label="train_acc for run " + str(i))
plt.title("Training Accuracy on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig(plot_dir + NAME + "/train_acc.png")


plt.figure()
for i in range(len(H)):
    plt.plot(np.arange(0, N), H[i].history["val_accuracy"], label="val_acc for run " + str(i))
plt.title("Validation Accuracy on pneumonia detection")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig(plot_dir + NAME + "/val_acc.png")

print("FINISHED")