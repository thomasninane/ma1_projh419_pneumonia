##############################################################################
#IMPORTS
##############################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from sklearn.utils import class_weight

##############################################################################
#PARAMETERS
##############################################################################
pd.set_option('display.expand_frame_repr', False)

img_dir = '../../OneDrive/Temp/MA1_PROJH419_pneumonia_data/flow_from_dir/'
test_img_dir = img_dir + 'test/'
train_img_dir = img_dir + 'train/'
val_img_dir = img_dir + 'val/'

csv_dir = 'csv/'
plot_dir = 'plots/'
model_dir = '../../OneDrive/Temp/MA1_PROJH419_pneumonia_data/models/'

nb_test_samples = 624
nb_train_samples = 5216
nb_val_samples = 16

EPOCHS = 20
BATCH_SIZE = 16
INIT_LR = 1e-3

WIDTH = 150
HEIGHT = 150

#unbalanced, classWeights
class_balance = 'unbalanced'
plot_name = 'flow_from_dir_' + class_balance + '_w' + str(WIDTH) + '_h' + str(HEIGHT) + '_e' + str(EPOCHS)
model_name = plot_name

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
    

##############################################################################
#
##############################################################################
    
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_img_dir,
                                                    target_size = (WIDTH, HEIGHT),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'binary'
                                                    )

val_generator = test_datagen.flow_from_directory(val_img_dir,
                                                 target_size = (WIDTH, HEIGHT),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'binary'
                                                 )

test_generator = test_datagen.flow_from_directory(test_img_dir,
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
                            steps_per_epoch = nb_train_samples // BATCH_SIZE,
                            epochs = EPOCHS,
                            validation_data = val_generator,
                            validation_steps = nb_val_samples // BATCH_SIZE,
                            class_weight = CLASS_WEIGHTS
                            )

else:
    H = model.fit_generator(train_generator,
                            steps_per_epoch = nb_train_samples // BATCH_SIZE,
                            epochs = EPOCHS,
                            validation_data = val_generator,
                            validation_steps = nb_val_samples // BATCH_SIZE
                            )


model.save(model_dir + model_name)

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
plt.savefig(plot_dir + plot_name + ".png")

print("Finished")