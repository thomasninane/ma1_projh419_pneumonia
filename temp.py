##############################################################################
#IMPORTS
##############################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cv2

##############################################################################
#PARAMETERS
##############################################################################
pd.set_option('display.expand_frame_repr', False)

img_dir = '../MA1_PROJH419_pneumonia_data/'
test_img_dir = '../MA1_PROJH419_pneumonia_data/test/'
train_img_dir = '../MA1_PROJH419_pneumonia_data/train/'
val_img_dir = '../MA1_PROJH419_pneumonia_data/val/'

csv_dir = 'csv/'

nb_test_samples = 624
nb_train_samples = 5216
nb_val_samples = 16


##############################################################################
#FUNCTIONS
##############################################################################

def open_dataframes():
    df_test_n = pd.read_csv(csv_dir + 'test_normal.csv').iloc[:,1:]
    df_test_p = pd.read_csv(csv_dir + 'test_pneumonia.csv').iloc[:,1:]
    
    df_train_n = pd.read_csv(csv_dir + 'train_normal.csv').iloc[:,1:]
    df_train_p = pd.read_csv(csv_dir + 'train_pneumonia.csv').iloc[:,1:]
    
    df_val_n = pd.read_csv(csv_dir + 'val_normal.csv').iloc[:,1:]
    df_val_p = pd.read_csv(csv_dir + 'val_pneumonia.csv').iloc[:,1:]
    
    return df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p

def add_bacteria_or_virus(df):
    res = []
    bact=0
    virus = 0
    na=0
    for name in df['name']:
        if 'bacteria' in name:
            res.append('bacteria')
            bact += 1
        elif 'virus' in name:
            res.append('virus')
            virus += 1
        else:
            res.append('na')
            na += 1
    print(bact, virus, na)
    
    df['bacteria/virus'] = res
    return df

##############################################################################
#
##############################################################################

df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p = open_dataframes()

df_test_p = add_bacteria_or_virus(df_test_p)
df_train_p = add_bacteria_or_virus(df_train_p)
df_val_p = add_bacteria_or_virus(df_val_p)