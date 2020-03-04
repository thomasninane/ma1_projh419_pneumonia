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

def get_names(directory):
    names = os.listdir(directory)
    return names

def create_df(directory):
    names = os.listdir(directory)
    df = pd.DataFrame(names, columns = ['name'])
    return df

def add_set_name(df, SET_NAME):
    set_name = []
    for i in range(df.shape[0]):
        set_name.append(SET_NAME)
    
    df['set_name'] = set_name
    return df

def add_norm_or_pneu(df, NORM_PNEU):
    norm_pneu = []
    for i in range(df.shape[0]):
        norm_pneu.append(NORM_PNEU)
        
    df['normal/pneumonia'] = norm_pneu
    return df
    
def add_img_dimensions(df, directory):
    samples = df.shape[0]
    width = []
    height = []
    
    i=0
    for name in df.loc[:, 'name']:
        i += 1
        img = cv2.imread(directory + name)
        height.append(img.shape[0])
        width.append(img.shape[1])
        
        print('Percentage:', i*100/samples, '%')
        
    df['width'] = width
    df['height'] = height
    return df 

def create_csv(directory, csv_dir, set_name, disease):
    name = set_name+'_'+disease
    df = create_df(directory)
    df = add_set_name(df, set_name)
    df = add_norm_or_pneu(df, disease)
    df = add_bacteria_or_virus(df)
    df = add_img_dimensions(df, directory)
    
    print(df.head())
    export_csv = df.to_csv(csv_dir + name +'.csv')
    return 0

def open_dataframes():
    df_test_n = pd.read_csv(csv_dir + 'test_normal.csv').iloc[:,1:]
    df_test_p = pd.read_csv(csv_dir + 'test_pneumonia.csv').iloc[:,1:]
    
    df_train_n = pd.read_csv(csv_dir + 'train_normal.csv').iloc[:,1:]
    df_train_p = pd.read_csv(csv_dir + 'train_pneumonia.csv').iloc[:,1:]
    
    df_val_n = pd.read_csv(csv_dir + 'val_normal.csv').iloc[:,1:]
    df_val_p = pd.read_csv(csv_dir + 'val_pneumonia.csv').iloc[:,1:]
    
    return df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p

def combine_dataframes(df1, df2, df3, df4, df5, df6):
    df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    export_csv = df.to_csv(csv_dir + 'combined.csv')
    return df

def add_bacteria_or_virus(df):
    res = []
    
    for name in df['name']:
        if 'bacteria' in name:
            res.append('bacteria')
        elif 'virus' in name:
            res.append('virus')
        else:
            res.append('normal')
    
    df['bacteria/virus/normal'] = res
    return df

##############################################################################
#TRAIN DF
##############################################################################
SET_NAME = 'train'

DISEASE = 'NORMAL' 
directory = img_dir + SET_NAME + '/' + DISEASE +'/'
create_csv(directory, csv_dir, SET_NAME, DISEASE)

DISEASE = 'PNEUMONIA' 
directory = img_dir + SET_NAME + '/' + DISEASE +'/'
create_csv(directory, csv_dir, SET_NAME, DISEASE)


##############################################################################
#TEST DF
##############################################################################

SET_NAME = 'test'

DISEASE = 'NORMAL' 
directory = img_dir + SET_NAME + '/' + DISEASE +'/'
create_csv(directory, csv_dir, SET_NAME, DISEASE)

DISEASE = 'PNEUMONIA' 
directory = img_dir + SET_NAME + '/' + DISEASE +'/'
create_csv(directory, csv_dir, SET_NAME, DISEASE)


##############################################################################
#VAL DF
##############################################################################

SET_NAME = 'val'

DISEASE = 'NORMAL' 
directory = img_dir + SET_NAME + '/' + DISEASE +'/'
create_csv(directory, csv_dir, SET_NAME, DISEASE)

DISEASE = 'PNEUMONIA' 
directory = img_dir + SET_NAME + '/' + DISEASE +'/'
create_csv(directory, csv_dir, SET_NAME, DISEASE)


##############################################################################
#COMBINE ALL DATAFRAMES
##############################################################################

df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p = open_dataframes()
df = combine_dataframes(df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p)
print(df.head())