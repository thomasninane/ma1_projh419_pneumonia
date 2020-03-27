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

IMG_DIR = '../../OneDrive/Temp/projh419_data/flow_from_dir/'

CSV_DIR = '../../OneDrive/Temp/projh419_data/csv/'


##############################################################################
#FUNCTIONS
##############################################################################


def create_data_frame(directory):
    """
    Creates a data frame
    Each row contains the name of an image located in the directory
    """
    names = os.listdir(directory)
    df = pd.DataFrame(names, columns = ['filename'])
    
    return df


def add_set_type(df, dataset_type):
    """Adds the type of the set (train, val, test) to the df"""
    res = []
    for i in range(df.shape[0]):
        res.append(dataset_type)
    
    df['dataset_type'] = res
    
    return df


def add_normal_or_pneumonia(df, normal_or_pneumonia):
    """Adds wheather the patient is normal or suffers from pneumonia to the df"""
    res = []
    for i in range(df.shape[0]):
        res.append(normal_or_pneumonia)
        
    df['normal/pneumonia'] = res
    
    return df
   

def add_img_dimensions(df, directory):
    '''Adds the image dimensions to the df'''
    samples = df.shape[0]
    width = []
    height = []
    
    i=0
    for name in df['filename']:
        i += 1
        img = cv2.imread(directory + name)
        height.append(img.shape[0])
        width.append(img.shape[1])
        
        print('Percentage:', i*100/samples, '%')
        
    df['width'] = width
    df['height'] = height
    
    return df 


def add_normal_or_bacteria_or_virus(df):
    '''Adds whether the patient is normal or suffers from bacterial/viral pneumonia to the df'''
    res = []
    
    for name in df['filename']:
        if 'bacteria' in name:
            res.append('bacteria')
        elif 'virus' in name:
            res.append('virus')
        else:
            res.append('normal')
    
    df['normal/bacteria/virus'] = res
    
    return df


def create_csv(set_type, normal_or_pneumonia):
    '''Creates a csv file'''
    directory = IMG_DIR + set_type +'/' + normal_or_pneumonia + '/'
    csv_name = set_type + '_' + normal_or_pneumonia

    df = create_data_frame(directory)
    df = add_set_type(df, set_type)
    df = add_normal_or_pneumonia(df, normal_or_pneumonia)
    df = add_normal_or_bacteria_or_virus(df)
    df = add_img_dimensions(df, directory)
    
    print(df.head())
    export_csv = df.to_csv(CSV_DIR + csv_name +'.csv')
    
    return 0


def open_data_frames():
    '''Opens and returns all 6 dataframes'''
    df_test_n = pd.read_csv(CSV_DIR + 'test_normal.csv').iloc[:,1:]
    df_test_p = pd.read_csv(CSV_DIR + 'test_pneumonia.csv').iloc[:,1:]
    
    df_train_n = pd.read_csv(CSV_DIR + 'train_normal.csv').iloc[:,1:]
    df_train_p = pd.read_csv(CSV_DIR + 'train_pneumonia.csv').iloc[:,1:]
    
    df_val_n = pd.read_csv(CSV_DIR + 'val_normal.csv').iloc[:,1:]
    df_val_p = pd.read_csv(CSV_DIR + 'val_pneumonia.csv').iloc[:,1:]
    
    return df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p


def combine_data_frames(df1, df2, df3, df4, df5, df6):
    df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    export_csv = df.to_csv(CSV_DIR + 'combined.csv')
    return df

##############################################################################
#TRAIN DF
##############################################################################
set_type = 'train'

normal_or_pneumonia = 'NORMAL' 
create_csv(set_type, normal_or_pneumonia)

normal_or_pneumonia = 'PNEUMONIA' 
create_csv(set_type, normal_or_pneumonia)

##############################################################################
#TEST DF
##############################################################################

set_type = 'test'

normal_or_pneumonia = 'NORMAL' 
create_csv(set_type, normal_or_pneumonia)

normal_or_pneumonia = 'PNEUMONIA' 
create_csv(set_type, normal_or_pneumonia)

##############################################################################
#VAL DF
##############################################################################

set_type = 'val'

normal_or_pneumonia = 'NORMAL' 
create_csv(set_type, normal_or_pneumonia)

normal_or_pneumonia = 'PNEUMONIA' 
create_csv(set_type, normal_or_pneumonia)

##############################################################################
#COMBINE ALL DATAFRAMES
##############################################################################

df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p = open_data_frames()
df = combine_data_frames(df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p)
print(df.head())