##############################################################################
#IMPORTS
##############################################################################

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


from collections import Counter

##############################################################################
#PARAMETERS
##############################################################################
pd.set_option('display.expand_frame_repr', False)

img_dir = '../MA1_PROJH419_pneumonia_data/flow_from_dir/'
test_img_dir = img_dir + 'test/'
train_img_dir = img_dir + 'train/'
val_img_dir = img_dir + 'val/'

csv_dir = 'csv/'


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

def create_df():
    df = pd.DataFrame(columns = ['df_name',
                                 'samples #',
                                 'max_width',
                                 'min_width',
                                 'mean_width',
                                 'median_width',
                                 'max_height',
                                 'min_height',
                                 'mean_height',
                                 'median_height'
                                 ]
                      )
    return df

def dataviz(df, name):
    res = []
    res.append(name)
    
    nb_samples = df.shape[0]
    res.append(nb_samples)
    
    max_width = max(df['width'])
    res.append(max_width)
    min_width = min(df['width'])
    res.append(min_width)
    
    mean_width = df['width'].mean()
    res.append(mean_width)
    median_width = df['width'].median()
    res.append(median_width)
    
    max_height = max(df['height'])
    res.append(max_height)
    min_height = min(df['height'])
    res.append(min_height)    
    
    mean_height = df['height'].mean()
    res.append(mean_height)
    median_height = df['height'].median()
    res.append(median_height)
    
    return res

def add_row(df, to_append):
    to_append_series = pd.Series(to_append, index = df.columns)
    df = df.append(to_append_series, ignore_index=True)
    return df
    
def plot_histogram(x, y, title, xlabel, ylabel, width):
    plt.figure()
    plt.bar(x, y, width)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return 0

def xy_calculator(column):
    data = column.to_numpy()
    data = Counter(data)
    xy = []
    for x, y in data.items():
        xy.append([x, y])
    xy.sort()
    return xy

def scatterplot(xy, title, xlabel, ylabel): 
    '''
    xy: [[x0, y0], [x1, y1], ...]
    xy is already sorted
    '''
    x = []
    y = []    
    for row in xy:
        x.append(row[0])
        y.append(row[1])
    
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return 0

def virus_bacteria_count(df):
    bacteria = 0
    virus = 0
    for element in df['bacteria/virus/normal']:
        if element=='bacteria':
            bacteria += 1
        else:
            virus += 1
    return bacteria, virus
    

##############################################################################
#
##############################################################################

df_test_n, df_test_p, df_train_n, df_train_p, df_val_n, df_val_p = open_dataframes()
df = pd.read_csv(csv_dir + 'combined.csv').iloc[:, 1:]
#print(df.head())

table = create_df()
res = dataviz(df, 'combined')
table = add_row(table, res)

res = dataviz(df_test_n, 'df_test_n')
table = add_row(table, res)

res = dataviz(df_test_p, 'df_test_p')
table = add_row(table, res)

res = dataviz(df_train_n, 'df_train_n')
table = add_row(table, res)

res = dataviz(df_train_p, 'df_train_p')
table = add_row(table, res)

res = dataviz(df_val_n, 'df_val_n')
table = add_row(table, res)

res = dataviz(df_val_p, 'df_val_p')
table = add_row(table, res)

print(table.head(10))


##############################################################################
#Histogram of NORMAL and PNEUMONIA for TRAIN set
##############################################################################

obs_n = df_train_n.shape[0]
obs_p = df_train_p.shape[0]

categories = ("NORMAL", "PNEUMONIA")
title = "NORMAL AND PNEUMONIA COUNT (TRAIN SET)"
xlabel = "Normal or Pneumonia"
ylabel = "Number of observations"
plot_histogram(categories, (obs_n, obs_p), title, xlabel, ylabel, 0.8)

##############################################################################
#Histogram of NORMAL and PNEUMONIA for TEST set
##############################################################################

obs_n = df_test_n.shape[0]
obs_p = df_test_p.shape[0]

categories = ("NORMAL", "PNEUMONIA")
title = "NORMAL AND PNEUMONIA COUNT (TEST SET)"
xlabel = "Normal or Pneumonia"
ylabel = "Number of observations"
plot_histogram(categories, (obs_n, obs_p), title, xlabel, ylabel, 0.8)

##############################################################################
#Histogram of NORMAL and PNEUMONIA for VALIDATION set
##############################################################################

obs_n = df_val_n.shape[0]
obs_p = df_val_p.shape[0]

categories = ("NORMAL", "PNEUMONIA")
title = "NORMAL AND PNEUMONIA COUNT (VALIDATION SET)"
xlabel = "Normal or Pneumonia"
ylabel = "Number of observations"
plot_histogram(categories, (obs_n, obs_p), title, xlabel, ylabel, 0.8)

##############################################################################
#Histogram of NORMAL and VIRUS and BACTERIA for TRAIN set
##############################################################################

obs_n = df_train_n.shape[0]
obs_b, obs_v = virus_bacteria_count(df_train_p)

categories = ("NORMAL", "BACTERIA", "VIRUS")
title = "NORMAL, BACTERIA AND VIRUS COUNT (TRAIN SET)"
xlabel = "Normal, Bacteria or Virus"
ylabel = "Number of observations"
plot_histogram(categories, (obs_n, obs_b, obs_v), title, xlabel, ylabel, 0.8)

##############################################################################
#Histogram of NORMAL and VIRUS and BACTERIA for TEST set
##############################################################################

obs_n = df_test_n.shape[0]
obs_b, obs_v = virus_bacteria_count(df_test_p)

categories = ("NORMAL", "BACTERIA", "VIRUS")
title = "NORMAL, BACTERIA AND VIRUS COUNT (TEST SET)"
xlabel = "Normal, Bacteria or Virus"
ylabel = "Number of observations"
plot_histogram(categories, (obs_n, obs_b, obs_v), title, xlabel, ylabel, 0.8)

##############################################################################
#Histogram of NORMAL and VIRUS and BACTERIA for VALIDATION set
##############################################################################

obs_n = df_val_n.shape[0]
obs_b, obs_v = virus_bacteria_count(df_val_p)

categories = ("NORMAL", "BACTERIA", "VIRUS")
title = "NORMAL, BACTERIA AND VIRUS COUNT (VALIDATION SET)"
xlabel = "Normal, Bacteria or Virus"
ylabel = "Number of observations"
plot_histogram(categories, (obs_n, obs_b, obs_v), title, xlabel, ylabel, 0.8)


##############################################################################
#Scatterplot of img_width (all images)
##############################################################################

xy_width = xy_calculator(df['width'])
title = "Number of observations in fuction of the width"
xlabel = "Width"
scatterplot(xy_width, title, xlabel, ylabel)

##############################################################################
#Scatterplot of img_height (all images)
##############################################################################

xy_height = xy_calculator(df['height'])
title = "Number of observations in fuction of the height"
xlabel = "Height"
scatterplot(xy_height, title, xlabel, ylabel)