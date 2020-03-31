import pandas as pd
import numpy as np

pixel_size = 256

df = pd.read_csv('C:\\Thomas_Data\\OneDrive\\Temp\\projh419_data\\csv\\train_pneumonia.csv')
print(df.head())
width = df['width'].to_numpy()
height = df['height'].to_numpy()
res = 0
for i in range(df.shape[0]):
    if ( width[i]<pixel_size ) or ( height[i]<pixel_size ):
        res +=1

print(res)