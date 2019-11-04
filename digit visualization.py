# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:44:22 2019

@author: Joule
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

"""Open file with training data"""
train_data = pd.read_csv("train.csv")
labels = train_data["label"]


"""Convert dataframe to list so that i can visualize with pyplotlib"""
pictures_array = []
labels = []
num_of_pic_shown = 15
for num_picture in range(num_of_pic_shown):
    picture = train_data.loc[num_picture].values.tolist()
    label = picture.pop(0)
    labels.append(label)
    """make the pixel values to an 28x28 array"""
    pic_pixels = []
    pic_pixels_row = []
    for pixel_value in picture:
        pic_pixels_row.append(pixel_value)
        if len(pic_pixels_row) == 28:
            pic_pixels.append(pic_pixels_row)
            pic_pixels_row = []

    pictures_array.append(pic_pixels)

"""Plot the drawn images"""
for num_of_picture in range(len(pictures_array)):
    plt.imshow(pictures_array[num_of_picture], cmap=plt.cm.binary)
    plt.show()
    print("This image is supposed to be",labels[num_of_picture])
    

