# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:00:52 2019

@author: Joule
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""Load in training and testing data"""
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

"""convert test data dataframe to 28x28 matrix of pixels"""
testdata_array = []
for num_picture in range(len(test_data)):
    print("Doing task",num_picture, "out of",len(test_data))
    picture = test_data.loc[num_picture].values.tolist()
    """make the pixel values to an 28x28 array"""
    pic_pixels = []
    pic_pixels_row = []
    for pixel_value in picture:
        pic_pixels_row.append(pixel_value)
        if len(pic_pixels_row) == 28:
            pic_pixels.append(pic_pixels_row)
            pic_pixels_row = []

    testdata_array.append(pic_pixels)


"""Convert training data dataframe to 28x28 matrix and labels so that NN can learn on it"""
trainingdata_array = []
labels = []
for num_picture in range(len(train_data)):
    print("Doing task",num_picture, "out of",len(train_data))
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

    trainingdata_array.append(pic_pixels)

"""Normalize data, and set it up"""
training_data = trainingdata_array
training_data = tf.keras.utils.normalize(training_data,axis=1)
training_predictions = np.array(train_data['label'].values.tolist())


"""Make model and add hidden layers and softmax at the end"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

"""Choose model parameters"""
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics = ["accuracy"]
              )

"""Fit and predict"""
print("Fitting")
model.fit(training_data,training_predictions,epochs=1) # epochs = 1 to avoid overfitting since we have so much data

print("Predicting")
testing_data = testdata_array
testing_data = tf.keras.utils.normalize(testing_data,axis=1)

"""Make predictions a one dimensional array"""
predictions = model.predict([testing_data])
for i in range(len(predictions)):
    print(np.argmax(predictions[i]))
    predictions[i] = np.argmax(predictions[i])
predictions = [int(line[0]) for line in predictions]



"""Plot the n first test images with their corresponding predictions"""
n = 10
for num_of_picture in range(n):
    plt.imshow(testdata_array[num_of_picture], cmap=plt.cm.binary)
    plt.show()
    print("the output for this image was",int(predictions[num_of_picture]))


"""Save the predictions on test-set in a csv document to submit to Kaggle"""
ImageId = [k for k in range(1, 1 + len(predictions))]
Label = predictions
output = pd.DataFrame()
output['ImageId'] = ImageId
output['Label'] = Label
output.to_csv('digit_recognized.csv', index=False)
    




