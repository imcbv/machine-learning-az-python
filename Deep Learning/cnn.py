#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 12:37:13 2017

@author: imcbv
"""

# Convolutional neural network

# Reset variables
%reset -f

# Part 1 - Build the CNN

# import libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import os, sys

# initialize CNN
classifier = Sequential()

# convolution
# input_shape order follows Tensor Flow specifications, 
# Theano would be the opposide
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# add a second convolutional layer to improve performances
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# flattening
classifier.add(Flatten())

# full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# compiling the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

trainining_set = train_datagen.flow_from_directory(
    'image_dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'image_dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        trainining_set,
        samples_per_epoch=8000,
        nb_epoch=25,
        validation_data=test_set,
        nb_val_samples=2000)


# Making the prediction
new_datagen = ImageDataGenerator(rescale=1./255)
new_set = new_datagen.flow_from_directory(
        'image_dataset/new_set',
        target_size=(64, 64),
        batch_size=6,
        class_mode=None,
        shuffle = False)

predictions = classifier.predict_generator(new_set, 6)

# Read filenames in the folder
path = "image_dataset/new_set/test"
filenames = os.listdir( path )

# Read classnames
path = "image_dataset/training_set"
classnames = os.listdir( path )
classnames = [i for i in classnames if i != '.DS_Store']

# Create the dataframe
import pandas as pd
import numpy as np
df = pd.DataFrame(filenames)
df.columns = ['Filename']
df['Prediction'] = np.round(predictions)
df['Predicted Class'] = [classnames[int(i)] for i in np.round(predictions)]
df['Confidence'] = predictions
  
temp = (df[['Prediction','Confidence']].sum(axis=1) - 1).abs()
df['Confidence'] = temp 
df.drop('Prediction', axis=1, inplace=True)
  
# Export to Excel
df.to_excel('Prediction_output.xls', index=True)
print('Done')
