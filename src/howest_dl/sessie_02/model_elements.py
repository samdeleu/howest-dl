import os
import pickle

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from skimage.io import imread
# skimage.io.imshow is deprecated use matplotlib.pyplot.imshow instead
from skimage import data, color, io, filters, morphology,transform, exposure, feature, util
from scipy import ndimage
#import Tensorflow namespaces

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg19 import VGG19


def display(*args, **kwargs):
    print(*args, **kwargs)


def model_builder_1():
    # Neural network parameters
    # -----------------------------------------------
    # -----------------------------------------------
    batch_size = 64  #
    epochs = 150  #
    # -----------------------------------------------
    # -----------------------------------------------
    num_classes = 10
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)  # drie-kleuren kanaal

    # Model
    model = Sequential()
    # -----------------------------------------------
    # -----------------------------------------------
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))  # Eerste parameter = aantal features die wordt gezocht
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # -----------------------------------------------
    model.add(Dropout(0.3))  # Value between 0 and 1
    model.add(BatchNormalization())
    # -----------------------------------------------
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # -----------------------------------------------
    model.add(Dropout(0.3))  # Value between 0 and 1
    # -----------------------------------------------
    model.add(BatchNormalization())
    # -----------------------------------------------
    # -----------------------------------------------
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    # -----------------------------------------------
    # -----------------------------------------------
    model.add(Dropout(0.2))  # Value between 0 and 1
    # -----------------------------------------------
    # -----------------------------------------------
    model.add(Dense(num_classes, activation='softmax'))

    # model.compile(loss='hinge',
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    display(model.summary())
    # plot_model(model)


def model_builder_vgg19():
    modelVGG19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    print(type(modelVGG19))
    
    model = Sequential()
    print(type(modelVGG19.layers))
    for layer in modelVGG19.layers:
        model.add(layer)
        
    print(model.summary())

if __name__ == '__main__':
    model_builder_1()
    model_builder_vgg19()