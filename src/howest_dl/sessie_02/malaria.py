from pathlib import Path
from datetime import datetime
import pytz
import time

# Graphics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from termcolor import (
    colored,
    cprint,
)
import matplotlib.image as mpimg
from skimage.io import imread, imshow

# Data
import numpy as np
import pandas as pd

# SKLearn
from sklearn.datasets import (
    make_blobs,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    LabelBinarizer,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    auc,
    roc_curve,
    RocCurveDisplay,
)

from sklearn.utils import (
    class_weight,
    shuffle,
)

# distributions
from scipy.stats import randint
from scipy.stats import uniform

# Tensorflow and Keras
import tensorflow as tf
from tensorflow.keras.models import (
    Sequential,
)

from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

from tensorflow.keras.optimizers import (
    SGD,
    Adam,
)

from tensorflow.keras.activations import (
    leaky_relu,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input,
    decode_predictions,
)


from howest_dl.sessie_02.helpers import (
    display_title,
    display_value,
    print_metrics,
    read_images,
)

# Reading the data
def read_malaria_input():
    ROOT_PATH = Path("/home/sam/howest/howest-dl/sessie_02/opdracht")

    infected_train_images, y_infected_train = read_images(ROOT_PATH / "./Malaria/train/infected", 10, 100, 1)
    infected_test_images, y_infected_test = read_images(ROOT_PATH / "./Malaria/test/infected", 10, 100, 1)
    uninfected_train_images, y_uninfected_train = read_images(ROOT_PATH / "./Malaria/train/uninfected", 5, 100, 0)
    uninfected_test_images, y_uninfected_test = read_images(ROOT_PATH / "./Malaria/test/uninfected", 5, 100, 0)

    print(len(infected_train_images), len(y_infected_train))
    print(len(infected_test_images), len(y_infected_test))
    print(len(uninfected_train_images), len(y_uninfected_train))
    print(len(uninfected_test_images), len(y_uninfected_test))

    X_train, y_train = shuffle(uninfected_train_images + infected_train_images,  y_uninfected_train + y_infected_train, random_state=0)
    print(y_train[0:20])
    X_test, y_test = shuffle(uninfected_test_images + infected_test_images,  y_uninfected_test + y_infected_test, random_state=1)
    print(y_test[0:20])

    return X_train, y_train, X_test, y_test

def preprocess_data():
    pass

def build_model(inputshape=(100, 100, 3)):
    # Model
    model = Sequential()

    # Input Layer
    model.add(Input(shape=inputshape))

    # Feature Extraction Layers
    # Strides=(1,1), padding="valid"
    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(Activation(tf.keras.activations.relu))
    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(Activation(tf.keras.layers.ReLU(negative_slope=0.0)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(Activation('relu'))


    # Prepare for classification
    model.add(Flatten())

    # Classification Layers
    model.add(Dense(units=128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # Explicit activation

    model.add(Dense(units=128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # Explicit activation

    # Add an output layer for binary classification
    model.add(Dense(units=1))  # Single unit for binary classification
    model.add(Activation('sigmoid'))

    return model

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = read_malaria_input()
    preprocess_data()
    model = build_model()
    print(model.summary())
