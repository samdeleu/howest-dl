import sys
import os
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
import tensorflow
#from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNomalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Rescaling,
)

from tensorflow.python.keras.activations import (
    relu,
    leaky_relu,
)
from tensorflow.python.keras.callbacks import (
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

# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.python.layers.normalization import BatchNormalization


from howest_dl.sessie_02.helpers import (
    display_title,
    display_value,
    collect_accuracy,
    print_binary_metrics,
    read_images,
)
from howest_dl.sessie_02.model_explorer import model_explorer

# Reading the data
def read_malaria_input(sample_size=1000, image_size=100):
    ROOT_PATH = Path("/home/sam/howest/howest-dl/sessie_02/opdracht")

    infected_train_images, y_infected_train = read_images(ROOT_PATH / "./Malaria/train/infected", sample_size, image_size, 1)
    infected_test_images, y_infected_test = read_images(ROOT_PATH / "./Malaria/test/infected", sample_size, image_size, 1)
    uninfected_train_images, y_uninfected_train = read_images(ROOT_PATH / "./Malaria/train/uninfected", sample_size, image_size, 0)
    uninfected_test_images, y_uninfected_test = read_images(ROOT_PATH / "./Malaria/test/uninfected",sample_size, image_size, 0)

    print(len(infected_train_images), len(y_infected_train))
    print(len(infected_test_images), len(y_infected_test))
    print(len(uninfected_train_images), len(y_uninfected_train))
    print(len(uninfected_test_images), len(y_uninfected_test))

    X_train, y_train = shuffle(uninfected_train_images + infected_train_images,  y_uninfected_train + y_infected_train, random_state=0)
    print(type(X_train), y_train[0:20])
    X_test, y_test = shuffle(uninfected_test_images + infected_test_images,  y_uninfected_test + y_infected_test, random_state=1)
    print(type(X_test), y_test[0:20])

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    print(type(X_train), y_train[0:20])
    return X_train, y_train, X_test, y_test

def preprocess_data():
    pass

def build_malaria_model(input_shape=(100, 100, 3)):
    # Model
    model = Sequential()

    # Input Layer
    model.add(Input(shape=input_shape))

    # Feature Extraction Layers
    # Strides=(1,1), padding="valid"
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation(tf.keras.activations.relu))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(tf.keras.activations.relu))

    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(tf.keras.activations.relu))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(tf.keras.layers.ReLU(negative_slope=0.0)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(tf.keras.layers.ReLU(negative_slope=0.0)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Prepare for classification
    model.add(Flatten())

    # Classification Layers
    model.add(Dense(units=32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # Explicit activation

    model.add(Dense(units=16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # Explicit activation

    # Add an output layer for binary classification
    model.add(Dense(units=1))  # Single unit for binary classification
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            # tf.keras.metrics.Accuracy(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.TruePositives(),
        ],
    )
    return model

def train_malaria_model(model, X_train_input, y_train_input, description, verbose=1):
    # Stop training when no further improvement is seen in the metric
    early_stop = EarlyStopping(
        monitor="val_loss",    # metric to monitor
        patience=10,           # stop when no improvement after 10 consecutive epochs
        mode="min",            # stop when metric stops decreasing
        restore_best_weights=True,
        verbose=verbose,             # display the actions taken
    )
    # Callback to save the Keras model or model weights at some frequency
    checkpoint = ModelCheckpoint(
        filepath="malaria.keras",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_freq="epoch",
        verbose=verbose,
    )
    # Reduce learning rate when a metric has stopped improving.
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        mode="min",
        # min_delta=0.0001,
        # cooldown=0,
        min_lr=1e-6,
        verbose=verbose,
    )
    # Save output to be used with tensorboard
    tensorboard = TensorBoard(
        log_dir=f"logs/malaria_{description}"
    )

    # Train the model
    start_timing = time.time()
    history = model.fit(
        x=X_train_input,
        y=y_train_input,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        verbose=verbose,
        callbacks=[early_stop, reduce_lr_on_plateau, tensorboard],
    )
    end_timing = time.time()
    return history, end_timing - start_timing

def build_malaria_model_based_on_vgg():
    modelVGG19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    type(modelVGG19)

if __name__ == '__main__':
    pass