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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
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
    relu,
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
    collect_accuracy,
    print_binary_metrics,
    read_images,
)

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

def model_explorer(name, input_shape, feature_extractor, classifier, skip=False):
    if skip:
        return None

    model = Sequential()
    # Input Layer
    model.add(Input(shape=input_shape))

    # Feature Extraction Layers
    for fl in feature_extractor:
        match fl:
            case ("C", filters):
                model.add(Conv2D(filters=filters, kernel_size=(3, 3)))
            case ("A", "relu"):
                model.add(Activation(tf.keras.activations.relu))
            case ("A", "leaky_relu"):
                model.add(Activation(tf.keras.activations.leaky_relu))
            case "B":
                model.add(BatchNormalization())
            case ("Dr", rate):
                model.add(Dropout(rate=rate))
            case ("P", "max"):
                model.add(MaxPooling2D(pool_size=(2, 2)))
            case ("P", "avg"):
                model.add(AveragePooling2D(pool_size=(2, 2)))
            case x:
                print(f"Unknown layer type {x}")

    # Prepare for classification
    # Flatten the output of the feature extraction layers
    model.add(Flatten())

    # Classification Layers
    for cl in classifier:
        match cl:
            case ("D", units):
                model.add(Dense(units=units))
            case ("A", activation):
                model.add(Activation(activation))
            case "B":
                model.add(BatchNormalization())
            case ("Dr", rate):
                model.add(Dropout(rate=rate))
            case x:
                print(f"Unknown layer type {x}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        # metrics=['accuracy'],
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
    setattr(model, "howest", name)

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

if __name__ == '__main__':
    image_size = 50
    sample_size = 1000
    X_train, y_train, X_test, y_test = read_malaria_input(sample_size=sample_size, image_size=image_size)
    print(type(X_train))

    skip = False
    if not skip:
        preprocess_data()
        malaria_model = build_malaria_model(input_shape=(image_size, image_size, 3))
        print(malaria_model.summary())

        history, timing = train_malaria_model(
            model=malaria_model,
            X_train_input=X_train, y_train_input=y_train,
            description="First attempt"
        )
        print(malaria_model.metrics_names)
        # Metrics on the training set
        print_binary_metrics(model=malaria_model, X_test_input=X_train, y_test_input=y_train, decision_boundary=0.5, title="Training Set")
        # Metrics on the test set
        print_binary_metrics(model=malaria_model, X_test_input=X_test, y_test_input=y_test, decision_boundary=0.5, title="Test Set")
        collect_accuracy(model=malaria_model, X_train_input=X_train, y_train_input=y_train, X_test_input=X_test, y_test_input=y_test, decision_boundary=0.5)

        # Metrics on the training set
        print_binary_metrics(model=malaria_model, X_test_input=X_train, y_test_input=y_train, decision_boundary=0.95, title="Training Set")
        # Metrics on the test set
        print_binary_metrics(model=malaria_model, X_test_input=X_test, y_test_input=y_test, decision_boundary=0.95, title="Test Set")
        collect_accuracy(model=malaria_model, X_train_input=X_train, y_train_input=y_train, X_test_input=X_test, y_test_input=y_test, decision_boundary=0.95)

        y_pred_proba = malaria_model.predict(X_test).flatten()
        y_pred_class = (y_pred_proba >= 0.95).astype(int)

        X_test_fp = X_test[(y_test == 0) & (y_pred_class == 1)]
        X_test_fn = X_test[(y_test == 1) & (y_pred_class == 0)]

        display_title("Examples of False Negatives")
        for i in range(min([3, len(X_test_fn)])):
            print(X_test_fn[i])

    image_shape=(image_size, image_size, 3)
    # Results: Name, threshold, "accuracy on training set", "accuracy on test set"
    malaria_models = [
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 01",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 128), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 128), "B", ("A", "relu"),
                ("D", 64), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 02",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 03",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 128), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "avg"),
                ("C", 64), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "avg"),
                ("C", 32), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "avg"),
            ],
            classifier=[
                ("D", 128), "B", ("A", "relu"),
                ("D", 64), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 04",
            input_shape=(image_size, image_size, 3),
            feature_extractor=[
                ("C", 128), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
                ("C", 32), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
            ],
            classifier=[
                ("D", 128), "B", ("A", "relu"),
                ("D", 64), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 05",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 128), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
                ("C", 32), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
            ],
            classifier=[
                ("D", 128), "B", ("A", "relu"),
                ("D", 64), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 06",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 256), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
                ("C", 128), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
                ("C", 32), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("P", "max"),
            ],
            classifier=[
                ("D", 128), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("D", 64), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 07",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 08",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 09",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 10",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 11",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 12",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 13",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 14",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 16), "B", ("A", "relu"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 64), "B", ("A", "relu"),
                ("C", 64), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 64), "B", ("A", "relu"),
                ("D", 32), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 15",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 8), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 16), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 1024), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("D", 1024), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 16",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 8), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 16), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 32), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 1024), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("D", 1024), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 17",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 8), "B", ("A", "leaky_relu"),
                ("P", "max"),
                ("C", 16), "B", ("A", "leaky_relu"),
                ("P", "max"),
                ("C", 32), "B", ("A", "leaky_relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 1024), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("D", 1024), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 18",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 128), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 128), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 256), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 128), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 1024), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("D", 128), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),
        model_explorer(
            name=f"Exploration ({image_size}x{image_size}) 19",
            input_shape=image_shape,
            feature_extractor=[
                ("C", 128), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 256), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 256), "B", ("A", "relu"),
                ("P", "max"),
                ("C", 128), "B", ("A", "relu"),
                ("P", "max"),
            ],
            classifier=[
                ("D", 1024), "B", ("A", "relu"),
                ("Dr", 0.3),
                ("D", 512), "B", ("A", "relu"),
                ("D", 64), "B", ("A", "relu"),
                ("D", 1),
                ("A", "sigmoid"),
            ]
        ),

    ]

    collected_results = {}
    for i, m in enumerate(malaria_models):
        if m is None:
            continue

        print(f"Session: {i}-{getattr(m, 'howest', 'xx')}")
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        y_train_copy = y_train.copy()
        y_test_copy = y_test.copy()
        training_history = train_malaria_model(
            model=m,
            X_train_input=X_train_copy, y_train_input=y_train_copy,
            description=f"{i}-{getattr(m, 'howest', 'xx')}",
            verbose=1,
        )

        for threshold in [0.5, 0.9, 0.99]:
            if collected_results.get(threshold) is None:
                collected_results[threshold] = []

            collected_results[threshold].append(
                collect_accuracy(
                    model=m,
                    X_train_input=X_train_copy, y_train_input=y_train_copy,
                    X_test_input=X_test_copy, y_test_input=y_test_copy,
                    decision_boundary=threshold,
                    verbose=0,
                )
            )

    for threshold, collected_results in collected_results.items():
        print("Treshold:", threshold)
        for n, t, train_a, test_a in collected_results:
            print(f"{n}\tthreshold:\t{t:.2f}\tTrain:\t{train_a:.2f}\tTest:\t{test_a:.2f})")


    """
    Summary results
    Treshold: 0.5
    Exploration (50x50) 01	threshold:	0.50	Train:	98.00	Test:	93.10)
    Exploration (50x50) 02	threshold:	0.50	Train:	98.10	Test:	92.55)
    Exploration (50x50) 03	threshold:	0.50	Train:	95.75	Test:	91.85)
    Exploration (50x50) 04	threshold:	0.50	Train:	57.95	Test:	55.95)
    Exploration (50x50) 05	threshold:	0.50	Train:	50.15	Test:	50.00)
    Exploration (50x50) 06	threshold:	0.50	Train:	51.45	Test:	50.90)
    Exploration (50x50) 07	threshold:	0.50	Train:	98.50	Test:	94.30)
    Exploration (50x50) 08	threshold:	0.50	Train:	98.50	Test:	93.80)
    Exploration (50x50) 09	threshold:	0.50	Train:	98.25	Test:	93.65)
    Exploration (50x50) 10	threshold:	0.50	Train:	98.70	Test:	94.40)
    Exploration (50x50) 11	threshold:	0.50	Train:	98.55	Test:	94.00)
    Exploration (50x50) 12	threshold:	0.50	Train:	98.05	Test:	93.35)
    Exploration (50x50) 13	threshold:	0.50	Train:	98.50	Test:	93.10)
    Exploration (50x50) 14	threshold:	0.50	Train:	98.50	Test:	93.60)
    Exploration (50x50) 15	threshold:	0.50	Train:	98.20	Test:	91.10)
    Exploration (50x50) 16	threshold:	0.50	Train:	97.60	Test:	89.25)
    Exploration (50x50) 17	threshold:	0.50	Train:	98.00	Test:	91.40)
    Exploration (50x50) 18	threshold:	0.50	Train:	98.20	Test:	93.60)
    Exploration (50x50) 19	threshold:	0.50	Train:	98.15	Test:	94.45)
    
    Treshold: 0.9
    Exploration (50x50) 01	threshold:	0.90	Train:	96.55	Test:	90.35)
    Exploration (50x50) 02	threshold:	0.90	Train:	96.35	Test:	89.15)
    Exploration (50x50) 03	threshold:	0.90	Train:	89.55	Test:	86.65)
    Exploration (50x50) 04	threshold:	0.90	Train:	50.00	Test:	50.00)
    Exploration (50x50) 05	threshold:	0.90	Train:	50.00	Test:	50.00)
    Exploration (50x50) 06	threshold:	0.90	Train:	50.00	Test:	50.00)
    Exploration (50x50) 07	threshold:	0.90	Train:	98.15	Test:	91.85)
    Exploration (50x50) 08	threshold:	0.90	Train:	96.80	Test:	91.50)
    Exploration (50x50) 09	threshold:	0.90	Train:	95.60	Test:	90.05)
    Exploration (50x50) 10	threshold:	0.90	Train:	98.60	Test:	92.20)
    Exploration (50x50) 11	threshold:	0.90	Train:	97.80	Test:	91.40)
    Exploration (50x50) 12	threshold:	0.90	Train:	97.25	Test:	91.90)
    Exploration (50x50) 13	threshold:	0.90	Train:	96.65	Test:	89.30)
    Exploration (50x50) 14	threshold:	0.90	Train:	96.05	Test:	89.60)
    Exploration (50x50) 15	threshold:	0.90	Train:	97.90	Test:	89.15)
    Exploration (50x50) 16	threshold:	0.90	Train:	97.55	Test:	88.50)
    Exploration (50x50) 17	threshold:	0.90	Train:	97.95	Test:	91.25)
    Exploration (50x50) 18	threshold:	0.90	Train:	98.00	Test:	94.10)
    Exploration (50x50) 19	threshold:	0.90	Train:	97.60	Test:	93.95)
    
    Treshold: 0.99
    Exploration (50x50) 01	threshold:	0.99	Train:	86.05	Test:	80.35)
    Exploration (50x50) 02	threshold:	0.99	Train:	73.40	Test:	70.00)
    Exploration (50x50) 03	threshold:	0.99	Train:	75.10	Test:	74.90)
    Exploration (50x50) 04	threshold:	0.99	Train:	50.00	Test:	50.00)
    Exploration (50x50) 05	threshold:	0.99	Train:	50.00	Test:	50.00)
    Exploration (50x50) 06	threshold:	0.99	Train:	50.00	Test:	50.00)
    Exploration (50x50) 07	threshold:	0.99	Train:	89.40	Test:	82.80)
    Exploration (50x50) 08	threshold:	0.99	Train:	84.60	Test:	79.05)
    Exploration (50x50) 09	threshold:	0.99	Train:	72.45	Test:	69.60)
    Exploration (50x50) 10	threshold:	0.99	Train:	94.00	Test:	86.90)
    Exploration (50x50) 11	threshold:	0.99	Train:	90.95	Test:	83.95)
    Exploration (50x50) 12	threshold:	0.99	Train:	89.85	Test:	84.90)
    Exploration (50x50) 13	threshold:	0.99	Train:	79.15	Test:	75.50)
    Exploration (50x50) 14	threshold:	0.99	Train:	75.65	Test:	73.20)
    Exploration (50x50) 15	threshold:	0.99	Train:	94.05	Test:	85.85)
    Exploration (50x50) 16	threshold:	0.99	Train:	97.05	Test:	86.05)
    Exploration (50x50) 17	threshold:	0.99	Train:	96.25	Test:	89.00)
    Exploration (50x50) 18	threshold:	0.99	Train:	96.25	Test:	92.55)
    Exploration (50x50) 19	threshold:	0.99	Train:	93.50	Test:	89.35) 
    """