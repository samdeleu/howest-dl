import os

# Numbers
import pandas as pd
import numpy as np

# Graphics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from termcolor import (
    colored,
    cprint,
)

# Modeling
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
)
import keras

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf

from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Activation,
    BatchNormalization,
)
from keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image


###### Voor Tensorflow-GPU ########
import re

#K.set_image_dim_ordering('tf')

# Make sure that we have pandas.DataFrames at each step in scikit-learn
from sklearn import set_config
set_config(transform_output="pandas")

# Multiprocessing options (joblib/loky)
os.environ['OMP_NUM_THREADS']='8'

# limit some output when printing pandas data
pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',200)
pd.set_option('mode.copy_on_write', True)
pd.set_option('future.no_silent_downcasting', True)

# Some defaults for matplotlib
plt.rcParams['image.cmap'] = 'gray'
# plt.tight_layout()
# plt.rcParams.update({
#     "axes.titlesize":"10",
#     "xtick.labelsize":"8",
#     "ytick.labelsize":"8",
#     "font.size": "8",
#     # "figure.figsize": [6,3],
# })