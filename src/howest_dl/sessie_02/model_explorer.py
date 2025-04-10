# Tensorflow and Keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

# from tensorflow.python.layers.normalization import (
#     BatchNormalization,
# )

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

