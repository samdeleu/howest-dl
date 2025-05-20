import numpy as np
import tensorflow as tf
from tensorflow import keras

def dice_coefficient(image1, image2):
    # Ensure the images are binary (0 or 1)
    print(image1)
    print(image2)
    image1 = (image1 > 0).astype(np.int32)
    image2 = (image2 > 0).astype(np.int32)

    # Calculate the intersection and the sizes of the images
    # i1_maal_i2 = image1 * image2
    # intersection = np.sum(i1_maal_i2)

    intersection = np.sum(image1 * image2)
    size_image1 = np.sum(image1)
    size_image2 = np.sum(image2)

    # Calculate the Dice coefficient
    if (size_image1 + size_image2) == 0:
        return 1.0  # If both images are empty, return 1.0 (perfect similarity)

    dice = (2. * intersection) / (size_image1 + size_image2)
    return dice


def dice_coefficient2(y_true, y_pred):
    """
    Compute the Dice coefficient between y_true and y_pred.

    Args:
    - y_true: Ground truth mask
    - y_pred: Predicted mask

    Returns:
    - dice_coeff: The computed Dice coefficient.
    """

    # Flatten the tensors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Explicitly cast to float32
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)

    # Compute the intersection and the sum of the two masks
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    # Compute Dice coefficient
    dice_coeff = (2. * intersection) / union

    return dice_coeff

def dice_coefficient3(y_true, y_pred):
    y_true_f = tensorflow.keras.flatten(y_true)
    y_pred_f = tensorflow.keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

if __name__ == '__main__':
    # Example binary images
    image1 = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    print(image1.shape)
    image2 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    print(image2.shape)

    # Method 1: Calculate the Dice coefficient
    dice_score = dice_coefficient(image1, image2)
    print(f"Dice Coefficient: {dice_score}")

    # Method 2: Calculate the Dice coefficient
    dice_score2 = dice_coefficient2(image1, image2)
    print(f"Dice Coefficient2: {dice_score2}")

    # Method 3: Calculate the Dice coefficient
    dice_score2 = dice_coefficient3(image1, image2)
    print(f"Dice Coefficient2: {dice_score3}")

    # A list of images
    i1 = np.array([image1, image1, image1, image1])
    i2 = np.array([image2, image2, image2, image2])
    print(i1.shape)

    dice_scores = []
    num_images = i1.shape[0]

    for i in range(num_images):
        score = dice_coefficient(i1[i], i2[i])
        dice_scores.append(score)

    for i, score in enumerate(dice_scores):
        print(f"Dice Coefficient for image {i + 1}: {score}")

    print(f"Dice Coefficient List: {dice_scores}")