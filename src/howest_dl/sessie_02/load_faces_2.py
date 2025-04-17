import tensorflow as tf
import os
from pathlib import Path

import matplotlib.pyplot as plt

# Data Reading
def get_label(file_path):
    # Extract the label from the filename
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(type(file_path), file_path)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    return tf.strings.split(file_path, os.sep)[-1]

def get_label_2(file_path):
    # Extract the filename from the file path
    print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
    print(type(file_path), file_path)
    print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
    filename = tf.strings.split(file_path, os.sep)[-1]
    # Remove specific characters (e.g., remove underscores) and transform to uppercase
    cleaned_filename = tf.strings.regex_replace(filename, '_', '')
    label = tf.strings.upper(cleaned_filename)
    return label

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [128, 128])

def process_path(file_path):
    label = get_label_2(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def load_dataset(data_dir):
    # Get list of all image file paths
    list_ds = tf.data.Dataset.list_files(str(Path(data_dir) / '*.jpg'))
    # Map the process_path function to each file path
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    return labeled_ds

# Data display
def show_examples_in_dataset(dataset, num_examples=5):
    for i, (image, label) in enumerate(dataset.take(num_examples)):
        print(f"--{i}--")
        print(type(label), label, type(image), image.shape)

def show_examples_in_batched(batched_dataset, num_batches=1):
    for batch in batched_dataset.take(num_batches):
        images, labels = batch
        # for i in range(len(images)):
        print(f"Aantal images: {len(images)}")
        for i, image in enumerate(images):
            print(f"--{i}--")
            img = images[i].numpy().astype("uint8")
            lbl = labels[i].numpy().decode('utf-8')
            print(type(lbl), lbl, type(img), img.shape)

if __name__ == '__main__':
    DATA_DIR = Path("/home/sam/howest/howest-dl/sessie_02/opdracht/Face_Recognition/2_Classes")
    dataset = load_dataset(DATA_DIR)
    print(dataset)
    dataset_batched = dataset.batch(2).prefetch(buffer_size=tf.data.AUTOTUNE)
    print(dataset_batched)

    show_examples_in_dataset(dataset)
    print("+++++++++++++++++++++++++++++++")
    show_examples_in_dataset(dataset_batched)