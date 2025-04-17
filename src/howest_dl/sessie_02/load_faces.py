import os
from pathlib import Path

import tensorflow as tf
from skimage.io import imread, imshow

DATA_DIR = Path("/home/sam/howest/howest-dl/sessie_02/opdracht/Face_Recognition/15_Classes")
VALID_IMAGE_FILE_EXTENSIONS = [".jpg",".gif",".png"]


def process_face_file_dataset(file_path):
    print(f"processing: {file_path}")
    file_name = file_path.numpy().decode('utf-8')
    f, ext = os.path.splitext(file_name)
    p, label = os.path.split(f)
    if ext not in VALID_IMAGE_FILE_EXTENSIONS:  # the extension
        return None, None
    raw_image_data = tf.io.read_file(file_name)
    image_data = imread(file_name)
    print(image_data.shape)

    return image_data, label


def process_face_file(file_path_tensor):
    print(f"processing: {type(file_path_tensor)}: {file_path_tensor}")
    file_name = str(file_path_tensor)
    print(type(file_name), file_name)
    parts = tf.strings.split(file_path_tensor, os.sep)
    label = parts[-1]
    # f, ext = os.path.splitext(parts[-1].numpy().decode('utf-8'))
    # p, label = os.path.split(f)
    # if ext not in VALID_IMAGE_FILE_EXTENSIONS:  # the extension
    #     return None, None
    raw_image_data = tf.io.read_file(file_path_tensor)
    # image_data = imread(file_name)
    # print(image_data.shape)

    return label, label


if __name__ == '__main__':
    list_ds = tf.data.Dataset.list_files(f"{DATA_DIR}/*")
    # for f in list_ds:
    #     img, label = process_face_file_dataset(f)
    #     print(label)

    labeled_ds = list_ds.map(process_face_file)

    for img, label in labeled_ds.take(3):
        print(label, img.shape)
