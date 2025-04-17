import os
import re
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from howest_dl.sessie_02.load_faces import VALID_IMAGE_FILE_EXTENSIONS


def create_keras_compatible_structure(source_dir, target_dir):
    def get_name(filename):
        VALID_IMAGE_FILE_EXTENSIONS = [".jpg"]
        name, ext = os.path.splitext(filename)
        if ext not in VALID_IMAGE_FILE_EXTENSIONS:
            return None

        name.replace(ext, "")
        name = re.sub(r'[^a-zA-Z\s]', "", name)
        name = name.strip()
        name = name.replace(" ", "_")
        return name

    def categorize_files(data_dir):
        files = os.listdir(data_dir)
        df_files = pd.DataFrame(files, columns=['filename'])
        df_files["name"] = df_files["filename"].apply(get_name)
        df_files = df_files[df_files["name"].notna()]

        categories = df_files["name"].unique()
        df_test_files = df_files.sample(frac=0.25,  random_state=101 )
        df_train_files = df_files.drop(df_test_files.index)

        return df_train_files, df_test_files, categories

    df_train_files, df_test_files, categories = categorize_files(source_dir)

    for d in categories:
        os.makedirs(os.path.join(target_dir, "train", d), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "test", d), exist_ok=True)

    # Create the train structure
    for _, entry in df_train_files.iterrows():
        print("TRAINING", entry["filename"])
        source_path = os.path.join(source_dir, entry["filename"])
        target_path = os.path.join(target_dir, "train", entry["name"], entry["filename"])
        shutil.copy2(source_path, target_path)

    # Create the test structure
    for _, entry in df_test_files.iterrows():
        print("TEST", entry["filename"])
        source_path = os.path.join(source_dir, entry["filename"])
        target_path = os.path.join(target_dir, "test", entry["name"], entry["filename"])
        shutil.copy2(source_path, target_path)

    return categories

if __name__ == '__main__':
    # print(get_name("123 AA 33 BB 321.jpg"))
    DATA_DIR = Path("/home/sam/howest/howest-dl/sessie_02/opdracht/Face_Recognition/2_Classes")
    WORKING_DATA_DIR = Path("/home/sam/howest/howest-dl/sessie_02/opdracht/Face_Recognition/2_Classes_keras")

    categories = create_keras_compatible_structure(source_dir=DATA_DIR, target_dir=WORKING_DATA_DIR)
    print(type(categories))

    # Inlezen van de data in een tf.data.Dataset voor training
    # split voor gebruik als validatie
    TRAINING_DATA_DIR = Path(WORKING_DATA_DIR) / Path("train")
    TEST_DATA_DIR = Path(WORKING_DATA_DIR) / Path("test")

    training_dataset, validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=str(TRAINING_DATA_DIR),
        labels="inferred",
        label_mode='categorical',
        color_mode='rgb',
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=101,
        validation_split=0.25,
        subset="both",
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True
    )

    print(training_dataset.class_names)

    for element in training_dataset.take(2):
        image, label = element
        print(f"{label}", label.shape)
        print("image size", image.shape)

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=str(TEST_DATA_DIR),
        labels="inferred",
        label_mode='categorical',
        color_mode='rgb',
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=101,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True
    )

    for element in test_dataset.take(2):
        image, label = element
        print(f"{label}", label.shape)
        print("image size", image.shape)