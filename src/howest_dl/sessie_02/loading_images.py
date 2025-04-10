from pathlib import Path

import tensorflow

DATA_DIR = Path("/home/sam/howest/howest-dl/udemy_1/TF_2_Notebooks_and_Data/04-CNNs/cell_images")

def loading():
    train_ds, validation_ds = tensorflow.keras.preprocessing.image_dataset_from_directory(
        directory=DATA_DIR / Path("train"),
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        color_mode="rgb",
        subset="both",
        image_size=(50, 50),
        shuffle=True,
        seed=123,
        batch_size=None,
    )

    return train_ds, validation_ds


if __name__ == '__main__':
    train_ds, val_ds = loading()

    example = train_ds.take(1)
    print(type(train_ds))
    print(train_ds.class_names)