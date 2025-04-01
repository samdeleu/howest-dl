import os
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    auc,
    roc_curve,
    RocCurveDisplay,
)

from skimage.io import imread
from skimage import transform

from termcolor import (
    colored,
    cprint,
)

ROOT_PATH = Path("/home/sam/howest/howest-dl/sessie_02/opdracht")
malaria_path = Path("/home/sam/howest/howest-dl/sessie_02/opdracht/Malaria/train/infected")

# Display functions
def display_title(title, value=None):
    if value is None:
        cprint(title, "black", "on_cyan")
    else:
        print(colored(title, "blue"))
        print(value)

def display_value(title, value):
    print(f"{colored(title, "blue")}: {value}")


# Metrics
def print_metrics(model, X_test, y_test, title="Results"):
    display_title(title)
    y_pred_proba = model.predict(X_test)
    y_pred_class = np.argmax(y_pred_proba, axis=1)

    display_title("classification report", classification_report(y_test, y_pred_class))
    display_title("confusion matrix", confusion_matrix(y_test, y_pred_class))
    display_value("Accuracy score", (accuracy_score(y_test, y_pred_class) * 100))

# Reading images
def read_images(path, nbr_images, image_size, label):
    valid_image_extensions = [".jpg", ".gif", ".png"]
    selected_images = []
    labels = []
    nbr_of_images_collected = 0
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_image_extensions:
            continue
        if nbr_of_images_collected >= nbr_images:
            break
        im = imread(os.path.join(path, f))
        im = transform.resize(im, (image_size, image_size), mode='constant', anti_aliasing=True)
        selected_images.append(im)
        labels.append(label)
        nbr_of_images_collected = nbr_of_images_collected+1

    return selected_images, labels

if __name__ == '__main__':
    infected_train_images, y_infected_train = read_images(ROOT_PATH / "./Malaria/train/infected", 4, 100, 1)
    infected_test_images, y_infected_test = read_images(ROOT_PATH / "./Malaria/test/infected", 5, 100, 1)
    uninfected_train_images, y_uninfected_train = read_images(ROOT_PATH / "./Malaria/train/uninfected", 6, 100, 0)
    uninfected_test_images, y_uninfected_test = read_images(ROOT_PATH / "./Malaria/test/uninfected", 7, 100, 0)
    print(len(infected_train_images), len(y_infected_train))
    print(len(infected_test_images), len(y_infected_test))
    print(len(uninfected_train_images), len(y_uninfected_train))
    print(len(uninfected_test_images), len(y_uninfected_test))
