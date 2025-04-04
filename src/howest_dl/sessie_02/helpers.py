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
import matplotlib.pyplot as plt

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

def display(value):
    print(value)

# Metrics
def print_categorical_metrics(model, X_test, y_test, title="Results"):
    display_title(title)
    y_pred_proba = model.predict(X_test)
    # find the index of the maximum predicted probability this equals the number of the predicted class
    y_pred_class = np.argmax(y_pred_proba, axis=1)

    display_title("classification report", classification_report(y_test, y_pred_class))
    display_title("confusion matrix", confusion_matrix(y_test, y_pred_class))
    display_value("Accuracy score", (accuracy_score(y_test, y_pred_class) * 100))

def plot_roc_curve(false_positive_rate, true_positive_rate, area_under_curve):
    display_value("AUC", auc(fpr, tpr))

    # ROC
    fig, axes = plt.subplots(1, 1, figsize = (6, 3), sharey=False)
    axes[0].set_title("Receiver Operating Characteristic: Infected TRUE")
    axes[0].plot(false_positive_rate, true_positive_rate, "b", label = f"AUC = {area_under_curve}")
    axes[0].plot([0, 1], [0, 1],'r--')
    axes[0].xlim([0, 1])
    axes[0].ylim([0, 1])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].grid()
    axes[0].legend(loc='upper right')

# Placeholder
def plot_roc_curve(model, X_test_input, y_test_input):

    y_pred_proba = model.predict(X_test_input).flatten()
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_input, y_pred_proba, pos_label=1)
    area_under_curve = auc(false_positive_rate, true_positive_rate)
    display_value("AUC", area_under_curve)

    # ROC
    # fig, ax = plt.subplots(figsize = (6, 3), sharey=False)
    # ax.set_title("Receiver Operating Characteristic: Infected TRUE")
    # ax.plot(false_positive_rate, true_positive_rate, color="blue", label = f"AUC = {area_under_curve}")
    # ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    # ax.set_xlim(0.0, 1.0)
    # ax.set_ylim(0.0, 1.0)
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.grid()
    # ax.legend(loc='upper right')
    # plt.show()

def print_binary_metrics(model, X_test_input, y_test_input, decision_boundary=0.5, title="Binary Results"):
    display_title(f"{title} - Treshold({decision_boundary})")
    y_pred_proba = model.predict(X_test_input).flatten()
    y_pred_class = (y_pred_proba >= decision_boundary).astype(int)

    display_title("Scikit learn metrics", "---------------------------")
    display_title("classification report", classification_report(y_test_input, y_pred_class))
    display_title("confusion matrix", confusion_matrix(y_test_input, y_pred_class))
    display_value("Accuracy score", (accuracy_score(y_test_input, y_pred_class) * 100))

    display_title("Tensorflow evaluate", "---------------------------")
    metrics = model.evaluate(X_test_input, y_test_input, return_dict=True)
    for m,v in metrics.items():
        display_value(m, v)
    display("-----------------------------------------")


def collect_accuracy(model, X_train_input, y_train_input, X_test_input, y_test_input, decision_boundary=0.5, verbose=1):
    # Train accuracy
    y_train_pred_proba = model.predict(X_train_input, verbose=verbose).flatten()
    y_train_pred_class = (y_train_pred_proba >= decision_boundary).astype(int)
    train_accuracy = accuracy_score(y_train_input, y_train_pred_class) * 100

    # Test accuracy
    y_test_pred_proba = model.predict(X_test_input, verbose=verbose).flatten()
    y_test_pred_class = (y_test_pred_proba >= decision_boundary).astype(int)
    test_accuracy = accuracy_score(y_test_input, y_test_pred_class) * 100

    display_value(f"Accuracy - ({getattr(model, "howest", "xx")}) - Threshold({decision_boundary:.2f})", f"Train({train_accuracy:.2f})\tTest({test_accuracy:.2f})")
    return getattr(model, "howest", "xx"), decision_boundary, train_accuracy, test_accuracy

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
