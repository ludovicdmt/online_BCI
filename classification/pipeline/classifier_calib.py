"""Train and save a classifier on the calibration data."""

import pickle
import joblib
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
    cross_val_predict,
)
from sklearn.metrics import confusion_matrix

from pipeline.fbcsp import get_FBCSP_model
from pipeline.csp import get_CSP_model
from pipeline.kernelsCSP import get_kernelCSP_model
from pipeline.riemann import get_MDM_model, get_TS_model
from pipeline.kernelTS import get_kernelTS_model


def load_data(file: str) -> Tuple[np.array, np.array]:
    """Load data from a pickle file.

    Args:
        file (str): Path to the pickle file.
    Returns:
        Tuple(np.array, np.array): Data and labels.
    """
    with open(file, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict["calib_data"], data_dict["labels"]


def train_model(
    model_name: str, X: np.array, y: np.array, sample_rate: int, kernels=None, lr=True
):
    """Train the specified model on the given data.

    Args:
        model_name (str): Name of the model to train.
        X (np.array): Data.
        y (np.array): Labels.
        sample_rate (int): Sampling rate of the data.
        lr (bool): Whether to use logistic regression as the classifier (default: False).
    Output:
        model: Trained model.
    """
    feat_name = model_name.split("_")[0]
    if feat_name == "CSP":
        model, search_space = get_CSP_model(lr=lr)
    elif feat_name == "FBCSP":
        model, search_space = get_FBCSP_model(fs=sample_rate)
    elif feat_name == "kernelCSP":
        model, search_space = get_kernelCSP_model(kernels=kernels)
    elif feat_name == "TS":
        model, search_space = get_TS_model()
    elif feat_name == "MDM":
        model, search_space = get_MDM_model()
    elif feat_name == "kernelTS":
        model, search_space = get_kernelTS_model()
    else:
        raise ValueError("Unknown model name.")

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(model, param_grid=search_space, cv=cv, n_jobs=-1)
    print("Training the model...")
    grid.fit(X, y)
    # Get the mean and std of the test scores for the best parameters
    scores = cross_val_score(grid.best_estimator_, X, y, cv=5)
    # Print the results
    print(
        "%0.2f accuracy with a standard deviation of %0.2f"
        % (scores.mean(), scores.std())
    )
    # Compute confusion matrix and save it
    y_pred = cross_val_predict(grid.best_estimator_, X, y, cv=5)
    conf_mat = confusion_matrix(y, y_pred)
    # Normalise
    conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
    # Plot confusion matrix
    ax = plt.subplot()
    cmn = pd.DataFrame(conf_mat, range(2), range(2))
    sns.heatmap(cmn, annot=True, cmap="YlGn", ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    # Set the names of the classes
    if "_stage1" in model_name:
        classes_names = ["Motor Intention", "Rebound"]
    else:
        classes_names = ["Right", "Left"]
    ax.xaxis.set_ticklabels(classes_names)
    ax.yaxis.set_ticklabels(classes_names)
    plt.title(
        f"Average score: {scores.mean():.2f} (+/-) {scores.std():.2f}", fontsize=15
    )
    plt.savefig(f"saved_models\\{model_name}_confusion_matrix.png")
    return grid


def save_model(
    model_name: str, sample_rate: int, data_path: str, file: str, kernels=None
):
    """Save the model in a pickle file.

    Args:
        model_name (str): Name of the model to train. Can be either "CS" or "FBCSP".
        sample_rate (int): Sampling rate of the data.
        data_path (str): Path to the data.
        kernels (np.aray): Kernels for the kernelCSP model (default: None).
    """
    X, y = load_data(data_path)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = np.array(y)[idx]
    # Assuming y is your numpy array
    unique, counts = np.unique(y, return_counts=True)
    print(f"Labels: {dict(zip(unique, counts))}")
    clf = train_model(model_name, X, y, sample_rate, kernels)
    file = f"saved_models\\{model_name}.pickle"
    joblib.dump(clf, file)
    print("Model saved.")


def load_model(file: str):
    """Load the model from a pickle file.

    Args:
        file (str): Path to the pickle file.
    Returns:
        clf: Trained model.
    """
    model = joblib.load(file)
    clf = model.best_estimator_
    return clf
