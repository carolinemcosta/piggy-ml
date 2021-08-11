import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from get_data import fetch_pig_data
from get_data import load_pig_data
from training_set import split_train_test_pig

from sklearn.preprocessing import StandardScaler


def prepare_pig():
    """ Loads data, splits into training and test sets, and into data, labels, and group names

        Parameters:

        Returns:
          train_data
          train_labels
          train_groups
          test_data
          test_labels
          test_groups
    """

    # fetch data from Dropbox folder
    fetch_pig_data()

    # load data to dataframe
    pig_data = load_pig_data()

    # split into training and testing sets
    train_set, test_set = split_train_test_pig(pig_data, 0.35)

    # split into data and labels
    train_data = train_set[["AMP", "DVDT", "ARI"]]
    train_labels = train_set[["TAG"]]
    train_groups = train_set["PIG"]

    test_data = test_set[["AMP", "DVDT", "ARI"]]
    test_labels = test_set[["TAG"]]
    test_groups = test_set["PIG"]

    return train_data, train_labels, train_groups, test_data, test_labels, test_groups


def prepare_pig_binary():
    """ Transform labels into binary to build binary classifiers

        Parameters:

        Returns:
          Pandas DataFrames with data, binary labels, and group names for the trainig test sets
    """

    # get data and labels separately
    train_data, train_labels, train_groups, test_data, test_labels, test_groups = prepare_pig()

    # make labels binary: scar = 1 and healthy = 0
    train_labels[train_labels == 2] = 0
    train_labels[train_labels > 2] = 1
    test_labels[test_labels == 2] = 0
    test_labels[test_labels > 2] = 1

    return train_data, train_labels, train_groups, test_data, test_labels, test_groups


def prepare_pig_scaled():
    """ Final step to prepares the data for ML algorithms.
        Transforms the training data to ensure a normal distribution and 0..1 range.
        Transform columns from DataFrames into Numpy arrays

        Parameters:

        Returns:
          Pandas DataFrames with transformed data, labels, and group names
    """

    # get data with binary labels
    train_data, train_labels, train_groups, test_data, test_labels, test_groups = prepare_pig_binary()

    # transform data
    std_scaler = StandardScaler()
    prepared_train_data = std_scaler.fit_transform(train_data)
    prepared_test_data = std_scaler.transform(test_data)

    # transformed data to numpy arrays
    prepared_train_labels = np.ravel(train_labels.to_numpy(dtype=int))
    prepared_test_labels = np.ravel(test_labels.to_numpy(dtype=int))
    prepared_train_groups = train_groups.to_numpy(dtype=int)
    prepared_test_groups = test_groups.to_numpy(dtype=int)

    # plot original and transformed data
    transformed_data = pd.DataFrame(prepared_train_data, columns=train_data.columns)
    plot_prepared_scaled(train_data, transformed_data)

    return prepared_train_data, prepared_train_labels, prepared_train_groups, prepared_test_data, prepared_test_labels, prepared_test_groups


def plot_prepared_scaled(original_data, transformed_data):
    """ Plot data before and after scaling

        Parameters:
          original_data (Pandas DataFrame): original data
          transformed_data (Pandas DataFrame): standardized data

        Returns:
          Shows plot on screen
    """

    features = original_data.columns
    n_features = len(original_data.columns)

    # plot data before and after feature scaling
    plt.rcParams['font.size'] = '16'
    fig, axs = plt.subplots(2, n_features)

    for idx in range(n_features):
        axs[0, idx].hist(original_data[features[idx]], bins=50, color='g')
        axs[0, 0].set_title("Original {}".format(features[idx]))

        axs[1, idx].hist(transformed_data[features[idx]], bins=50, color='g')
        axs[1, 0].set_title("Transformed {}".format(features[idx]))

    fig.suptitle("Feature Scaling", fontsize=24)
    plt.savefig("{}/feature_scaling".format(os.getcwd()))  # save to current directory
    plt.close()
