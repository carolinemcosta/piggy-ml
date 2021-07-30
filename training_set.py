import numpy as np

def split_train_test(data, test_ratio):
    """ Splits the data into traning and test sets using a random permutation.

      Parameters:
        data (Pandas DataFrame): Data prior to splitting into training and testing
        test_ratio (float): the ratio to split the data into. Typically 20% with test_ratio = 0.2

      Returns:
        Two Pandas DataFrames, one with the trainig set and one with testing set
  """

    # generate the random permutations
    np.random.seed(42)  # set a seed to always get the same split
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)

    # split the indices
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # return split data
    return data.iloc[train_indices], data.iloc[test_indices]


def split_train_test_pig(data, test_ratio):
    """ Splits the data into training and test sets using a random permutation,
      but ensuring that at least one whole pig dataset is in the testing set.

      Parameters:
        data (Pandas DataFrame): Data prior to splitting into training and testing
        test_ratio (float): the ratio to split the data into. Typically 20% with test_ratio = 0.2

      Returns:
        Two Pandas DataFrames, one with the training set and one with testing set
    """

    # number of unique pigs
    pigs = data.PIG.unique()
    total_pigs = len(pigs)

    # generate the random permutations
    np.random.seed(42)  # set a seed to always get the same split
    shuffled_indices = np.random.permutation(total_pigs)
    test_set_size = round(total_pigs * test_ratio)  # round to nearest integer
    # print(test_set_size)

    # split the indices
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # split the data
    test_pigs = pigs[test_indices]
    test_data = data[data.PIG.isin(test_pigs)]
    train_pigs = pigs[train_indices]
    train_data = data[data.PIG.isin(train_pigs)]

    return train_data, test_data
