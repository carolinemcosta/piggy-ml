import os
import pandas as pd
import matplotlib.pyplot as plt

from get_data import fetch_pig_data
from get_data import load_pig_data
from training_set import split_train_test_pig

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

def prepare_pig():
  # fetch data from Dropbox folder
  fetch_pig_data()
  
  # load data to dataframe
  pig_data = load_pig_data()
  
  # split into training and testing sets
  train_set, test_set = split_train_test_pig(pig_data, 0.2)

  # split into data and labels
  train_data = train_set[["PIG","AMP","DVDT","ARI"]]
  train_labels = train_set[["TAG"]]

  test_data = test_set[["PIG","AMP","DVDT","ARI"]]
  test_labels = test_set[["TAG"]]
  
  return train_data, train_labels, test_data, test_labels
  
def prepare_pig_binary():  
  # get data and labels separately
  train_data, train_labels, test_data, test_labels = prepare_pig()
  #print(train_data.head())
  
  # make labels binary: scar = 1 and healthy = 0
  train_labels[train_labels==2] = 0
  train_labels[train_labels>2] = 1
  test_labels[test_labels==2] = 0
  test_labels[test_labels>2] = 1
  
  return train_data, train_labels, test_data, test_labels

