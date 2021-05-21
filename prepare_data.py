import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from get_data import fetch_pig_data
from get_data import load_pig_data
from training_set import split_train_test_pig

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

def prepare_pig():
  ''' Loads data, splits into traning and test sets, and into data, labels, and group names
      
      Parameters: 
        
      Returns:
        Pandas DataFrames with data, labels, and group names for the trainig test sets
  '''
  
  # fetch data from Dropbox folder
  fetch_pig_data()
  
  # load data to dataframe
  pig_data = load_pig_data()
  
  # split into training and testing sets
  train_set, test_set = split_train_test_pig(pig_data, 0.2)

  # split into data and labels
  train_data = train_set[["AMP","DVDT","ARI"]]
  train_labels = train_set[["TAG"]]
  train_groups = train_set["PIG"]

  test_data = test_set[["AMP","DVDT","ARI"]]
  test_labels = test_set[["TAG"]]
  test_groups = test_set["PIG"]
  
  return train_data, train_labels, train_groups, test_data, test_labels, test_groups
  
def prepare_pig_binary():  
  ''' Transform labels into binary to build binary classifiers
      
      Parameters: 
        
      Returns:
        Pandas DataFrames with data, binary labels, and group names for the trainig test sets
  '''
  
  # get data and labels separately
  train_data, train_labels, train_groups, test_data, test_labels, test_groups = prepare_pig()
  
  # make labels binary: scar = 1 and healthy = 0
  train_labels[train_labels==2] = 0
  train_labels[train_labels>2] = 1
  test_labels[test_labels==2] = 0
  test_labels[test_labels>2] = 1
  
  return train_data, train_labels, train_groups, test_data, test_labels, test_groups

def prepare_pig_scaled():
  ''' Final step to prepares the data for ML algorithms.
      Tranforms the training data to ensure a normal distribution and 0..1 range. 
      Transform colums from DataFrames into Numpy arrays
      
      Parameters: 
        
      Returns:
        Pandas DataFrames with transformed data, labels, and group names
  '''  
  # get data with binary labels
  train_data, train_labels, train_groups, test_data, test_labels, test_groups = prepare_pig_binary()
  
  # transform training data
  qnorm_data = QuantileTransformer(output_distribution='normal').fit_transform(train_data) # normal distribution
  prepared_train_data = MinMaxScaler().fit_transform(qnorm_data) # set range to 0..1
  
  # transformed data to numpy arrays
  prepared_train_labels = np.ravel(train_labels.to_numpy())
  prepared_train_groups = train_groups

  #prepared_test_data = np.ravel(test_data.to_numpy())
  #prepared_test_labels = np.ravel(test_labels.to_numpy())    
  #prepared_test_groups = np.ravel(test_groups.to_numpy())
  
  # plot original and transformed data
  transformed_data = pd.DataFrame(prepared_train_data, columns=train_data.columns)
  plot_prepared_scaled(train_data, transformed_data)
  
  return prepared_train_data, prepared_train_labels, prepared_train_groups, test_data, test_labels, test_groups

def plot_prepared_scaled(original_data, transformed_data):
  ''' Plot data before and after scaling
      
      Parameters: 
        original_data: Pandas DataFrame with original data
        transformed_data: Pandas DataFrame with scaled data
        
      Returns:
        Shows plot on screen
  '''    
  
  features = original_data.columns
  nfeatures = len(original_data.columns)

  # plot data before and after feature scaling
  plt.rcParams['font.size'] = '16'
  fig, axs = plt.subplots(2,nfeatures)
  
  for idx in range(nfeatures):  
    axs[0,idx].hist(original_data[features[idx]], bins=50, color='g')
    axs[0,0].set_title("Original %s"%features[idx])

    axs[1,idx].hist(transformed_data[features[idx]], bins=50, color='g')
    axs[1,0].set_title("Transformed %s"%features[idx])

  fig.suptitle("Feature Scaling", fontsize=24)
  plt.savefig("%s/feature_scaling"%os.getcwd()) # save to current directory
  plt.close() 
