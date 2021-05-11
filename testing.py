import os
import pandas as pd
import get_data as gd
import matplotlib.pyplot as plt
from training_set import split_train_test_pig

def main():
  # fetch data from Dropbox folder
  gd.fetch_pig_data()
  
  # load data to dataframe
  pig_data = gd.load_pig_data()
  #print(pig_data.head())
  
  train_set, test_set = split_train_test_pig(pig_data, 0.2)
  #print(train_set.head())
  #print(test_set.head())
  
  #print(len(train_set), len(test_set))

  # plot the data
  data = train_set[["AMP","DVDT","ARI"]]
  data.hist(bins=50,figsize=(15,15))
  plt.show()
  
if __name__== "__main__":
  main()
