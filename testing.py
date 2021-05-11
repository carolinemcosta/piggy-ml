import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from get_data import fetch_pig_data
from get_data import load_pig_data
from training_set import split_train_test_pig
from prepare_data import prepare_pig_binary

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error

def main():
  
  train_data, train_labels, test_data, test_labels = prepare_pig_binary()
  
  #train_data.hist(bins=50)
  #plt.show()
  
  # testing different scalers on traning data
  #minmax_data = MinMaxScaler().fit_transform(train_data)
  qnorm_data = QuantileTransformer(output_distribution='normal').fit_transform(train_data) # force data to have normal distribution
  qnorm_minmax_data = MinMaxScaler().fit_transform(qnorm_data) # set range to 0..1
  
  # plot transformed data
  #pd.DataFrame(minmax_data).hist(bins=50)
  #plt.show()  
  #pd.DataFrame(qnorm_data).hist(bins=50)
  #plt.show()  
  #pd.DataFrame(qnorm_minmax_data).hist(bins=50)
  #plt.show()  
  
  # select scaled data and prepare labels
  prepared_train_data = qnorm_minmax_data
  prepared_train_labels = np.ravel(train_labels.to_numpy())

  # train random forest classifier 
  rnd_clf = RandomForestClassifier()
  rnd_clf.fit(prepared_train_data, prepared_train_labels)
  
  # try on some traning set data
  some_data = prepared_train_data[:10]
  some_labels = prepared_train_labels[:10]
  
  print("Predictions:", rnd_clf.predict(some_data))
  print("Labels:", some_labels)
  
  # evaluate
  pig_predictions = rnd_clf.predict(prepared_train_data)
  rnd_mse = mean_squared_error(prepared_train_labels, pig_predictions)
  rnd_rmse = np.sqrt(rnd_mse)
  print("Root mean square error:", rnd_rmse) # probably overfitting!!!
  
  
  
if __name__== "__main__":
  main()
