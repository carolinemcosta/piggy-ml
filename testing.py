import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from get_data import fetch_pig_data
from get_data import load_pig_data
from training_set import split_train_test_pig
from prepare_data import prepare_pig_binary
from evaluate_model import evaluate_classifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier


from sklearn.metrics import mean_squared_error

def main():
  
  train_data, train_labels, test_data, test_labels = prepare_pig_binary()
  feature_names = ["AMP","DVDT","ARI"]
  
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
  
  # plot data before and after feature scaling
  pd.DataFrame(train_data, columns=feature_names).hist(bins=50)
  plt.show()
  pd.DataFrame(prepared_train_data, columns=feature_names).hist(bins=50)
  plt.show()

  ## random forest classifier 
  #print("\nRANDOM FOREST CLASSIFIER\n")
  #rnd_clf = RandomForestClassifier()
  #rnd_clf.fit(prepared_train_data, prepared_train_labels)
  #evaluate_classifier(rnd_clf, prepared_train_data, prepared_train_labels, feature_names, rnd_clf.feature_importances_)
  
  ## linear SVM classifier
  #print("\nLINEAR SVM CLASSIFIER\n")
  #lsv_clf = svm.SVC(kernel='linear')
  #lsv_clf.fit(prepared_train_data, prepared_train_labels)  
  #evaluate_classifier(lsv_clf, prepared_train_data, prepared_train_labels, feature_names, [])
  
  ## Stochastic CG classifier
  #print("\nSTOCHASTIC GD CLASSIFIER\n")
  #sgd_clf = SGDClassifier(random_state=42)
  #sgd_clf.fit(prepared_train_data, prepared_train_labels)  
  #evaluate_classifier(sgd_clf, prepared_train_data, prepared_train_labels, feature_names, [])  
  
  
if __name__== "__main__":
  main()