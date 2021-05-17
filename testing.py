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
from sklearn.metrics import plot_roc_curve

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate

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
  prepared_test_data = np.ravel(test_data.to_numpy())
  prepared_test_labels = np.ravel(test_labels.to_numpy())  
  
  # split data for grouped k-fold cross validation
  gkf = GroupKFold(n_splits=5)
  pignames = train_data.PIG
  #scoring = ['precision', 'recall', 'roc_auc', 'f1']
  #scoring = ['accuracy', 'roc_auc']
  scoring = ['roc_auc']

  #for train, test in gkf.split(prepared_train_data, train_labels, groups=pignames):
    #print("%s %s" % (train, test))
  
  # random forest classifier 
  print("\nRANDOM FOREST CLASSIFIER\n")
  rnd_clf = RandomForestClassifier()
  #rnd_clf.fit(prepared_train_data, prepared_train_labels)
  #evaluate_classifier(rnd_clf, prepared_train_data, prepared_train_labels, feature_names, rnd_clf.feature_importances_)
  scores = cross_validate(rnd_clf, prepared_train_data, prepared_train_labels, scoring=scoring, cv=gkf, groups=pignames, return_train_score=True)
  #print("Precision:", np.mean(scores['test_precision']))
  #print("Recall:", np.mean(scores['test_recall']))
  #print("Accuracy:", np.mean(scores['test_accuracy']))
  print("ROC AUC:", np.mean(scores['test_roc_auc']))
  #print("f1:", np.mean(scores['test_f1']))
  
  # linear SVM classifier
  print("\nLINEAR SVM CLASSIFIER\n")
  lsv_clf = svm.SVC(kernel='linear')
  #lsv_clf.fit(prepared_train_data, prepared_train_labels)  
  #evaluate_classifier(lsv_clf, prepared_train_data, prepared_train_labels, feature_names, [])
  scores = cross_validate(lsv_clf, prepared_train_data, prepared_train_labels, scoring=scoring, cv=gkf, groups=pignames, return_train_score=True)
  #print("Accuracy:", np.mean(scores['test_accuracy']))
  print("ROC AUC:", np.mean(scores['test_roc_auc']))
  
  ## Stochastic CG classifier
  print("\nSTOCHASTIC GD CLASSIFIER\n")
  sgd_clf = SGDClassifier(random_state=42)
  #sgd_clf.fit(prepared_train_data, prepared_train_labels)  
  #evaluate_classifier(sgd_clf, prepared_train_data, prepared_train_labels, feature_names, [])  
  scores = cross_validate(sgd_clf, prepared_train_data, prepared_train_labels, scoring=scoring, cv=gkf, groups=pignames, return_train_score=True)
  #print("Accuracy:", np.mean(scores['test_accuracy']))
  print("ROC AUC:", np.mean(scores['test_roc_auc']))


  # evaluate models on test set
  #plot_roc_curve(rnd_clf, prepared_test_data, prepared_test_labels)  
  #plt.show()
  ## plot data before and after feature scaling
  #plt.rcParams['font.size'] = '16'
  #fig, axs = plt.subplots(2,3)
  #original_data = pd.DataFrame(train_data, columns=feature_names)
  #transformed_data = pd.DataFrame(prepared_train_data, columns=feature_names)
  
  #axs[0,0].hist(original_data["AMP"], bins=50)
  #axs[0,1].hist(original_data["DVDT"], bins=50)
  #axs[0,2].hist(original_data["ARI"], bins=50)  
  #axs[0,0].set_title("Original AMP")
  #axs[0,1].set_title("Original DVDT")
  #axs[0,2].set_title("Original ARI")

  #axs[1,0].hist(transformed_data["AMP"], bins=50)
  #axs[1,1].hist(transformed_data["DVDT"], bins=50)
  #axs[1,2].hist(transformed_data["ARI"], bins=50)  
  #axs[1,0].set_title("Transformed AMP")
  #axs[1,1].set_title("Transformed DVDT")
  #axs[1,2].set_title("Transformed ARI")

  #fig.suptitle("Feature Scaling", fontsize=24)
  #plt.show()

  
if __name__== "__main__":
  main()
