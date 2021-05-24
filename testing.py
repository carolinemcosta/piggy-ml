import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import joblib

from prepare_data import prepare_pig_scaled
from evaluate_model import evaluate_classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.model_selection import GroupKFold


from sklearn.metrics import auc, roc_curve, accuracy_score

def main():
  ''' This is where I test things ''' 

  # get prepared data
  train_data, train_labels, train_groups, test_data, test_labels, test_groups = prepare_pig_scaled()

  # split data for grouped k-fold cross validation: 2 splits gives the best AUC and accuracy (tested)
  group_splits = GroupKFold(n_splits=2)

  # define scoring metrics
  scoring = ['accuracy', 'roc_auc']
  #scoring = ['roc_auc']
  refit = 'roc_auc'

  # random forest classifier 
  modelname = "%s/trained_rnd_clf_balanced_2splits.sav"%os.getcwd()
  if not os.path.isfile(modelname):
    print("\nRANDOM FOREST CLASSIFIER\n")
    rnd_clf = RandomForestClassifier()

    #param_grid = [{'n_estimators': [100,500], 'max_features': [1,2,3], 'warm_start':[True, False], 'criterion': ['gini', 'entropy'], 'class_weight':['balanced', 'balanced_subsample']}]
    param_grid = [{'n_estimators': [100,500, 800, 1000], 'criterion': ['gini', 'entropy'], 'class_weight':['balanced', 'balanced_subsample']}]

    trained_rnd_clf = evaluate_classifier(rnd_clf, param_grid, scoring, refit, group_splits, train_data, train_labels, train_groups)

    joblib.dump(trained_rnd_clf, modelname)
  else:
    trained_rnd_clf = joblib.load(modelname)
    
  print(trained_rnd_clf)

  ## SVM classifier
  #modelname = "%s/trained_svm_clf.sav"%os.getcwd()
  #if not os.path.isfile(modelname):
    #print("\nSVM CLASSIFIER\n")
    #svm_clf = svm.SVC()
    #param_grid = [{'kernel':('linear', 'poly', 'rbf'), 'C':[0.001, 0.1, 1, 10, 100]}]

    #trained_svm_clf = evaluate_classifier(svm_clf, param_grid, scoring, refit, group_splits, train_data, train_labels, train_groups)

    #joblib.dump(trained_svm_clf, modelname)
  #else:
    #trained_svm_clf = joblib.load(modelname)

  # build "classifer" using amplitude threshold of 1.5mV
  amp = train_data[:,0]
  amp_pred = amp.copy()
  amp_pred[amp<1.5] = 1 # scar
  amp_pred[amp>=1.5] = 0 # healthy
  amp_fpr, amp_tpr, _ = roc_curve(train_labels, amp_pred)
  print("Amplitude threshold accuracy:", accuracy_score(train_labels, amp_pred))

  # evaluate models on traning set
  # Random Forest
  rdn_pred = trained_rnd_clf.predict(train_data)
  rdn_fpr, rdn_tpr, _ = roc_curve(train_labels, rdn_pred)
  print("Random Forest accuracy:", accuracy_score(train_labels, rdn_pred))
  ## SVM 
  #svm_pred = trained_svm_clf.predict(train_data)
  #svm_fpr, svm_tpr, _ = roc_curve(train_labels, svm_pred)

  fig, axs = plt.subplots(1,1)
  lw = 2
  plt.plot(amp_fpr, amp_tpr, color='darkblue', lw=lw, label='Amplitude threshold (area = %0.2f)' % auc(amp_fpr, amp_tpr))
  plt.plot(rdn_fpr, rdn_tpr, color='darkgreen', lw=lw, label='Random Forest (area = %0.2f)' % auc(rdn_fpr, rdn_tpr))
  #plt.plot(svm_fpr, svm_tpr, color='darkorange', lw=lw, label='SVM (area = %0.2f)' % auc(svm_fpr, svm_tpr))
  plt.plot([0, 1], [0, 1], color='darkgrey', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC curve')
  plt.legend(loc="lower right")
  axs.grid(alpha=0.5)
  plt.savefig("%s/roc_curves"%os.getcwd()) # save to current directory
  plt.close() 
  


  
if __name__== "__main__":
  main()
