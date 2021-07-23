import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import joblib

from prepare_data import prepare_pig_scaled
from prepare_data import prepare_pig_binary
from evaluate_model import evaluate_classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier

from sklearn.model_selection import GroupKFold, cross_val_predict

from sklearn.metrics import auc, roc_curve, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve

def print_scores(clf_name, labels, predictions):
  print(clf_name)
  print("F1:", f1_score(labels, predictions))
  print("Accuracy:", accuracy_score(labels, predictions))
  print("Confusion matrix:", confusion_matrix(labels, predictions))
  print("Precision:", precision_score(labels, predictions))
  print("Recall:", recall_score(labels, predictions))  

def main():
  ''' This is where I test things ''' 
  # get raw data 
  raw_train_data, _, _, raw_test_data, _, _ = prepare_pig_binary()

  # get prepared data
  train_data, train_labels, train_groups, test_data, test_labels, test_groups = prepare_pig_scaled()
  ptrain_data, ptrain_labels, _, _, _, _= prepare_pig_scaled()
  #save_train_data = train_data
  #save_test_data = test_data
  
  ## trying without ARI - no significant difference
  #train_data = save_train_data[:,0:1]
  #test_data = save_test_data[:,0:1]

  # split data for grouped k-fold cross validation: 2 splits gives the best AUC and accuracy (tested)
  group_splits = GroupKFold(n_splits=2)

  # define scoring metrics
  #scoring = 'accuracy'
  #scoring = 'precision'
  scoring = 'f1'
    
  # random forest classifier 
  modelname = "%s/trained_rnd_clf_balanced_f1.sav"%os.getcwd()
  if not os.path.isfile(modelname):
    print("\nRANDOM FOREST CLASSIFIER\n")
    rnd_clf = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = [{'n_estimators': [10, 50, 100, 500, 1000]}]

    trained_rnd_clf = evaluate_classifier(rnd_clf, param_grid, scoring, group_splits, train_data, train_labels, train_groups, "%s/rf_cross_training_balanced_f1.txt"%os.getcwd())

    joblib.dump(trained_rnd_clf, modelname)
  else:
    trained_rnd_clf = joblib.load(modelname)
    
  # SVM classifier
  modelname = "%s/trained_svm_clf_balanced_f1.sav"%os.getcwd()
  if not os.path.isfile(modelname):
    print("\nSVM CLASSIFIER\n")
    svm_clf = svm.SVC(random_state=42, kernel='poly', class_weight='balanced')
    param_grid = [{'C':[0.1, 1, 10]}]

    trained_svm_clf = evaluate_classifier(svm_clf, param_grid, scoring, group_splits, train_data, train_labels, train_groups, "%s/svm_cross_training_balanced_f1.txt"%os.getcwd())

    joblib.dump(trained_svm_clf, modelname)
  else:
    trained_svm_clf = joblib.load(modelname)
    
  # XGBoost classifier
  modelname = "%s/trained_xgb_clf_f1.sav"%os.getcwd()
  if not os.path.isfile(modelname):
    print("\nXGBoost CLASSIFIER\n")
    #scale_pos_weight = np.count_nonzero(train_labels)/np.count_nonzero(~train_labels)
    #print(scale_pos_weight)
    xgb_clf = XGBClassifier(objective='binary:hinge', random_state=42, use_label_encoder=False, subsample=0.5)#, scale_pos_weight=scale_pos_weight)
    param_grid = [{'max_depth':[2, 3, 4, 5, 6], 'learning_rate':[0.03, 0.1, 0.3, 1]}]

    trained_xgb_clf = evaluate_classifier(xgb_clf, param_grid, scoring, group_splits, train_data, train_labels, train_groups, "%s/xgb_cross_training_f1.txt"%os.getcwd())

    joblib.dump(trained_xgb_clf, modelname)
  else:
    trained_xgb_clf = joblib.load(modelname)

  print("\nTRAINING SET\n")
  # build "classifer" using amplitude threshold of 1.5mV
  amp = raw_train_data['AMP'] # train_data[:,0]
  amp_train_pred = amp.copy()
  amp_train_pred[amp<1.5] = 1 # scar
  amp_train_pred[amp>=1.5] = 0 # healthy
  print_scores("Amplitude", train_labels, amp_train_pred)

  # Random Forest
  rdn_train_pred = trained_rnd_clf.predict(train_data)
  print_scores("Random Forest", train_labels, rdn_train_pred)
  
  # SVM 
  svm_train_pred = trained_svm_clf.predict(train_data)
  print_scores("SVM", train_labels, svm_train_pred)

  # XGBoost
  xgb_train_pred = trained_xgb_clf.predict(train_data)
  print_scores("XGBoost", train_labels, xgb_train_pred)

  # evaluate models on test set  
  print("\nTEST SET\n")
  # amplitude
  amp_test = raw_test_data['AMP']
  amp_test_pred = amp_test.copy()
  amp_test_pred[amp_test<1.5] = 1 # scar
  amp_test_pred[amp_test>=1.5] = 0 # healthy
  print_scores("Amplitude", test_labels, amp_test_pred)
 
  # Random Forest
  rdn_test_pred = trained_rnd_clf.predict(test_data)
  print_scores("Random Forest", test_labels, rdn_test_pred)
  
  # SVM 
  svm_test_pred = trained_svm_clf.predict(test_data)
  print_scores("SVM", test_labels, svm_test_pred)

  # XGBoost
  xgb_test_pred = trained_xgb_clf.predict(test_data)
  print_scores("XGBoost", test_labels, xgb_test_pred)

  ## Precision and recall curves
  #rdn_train_score = cross_val_predict(trained_rnd_clf, train_data, train_labels, cv=group_splits, 
                                      #groups=train_groups, method="predict_proba")
  #rdn_train_score = rdn_train_score[:,1]
  #rdn_train_avrg_pre = average_precision_score(train_labels, rdn_train_score)
  #rdn_train_pre, rdn_train_rec, rdn_train_thr = precision_recall_curve(train_labels, rdn_train_score)
  
  #svm_train_score = cross_val_predict(trained_svm_clf, train_data, train_labels, cv=group_splits, 
                                      #groups=train_groups, method="decision_function")
  #svm_train_avrg_pre = average_precision_score(train_labels, svm_train_score)
  #svm_train_pre, svm_train_rec, svm_train_thr = precision_recall_curve(train_labels, svm_train_score)
  
  
  #fig, axs = plt.subplots(1,1, figsize=(10,10))
  #lw = 2
  #ls='solid'

  #plt.plot(rdn_train_pre[:-1], rdn_train_rec[:-1],color='darkgreen', lw=lw, linestyle='dashed', label='RF train')
  #plt.plot(svm_train_pre[:-1], svm_train_rec[:-1],color='darkred', lw=lw, linestyle='dashed', label='SVM train')

  ##plt.plot(rdn_train_thr, rdn_train_pre[:-1],color='darkgreen', lw=lw, linestyle='solid', label='RF train precision')
  ##plt.plot(rdn_train_thr, rdn_train_rec[:-1],color='darkgreen', lw=lw, linestyle='dashed', label='RF train recall')

  ##plt.plot(svm_train_thr, svm_train_pre[:-1],color='darkred', lw=lw, linestyle='solid', label='SVM train precision')
  ##plt.plot(svm_train_thr, svm_train_rec[:-1],color='darkred', lw=lw, linestyle='dashed', label='SVM train recall')

  ##plt.plot(rdn_train_pre, rdn_train_rec, color='darkgreen', lw=lw, linestyle=ls, label='RF train (Average Precision = %0.2f)' % rdn_train_avrg_pre)
  ##plt.plot(svm_train_pre, svm_train_rec, color='darkred', lw=lw, linestyle=ls, label='SVM train (Average Precision = %0.2f)' % svm_train_avrg_pre)

  ##ls='dashed'
  ##plt.plot(rdn_test_pre, rdn_test_rec, color='darkgreen', lw=lw, linestyle=ls, label='RF test (Average Precision = %0.2f)' % rdn_test_avrg_pre)
  ##plt.plot(svm_test_pre, svm_test_rec, color='darkorange', lw=lw, linestyle=ls, label='SVM test (Average Precision = %0.2f)' % svm_test_avrg_pre)

  ##plt.plot([0, 1], [0, 1], color='darkgrey', lw=lw, linestyle='--')

  #plt.xlim([0.0, 1.0])
  #plt.ylim([0.0, 1.05])
  #plt.xlabel('Precision')
  #plt.ylabel('Recall')
  #plt.title('Precision-Recall curve')
  #plt.legend(loc="lower right")
  #axs.grid(alpha=0.5)
  #plt.legend(loc='upper right')
  #plt.savefig("%s/pr_curves_train"%os.getcwd()) # save to current directory
  #plt.close() 
  
  # ROC curves - training set
  #lr_train_pred_prob = trained_lr_clf.predict_proba(train_data)
  #lr_train_pred_prob = lr_train_pred_prob[:,1]

  #amp_train_fpr, amp_train_tpr, _ = roc_curve(train_labels, amp_train_pred) # binary
  #lr_train_fpr, lr_train_tpr, _   = roc_curve(train_labels, lr_train_pred)  
  #rdn_train_fpr, rdn_train_tpr, _ = roc_curve(train_labels, rdn_train_pred)  
  #svm_train_fpr, svm_train_tpr, _ = roc_curve(train_labels, svm_train_pred)  

  #amp_test_fpr, amp_test_tpr, _ = roc_curve(test_labels, amp_test_pred) # binary
  #lr_test_fpr, lr_test_tpr, _   = roc_curve(test_labels, lr_test_pred)  
  #rdn_test_fpr, rdn_test_tpr, _ = roc_curve(test_labels, rdn_test_pred)  
  #svm_test_fpr, svm_test_tpr, _ = roc_curve(test_labels, svm_test_pred)  

  #fig, axs = plt.subplots(1,1)
  #lw = 2
  #ls='solid'
  #plt.plot(amp_train_fpr, amp_train_tpr, color='darkblue', lw=lw, linestyle=ls, label='Amp train (area = %0.2f)' % auc(amp_train_fpr, amp_train_tpr))
  #plt.plot(rdn_train_fpr, rdn_train_tpr, color='darkgreen', lw=lw, linestyle=ls, label='RF train (area = %0.2f)' % auc(rdn_train_fpr, rdn_train_tpr))
  #plt.plot(lr_train_fpr, lr_train_tpr, color='darkred', lw=lw, linestyle=ls, label='LR train (area = %0.2f)' % auc(lr_train_fpr, lr_train_tpr))
  #plt.plot(svm_train_fpr, svm_train_tpr, color='darkorange', lw=lw, linestyle=ls, label='SVM train (area = %0.2f)' % auc(svm_train_fpr, svm_train_tpr))

  #ls='dashed'
  #plt.plot(amp_test_fpr, amp_test_tpr, color='darkblue', lw=lw, linestyle=ls, label='Amp test (area = %0.2f)' % auc(amp_test_fpr, amp_test_tpr))
  #plt.plot(rdn_test_fpr, rdn_test_tpr, color='darkgreen', lw=lw, linestyle=ls, label='RF test (area = %0.2f)' % auc(rdn_test_fpr, rdn_test_tpr))
  #plt.plot(lr_test_fpr, lr_test_tpr, color='darkred', lw=lw, linestyle=ls, label='LR test (area = %0.2f)' % auc(lr_test_fpr, lr_test_tpr))
  #plt.plot(svm_test_fpr, svm_test_tpr, color='darkorange', lw=lw, linestyle=ls, label='SVM test (area = %0.2f)' % auc(svm_test_fpr, svm_test_tpr))

  #plt.plot([0, 1], [0, 1], color='darkgrey', lw=lw, linestyle='--')
  #plt.xlim([0.0, 1.0])
  #plt.ylim([0.0, 1.05])
  #plt.xlabel('False Positive Rate')
  #plt.ylabel('True Positive Rate')
  #plt.title('ROC curve')
  #plt.legend(loc="lower right")
  #axs.grid(alpha=0.5)
  #plt.savefig("%s/roc_curves_train_test"%os.getcwd()) # save to current directory
  #plt.close() 
  


  
if __name__== "__main__":
  main()
