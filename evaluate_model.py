import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

def evaluate_classifier(clf, param_grid, scoring, refit, splits, data, labels, groups):
  ''' Tune hyperparameters and evaluate model with cross validation
      
      Parameters: 
        clf (Object): instance of the classifier
        param_grid: dictionary defining search space
        scoring: list of metrics for evaluation
        refit: metric to use when re-fitting the model. Can be a string or 'False' if not refitting.
        splits: object with splits for cross validation
        data (Numpy ND array): the data used for training
        labels (Numpy 1D array): the labels used for training
        groups: list with data groups for groupKfold cross-validation
      Returns:
        Prints scores on the screen
        trained_clf: trained and evaluated classifier
  '''    
  # tune hyperparameters
  tuned_clf = GridSearchCV(clf, param_grid, scoring=scoring, refit=refit, cv=splits, return_train_score=True)  
  tuned_clf.fit(data, labels, groups=groups)
  
  print(tuned_clf.best_params_)
  # evaluate best estimator
  scores = cross_validate(tuned_clf.best_estimator_, data, labels, scoring=scoring, cv=splits, groups=groups, return_train_score=True)

  # print scores
  for scr in scoring:
    print("%s:"%scr, scores["test_%s"%scr])
    print("mean %s:"%scr, np.mean(scores["test_%s"%scr]))
    
  
  return tuned_clf.best_estimator_
  

