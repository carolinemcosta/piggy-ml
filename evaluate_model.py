from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_classifier(clf, data, labels, feature_names, feature_importances):
  ''' Evaluates a trained classifier using the training set
      
      Parameters: 
        clf (Object): instance of the classifier
        data (Numpy ND array): the data used for training
        labels (Numpy 1D array): the labels used for training
        feature_names (List): list of strings with feature names 
        feature_importances (List): list with features importances - only available for some classifiers like Random Forest
      Returns:
        Prints results on the screen
  '''    
  # try on some traning set data
  some_data = data[:100]
  some_labels = labels[:100]
  
  some_predictions = clf.predict(some_data)
  print("Predictions:", some_predictions)
  print("Labels:", some_labels)
  
  # evaluate
  rnd_mse = mean_squared_error(some_labels, some_predictions)
  rnd_rmse = np.sqrt(rnd_mse)
  print("Root mean square error:", rnd_rmse)
  
  # feature importances: not all classifiers have this
  if len(feature_importances) > 0:
    print("Feature importances:")
    for name, score in zip(feature_names, feature_importances):
      print(name, score)
