import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


def evaluate_classifier(clf, param_grid, scoring, splits, data, labels, groups):
    """ Tune hyperparameters and evaluate model with cross validation

      Parameters:
        clf (Object): instance of the classifier
        param_grid (dict): search space
        scoring (list): metrics for evaluation
        splits (object): splits for cross validation
        data (Numpy ND array): the data used for training
        labels (Numpy 1D array): the labels used for training
        groups (list): data groups for groupKfold cross-validation
      Returns:
        Prints scores on the screen
        trained_clf (object): trained and evaluated classifier
    """

    # tune hyperparameters
    tuned_clf = GridSearchCV(clf, param_grid, scoring=scoring, cv=splits, return_train_score=True)
    tuned_clf.fit(data, labels, groups=groups)

    print(tuned_clf.best_params_)
    # evaluate best estimator
    scores = cross_validate(tuned_clf.best_estimator_, data, labels, scoring=scoring, cv=splits, groups=groups,
                            return_train_score=True)

    print(scores["train_score"], scores["test_score"])
    print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))

    return tuned_clf.best_estimator_
