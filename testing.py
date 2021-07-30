import os
import joblib

from prepare_data import prepare_pig_scaled
from prepare_data import prepare_pig_binary
from evaluate_model import evaluate_classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier

from sklearn.model_selection import GroupKFold

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


def print_scores(clf_name, labels, predictions):
    print(clf_name)
    print("F1:", f1_score(labels, predictions))
    print("Accuracy:", accuracy_score(labels, predictions))
    print("Confusion matrix:", confusion_matrix(labels, predictions))
    print("Precision:", precision_score(labels, predictions))
    print("Recall:", recall_score(labels, predictions))


def main():
    """ This is where I test things """
    # get raw data
    raw_train_data, _, _, raw_test_data, _, _ = prepare_pig_binary()

    # get prepared data
    train_data, train_labels, train_groups, test_data, test_labels, test_groups = prepare_pig_scaled()

    # split data for grouped k-fold cross validation: 2 splits gives the best AUC and accuracy (tested)
    group_splits = GroupKFold(n_splits=2)

    # define scoring metrics
    scoring = 'f1'

    # random forest classifier
    model_name = "{}/trained_rnd_clf_balanced_f1_reg.sav".format(os.getcwd())
    if not os.path.isfile(model_name):
        print("\nRANDOM FOREST CLASSIFIER\n")
        rnd_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = [{'n_estimators': [10, 20, 30]}]
        trained_rnd_clf = evaluate_classifier(rnd_clf, param_grid, scoring, group_splits,
                                              train_data, train_labels, train_groups)
        joblib.dump(trained_rnd_clf, model_name)
    else:
        trained_rnd_clf = joblib.load(model_name)

    # SVM classifier
    model_name = "{}/trained_svm_clf_balanced_f1_reg.sav".format(os.getcwd())
    if not os.path.isfile(model_name):
        print("\nSVM CLASSIFIER\n")
        svm_clf = svm.SVC(random_state=42, kernel='poly', class_weight='balanced')
        param_grid = [{'C': [0.1, 0.5]}]
        trained_svm_clf = evaluate_classifier(svm_clf, param_grid, scoring, group_splits,
                                              train_data, train_labels, train_groups)
        joblib.dump(trained_svm_clf, model_name)
    else:
        trained_svm_clf = joblib.load(model_name)

    # XGBoost classifier
    model_name = "{}/trained_xgb_clf_f1_reg.sav".format(os.getcwd())
    if not os.path.isfile(model_name):
        print("\nXGBoost CLASSIFIER\n")
        # scale_pos_weight = np.count_nonzero(train_labels)/np.count_nonzero(~train_labels)
        # print(scale_pos_weight)
        xgb_clf = XGBClassifier(objective='binary:hinge', random_state=42, use_label_encoder=False,
                                subsample=0.5)  # , scale_pos_weight=scale_pos_weight)
        param_grid = [{'max_depth': [2, 3, 4, 5, 6], 'learning_rate': [0.03, 0.1, 0.3]}]
        trained_xgb_clf = evaluate_classifier(xgb_clf, param_grid, scoring, group_splits,
                                              train_data, train_labels, train_groups)
        joblib.dump(trained_xgb_clf, model_name)
    else:
        trained_xgb_clf = joblib.load(model_name)

    print("\nTRAINING SET\n")
    # build "classifier" using amplitude threshold of 1.5mV
    amp = raw_train_data['AMP']  # train_data[:,0]
    amp_train_pred = amp.copy()
    amp_train_pred[amp < 1.5] = 1  # scar
    amp_train_pred[amp >= 1.5] = 0  # healthy
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
    amp_test_pred[amp_test < 1.5] = 1  # scar
    amp_test_pred[amp_test >= 1.5] = 0  # healthy
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


if __name__ == "__main__":
    main()
