from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

import sys
import os
path_to_parent = os.path.dirname(os.getcwd())
print(path_to_parent)
sys.path.append(path_to_parent)
from scripts import utils as bdt

def mlp_model(_remove_extra_classes = False, metric="accuracy"):

    print("Running MLP model| Limited dataset:{}".format(_remove_extra_classes))
    dt = bdt.getOneHotEncodedDataset(remove_extra_classes=_remove_extra_classes)

    X  = dt.iloc[ : , 1:31 ]
    y = dt.iloc[ : , 31 ]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    train = bdt.getBalancedDataset(X_train, y_train)
    X_train  = train.iloc[ : , :-1 ]
    y_train = train.iloc[ : , -1]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90, 40), random_state=1)
    clf.fit(X_train, y_train)

    scores = cross_val_score(clf, X_test, y_test, cv=5, scoring=metric)

    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    print('{} testing : {:.3f} (+-{:.3f})'.format(metric, scores.mean(), scores.std()))
    print("confusion matrix:\n", conf_matrix)
    print("True Negative:{0}, False Positive:{1} \nFalse Negative:{2}, True Positive:{3}".format(tn, fp, fn, tp))

    print("###Finished running MLP model")

    return

def gradient_boosting_model(_remove_extra_classes = False, metric="accuracy"):

    print("###Running Gradient Boosting model| Limited dataset:{}".format(_remove_extra_classes))
    dt = bdt.getOneHotEncodedDataset(remove_extra_classes=_remove_extra_classes)

    X  = dt.iloc[ : , 1:31 ]
    y = dt.iloc[ : , 31 ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    train = bdt.getBalancedDataset(X_train, y_train)
    X_train  = train.iloc[ : , :-1 ]
    y_train = train.iloc[ : , -1]

    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=42)
    gb_clf.fit(X_train, y_train)

    scores = cross_val_score(gb_clf, X_test, y_test, cv=5, scoring=metric)

    y_pred = gb_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    print('{} testing : {:.3f} (+-{:.3f})'.format(metric, scores.mean(), scores.std()))
    print("confusion matrix:\n", conf_matrix)
    print("True Negative:{0}, False Positive:{1} \nFalse Negative:{2}, True Positive:{3}".format(tn, fp, fn, tp))

    print("###Finished running Gradient Boosting model")

    return

def svm_model(_remove_extra_classes = False, metric="accuracy"):

    print("###Running SVM model| Limited dataset:{}".format(_remove_extra_classes))
    dt = bdt.getOneHotEncodedDataset(remove_extra_classes=_remove_extra_classes)

    X  = dt.iloc[ : , 1:31 ]
    y = dt.iloc[ : , 31 ]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    train = bdt.getBalancedDataset(X_train, y_train)
    X_train  = train.iloc[ : , :-1 ]
    y_train = train.iloc[ : , -1]

    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', C=1.0, cache_size=200, kernel='rbf')
    clf.fit(X_train, y_train)

    scores = cross_val_score(clf, X_test, y_test, cv=5, scoring=metric)

    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print('{} testing : {:.3f} (+-{:.3f})'.format(metric, scores.mean(), scores.std()))
    print("confusion matrix:\n", conf_matrix)
    print("True Negative:{0}, False Positive:{1} \nFalse Negative:{2}, True Positive:{3}".format(tn, fp, fn, tp))
    print("###Finished running SVM model")

    return

def main():
    met = "balanced_accuracy"
    mlp_model(False, metric=met)
    print("###--------\n")
    mlp_model(True, metric=met)

    print("###--------\n")
    gradient_boosting_model(False, metric=met)
    print("###--------\n")
    gradient_boosting_model(True, metric=met)

    print("###--------\n")
    svm_model(False, metric=met)
    print("###--------\n")
    svm_model(True, metric=met)



    return



if __name__ == "__main__":
    main()
