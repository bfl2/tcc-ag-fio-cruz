from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.model_selection import KFold
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

import sys
import os
path_to_parent = os.path.dirname(os.getcwd())
path_to_parent2 = os.path.dirname(path_to_parent)

sys.path.append(path_to_parent)
sys.path.append(path_to_parent2)

from scripts import utils
import plot_utils

## Global Variables
metric = "roc_auc"

def getIndivDataset(individual):
    indiv_dataset = utils.getFilteredDataset(individual.genes)

    return indiv_dataset

def getScore(y_test, y_pred, scoring=metric):

    if (scoring == "accuracy"):
        score = accuracy_score(y_test, y_pred, normalize=True)
    elif (scoring == "balanced_accuracy"):
        score = balanced_accuracy_score(y_test, y_pred)
    elif (scoring == "roc_auc"):
        score = roc_auc_score(y_test, y_pred)
    return score


def runConfiguration(individual):

    classifier = getClassifier(individual.parameters['model'])
    score = runModel(individual, classifier, individual.parameters["verbose"])

    return score

def getClassifier(model):
    if(model == "svm"):
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90, 40), random_state=1)
    elif(model == "mlp"):
        classifier = svm.SVC(gamma='scale', decision_function_shape='ovo', C=1.0, cache_size=200, kernel='rbf')
    elif(model == "gradient_boosting"):
        classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=42)
    elif(model == "random_forest"):
        classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
    else:
        classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)


    return classifier

def runModel(individual, classifier, verbose=False):
    dataset = getIndivDataset(individual)
    X = dataset.iloc[ : , 1:-1] #removing ID column
    y = dataset.iloc[ : , -1:]
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    kf.get_n_splits(X)
    scores = []
    conf_matrixes = []
    for train_index, test_index in kf.split(X):
        if(verbose):
            print("TRAIN-SIZE:", len(train_index), "TEST-SIZE:", len(test_index))

        train = utils.getBalancedDataset(X.iloc[train_index], y.iloc[train_index])
        X_train  = train.iloc[ : , :-1 ]
        y_train = train.iloc[ : , -1]

        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        score = getScore(y_test, y_pred)
        scores.append(score)
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrixes.append(conf_matrix)

    scores = np.array(scores)
    conf_matrixes = np.array(conf_matrixes)
    if(verbose):
        print("Model:{} | Metric:{}\nScore:{:.3f} (+-{:.3f})".format(individual.parameters['model'], metric, scores.mean(), scores.std()))
        plot_utils.printConfusionMatrix(conf_matrixes)

    return round(scores.mean(), 4)




