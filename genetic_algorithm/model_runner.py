from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import OneClassSVM

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
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

import ga_utils
import plot_utils

## Disabling DataConversion Warning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

## Global Variables
metric = "roc_auc"
empty_feature_indiv = {"genes":[0,0,0,0,0,0,0,0,0,0], "score":0}
calculated_individuals = []
calculated_individuals.append(empty_feature_indiv)

def getIndivScoredFromMatch(individual):
    genes = individual.genes
    for indiv in calculated_individuals:
        if indiv['genes'] == genes:
            return indiv

    return None

def addNewIndivScore(individual, score):
    global calculated_individuals
    new_result = {"genes":individual.genes, "score":score}
    calculated_individuals.append(new_result)
    calculated_individuals = sorted(calculated_individuals, key=lambda k: k['score'])
    return

def printAdditionalMetrics(scores, indiv):
    accuracy = []
    balanced_accuracy = []
    auc_roc = []
    for score in scores:
        accuracy.append(score[0])
        balanced_accuracy.append(score[1])
        auc_roc.append(score[2])

    accuracy = np.array(accuracy)
    balanced_accuracy = np.array(balanced_accuracy)
    auc_roc = np.array(auc_roc)

    print("Model:{} ".format(indiv.parameters["model"]))
    print("Features:{} | {}".format(ga_utils.getFeaturesTextFromGenes(indiv.genes), indiv.genes))
    print("Accuracy:{:.3f} (+-{:.3f})".format(accuracy.mean(), accuracy.std()))
    print("Balanced Accuracy:{:.3f} (+-{:.3f})".format(balanced_accuracy.mean(), balanced_accuracy.std()))
    print("AUC ROC:{:.3f} (+-{:.3f})".format(auc_roc.mean(), auc_roc.std()))

    return round(auc_roc.mean(),4)

def getIndivDataset(individual):

    indiv_dataset = ga_utils.getFilteredDataset(individual.genes, classes_config=individual.parameters["classes_config"])

    return indiv_dataset

def getScore(y_test, y_pred, scoring=metric, additional_metrics=False):
    score = []
    if(additional_metrics):
        score.append(accuracy_score(y_test, y_pred, normalize=True))
        score.append(balanced_accuracy_score(y_test, y_pred))
        score.append(roc_auc_score(y_test, y_pred))

    else:
        if (scoring == "accuracy"):
            score = accuracy_score(y_test, y_pred, normalize=True)
        elif (scoring == "balanced_accuracy"):
            score = balanced_accuracy_score(y_test, y_pred)
        elif (scoring == "roc_auc"):
            score = roc_auc_score(y_test, y_pred)

    return score


def runConfiguration(individual):
    foldType = individual.parameters["fold_type"]
    classifier = getClassifier(individual.parameters['model'])
    indiv_match = getIndivScoredFromMatch(individual)

    if(indiv_match == None or (individual.parameters["additional_metrics"] or individual.parameters["verbose"])):
        if(foldType == "leave_one_out"):
            score = runModelLeaveOneOut(individual, classifier, individual.parameters["verbose"], additional_metrics=individual.parameters["additional_metrics"])
        else: ### Default | KFold = 5
            score = runModel(individual, classifier, individual.parameters["verbose"], additional_metrics=individual.parameters["additional_metrics"])

        if(indiv_match == None):
            addNewIndivScore(individual, score)

    else:
        score = indiv_match["score"]

    return score

def getClassifier(model):
    if(model == "mlp"):
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90, 40), random_state=42)
    elif(model == "svm"):
        classifier = svm.SVC(gamma='scale', decision_function_shape='ovo', C=1.0, cache_size=200, kernel='rbf')
    elif(model == "gradient_boosting"):
        classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=42)
    elif(model == "random_forest"):
        classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
    elif(model == "one_class_svm"):
        classifier = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma="auto", random_state=42)
    else:
        classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

    return classifier

def runModelLeaveOneOut(individual, classifier, verbose=False, additional_metrics=False, debug=False):
    dataset = getIndivDataset(individual)
    X = dataset.iloc[ : , 1:-1] #removing ID column
    y = dataset.iloc[ : , -1:]
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    scores = []
    conf_matrixes = []
    y_pred = []
    y_tt =[]
    for train_index, test_index in loo.split(X):
        if(verbose and debug):
            print("TRAIN-SIZE:", len(train_index), "TEST-SIZE:", len(test_index))

        if(individual.parameters["balance_method"] == "one_class"):
            X_train, y_train = ga_utils.getOneClassDataset(X.iloc[train_index], y.iloc[train_index])
        elif(individual.parameters["balance_method"] == "integer_balanced"):
            train = ga_utils.getBalancedDataset(X.iloc[train_index], y.iloc[train_index])
            X_train  = train.iloc[ : , :-1 ]
            y_train = train.iloc[ : , -1]
        else: ## float_balanced // default
            X_train, y_train = ga_utils.getBalancedDatasetROS(X.iloc[train_index], y.iloc[train_index])

        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)
        y_p = classifier.predict(X_test)
        if(individual.parameters["model"] == "one_class_svm"):
            y_p = ga_utils.convertOneClassResullts(y_p)
        y_pred.extend(y_p)
        y_tt.extend(y_test)

    score = getScore(y, y_pred, additional_metrics=additional_metrics)
    scores.append(score)
    conf_matrix = confusion_matrix(y, y_pred)
    conf_matrixes.append(conf_matrix)

    if(additional_metrics):
        mean_score = printAdditionalMetrics(scores, individual)
        plot_utils.printConfusionMatrix(conf_matrixes, plot_matrix=True)
        return mean_score
    else:
        scores = np.array(scores)
        conf_matrixes = np.array(conf_matrixes)
        if(verbose):
            print("\n### Model:{} | Metric:{}\nScore:{:.3f} (+-{:.3f})".format(individual.parameters['model'], metric, scores.mean(), scores.std()))
        return round(scores.mean(), 4)


def runModel(individual, classifier, verbose=False, additional_metrics=False, debug=False):
    dataset = getIndivDataset(individual)
    X = dataset.iloc[ : , 1:-1] #removing ID column
    y = dataset.iloc[ : , -1:]
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    kf.get_n_splits(X)
    scores = []
    conf_matrixes = []
    for train_index, test_index in kf.split(X):
        if(verbose and debug):
            print("     TRAIN-SIZE:", len(train_index), "TEST-SIZE:", len(test_index))

        if(individual.parameters["balance_method"] == "one_class"):
            X_train, y_train = ga_utils.getOneClassDataset(X.iloc[train_index], y.iloc[train_index])
        elif(individual.parameters["balance_method"] == "integer_balanced"):
            train = ga_utils.getBalancedDataset(X.iloc[train_index], y.iloc[train_index])
            X_train  = train.iloc[ : , :-1 ]
            y_train = train.iloc[ : , -1]
        else: ## float_balanced // default
            X_train, y_train = ga_utils.getBalancedDatasetROS(X.iloc[train_index], y.iloc[train_index])

        if(verbose and debug):
            print("     Class balance:0|1 - {}|{}".format(  (y_train == 0).sum(), (y_train == 1).sum()))
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        if(individual.parameters["model"] == "one_class_svm"):
            y_pred = ga_utils.convertOneClassResullts(y_pred)

        score = getScore(y_test, y_pred, additional_metrics=additional_metrics)
        scores.append(score)
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrixes.append(conf_matrix)

    if(additional_metrics):
        mean_score = printAdditionalMetrics(scores, individual)
        plot_utils.printConfusionMatrix(conf_matrixes, plot_matrix=True)
        return mean_score
    else:
        scores = np.array(scores)
        conf_matrixes = np.array(conf_matrixes)
        if(verbose):
            print("\n   Model:{} | Metric:{}\n  Score:{:.3f} (+-{:.3f})".format(individual.parameters['model'], metric, scores.mean(), scores.std()))
        return round(scores.mean(), 4)
