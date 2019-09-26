
import utils as bdt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("Running MLP model")
    dt = bdt.getOneHotEncodedDataset(remove_extra_classes=False)
    #dt = bdt.getBalancedDataset(dt)

    X  = dt.iloc[ : , 1:31 ]
    y = dt.iloc[ : , 31 ]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100, stratify=y)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90, 40), random_state=1)
    clf.fit(X_train, y_train)

    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

    print('Accuracy testing : {:.3f} (+-{:.3f})'.format(scores.mean(), scores.std()))

    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    print('Accuracy testing : {:.3f} (+-{:.3f})'.format(scores.mean(), scores.std()))
    print("confusion matrix:", conf_matrix)
    print("True Negative:{0}, False Positive:{1} \nFalse Negative:{2}, True Positive:{3}".format(tn, fp, fn, tp))

    print("###Finished running MLP model")

    return



if __name__ == "__main__":
    main()
