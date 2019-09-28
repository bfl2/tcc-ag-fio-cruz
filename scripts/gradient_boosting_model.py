import utils as bdt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def main():

    print("Running Gradient Boosting model")

    dt = bdt.getOneHotEncodedDataset(remove_extra_classes=False)

    X  = dt.iloc[ : , 1:31 ]
    y = dt.iloc[ : , 31 ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    train = bdt.getBalancedDataset(X_train, y_train)
    X_train  = train.iloc[ : , :-1 ]
    y_train = train.iloc[ : , -1]

    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=42)
    gb_clf.fit(X_train, y_train)

    metric = 'accuracy'
    scores = cross_val_score(gb_clf, X_test, y_test, cv=5, scoring=metric)

    y_pred = gb_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    print('{} testing : {:.3f} (+-{:.3f})'.format(metric, scores.mean(), scores.std()))
    print("confusion matrix:\n", conf_matrix)
    print("True Negative:{0}, False Positive:{1} \nFalse Negative:{2}, True Positive:{3}".format(tn, fp, fn, tp))

    print("Running Gradient Boosting  model")

    return



if __name__ == "__main__":
    main()
