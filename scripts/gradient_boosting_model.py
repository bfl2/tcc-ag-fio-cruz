import utils as bdt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def main():

    print("Running Gradient Boosting model")

    dt = bdt.getOneHotEncodedDataset()

    X  = dt.iloc[ : , 1:31 ]
    y = dt.iloc[ : , 31 ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100, stratify=y)
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=100)
    gb_clf.fit(X_train, y_train)

    scores = cross_val_score(gb_clf, X_test, y_test, cv=5, scoring='accuracy')
    print("Score: {0:.3f} +-({1:.3f}))".format(scores.mean(), scores.std()))

    print("Running Gradient Boosting  model")

    return



if __name__ == "__main__":
    main()
