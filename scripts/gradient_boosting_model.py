import generate_base_dataset as bdt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from operator import itemgetter

def main():
    dt = bdt.getNumberFormatedDataset()
    dataset = bdt.getBalancedDataset(dt)


    train, val, test =  bdt.trainValTestSplit(dataset)

    X = train.iloc[ : , 1:41 ]
    y = train.iloc[ : , 42 ]

    X_val = val.iloc[ : , 1:41 ]
    y_val = val.iloc[ : , 42 ]

    X_test = test.iloc[ : , 1:41 ]
    y_test = test.iloc[ : , 42 ]

    print(X)
    print(y)
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    models_score = []

    for learning_rate in lr_list:
        gb_clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=learning_rate, max_features=40, max_depth=3, random_state=100)
        gb_clf.fit(X, y)

        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X, y)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))
        models_score.append([gb_clf, gb_clf.score(X_val, y_val)])

    max_score = max(models_score, key=itemgetter(1))
    index = models_score.index(max_score)
    print("max score(validation): ", max_score[1], " with learning rate: ", lr_list[index])
    gb_clf = max_score[0]

    predict_result_pairs = list(zip(gb_clf.predict(X_test), y_test))
    print(predict_result_pairs)
    print('Accuracy testing : {:.3f}'.format(gb_clf.score(X_test, y_test)))

    return



if __name__ == "__main__":
    main()
