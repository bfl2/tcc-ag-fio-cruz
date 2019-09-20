
import generate_base_dataset as bdt
from sklearn.neural_network import MLPClassifier

def main():
    dt = bdt.getNumberFormatedDataset()
    dataset = bdt.getBalancedDataset(dt)


    train, val, test =  bdt.trainValTestSplit(dataset)

    X = train.iloc[ : , 1:41 ]
    y = train.iloc[ : , 42 ]

    X_test = test.iloc[ : , 1:41 ]
    y_test = test.iloc[ : , 42 ]

    print(X)
    print(y)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(90, 40), random_state=1)
    clf.fit(X, y)

    print(clf.predict(X_test))
    print(y_test)
    print('Accuracy testing : {:.3f}'.format(clf.score(X_test, y_test)))

    return



if __name__ == "__main__":
    main()
