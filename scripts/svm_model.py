import generate_base_dataset as bdt
from sklearn import svm

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

    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', C=1.0, cache_size=200, kernel='rbf')
    clf.fit(X, y)

    #print(clf.predict(X_test))
    #print(y_test)
    predict_result_pairs = list(zip(clf.predict(X_test), y_test))
    print(predict_result_pairs)
    print('Accuracy testing : {:.3f}'.format(clf.score(X_test, y_test)))

    return



if __name__ == "__main__":
    main()
