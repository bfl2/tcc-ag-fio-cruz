import utils as bdt
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def main():
    print("###Running SVM model")
    dt = bdt.getOneHotEncodedDataset(remove_extra_classes=True)

    X  = dt.iloc[ : , 1:31 ]
    y = dt.iloc[ : , 31 ]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100, stratify=y)


    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', C=1.0, cache_size=200, kernel='rbf')
    clf.fit(X, y)

    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print('Accuracy testing : {:.3f} (+-{:.3f})'.format(scores.mean(), scores.std()))
    print("###Finished running SVM model")

    return



if __name__ == "__main__":
    main()
