from scripts import generate_base_dataset as bdt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def main():
    dt = bdt.getOneHotEncodedDataset()

    X  = dt.iloc[ : , 1:31 ]
    y = dt.iloc[ : , 31 ]

    subsets = [
    ["IL-10 -592=CA","TNF-308=GG"],
    ["MBL -221=YX", "IL-10 -819=CT", "TNF-308=GG"],
    ["TNF-308=GG"],
    ["PTX3 rs2305619=GG", "MPO C-463T=GG"],
    ["PTX3 rs2305619=AA", "IL-10 -592=CA"],
    ["PTX3 rs2305619=AA"],
    ["IL-10 -819=CT", "MPO C-463T=GG"],
    ["PTX3 rs1840680=AA", "IL-28b rs12979860=CT"],
    ["MPO C-463T=GG"],
    ["PTX3 rs1840680=AA", "MBL -221=XX"]
    ]

    for sset in subsets:
        X_sset = X[sset]
        print(X_sset.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_sset, y, test_size=0.33, random_state=100, stratify=y)

        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3, random_state=100)
        gb_clf.fit(X_train, y_train)
        scores = cross_val_score(gb_clf, X_train, y_train, cv=5, scoring='accuracy')
        print("Score: {0:.3f} +-({1:.3f}))".format(scores.mean(), scores.std()))

    return



if __name__ == "__main__":
    main()
