import pandas as pd
import numpy as np
### Genetic algorithm for generating the 10 SNPs genes and its probability of being in a certain class ###

def trainTestSplit(dataset_features):
    trainValFrac = 0.75
    testFrac = 1 -trainValFrac
    len = dataset_features.shape[0]
    divisionIndex = int(np.floor(trainValFrac*len))

    dataset_train_val = dataset_features.iloc[ :divisionIndex, : ]
    dataset_test = dataset_features.iloc[ divisionIndex:, : ]

    print(dataset_train_val)
    print(dataset_test)


    return dataset_train_val, dataset_test

def getFormatedDataset():
    base_dataset_path = "data/PLANILHA_HCV-RECORTE.csv"
    base_dataset = pd.read_csv(base_dataset_path)

    dataset_features = base_dataset.iloc[:, :16]
    dataset_features = dataset_features.drop(["PACIENTE", "ID.1", "SEXO", "At. Inflam. 1"], axis = 1)
    dataset_features = dataset_features.sample(frac=1, random_state = 100)
    #print(dataset_features)
    dataset_features.to_csv("data/base/dataset_base.csv")

    dataset_train_val, dataset_test = trainTestSplit(dataset_features)

    return dataset_features

def main():

    print("Genetic algorithm to generate indiduals with genes and associated chance to be part of a group")
    formattedDataset = getFormatedDataset()
    return



if __name__ == "__main__":
    main()
