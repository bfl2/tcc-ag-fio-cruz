import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

### Script for formatting dataset into reduced format ###

data_path = "../data/"

SNPs = ["PTX3 rs1840680","PTX3 rs2305619","MBL -221","IL-10 -1082","IL-10 -819","IL-10 -592","TNF-308","SOD2","MPO C-463T","IL-28b rs12979860"]

Genes_comb = {"AG":["AA", "AG", "GG"], "GA":["AA", "GA", "GG"], "CA":["AA", "CA", "CC"], "YX":["XX", "YX", "YY"], "CT":["CC", "CT", "TT"]}

SNPs_Values = [Genes_comb["AG"], Genes_comb["AG"], Genes_comb["YX"], Genes_comb["GA"], Genes_comb["CT"], Genes_comb["CA"], Genes_comb["GA"], Genes_comb["GA"], Genes_comb["GA"], Genes_comb["CT"]]


def trainValTestSplit(dataset_features):
    dataset_features = dataset_features.sample(frac=1, random_state = 100)
    trainFrac = 0.50
    valFrac = 0.25
    testFrac = 1 - trainFrac - valFrac
    len = dataset_features.shape[0]
    divisionIndexTrain = int(np.floor(trainFrac*len))
    divisionIndexVal = divisionIndexTrain + int(np.floor(valFrac*len))

    dataset_train = dataset_features.iloc[ :divisionIndexTrain, : ]
    dataset_val = dataset_features.iloc[ divisionIndexTrain:divisionIndexVal, : ]
    dataset_test = dataset_features.iloc[ divisionIndexVal:, : ]

    return dataset_train, dataset_val, dataset_test

def getBalancedDataset(dataset):

    y_feature = "IsHCC"
    class0Filter = dataset[y_feature] == 0
    class1Filter = dataset[y_feature] == 1

    datasetClass0 = dataset[class0Filter]
    datasetClass1 = dataset[class1Filter]

    classRatios = int(len(datasetClass0.index)/len(datasetClass1.index))
    print("Class ratios:", classRatios)
    balanceDataset = pd.DataFrame()

    balanceDataset = balanceDataset.append([datasetClass0], ignore_index=True)
    balanceDataset = balanceDataset.append([datasetClass1]*(classRatios), ignore_index=True)
    balanceDataset = balanceDataset.sample(frac = 1, random_state=100)

    return balanceDataset

def getFormatedDataset():
    base_dataset_path = data_path + "PLANILHA_HCV-RECORTE.csv"
    base_dataset = pd.read_csv(base_dataset_path)

    dataset_features = base_dataset.iloc[:, :16]
    dataset_features = dataset_features.drop(["PACIENTE", "ID.1", "SEXO", "At. Inflam. 1"], axis = 1)

    dataset_features.to_csv(data_path + "base/dataset_base.csv")

    return dataset_features

def create_new_columns(number_formatted_dataset, column, feature):

    index = SNPs.index(feature)
    values = SNPs_Values[index]

    for value in values:
        number_formatted_dataset[feature + "=" + value] = (column == value).fillna(0.0).astype(np.int64)

    return number_formatted_dataset

def getNumberFormatedDataset():

    pre_formatted_dataset = getFormatedDataset()
    base_len = pre_formatted_dataset.shape[0]

    number_formatted_dataset = pd.DataFrame()
    number_formatted_dataset["ID"] = pre_formatted_dataset.iloc[:, 0]

    for feature in SNPs:
       number_formatted_dataset = create_new_columns(number_formatted_dataset, pre_formatted_dataset.loc[ : , feature ], feature)

    number_formatted_dataset["Fibrose 1"] = pre_formatted_dataset.loc[ : , "Fibrose 1" ]
    number_formatted_dataset["IsHCC"] = ( pre_formatted_dataset.loc[ : , "Fibrose 1" ] == "HCC").astype(np.int)

    removed_classes = ['F2', 'F3', 'F4']
    removed_cases = number_formatted_dataset.loc[number_formatted_dataset['Fibrose 1'].isin(removed_classes)]

    number_formatted_dataset.drop(removed_cases.index, inplace=True)
    number_formatted_dataset = number_formatted_dataset.drop("Fibrose 1", axis = 1)
    number_formatted_dataset.to_csv(data_path + "base/dataset_integer_base.csv")
    return number_formatted_dataset


def main():

    print("#### Formatting Dataset:")
    FormattedNumberDataset = getNumberFormatedDataset()
    print(FormattedNumberDataset)
    #balancedDt = getBalancedDataset(FormattedNumberDataset)
    #balancedDt.to_csv(data_path + "base/dataset_balanced.csv")
    print("#### Dataset formatting complete ####")
    return



if __name__ == "__main__":
    main()
