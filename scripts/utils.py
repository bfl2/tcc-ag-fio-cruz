import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from sklearn import datasets, linear_model

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies

from imblearn.over_sampling import RandomOverSampler

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

def getBalancedDataset(*args):
    if(len(args) == 1):
        dataset = getBalancedDatasetFromDt(args[0])
    elif(len(args) == 2):
       dataset = getBalancedDatasetXY(args[0], args[1])
    else:
       dataset = getBalancedDatasetFromDt(args[0])

    return dataset

def getBalancedDatasetROS(X, y):
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

def getBalancedDatasetFromDt(dataset, verbose=False):

    y_feature = "IsHCC"
    class0Filter = dataset[y_feature] == 0
    class1Filter = dataset[y_feature] == 1

    datasetClass0 = dataset[class0Filter]
    datasetClass1 = dataset[class1Filter]

    classRatios = int(len(datasetClass0.index)/len(datasetClass1.index))
    invClassRatios = int(len(datasetClass1.index)/len(datasetClass0.index))
    if(verbose):
        print("Class ratios:", classRatios, "Inverse Class ratios:", invClassRatios)
    balanceDataset = pd.DataFrame()
    if(classRatios > 0):
        balanceDataset = balanceDataset.append([datasetClass0], ignore_index=True)
        balanceDataset = balanceDataset.append([datasetClass1]*(classRatios), ignore_index=True)
        balanceDataset = balanceDataset.sample(frac = 1, random_state=42)
    else:
        balanceDataset = balanceDataset.append([datasetClass0]*(invClassRatios), ignore_index=True)
        balanceDataset = balanceDataset.append([datasetClass1], ignore_index=True)
        balanceDataset = balanceDataset.sample(frac = 1, random_state=42)

    return balanceDataset

def getBalancedDatasetXY(X_columns, y_column):
    dataset = getDatasetFromXY(X_columns, y_column)
    dataset = getBalancedDataset(dataset)

    return dataset

def getDatasetFromXY(X_columns, y_column):

    dataset = X_columns.copy()
    dataset["IsHCC"] = y_column

    return dataset

def getXYFromDataset(dataset):
    y = dataset["IsHCC"]
    X = dataset.drop(["IsHCC"])
    return X, y

def getFormatedDataset(write_to_file=False):
    base_dataset_path = data_path + "PLANILHA_HCV-RECORTE.csv"
    base_dataset = pd.read_csv(base_dataset_path)

    dataset_features = base_dataset.iloc[:, :16]
    dataset_features = dataset_features.drop(["PACIENTE", "ID.1", "SEXO", "At. Inflam. 1"], axis = 1)

    if(write_to_file):
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

def getOneHotEncodedDataset(pre_formatted_dataset=getFormatedDataset(), write_to_file=False, remove_extra_classes=False):
    base_len = pre_formatted_dataset.shape[0]
    encoded_dataset = pd.DataFrame()
    encoded_dataset["ID"] = pre_formatted_dataset["ID"]
    categorical_features = SNPs

    ### One Hot Encoding Dataset
    for feature in categorical_features:
        encoded_features = pd.get_dummies(pre_formatted_dataset[feature], dtype=np.int64, prefix=(feature), prefix_sep="=")
        encoded_dataset = pd.concat([encoded_dataset, encoded_features], axis=1)

    encoded_dataset["IsHCC"] = ( pre_formatted_dataset.loc[ : , "Fibrose 1" ] == "HCC").astype(np.int64)

    ### Removing Classes: F2, F3, F4
    if(remove_extra_classes):
        encoded_dataset["Fibrose 1"] = pre_formatted_dataset.loc[ : , "Fibrose 1" ]
        removed_classes = ['F2', 'F3', 'F4']
        removed_cases = encoded_dataset.loc[encoded_dataset['Fibrose 1'].isin(removed_classes)]

        encoded_dataset.drop(removed_cases.index, inplace=True)
        encoded_dataset = encoded_dataset.drop("Fibrose 1", axis = 1)

    if(write_to_file):
        encoded_dataset.to_csv(data_path + "base/dataset_base_encoded.csv")

    return encoded_dataset

def generateTargetColumn(label_encoded_dataset, pre_formatted_dataset, classes_config):

    label_encoded_dataset["Fibrose 1"] = pre_formatted_dataset.loc[ : , "Fibrose 1" ]
    ### Remove F2-F4 classes config
    ### Standard - F0, F1, F2, F3, F4 X HCC | 0 X 1
    if(classes_config == "standard"):
        label_encoded_dataset["IsHCC"] = ( pre_formatted_dataset.loc[ : , "Fibrose 1" ] == "HCC").astype(np.int)
    elif(classes_config == "F4XHCC"):
        removed_classes = ['F0', 'F1', 'F2', 'F3']
        removed_cases = label_encoded_dataset.loc[label_encoded_dataset['Fibrose 1'].isin(removed_classes)]
        label_encoded_dataset.drop(removed_cases.index, inplace=True)
        label_encoded_dataset["IsHCC"] = ( pre_formatted_dataset.loc[ : , "Fibrose 1" ] == "HCC").astype(np.int)
    elif(classes_config == "F0,F1,F2,F3XHCC"):
        removed_classes = ['F4']
        removed_cases = label_encoded_dataset.loc[label_encoded_dataset['Fibrose 1'].isin(removed_classes)]
        label_encoded_dataset.drop(removed_cases.index, inplace=True)
        label_encoded_dataset["IsHCC"] = ( pre_formatted_dataset.loc[ : , "Fibrose 1" ] == "HCC").astype(np.int)
    elif(classes_config == "F0,F1,F2,F3XF4,HCC"):
        target_1_class = ["HCC", "F4"]
        label_encoded_dataset["IsHCC"] = label_encoded_dataset['Fibrose 1'].apply(lambda i: 1 if i in target_1_class else 0)
    else:
        label_encoded_dataset["IsHCC"] = ( pre_formatted_dataset.loc[ : , "Fibrose 1" ] == "HCC").astype(np.int)

    label_encoded_dataset = label_encoded_dataset.drop("Fibrose 1", axis = 1)
    return label_encoded_dataset

def getLabelEncodedDataset(pre_formatted_dataset=getFormatedDataset(), classes_config="standard", write_to_file=False):
    base_len = pre_formatted_dataset.shape[0]
    label_encoded_dataset = pd.DataFrame()
    label_encoded_dataset["ID"] = pre_formatted_dataset["ID"]
    categorical_features = SNPs
    le = LabelEncoder()
    for feature in categorical_features:
        label_encoded_dataset[feature] = le.fit_transform(pre_formatted_dataset[feature])
        label_encoded_dataset[feature] = label_encoded_dataset[feature].apply(lambda x: round(((x+1)/3), 3))

     ### Creating isHCC target column based on classesConfig
    label_encoded_dataset = generateTargetColumn(label_encoded_dataset, pre_formatted_dataset, classes_config)
    ###

    if(write_to_file):
        label_encoded_dataset.to_csv(data_path + "base/dataset_base_label_encoded.csv")

    return label_encoded_dataset


def getFilteredDataset(genes, classes_config="standard"):
    filtered_dataset = getLabelEncodedDataset(classes_config=classes_config)
    index = 0
    if(len(genes) > len(SNPs)):
        print("Error: Passing filter larger than SNPs list")

    for gene in genes:
        if(gene == 0):
            filtered_dataset = filtered_dataset.drop(SNPs[index], axis=1)
        index+=1
    return filtered_dataset

def getFeaturesTextFromGenes(genes):
    features = []
    i = 0
    for gene in genes:
        if(gene == 1):
            features.append(SNPs[i])
        i += 1

    return features

def convertOneClassResullts(y_pred):
    i = 0
    for e in y_pred:
        if(e == -1):
            y_pred[i] = 0
        i += 1
    return y_pred

def getOneClassDataset(X, y):
    y_filt = y.loc[y['IsHCC']==0]
    y_one = y.drop(y_filt.index)
    X_one = X.drop(y_filt.index)
    return X_one, y_one

def main():

    print("#### Formatting Dataset:")
    print( getLabelEncodedDataset(write_to_file=True))
    print("#### Dataset formatting complete ####")
    return
