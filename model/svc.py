import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os


def read_from_csv(s):
    read_path = '../data/' + s
    data_path = os.path.join(read_path)
    df = pd.read_csv(data_path)
    df = df.fillna(0)
    return df


def transfer_to_TrainArray(df):
    length = df.shape[0]
    width = df.shape[1]
    # trainSet contains all data except the class
    train_set = np.zeros([length, width - 1])
    for i in range(length):
        train_set[i] = df.iloc[i, :-1].to_numpy()
    return train_set


def checkInf(df):
    print(df.shape)
    length = df.shape[0]
    for i in range(length):
        currentArray = df.iloc[i, :-1].to_numpy()
        arrayLength = len(currentArray)
        for element in range(arrayLength):
            currElement = currentArray[element]
            if currElement == float('inf'):
                df = df.drop([i])
                #print(i)
    print(df.shape)
    return df


def transfer_to_TestArray(df):
    length = df.shape[0]
    # trainSet contains all data except the class
    test_set = np.zeros(length)
    for i in range(length):
        if df.iloc[i, -1] == 'benign':
            test_set[i] = 0
        else:
            test_set[i] = 1
    return test_set


if __name__ == "__main__":
    test_df = read_from_csv('Malware.csv')
    #test_df = test_df[-2:]
    test_df = checkInf(test_df)
    trainSet = transfer_to_TrainArray(test_df)
    testSet = transfer_to_TestArray(test_df)
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(trainSet, testSet, random_state=1, test_size=0.2)
    print("train_data:",x_train[:10])
    print("length",len(x_train))
    print("type",type(x_train))

    print("y_train:",y_train[:10])
    print("length",len(y_train))
    print("type",type(y_train))

    svmclassifier = svm.SVC(kernel='linear', gamma=0.1, decision_function_shape='ovo', C=0.1)
    svmclassifier.fit(x_train, y_train)
    print(svmclassifier.score(x_train, y_train))
    rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    rf0.fit(x_train, y_train)
    print(rf0.oob_score_)