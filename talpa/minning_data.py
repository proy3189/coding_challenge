import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from talpa.visualization import *
from sklearn.model_selection import StratifiedShuffleSplit
from talpa.classifiers import *
from talpa.metrics import *


def read_data(filename):
    path = os.path.join(os.getcwd(), "data")
    print(path)
    filename = os.path.join(path, filename)
    print(filename)


    df = pd.read_csv(filename, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    print("Null Values :", df.isnull().values.any())
    print("Count of Null values:",df.isnull().sum().sum())

    # Heat map to show the Null values
    df = df.fillna(0)

    # Plot the distribution of labels
    label_distribution(df)
    # Plot the distribution of data
    #data_distribution(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]


    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(type(y_train))#
    df1= y_train.to_frame()
    print(df1)
    df_tr = pd.DataFrame(data=df1, columns=['activity'])
    label_distribution(df_tr)


    # clf = GradientBoostDetector(max_depth=5, n_estimators=100, random_state=0)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    #
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Classification_report:", classification_report(y_test, y_pred))

    rf = RandomForestDetector(max_depth=5, n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification_report:", classification_report(y_test, y_pred))




if __name__ =='__main__':
    start = time.time()
    filename = 'data_case_study.csv'
    read_data(filename)
    print("Execution time taken in sec :", time.time()-start)