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
from sklearn.preprocessing import StandardScaler


def read_data(filename):
    path = os.path.join(os.getcwd(), "data")
    print(path)
    filename = os.path.join(path, filename)
    print(filename)


    df = pd.read_csv(filename, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    print(df.info())

    print("Null Values :", df.isnull().values.any())
    print("Count of Null values:",df.isnull().sum().sum())

    # Heat map to show the Null values
    df = df.fillna(0)

    # Plot the distribution of labels
    label_distribution(df)
    #Plot the distribution of data
    #data_distribution(df)

    execute_model("RandomForest", df)

    # print(type(y_train))#
    # df1= y_train.to_frame()
    # print(df1)
    # df_tr = pd.DataFrame(data=df1, columns=['activity'])
    # label_distribution(df_tr)

def execute_model(modelname, df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if modelname == "LogisticRegression" or modelname == "KNN":
        print(modelname)
        scaler = StandardScaler()
        scaler.fit(df.drop('activity', axis=1))

        scaled_features = scaler.transform(df.drop('activity', axis=1))
        df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1], index=df.index)
        # df_feat.head()
        X = df_feat
        y = df.iloc[:, -1]


    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    accuracy = []
    f1scr =[]

    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if modelname == 'GradientBoost':
            clf = GradientBoostDetector(max_depth=5, n_estimators=100, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy.append(accuracy_score(y_test, y_pred))
            f1scr.append(f1_measure(y_test, y_pred))
            print("GradientBoost Accuracy:", accuracy, np.array(accuracy).mean())

        elif modelname == 'RandomForest':

            rf = RandomForestDetector(max_depth=5, n_estimators=100, random_state=0)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
            f1scr.append(f1_measure(y_test, y_pred))
            print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

        elif modelname == 'LogisticRegression':
            logreg = LogisticRegressionDetector(random_state=0)
            logreg.fit(X_train, y_train)
            y_pred = logreg.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
            f1scr.append(f1_measure(y_test, y_pred))
            print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

        elif modelname == 'KNN':
            knn = KNeighborsDetector(n_neighbors=3)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
            f1scr.append(f1_measure(y_test, y_pred))
            print("KNN Accuracy:", accuracy_score(y_test, y_pred))
    acc_mean = np.array(accuracy).mean()
    f1_mean = np.array(f1scr).mean()
    df_metrics = pd.DataFrame({'Accuracy':[acc_mean] , 'F1score': [f1_mean]})
    plot_metrics(modelname, df_metrics)
    print("Final Accuracy of ", modelname , np.array(accuracy).mean())



if __name__ =='__main__':
    start = time.time()
    filename = 'data_case_study.csv'
    read_data(filename)
    print("Execution time taken in sec :", time.time()-start)