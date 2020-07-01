import os
import logging
import inspect
import time
import pandas as pd
import numpy as np
from talpa.visualization import *
from sklearn.model_selection import StratifiedShuffleSplit
from talpa.classifiers import *
from talpa.metrics import *
from sklearn.preprocessing import StandardScaler
from talpa.core.data_checks import _check_data, is_numeric


logging.basicConfig(filename='reader1.log' ,level=logging.INFO, format='%(asctime)s %(name)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

class DatasetReader():

    def __init__(self, dataset_folder="",filename=" ", **kwargs):
        """
        The generic dataset parser for parsing datasets for solving different learning problems.

        Parameters
        ----------
        dataset_folder: string
            Name of the folder containing the datasets
        kwargs:
        Keyword arguments for the dataset parser
        """
        #self.dr_logger =logging.basicConfig(level=logging.INFO)
        self.dr_logger = logging.getLogger(DatasetReader.__name__)

        if dataset_folder is not None:
            self.dirname = os.path.join(os.getcwd(), dataset_folder)
            self.dr_logger.info("Dataset Folder path {}".format(self.dirname))
            if not os.path.exists(self.dirname):
                self.dr_logger("Path given for dataset does not exist {}".format(self.dirname))
                self.dirname = None
            else:
                self.filename = os.path.join(self.dirname, filename)
                self.dr_logger.info("Dataset Filepath {}".format(self.filename))



    def execute_model(self, modelname, df):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        if modelname == "LogisticRegression" or modelname == "KNN":
            print(modelname)
            scaler = StandardScaler()
            scaler.fit(df.drop('activity', axis=1))

            scaled_features = scaler.transform(df.drop('activity', axis=1))
            df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1], index=df.index)
            X = df_feat
            y = df.iloc[:, -1]


        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        accuracy = []
        f1scr =[]

        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if modelname == 'GradientBoost':
                clf = GradientBoostDetector(max_depth=5, n_estimators=100, random_state=0)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                accuracy.append(accuracy_score(y_test, y_pred))
                f1scr.append(f1_measure(y_test, y_pred))
                self.dr_logger.info("GradientBoost Accuracy and F1score in each split {}, {}:".format( accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))

            elif modelname == 'RandomForest':

                rf = RandomForestDetector(max_depth=7, n_estimators=50, random_state=0)
                #rf = RandomForestDetector(max_depth=5, random_state=0)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                accuracy.append(accuracy_score(y_test, y_pred))
                f1scr.append(f1_measure(y_test, y_pred))
                self.dr_logger.info("Random Forest Accuracy and F1score in each split {}, {}:".format( accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))
                #print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

            elif modelname == 'LogisticRegression':
                logreg = LogisticRegressionDetector(random_state=0)
                logreg.fit(X_train, y_train)
                y_pred = logreg.predict(X_test)
                accuracy.append(accuracy_score(y_test, y_pred))
                f1scr.append(f1_measure(y_test, y_pred))
                self.dr_logger.info("Logistic Regression Accuracy and F1score in each split {}, {}:".format( accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))
                #print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

            elif modelname == 'KNN':
                knn = KNeighborsDetector(n_neighbors=3)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                accuracy.append(accuracy_score(y_test, y_pred))
                f1scr.append(f1_measure(y_test, y_pred))
                self.dr_logger.info("KNN Accuracy and F1score in each split {}, {}:".format( accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))
                print("KNN Accuracy:", accuracy_score(y_test, y_pred))

            elif modelname == 'XGBoost':
                xgb = XGBClassification()
                xgb.fit(X_train, y_train)
                y_pred = xgb.predict(X_test)
                accuracy.append(accuracy_score(y_test, y_pred))
                f1scr.append(f1_measure(y_test, y_pred))
                self.dr_logger.info("XGB Accuracy and F1score in each split {}, {}:".format( accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')))
                print("XGB Accuracy:", accuracy_score(y_test, y_pred))
        acc_mean = np.array(accuracy).mean()
        f1_mean = np.array(f1scr).mean()
        df_metrics = pd.DataFrame({'Accuracy':[acc_mean] , 'F1score': [f1_mean]})
        plot_metrics(modelname, df_metrics)
        print("Final Accuracy of ", modelname , np.array(accuracy).mean(), "F1Score:", f1_mean)

    def check_data_validity(self, model_name):
        df = pd.read_csv(self.filename, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        visualise_mising_values(df)
        df = df.fillna(0)
        # Check if dataset has null values and if it has all numeric values
        _check_data(df)
        print(df.info())

        # Plot the distribution of labels to check if it is has imabalanced or balanced distribution of classes
        # targetclass_distribution(df)

        # Plot the distribution of dataset
        # data_distribution(df)

        self.execute_model(model_name, df)


if __name__ =='__main__':
    start = time.time()
    filename = 'data_case_study.csv'
    model_name = "LogisticRegression"

    read_data = DatasetReader("dataset", filename)
    read_data.check_data_validity(model_name)
    print("Execution time taken:", time.time()-start, "sec")