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
from talpa.core.data_checks import check_data, is_numeric

logfile_dir =os.path.join(os.getcwd(), "logs")
logfile = os.path.join(logfile_dir, 'Logs.log')
logging.basicConfig(filename=logfile ,level=logging.INFO, format='%(asctime)s %(name)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

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
        self.accuracy =  []
        self.f1scr =[]
        if dataset_folder is not None:
            self.dirname = os.path.join(os.getcwd(), dataset_folder)
            self.dr_logger.info("Dataset Folder path {}".format(self.dirname))
            if not os.path.exists(self.dirname):
                self.dr_logger.info("Path given for dataset does not exist {}".format(self.dirname))
                self.dirname = None
            else:
                self.filename = os.path.join(self.dirname, filename)
                self.dr_logger.info("Dataset Filepath {}".format(self.filename))


    def fit_predict(self, model, X_train, y_train, X_test, y_test, classifier_name):
        '''

        :param model: Classifier to be fitted
        :param X_train:   Dataframe of shape (n_samples, n_features)
        :param y_train:   Dataframe of shape (n_samples, 1)
        :param X_test:    Dataframe of shape (n_samples, n_features)
        :param y_test:    Dataframe of shape (n_samples, 1)
        :param classifier_name:  Name of the classifier model
        :return: Numpy array of accuracy and f1score
        '''
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1score = f1_measure(y_test, y_pred)
        self.dr_logger.info("Accuracy and F1score in each split for {} {}, {}:".format(classifier_name, acc, f1score))

        return acc, f1score


    def execute_model(self, modelname, df):
        '''
        This function will execute the model based on the model name provided and the dataframe of the datasets
        :param modelname: Classifier name that needs to be run
        :param df:  Dataframe of the datasets
        :return:
        '''
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        if modelname == "LogisticRegression" or modelname == "KNN":
            self.dr_logger.info("Performing feature scaling before executing {}".format (modelname))
            scaler = StandardScaler()
            scaler.fit(df.drop('activity', axis=1))

            scaled_features = scaler.transform(df.drop('activity', axis=1))
            df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1], index=df.index)
            X = df_feat
            y = df.iloc[:, -1]


        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if modelname == 'GradientBoost':
                clf = GradientBoostDetector(max_depth=5, n_estimators=100, random_state=0)
                acc, f1 = self.fit_predict(clf, X_train, y_train, X_test, y_test, modelname)
                self.accuracy.append(acc)
                self.f1scr.append(f1)

            elif modelname == 'RandomForest':
                rf = RandomForestDetector(max_depth=7, n_estimators=50, random_state=0)
                acc, f1 =self.fit_predict(rf, X_train, y_train, X_test, y_test, modelname)
                self.accuracy.append(acc)
                self.f1scr.append(f1)

            elif modelname == 'LogisticRegression':
                logreg = LogisticRegressionDetector(random_state=0)
                acc, f1 = self.fit_predict(logreg, X_train, y_train, X_test, y_test, modelname)
                self.accuracy.append(acc)
                self.f1scr.append(f1)

            elif modelname == 'KNN':
                knn = KNeighborsDetector(n_neighbors=3)
                acc, f1 =self.fit_predict(knn, X_train, y_train, X_test, y_test, modelname)
                self.accuracy.append(acc)
                self.f1scr.append(f1)

            elif modelname == 'XGBoost':
                xgb = XGBClassification()
                acc, f1 =self.fit_predict(xgb, X_train, y_train, X_test, y_test, modelname)
                self.accuracy.append(acc)
                self.f1scr.append(f1)

        acc_mean = np.array(self.accuracy).mean()
        f1_mean = np.array(self.f1scr).mean()
        df_metrics = pd.DataFrame({'Accuracy':[acc_mean] , 'F1score': [f1_mean]})
        plot_metrics(modelname, df_metrics)
        print("Final Accuracy of ", modelname , acc_mean, "F1Score:", f1_mean)

                                                
    def check_data_validity(self, model_name):
        '''
        This function will check if the dataset consist of missing or null values and then call the model execute function
        :param model_name: Classifier name that needs to be run
        :return:
        '''

        df = pd.read_csv(self.filename, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        visualise_mising_values(df)
        df = df.fillna(0)
        # Check if dataset has null values and if it has all numeric values
        check_data(df)
        #self.dr_logger.info("Dataframe information : {}".format(df.info()))

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