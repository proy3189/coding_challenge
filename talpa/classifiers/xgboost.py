import xgboost as xgb
from talpa.core import MetaClassifier
import  pandas as pd


class XGBClassification(MetaClassifier):

    def __init__(self, objective="multi:softprob", random_state=42):
        '''
        XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
        It implements machine learning algorithms under the Gradient Boosting framework.
        XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many dataset science problems in a fast and accurate way.

        Parameters
        ----------
        :param objective: multi:softprob
                set XGBoost to do multiclass classification using the softmax objective
        :param random_state: int or RandomState, default=None
               Controls the random seed given to each Tree estimator at each boosting iteration
        :param kwargs: Keyword arguments for classification of the given dataset

        References
        ----------
        [1]More information: https://xgboost.readthedocs.io/en/latest/index.html
        '''

        super().__init__()
        self._objective = objective
        self._random_state = random_state
        self.model = None


    def fit(self, data, labels=None, **kwargs):
        '''
        Fit gradient boosting classifier
        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param labels:  DataFrame, shape (n_samples, 1)
                Target values (strings or integers in classification, real numbers in regression)
        :param kwargs: Keyword arguments for classification of the given dataset
        :return: self : object
        '''

        self.model = xgb.XGBClassifier(objective=self._objective, random_state=self._random_state)
        print("Par", self.model.get_params())

        X = data.values
        if labels is not None:
            y = labels.values
        else:
            y=None

        self.model.fit(X,y, **kwargs)


    def predict(self, data, **kwargs):
        '''
        Predict class for dataset.
        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param kwargs: Keyword arguments for classification of the given dataset

        :return: Predicted values
        '''

        X = data.values
        preds = self.model.predict(X)
        predictions = pd.DataFrame(preds, index=data.index, columns=['labels'])

        return predictions


    def predict_scores(self, data, **kwargs):
        '''
        Predict class probabilities at each stage for data

        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param kwargs: Keyword arguments for classification of the given dataset

        :return: Predicted class probabilities for data.
        '''

        X = data.values
        scores = self.model.predict_proba(X)
        scores = pd.DataFrame(scores, index=data.index, columns=['score'])

        return scores
