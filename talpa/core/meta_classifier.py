from abc import ABC, abstractmethod


class MetaClassifier(ABC):
    def __init__(self):
        '''
        Base class for all types of classifiers
        '''
        self.name = self.__class__.__name__



    @abstractmethod
    def fit(self, data, labels=None, **kwargs):
        '''
        Fit the model.
        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param labels:  DataFrame, shape (n_samples, 1)
            Target values (strings or integers in classification, real numbers in regression)
        :param kwargs: Keyword arguments for classification of the given dataset
        :return: self : object
        '''
        raise NotImplemented

    @abstractmethod
    def predict(self, data, **kwargs):
        '''
        Predict class for dataset.
        :param data: Dataframe of shape (n_samples, n_features)
                The input samples.
        :param kwargs: Keyword arguments for classification of the given dataset

        :return: Predicted values
        '''
        raise NotImplemented

    @abstractmethod
    def predict_scores(self, data, **kwargs):
        '''
        Predict class probabilities at each stage for the given dataset

        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param kwargs: Keyword arguments for classification of the given dataset

        :return: Predict class probabilities for dataset.
        '''
        raise NotImplemented
