from abc import ABC, abstractmethod


class MetaClassifier(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
        pass


    @abstractmethod
    def fit(self, data, labels=None, **kwargs):
        raise NotImplementedError


    def predict(self, data, **kwargs):
        raise NotImplementedError


    def predict_scores(self, data, **kwargs):
        raise NotImplementedError
