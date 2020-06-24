from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from talpa.core import MetaClassifier

class KNeighborsDetector(MetaClassifier):

    def __init__(self, n_neighbors, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs):

        super().__init__()
        self._n_neighbors = n_neighbors
        self._weights = weights
        self._algorithm = algorithm
        self._leaf_size = leaf_size
        self._metric = metric
        self._p = p
        self._metric_params = metric_params
        self._n_jobs = n_jobs
        self.model = None

    def fit(self, data, labels=None, **kwargs):

        self.model = KNeighborsClassifier(n_neighbors=self._n_neighbors)
        X = data.values
        if labels is not None:
            y = labels.values
        else:
            y=None
        self.model.fit(data, y)


    def predict(self, data, **kwargs):

        X = data.values
        preds = self.model.predict(X)
        predictions = pd.DataFrame(preds, index=data.index, columns=['labels'])
        return predictions


    def predict_scores(self, data,y, **kwargs):

        X = data.values
        scores =self.model.score(X, y)
        scores = pd.DataFrame(scores, index=data.index, columns=['score'])
        return scores