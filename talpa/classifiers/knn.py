from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from talpa.core import MetaClassifier

class KNeighborsDetector(MetaClassifier):

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs):
        ''' Classifier implementing the k-nearest neighbors vote

        Parameters
        ----------
        :param n_neighbors: int, default=5
            Number of neighbors to use by default for kneighbors queries.
        :param weights:{‘uniform’, ‘distance’} or callable, default=’uniform’
            weight function used in prediction.
        :param algorithm:{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
            Algorithm used to compute the nearest neighbors:
        :param leaf_size: int, default=30
            Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query,
            as well as the memory required to store the tree.
        :param p:int, default=2
            Power parameter for the Minkowski metric.
            When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
        :param metric: str or callable, default=’minkowski’
            the distance metric to use for the tree. The default metric is minkowski,
            and with p=2 is equivalent to the standard Euclidean metric
        :param metric_params: dict, default=None
            Additional keyword arguments for the metric function.
        :param n_jobs: int, default=None
            The number of parallel jobs to run for neighbors search.
        :param kwargs: Keyword arguments for the algorithms

        References
        ----------
        [1] More information: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

        '''
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
        '''
        Fit the model using X as training dataset and y as target values
        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param labels: DataFrame, shape (n_samples, 1)
            Target values of shape = [n_samples] or [n_samples, n_outputs]
        :param kwargs: Keyword arguments for the algorithms

        :return: self : object
        '''

        self.model = KNeighborsClassifier(n_neighbors=self._n_neighbors)
        X = data.values
        if labels is not None:
            y = labels.values
        else:
            y=None
        self.model.fit(data, y)


    def predict(self, data, **kwargs):
        '''
        Predict the class labels for the provided data.

        :param data: Dataframe of shape (n_samples, n_features).
            Test samples.
        :param kwargs: Keyword arguments for classification of the given dataset
        :return: Class labels for each dataset sample.
        '''

        X = data.values
        preds = self.model.predict(X)
        predictions = pd.DataFrame(preds, index=data.index, columns=['labels'])
        return predictions


    def predict_scores(self, data, **kwargs):
        '''
        Return probability estimates for the test dataset X.

        :param data: Dataframe of shape (n_samples, n_features).
            Test samples.
        :param kwargs: Keyword arguments for classification of the given dataset
        :return: Return probability estimates for the test dataset X.
        '''

        X = data.values
        scores =self.model.predict_proba(X)
        scores = pd.DataFrame(scores, index=data.index, columns=['score'])
        return scores