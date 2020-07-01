from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from talpa.core import MetaClassifier

class RandomForestDetector(MetaClassifier):

    def __init__(self,  max_depth = None,  n_estimators = 100, criterion='gini', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        '''
        A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset
        and uses averaging to improve the predictive accuracy and control over-fitting.

        Parameters
        ----------
        :param max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure
            or until all leaves contain less than min_samples_split samples.
        :param n_estimators: int, default=100
            The number of trees in the forest.
        :param criterion: {“gini”, “entropy”}, default=”gini”
            The function to measure the quality of a split.
        :param min_samples_split: int or float, default=2
            The minimum number of samples required to split an internal node.
        :param min_samples_leaf: int or float, default=1
            The minimum number of samples required to be at a leaf node.
        :param min_weight_fraction_leaf: float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
        :param max_features: {“auto”, “sqrt”, “log2”}, int or float, default=”auto”
            The number of features to consider when looking for the best split.
        :param max_leaf_nodes: int, default=None
            Grow trees with max_leaf_nodes in best-first fashion.
        :param min_impurity_decrease: float, default=0.0
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        :param min_impurity_split: float, default=None
            Threshold for early stopping in tree growth.
        :param bootstrap: bool, default=True
            Whether bootstrap samples are used when building trees.
        :param random_state: int or RandomState, default=None
            Controls both the randomness of the bootstrapping of the samples used when building trees
        :param verbose: int, default=0
            Controls the verbosity when fitting and predicting.
        :param warm_start: bool, default=False
            When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
        :param class_weight: {“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
        :param ccp_alpha: non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning.
        :param max_samples: int or float, default=None
            If bootstrap is True, the number of samples to draw from X to train each base estimator.

        References
        ----------
        [1] More information: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        '''
        super().__init__()

        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth =max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._bootstrap = bootstrap
        self._random_state = random_state
        self._verbose  = verbose
        self._warm_start=warm_start
        self._class_weight =class_weight
        self._ccp_alpha = ccp_alpha
        self._max_samples = max_samples
        self.model = None


    def fit(self, data, labels=None, **kwargs):
        '''
        Fit the random forest model.
        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param labels:  DataFrame, shape (n_samples, 1)
            Target values (strings or integers in classification, real numbers in regression)
        :param kwargs: Keyword arguments for classification of the given dataset
        :return: self : object
        '''

        self.model = RandomForestClassifier(n_estimators=self._n_estimators, max_depth=self._max_depth, criterion=self._criterion)

        X = data.values
        if labels is not None:
            y = labels.values
        else:
            y=None

        self.model.fit(X,y, **kwargs)


    def predict(self, data, **kwargs):
        '''
        Predict class for data.
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
        Predict class probabilities at each stage for dataset

        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param kwargs: Keyword arguments for classification of the given dataset

        :return: Predicted class probabilities for dataset.
        '''
        X = data.values
        scores = self.model.predict_proba(X)
        scores = pd.DataFrame(scores, index=data.index, columns=['score'])

        return scores
