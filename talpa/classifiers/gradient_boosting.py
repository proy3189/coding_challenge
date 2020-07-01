from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from talpa.core import MetaClassifier

class GradientBoostDetector(MetaClassifier):

    def __init__(self, max_depth =3 , n_estimators =100, random_state = None, learning_rate =0.1 ,loss='deviance', subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, min_impurity_split=None, init=None,  max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, tol=0.0001, ccp_alpha=0.0):

        '''
        Gradient Tree Boosting is a generalization of boosting to arbitrary differentiable loss functions.
        GBDT is an accurate and effective off-the-shelf procedure that can be used for both regression and classification problems.
        GradientBoostingClassifier supports both binary and multi-class classification

        Parameters
        ----------
        :param max_depth: int, default=3
            maximum depth of the individual regression estimators.
            The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance.
        :param n_estimators: int, default=100
            The number of boosting stages to perform.
            Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        :param random_state: int or RandomState, default=None
            Controls the random seed given to each Tree estimator at each boosting iteration.
        :param learning_rate: float, default=0.1
            learning rate shrinks the contribution of each tree by learning_rate
        :param loss: {‘deviance’, ‘exponential’}, default=’deviance’
            loss function to be optimized.
        :param subsample: float, default=1.0
            The fraction of samples to be used for fitting the individual base learners.
        :param criterion: {‘friedman_mse’, ‘mse’, ‘mae’}, default=’friedman_mse’
            The function to measure the quality of a split.
        :param min_samples_split: int or float, default=2
            The minimum number of samples required to split an internal node:
        :param min_samples_leaf: int or float, default=1
            The minimum number of samples required to be at a leaf node
        :param min_weight_fraction_leaf: float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
        :param min_impurity_decrease: float, default=0.0
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        :param min_impurity_split: float, default=None
            Threshold for early stopping in tree growth.
        :param init: estimator or ‘zero’, default=None
            An estimator object that is used to compute the initial predictions.
        :param max_features: {‘auto’, ‘sqrt’, ‘log2’}, int or float, default=None
            The number of features to consider when looking for the best split
        :param verbose: int, default=0
            Enable verbose output.
        :param max_leaf_nodes: int, default=None
            Grow trees with max_leaf_nodes in best-first fashion
        :param warm_start: bool, default=False
            When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble,
            otherwise, just erase the previous solution.
        :param tol: float, default=1e-4
            Tolerance for the early stopping.
        :param ccp_alpha: non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning.

        References
        ----------
        [1] More information: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        '''
        super().__init__()
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._criterion = criterion
        self._max_depth =max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._random_state = random_state
        self._verbose  = verbose
        self._warm_start=warm_start
        self._tol = tol
        self._ccp_alpha = ccp_alpha
        self._loss = loss
        self._subsample =subsample
        self._init = init
        self.model = None

    def fit(self, data, labels=None, **kwargs):
        '''
        Fit the gradient boosting model.
        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param labels:  DataFrame, shape (n_samples, 1)
                Target values (strings or integers in classification, real numbers in regression)
        :param kwargs: Keyword arguments for classification of the given data
        :return: self : Fitted estimator.
        '''
        self.model = GradientBoostingClassifier(n_estimators=self._n_estimators, max_depth=self._max_depth, random_state=self._random_state)
        X = data.values
        if labels is not None:
            y = labels.values
        else:
            y=None
        self.model.fit(data,y, **kwargs)


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
        Predict class probabilities at each stage for data

        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param kwargs: Keyword arguments for classification of the given dataset

        :return: Predicted class probabilities for given data.
        '''

        X = data.values
        scores =self.model.predict_proba(X)
        scores = pd.DataFrame(scores, index=data.index, columns=['score'])
        return scores