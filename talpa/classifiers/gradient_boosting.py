from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from talpa.core import MetaClassifier

class GradientBoostDetector(MetaClassifier):

    def __init__(self, max_depth , n_estimators, random_state, learning_rate =0.1 ,loss='deviance', subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, min_impurity_split=None, init=None,  max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, tol=0.0001, ccp_alpha=0.0):

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
        self.model = None

    def fit(self, data, labels=None, **kwargs):

        self.model = GradientBoostingClassifier(n_estimators=self._n_estimators, max_depth=self._max_depth, random_state=self._random_state)
        X = data.values
        if labels is not None:
            y = labels.values
        else:
            y=None
        self.model.fit(data,y, **kwargs)


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