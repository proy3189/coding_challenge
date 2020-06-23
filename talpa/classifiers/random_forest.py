from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from talpa.core import MetaClassifier

class RandomForestDetector(MetaClassifier):

    def __init__(self, n_estimators,  max_depth, criterion='gini', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):

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

        self.model = RandomForestClassifier(n_estimators=self._n_estimators, max_depth=self._max_depth, criterion=self._criterion)

        X = data.values
        if labels is not None:
            y = labels.values
        else:
            y=None

        self.model.fit(X,y, **kwargs)


    def predict(self, data, **kwargs):

        X = data.values
        preds = self.model.predict(X)
        predictions = pd.DataFrame(preds, index=data.index, columns=['labels'])

        return predictions


    def predict_scores(self, data, **kwargs):

        X = data.values
        scores = self.model.predict_proba(X)
        scores = pd.DataFrame(scores, index=data.index, columns=['score'])

        return scores
