from sklearn.linear_model import LogisticRegression
import pandas as pd
from talpa.core import MetaClassifier

class LogisticRegressionDetector(MetaClassifier):

    def __init__(self,random_state, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,solver='lbfgs', max_iter=100, multi_class='auto',
                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):

        super().__init__()
        self._penalty = penalty
        self._dual = dual
        self._tol =tol
        self._C = C
        self._fit_intercept = fit_intercept
        self._intercept_scaling = intercept_scaling
        self._class_weight= class_weight
        self._solver = solver
        self._max_iter = max_iter
        self._multi_class = multi_class
        self._n_jobs = n_jobs
        self._random_state = random_state
        self._verbose  = verbose
        self._warm_start=warm_start
        self._class_weight =class_weight
        self._l1_ratio = l1_ratio
        self.model = None


    def fit(self, data, labels=None, **kwargs):

        self.model = LogisticRegression(random_state=self._random_state)

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
