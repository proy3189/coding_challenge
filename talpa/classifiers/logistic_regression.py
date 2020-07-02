from sklearn.linear_model import LogisticRegression
import pandas as pd
from talpa.core import MetaClassifier
import logging

class LogisticRegressionDetector(MetaClassifier):

    def __init__(self,random_state=None, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,solver='lbfgs', max_iter=100, multi_class='auto',
                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        '''
        Logistic regression is a classification algorithm used to assign observations to a discrete set of classes.
        :param penalty: {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
            Used to specify the norm used in the penalization.
        :param random_state: int, RandomState instance, default=None
            Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the dataset.
        :param dual: bool, default=False
            Dual or primal formulation.
        :param tol: float, default=1e-4
            Tolerance for stopping criteria.
        :param C: float, default=1.0
            Inverse of regularization strength; must be a positive float.
        :param fit_intercept: bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
        :param intercept_scaling: float, default=1
            Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True.
        :param class_weight: dict or ‘balanced’, default=None
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
        :param solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
            Algorithm to use in the optimization problem.
        :param max_iter: int, default=100
            Maximum number of iterations taken for the solvers to converge.
        :param multi_class: {‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
            If the option chosen is ‘ovr’, then a binary problem is fit for each label.
        :param verbose: int, default=0
           For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
        :param warm_start: bool, default=False
            When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
        :param n_jobs: int, default=None
            Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”.
        :param l1_ratio: float, default=None
            The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.

        References
        ----------
        [1] More information: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        '''
        self.logger = logging.getLogger(LogisticRegressionDetector.__name__)

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
        self.logger.info("Intialising classifier")


    def fit(self, data, labels=None, **kwargs):
        '''
        Fit the model according to the given training dataset.
        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param labels: DataFrame, shape (n_samples, 1)
                Target values (strings or integers in classification, real numbers in regression)
        :param kwargs: Keyword arguments for classification of the given dataset.
        :return: self: Fitted estimator.
        '''

        self.model = LogisticRegression(random_state=self._random_state)

        X = data.values
        if labels is not None:
            y = labels.values
        else:
            y=None

        self.model.fit(X,y, **kwargs)


    def predict(self, data, **kwargs):
        '''
        Predict class labels for samples in data
        :param data: Dataframe of shape (n_samples, n_features)
                The input samples.
        :param kwargs: Keyword arguments for classification of the given dataset

        :return: Predicted class label per sample.
        '''
        X = data.values
        preds = self.model.predict(X)
        predictions = pd.DataFrame(preds, index=data.index, columns=['labels'])

        return predictions


    def predict_scores(self, data, **kwargs):
        '''
        Probability estimates. The returned estimates for all classes are ordered by the label of classes.
        :param data: Dataframe of shape (n_samples, n_features)
            The input samples.
        :param kwargs: Keyword arguments for classification of the given dataset

        :return: Returns the probability of the sample for each class in the model, where classes are ordered as they are in self.classes_.
        '''

        X = data.values
        scores = self.model.predict_proba(X)
        scores = pd.DataFrame(scores, index=data.index, columns=['score'])

        return scores
