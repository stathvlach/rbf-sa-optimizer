import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from .transformers import RBFTransformer

class RBFEstimator(BaseEstimator, RegressorMixin):
    """
    """
    def __init__(self, kernel, clustering_func, alpha=1.0, clustering_params=None):
        self.kernel = kernel
        self.alpha = alpha
        self.pipeline_ = None
        self.clustering_func = clustering_func
        self.clustering_params = clustering_params or {}

    def fit(self, X, y):
        """
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        self.pipeline_ = Pipeline([
            ('transformer', RBFTransformer( 
                kernel=self.kernel,
                clustering_func = self.clustering_func,
                clustering_params = self.clustering_params
            )),
            ('regressor', Ridge(alpha=self.alpha))
        ])
        
        self.pipeline_.fit(X, y)
        self.centers_ = self.pipeline_.named_steps['transformer'].centers_
        self.n_features_in_ = X.shape[1]
        
        return self

    def predict(self, X):
        """
        """
        check_is_fitted(self, ['pipeline_'])
        X = check_array(X, accept_sparse=False)
        
        return self.pipeline_.predict(X)