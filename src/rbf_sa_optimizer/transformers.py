import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .interfaces import RBFKernel
from .interfaces import RBFClustering

class RBFTransformer(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, kernel: RBFKernel, clustering_func: RBFClustering, clustering_params):
        self.kernel = kernel
        self.clustering_func = clustering_func
        self.clustering_params = clustering_params or {}

    def fit(self, X, y=None):
        """
        """
        if not callable(self.clustering_func):
            raise TypeError("clustering_func must be a callable following RBFClustering protocol")

        X = check_array(X)

        self.n_features_in_ = X.shape[1]
        self.centers_ = self.clustering_func(X, **self.clustering_params)
        
        return self

    def transform(self, X):
        """
        """
        check_is_fitted(self, ['centers_'])
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features than during fit.")

        return self.kernel(X, self.centers_)