import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class RBFModel(BaseEstimator, RegressorMixin):
    """
    """
    def __init__(self, clustering_func, kernel_func, alpha):
        """
        """
        self.clustering_func = clustering_func
        self.kernel_func = kernel_func
        self.alpha = alpha
        

    def fit(self, X, y):
        """
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X .shape[1]
        
        # Generate centers
        self.centers_ = self.clustering_func(X)

        phi = self.kernel_func(X, self.centers_)

        self.regressor_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.regressor_.fit(phi, y)
        self.intercept_ = self.regressor_.intercept_
        self.weights_ = self.regressor_.coef_

        return self


    def predict(self, X):
        """
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        phi = self.kernel_func(X, self.centers_)

        return self.regressor_.predict(phi)