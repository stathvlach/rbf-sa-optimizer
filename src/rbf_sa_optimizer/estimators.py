import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from .transformers import FuzzyMeansTransformer

class RBFEstimator(BaseEstimator, RegressorMixin):
    """
    Radial basis function (RBF) regressor with fuzzy-means center selection.

    An end-to-end regression model that combines:
    1. Deterministic center discovery (fuzzy-means algorithm).
    2. Nonlinear feature engineering (RBF kernel applied to centers).
    3. Linear regression on the RBF features (Ridge regression).

    This pipeline approach separates concerns: the fuzzy-means algorithm finds
    compact, data-driven centers; the kernel maps samples into the RBF feature
    space; Ridge regression learns weights. The resulting model is interpretable
    (centers can be inspected), computationally efficient (no iterative optimization),
    and regularized (Ridge penalty prevents overfitting).

    The estimator follows the sklearn API, supporting fit/predict, pipelines,
    cross-validation, and hyperparameter tuning.

    Parameters
    ----------
    partitions : ndarray of shape (n_features,)
        Number of fuzzy partitions per input dimension. Controls the granularity
        of the fuzzy grid used by the fuzzy-means algorithm. Typical range: 4-12.
        Higher values yield more centers and finer approximations; lower values
        yield fewer centers and smoother approximations.

    kernel : RBFKernel
        An RBF kernel object (e.g., GaussianKernel) that computes basis function
        values. The kernel is responsible for mapping samples to the feature space.

    alpha : float, default=1.0
        Ridge regression regularization strength. Controls the L2 penalty on
        regression coefficients. Higher values increase regularization (simpler model,
        potentially higher bias); lower values decrease regularization (complex model,
        potentially higher variance). Use cross-validation to tune.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The fitted sklearn Pipeline containing the transformer and regressor.
        Access via estimator.named_steps['transformer'] or
        estimator.named_steps['regressor'].

    centers_ : ndarray of shape (n_centers, n_features)
        RBF centers discovered during fit. Exposes the learned structure for
        inspection, visualization, or reuse in other models.

    n_features_in_ : int
        Number of features seen during fit. Used for validation during predict.

    Notes
    -----
    - Input data should be normalized (e.g., via MinMaxScaler or StandardScaler)
      to ensure the fuzzy partition operates on a consistent scale and to
      stabilize numerical computation.
    - The number of centers (n_centers) is determined automatically by the
      fuzzy-means algorithm and is typically much smaller than n_samples.
    - The model is deterministic: the same training data will always produce
      the same centers, features, and predictions.
      
    See Also
    --------
    FuzzyMeansTransformer : The transformer component that discovers centers
                            and generates RBF features.
    GaussianKernel : A Gaussian RBF kernel implementation.
    compute_adaptive_sigma : Strategy for computing per-center width parameters.
    """
    def __init__(self, partitions, kernel, alpha=1.0):
        self.partitions = partitions
        self.kernel = kernel
        self.alpha = alpha
        self.pipeline_ = None

    def fit(self, X, y):
        """
        Fit the RBF regression model.

        Discovers RBF centers via fuzzy-means, computes kernel features, and
        fits Ridge regression on the features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples. Should be normalized to a consistent range
            (e.g., [-1, 1]) for best results.

        y : array-like of shape (n_samples,)
            Target values (real numbers).

        Returns
        -------
        self : RBFEstimator
            Returns self for method chaining.

        Notes
        -----
        - Internally creates and fits a Pipeline containing FuzzyMeansTransformer
          and Ridge regressor.
        - Stores discovered centers in self.centers_ for inspection.
        - Does not modify X or y.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        
        self.pipeline_ = Pipeline([
            ('transformer', FuzzyMeansTransformer(
                partitions=self.partitions, 
                kernel=self.kernel
            )),
            ('regressor', Ridge(alpha=self.alpha))
        ])
        
        self.pipeline_.fit(X, y)
        self.centers_ = self.pipeline_.named_steps['transformer'].centers_
        self.n_features_in_ = X.shape[1]
        
        return self

    def predict(self, X):
        """
        Predict target values for new samples.

        Applies the learned RBF kernel and Ridge regression to generate
        predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict. Must have the same number of features as the
            training data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        NotFittedError
            If fit has not been called yet.

        Notes
        -----
        - Predictions are computed as a linear combination of RBF basis
          functions: y_pred = Ridge weights * Kernel(X, centers).
        - If the original targets were scaled (e.g., via MinMaxScaler),
          remember to inverse-transform predictions back to the original range.
        """
        check_is_fitted(self, ['pipeline_'])
        X = check_array(X, accept_sparse=False)
        
        return self.pipeline_.predict(X)