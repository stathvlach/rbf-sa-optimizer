import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .utils import fuzzymeans
from .interfaces import RBFKernel

class FuzzyMeansTransformer(BaseEstimator, TransformerMixin):
    """
    Feature transformer: discover RBF centers and apply kernel to generate features.

    This transformer integrates two steps:
    1. Center discovery via the fuzzy-means algorithm (during fit).
    2. Kernel feature generation (during transform).

    The fuzzy-means algorithm partitions the input space into a regular grid of
    fuzzy subspaces and selects centers incrementally, ensuring that every sample
    is "covered" by at least one fuzzy subspace according to a relative distance
    criterion. This deterministic, one-pass approach yields compact, data-driven
    center configurations without iterative optimization.

    Once centers are discovered, the transformer applies a provided RBF kernel
    (typically Gaussian) to map each sample into a higher-dimensional feature space
    spanned by the basis functions. This nonlinear feature representation is then
    fed to a downstream regressor (usually Ridge regression).

    Parameters
    ----------
    partitions : ndarray of shape (n_features,)
        Number of fuzzy partitions per input dimension. Determines the granularity
        of the initial fuzzy grid. Empirically, 4-7 partitions work well for most
        problems; use up to 12 only for highly nonlinear dimensions.

    kernel : RBFKernel
        An RBF kernel object (e.g., GaussianKernel) that implements __call__(X, centers)
        and returns a kernel matrix of shape (n_samples, n_centers). The kernel is
        responsible for computing basis function values.

    Attributes
    ----------
    centers_ : ndarray of shape (n_centers, n_features)
        RBF centers discovered during fit. The number of centers (n_centers) is
        determined automatically by the fuzzy-means algorithm based on data coverage.

    n_features_in_ : int
        Number of features seen during fit (for validation during transform).

    Notes
    -----
    - The transformer assumes X is normalized (e.g., via MinMaxScaler) to ensure
      the fuzzy partition operates on a consistent scale.
    - The number of discovered centers is typically much smaller than n_samples
      and depends on data density and partition granularity.
    - This transformer is deterministic: the same input data will always produce
      the same centers and features.
    """
    def __init__(self, partitions: np.ndarray, kernel: RBFKernel):
        self.partitions = partitions
        self.kernel = kernel

    def fit(self, X, y=None):
        """
        Discover RBF centers using the fuzzy-means algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples. Typically normalized to a consistent range
            (e.g., [-1, 1]) before passing to fit.

        y : ignored
            Present for sklearn Pipeline compatibility.

        Returns
        -------
        self : FuzzyMeansTransformer
            Returns self for method chaining.

        Notes
        -----
        - Calls fuzzymeans(X, self.partitions) to determine centers.
        - Centers are stored in self.centers_ for later inspection or reuse.
        - The kernel is *not* evaluated during fit; it is applied only during
          transform (lazy evaluation).
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.centers_ = fuzzymeans(X, self.partitions)
        
        return self

    def transform(self, X):
        """
        Apply RBF kernel to generate feature representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform. Must have the same number of features as the
            training data used in fit.

        Returns
        -------
        X_features : ndarray of shape (n_samples, n_centers)
            Kernel feature matrix. X_features[i, j] is the value of the j-th
            basis function evaluated at sample i. These features are typically
            fed to a regression or classification algorithm.

        Raises
        ------
        NotFittedError
            If fit has not been called yet.

        ValueError
            If X has a different number of features than training data.

        Notes
        -----
        - The kernel is evaluated using the centers discovered during fit.
        - The transformation is nonlinear; even linear downstream models can
          capture nonlinear relationships in the original space.
        """
        check_is_fitted(self, ['centers_'])
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features than during fit.")

        return self.kernel(X, self.centers_)