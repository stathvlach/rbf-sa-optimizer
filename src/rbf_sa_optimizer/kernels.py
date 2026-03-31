import numpy as np
from scipy.spatial.distance import cdist
from .interfaces import RBFKernel, SigmaStrategy

class GaussianKernel(RBFKernel):
    """
    Gaussian radial basis function (RBF) kernel with adaptive per-center widths.

    Computes the Gaussian kernel matrix k(x, c) = exp(-||x - c||² / (2σ²))
    where σ is a width parameter that may differ for each center. This allows
    different centers to have different "reaches" - some narrow, some broad-based
    on local data density or other criteria embedded in the sigma strategy.

    The kernel is a two-stage process:
    1. On first call, compute sigma values from centers using the provided strategy.
    2. On subsequent calls, apply the cached sigma values to compute kernel values.

    This lazy initialization pattern allows the sigma strategy to inspect the
    discovered centers (e.g., their pairwise distances) before making decisions.

    Parameters
    ----------
    sigma_strategy : callable
        A function that accepts a center array of shape (n_centers, n_features)
        and returns an array of shape (n_centers,) containing positive sigma values.
        Common strategy: compute_adaptive_sigma(centers, n_neighbors=2).

        The strategy is called *once* during the first kernel evaluation and the
        result is cached. All subsequent calls use the same sigma values.

    Attributes
    ----------
    sigmas_ : ndarray of shape (n_centers,), or None
        Cached width parameters computed from centers on first call.
        None until the kernel is first evaluated.

    Raises
    ------
    ValueError
        If the sigma strategy returns any non-positive values. Gaussian kernels
        require strictly positive sigma to avoid numerical instability.

    Notes
    -----
    - The kernel is evaluated as a matrix of shape (n_samples, n_centers).
    - Each row corresponds to one sample; each column to one center.
    - Broadcasting: sigmas are reshaped to (1, n_centers) so they apply
      independently to each sample.
    - The computation uses squared Euclidean distance (scipy.spatial.distance.cdist
      with metric='sqeuclidean') for efficiency.
    """
    def __init__(self, sigma_strategy: SigmaStrategy):
        self.sigma_strategy = sigma_strategy
        self.sigmas_ = None

    def __call__(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian kernel matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Sample points at which to evaluate the kernel.

        centers : ndarray of shape (n_centers, n_features)
            RBF center locations. On first call, these are passed to the sigma
            strategy to compute adaptive width parameters.

        Returns
        -------
        kernel_matrix : ndarray of shape (n_samples, n_centers)
            Gaussian kernel values. kernel_matrix[i, j] = exp(-r²_ij / (2σ_j²))
            where r²_ij is the squared Euclidean distance from sample i to
            center j, and σ_j is the width of center j.
        """
        if self.sigmas_ is None:
            self.sigmas_ = self.sigma_strategy(centers)
            
            if np.any(self.sigmas_ <= 0):
                raise ValueError("GaussianKernel: all sigmas must be positive.")

        r_squared = cdist(X, centers, metric='sqeuclidean')
        
        return np.exp(-r_squared / (2.0 * self.sigmas_[None, :]**2))
