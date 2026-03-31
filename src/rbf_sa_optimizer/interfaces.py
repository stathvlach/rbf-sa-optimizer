from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class RBFKernel(Protocol):
    """
    Protocol for radial basis function (RBF) kernels.

    An RBF kernel computes basis function values, mapping samples and centers
    into a feature space where the model can fit nonlinear patterns linearly.
    Any callable object satisfying this protocol can be used as a kernel in
    RBFEstimator, FuzzyMeansTransformer, or other RBF-based components.

    This protocol defines the structural contract: a kernel must accept sample
    and center arrays and return a kernel matrix. Implementations may differ
    in their basis function shape (Gaussian, multiquadric, inverse multiquadric,
    etc.) and how they compute width parameters (fixed, adaptive, learned, etc.).

    Implementations
    ----------------
    GaussianKernel : Gaussian RBF with adaptive per-center widths.

    Notes
    -----
    - The protocol is runtime-checkable via isinstance() thanks to @runtime_checkable.
    - Implementations should return strictly positive values for most basis
      functions (Gaussian, multiquadric) to ensure numerical stability downstream.
    - The kernel is stateless in terms of data (it does not learn from X or y),
      but may maintain internal state (e.g., cached sigma values) across calls.
    """
    def __call__(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute RBF kernel matrix.

        Evaluates the RBF basis functions at given sample-center pairs,
        producing a nonlinear feature representation. The kernel should be
        pure in spirit: given the same X and centers, it returns the same result.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Sample points at which to evaluate basis functions. Typically comes
            from training or test data, possibly normalized to a standard range.

        centers : ndarray of shape (n_centers, n_features)
            RBF center locations. Typically discovered by a center-selection
            algorithm (e.g., fuzzy-means). The number of centers determines the
            dimensionality of the feature space.

        Returns
        -------
        Phi : ndarray of shape (n_samples, n_centers)
            Kernel matrix. Phi[i, j] is the value of the j-th basis function
            evaluated at sample i. Each element is typically a positive scalar
            representing the "activation" of that basis function.

        Notes
        -----
        - The kernel matrix is dense and of size (n_samples, n_centers).
        - For efficiency, use vectorized operations (e.g., scipy.spatial.distance.cdist)
          rather than loops over samples and centers.
        - If the kernel maintains state (e.g., width parameters), it may initialize
          or cache them on first call.
        """
        ...

@runtime_checkable
class SigmaStrategy(Protocol):
    """
    Protocol for strategies that estimate RBF width parameters (sigma).

    An RBF width parameter (sigma) controls the "reach" or "width" of a basis
    function. Different strategies may estimate sigma in different ways:
    - Based on local neighbor density (compute_adaptive_sigma)
    - From the range of data (fixed percentage of domain width)
    - From the position of centers (e.g., distance to nearest neighbor)
    - Via cross-validation or user specification

    A SigmaStrategy is a callable that inspects the discovered RBF centers and
    returns width parameters—one per center. This separation of concerns allows
    the center-discovery algorithm (fuzzy-means) to be independent of width
    estimation, and allows different width strategies to be swapped without
    changing the rest of the pipeline.

    Implementations
    ----------------
    compute_adaptive_sigma : Estimates sigma from k-nearest neighbor distances
                             among centers. Adapts to local center density.

    Notes
    -----
    - The strategy is called once during kernel initialization, typically when
      the kernel is first applied to data.
    - All returned sigma values must be strictly positive to ensure numerical
      stability in the Gaussian exponential computation.
    - Strategies are deterministic: the same centers should always yield the
      same sigma values.
    - A strategy may make assumptions about center geometry (e.g., that centers
      are well-separated).
    """
    def __call__(self, centers: np.ndarray) -> np.ndarray:
        """
        Estimate RBF width parameters from center locations.

        Given a set of discovered RBF centers, estimate an appropriate width
        (sigma) for each center's basis function. The strategy may examine
        center geometry (distances, density, clustering) or apply fixed rules
        to determine sigma.

        Parameters
        ----------
        centers : ndarray of shape (n_centers, n_features)
            RBF center locations, typically discovered by the fuzzy-means
            algorithm or another center-selection method. May represent
            different regions of the input space with varying local density.

        Returns
        -------
        sigmas : ndarray of shape (n_centers,)
            Estimated width parameter for each center. All values must be
            strictly positive (> 0) to ensure numerical stability. sigmas[i]
            is the width of the basis function centered at centers[i].

            Interpretation:
            - Small sigma: narrow, local basis function; activates only near centers[i].
            - Large sigma: broad, global basis function; influences distant samples.

        Notes
        -----
        - The returned array must have the same length as the number of centers.
        - All elements must be finite and strictly positive (typically enforced
          upstream, but the strategy should avoid returning zero, negative, NaN,
          or infinite values).
        - The strategy is deterministic and should be independent of training
          data; it depends only on center geometry.
        - If the strategy needs to handle edge cases (e.g., single center,
          identical centers, empty centers), document this behavior clearly.
        """
        ...