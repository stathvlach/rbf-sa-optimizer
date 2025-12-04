import numpy as np
from .utils import compute_adaptive_sigma

class RBFModel:
    """
    Simple Radial Basis Function (RBF) regression model.
    
    The centers of the gaussian functions are provided externally.
    The model does not perform normalization.
    """


    def __init__(self, centers, n_neighbors: int = 7):
        """
        Parameters
        ----------
        centers : np.ndarray, shape (M, D)
           Centers of the gaussian functions.

        n_neighbors : int
            Number of nearest neighbor centers used to compute adaptive sigmas.
            Determines one sigma per center.
        """

        # --- User-provided parameters ---
        self.centers = centers              # (M, D)
        self.n_neighbors = n_neighbors      # K-nearest neighbors for sigma estimation

        # --- Attributes computed during fit() ---
        self.sigmas_ = None                 # (M,) adaptive sigma per center
        self.weights_ = None                # (M+1, 1) RBF weights + bias

        # --- Metadata ---
        self.n_centers_ = centers.shape[0]  # M
        self.n_dims_ = centers.shape[1]     # D

        # Fit status flag
        self.fitted_ = False


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the RBF model weights given training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples. Data should already be preprocessed/scaled as needed.

        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : RBFModel
            Fitted model.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Basic shape checks
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")

        if X.shape[1] != self.n_dims_:
            raise ValueError(
                f"Incompatible feature dimension: X has {X.shape[1]} features, "
                f"but centers have {self.n_dims_}."
            )

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2 and y.shape[1] == 1:
            pass  # already (n_samples, 1)
        else:
            raise ValueError(
                f"y must be 1D or 2D with a single output (n_samples,) or (n_samples, 1), "
                f"got shape {y.shape}"
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have incompatible number of samples: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )

        # Compute distances from samples to centers
        diff = X[:, None, :] - self.centers[None, :, :]   # (n_samples, n_centers, n_features)
        r = np.linalg.norm(diff, axis=2)                  # (n_samples, n_centers)

        # Compute adaptive sigma per center
        self.sigmas_ = compute_adaptive_sigma(self.centers, self.n_neighbors)  # (n_centers,)

        # Radial basis function (Gaussian kernel)
        phi = np.exp(-(r ** 2) / (2.0 * self.sigmas_[None, :] ** 2))  # (n_samples, n_centers)

        # Design matrix with bias term
        Z = np.hstack([phi, np.ones((X.shape[0], 1), dtype=float)])   # (n_samples, n_centers + 1)

        # Solve the linear system in least-squares sense
        weights, *_ = np.linalg.lstsq(Z, y, rcond=None)  # (n_centers + 1, 1)

        self.weights_ = weights
        self.fitted_ = True

        return self


    def predict(self, X: np.ndarray):
        """
        Predict target values for the given input samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples. Should be preprocessed/scaled in the same way as during fit().

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples, 1)
            Predicted target values.
        """
        if not self.fitted_:
            raise ValueError("This RBFModel instance is not fitted yet. "
                             "Call 'fit' before using 'predict'.")

        X = np.asarray(X, dtype=float)

        # Basic shape checks
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")

        if X.shape[1] != self.n_dims_:
            raise ValueError(
                f"Incompatible feature dimension: X has {X.shape[1]} features, "
                f"but centers have {self.n_dims_}."
            )

        # Compute distances from samples to centers
        diff = X[:, None, :] - self.centers[None, :, :]   # (n_samples, n_centers, n_features)
        r = np.linalg.norm(diff, axis=2)                  # (n_samples, n_centers)

        # Radial basis function (Gaussian)
        phi = np.exp(-(r ** 2) / (2.0 * self.sigmas_[None, :] ** 2))  # (n_samples, n_centers)

        # Add bias term
        Z = np.hstack([phi, np.ones((X.shape[0], 1), dtype=float)])   # (n_samples, n_centers + 1)

        # Compute predictions
        y_pred = Z @ self.weights_    # (n_samples, 1)

        return y_pred

