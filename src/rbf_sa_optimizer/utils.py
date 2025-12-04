import numpy as np
from sklearn.cluster import KMeans


def kmeans_clustering(X: np.ndarray, n_clusters: int):
    """
    Compute RBF centers using KMeans clustering.
    This function acts as a simple wrapper around sklearn's KMeans and is used
    only for testing purposes (e.g., comparing fuzzy-means centers vs. KMeans centers).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input samples. Data should already be preprocessed or scaled as needed.

    n_clusters : int
        The number of clusters to form.

    Returns
    -------
    centers : np.ndarray of shape (n_clusters, n_features)
        The cluster centers found by KMeans.
    """
    X = np.asarray(X, dtype=float)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=0
    )

    kmeans.fit(X)
    return kmeans.cluster_centers_


def compute_adaptive_sigma(centers: np.ndarray, n_neighbors: int):
    """
    Compute an adaptive sigma value for each RBF center using the mean distance
    to its k nearest neighboring centers.

    Parameters
    ----------
    centers : np.ndarray of shape (n_centers, n_features)
        The RBF centers.

    n_neighbors : int
        Number of nearest centers to use when estimating each sigma.

    Returns
    -------
    sigmas : np.ndarray of shape (n_centers,)
        Adaptive sigma values, one per center.
    """
    centers = np.asarray(centers, dtype=float)

    # Pairwise distances between centers (M, M)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    # Prevent selecting the center itself by replacing the diagonal with +inf
    np.fill_diagonal(dist, np.inf)

    # Indices of k nearest neighbors for each center (M, k)
    nearest_idx = np.argpartition(dist, kth=n_neighbors, axis=1)[:, :n_neighbors]

    # Corresponding neighbor distances (M, k)
    nearest_dists = np.take_along_axis(dist, nearest_idx, axis=1)

    # Adaptive sigma_i = mean distance to k nearest neighbors
    sigmas = nearest_dists.mean(axis=1)

    return sigmas
