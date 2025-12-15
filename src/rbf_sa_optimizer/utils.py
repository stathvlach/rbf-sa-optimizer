import numpy as np
from sklearn.cluster import KMeans


def fuzzymeans(X, partitions):
    """
    Compute RBF centers using the fuzzy-means algorithm based on a non-symmetric
    fuzzy partition of the input space.

    The method constructs a multidimensional grid by partitioning each input
    dimension into a user-defined number of fuzzy sets and considers the centers
    of the resulting fuzzy subspaces as candidate RBF centers. Centers are selected
    incrementally so that each input sample is covered by at least one fuzzy
    subspace, according to a relative distance criterion. The number of centers
    is determined automatically by the algorithm.

    The relative distance is evaluated using a hyper-ellipsoidal metric whose
    axes are proportional to the partition widths of each input dimension.
    This corresponds to the non-symmetric fuzzy-means formulation, which allows
    different numbers of partitions per dimension and typically yields more
    compact RBF network structures.

    The algorithm is deterministic and does not rely on iterative optimization
    or random initialization. It performs a single pass over the data and has
    low computational complexity compared to clustering-based center selection
    methods such as k-means.

    This implementation follows the fuzzy-means methodology introduced in:
    - Sarimveis et al., “A fast and efficient algorithm for training radial basis
      function neural networks based on a fuzzy partition of the input space” (2002)
    and its non-symmetric extension described in:
    - Alexandridis & Sarimveis, “A Radial Basis Function network training algorithm
      using a non-symmetric partition of the input space” (2011)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input samples used to determine the locations of the RBF centers.

    partitions : array-like of shape (n_features,)
        Number of fuzzy partitions per input dimension. Each entry specifies the
        number of fuzzy sets used to partition the corresponding input variable.

        In practice, the number of fuzzy partitions per dimension should remain small.
        Empirical evidence suggests using 4-7 partitions in most cases, and up to 12
        only for highly non-linear input dimensions.

    Returns
    -------
    centers : ndarray of shape (n_centers, n_features)
        The selected RBF centers corresponding to the centers of the chosen fuzzy
        subspaces.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}.")
    if X.shape[0] < 1:
        raise ValueError("X must contain at least one sample.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values (no NaN/inf).")

    partitions = np.asarray(partitions)
    if partitions.ndim != 1:
        raise ValueError(
            f"partitions must be a 1D array of length n_features={X.shape[1]}, "
            f"got shape {partitions.shape}."
        )
    if partitions.shape[0] != X.shape[1]:
        raise ValueError(
            f"partitions must have length n_features={X.shape[1]}, "
            f"got length {partitions.shape[0]}."
        )

    if np.issubdtype(partitions.dtype, np.integer):
        partitions = partitions.astype(int, copy=False)
    elif np.issubdtype(partitions.dtype, np.floating):
        if not np.isfinite(partitions).all() or not np.all(partitions == np.round(partitions)):
            raise TypeError("partitions must be integer-valued.")
        partitions = partitions.astype(int)
    else:
        raise TypeError("partitions must be an array of integers.")

    if (partitions < 2).any():
        raise ValueError("Each partitions[d] must be >= 2.")

    n_obs, n_dims = X.shape
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    # Initialize matrix for output centers
    cluster_centers = np.zeros((n_obs, n_dims), dtype=float)

    # Preallocate fuzzy grid per dimension
    cc = np.zeros((int(partitions.max()), n_dims), dtype=float)

    # Distance scratch vector
    dr = np.zeros(n_obs, dtype=float)

    # Step 1 – fuzzy partitions per dimension
    da = (x_max - x_min) / (partitions - 1)  # step per dimension

    for j in range(n_dims):
        # cc(0:p[j], j) = x_min[j] + [0,1,2,...,p[j]-1] * da[j]
        p_j = int(partitions[j])
        cc[:p_j, j] = x_min[j] + np.arange(p_j, dtype=float) * da[j]

    # Step 2 – center selection
    # First center from first data point
    m = 0
    for dim in range(n_dims):
        p_d = int(partitions[dim])

        # If da[dim] == 0, all grid points coincide; argmin fallback handles it safely.
        candidates = np.abs(cc[:p_d, dim] - X[0, dim]) <= (da[dim] / 2.0)
        idx = np.nonzero(candidates)[0]

        if idx.size > 0:
            cluster_centers[m, dim] = cc[idx[0], dim]
        else:
            j_best = np.argmin(np.abs(cc[:p_d, dim] - X[0, dim]))
            cluster_centers[m, dim] = cc[j_best, dim]

    # Ellipse radii r(i) = n_dims * da(i)^2
    r = n_dims * (da ** 2)

    # Iterate over all samples (k=1..N-1)
    for k in range(1, n_obs):
        # Vectorized distance computation to all existing centers (0..m)
        diff = cluster_centers[: m + 1, :] - X[k, :]  # (m+1, D)

        # Avoid 0/0 when a dimension is constant (da == 0 => r == 0).
        # In that case, only exact match in that dimension is acceptable.
        scaled_sq = np.empty_like(diff)
        for d in range(n_dims):
            if r[d] == 0.0:
                scaled_sq[:, d] = np.where(diff[:, d] == 0.0, 0.0, np.inf)
            else:
                scaled_sq[:, d] = (diff[:, d] ** 2) / r[d]

        dr[: m + 1] = np.sqrt(scaled_sq.sum(axis=1))  # (m+1,)

        # If no center covers the point (dr > 1), add new center
        eps = 1e-12 # Numerical tolerance to avoid boundary artifacts at dr == 1.0
        if dr[: m + 1].min() > 1.0 + eps:
            m += 1
            for dim in range(n_dims):
                p_d = int(partitions[dim])
                candidates = np.abs(cc[:p_d, dim] - X[k, dim]) <= (da[dim] / 2.0)
                idx = np.nonzero(candidates)[0]

                if idx.size > 0:
                    cluster_centers[m, dim] = cc[idx[0], dim]
                else:
                    j_best = np.argmin(np.abs(cc[:p_d, dim] - X[k, dim]))
                    cluster_centers[m, dim] = cc[j_best, dim]

    # Keep only the discovered centers
    return cluster_centers[: m + 1, :]


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
