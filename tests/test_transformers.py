import numpy as np
import pytest
from sklearn.datasets import make_regression
from rbf_sa_optimizer.transformers import FuzzyMeansTransformer
from rbf_sa_optimizer.kernels import GaussianKernel
from rbf_sa_optimizer.utils import compute_adaptive_sigma
from functools import partial

def test_fuzzy_means_transformer_integration():
    X, _ = make_regression(n_samples=100, n_features=2, random_state=42)
    
    partitions = np.array([3, 3])
    sigma_strat = partial(compute_adaptive_sigma, n_neighbors=2)
    kernel = GaussianKernel(sigma_strategy=sigma_strat)
    
    transformer = FuzzyMeansTransformer(partitions=partitions, kernel=kernel)
    transformer.fit(X)
    
    n_centers_found = transformer.centers_.shape[0]
    X_transformed = transformer.transform(X)
    
    assert X_transformed.shape == (100, n_centers_found)
    assert n_centers_found <= 9
    assert n_centers_found > 0
