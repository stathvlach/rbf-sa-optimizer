import pytest
import numpy as np
from rbf_sa_optimizer.kernels import GaussianKernel
from rbf_sa_optimizer.interfaces import RBFKernel

def test_gaussian_kernel_shape():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]) # 3 δείγματα
    centers = np.array([[0.0, 0.0], [1.0, 1.0]])      # 2 κέντρα
    
    mock_sigma_strat = lambda c: np.ones(len(c))
    
    kernel = GaussianKernel(sigma_strategy=mock_sigma_strat)
    
    phi = kernel(X, centers)
    
    # Check the shape (n_samples, n_centers)
    assert phi.shape == (3, 2)
    
    # Check if the activation for the center [0,0] and sample [0,0] is 1.0
    assert np.isclose(phi[0, 0], 1.0)

def test_gaussian_kernel_invalid_sigma():
    X = np.array([[0.0, 0.0]])
    centers = np.array([[0.0, 0.0]])
    
    # Check for zero or negative sigma (non valid case)
    bad_sigma_strat = lambda c: np.array([0.0])
    
    kernel = GaussianKernel(sigma_strategy=bad_sigma_strat)
    
    with pytest.raises(ValueError, match="GaussianKernel: all sigmas must be positive."):
        kernel(X, centers)

def test_kernel_implements_protocol():
    # Check types consistency
    mock_sigma_strat = lambda c: np.ones(len(c))
    kernel = GaussianKernel(sigma_strategy=mock_sigma_strat)
    
    assert isinstance(kernel, RBFKernel)