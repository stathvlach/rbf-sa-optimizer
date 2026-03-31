import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from rbf_sa_optimizer.estimators import RBFEstimator
from rbf_sa_optimizer.kernels import GaussianKernel
from rbf_sa_optimizer.utils import compute_adaptive_sigma, fuzzymeans
from functools import partial


def test_rbf_estimator_basic_flow():
    """
    """
    X, y = make_regression(n_samples=50, n_features=2, noise=0.1, random_state=42)
    
    params = {'partitions': np.array([3, 3])}
    sigma_strat = partial(compute_adaptive_sigma, n_neighbors=2)
    kernel = GaussianKernel(sigma_strategy=sigma_strat)
    
    model = RBFEstimator(kernel=kernel, clustering_func=fuzzymeans, clustering_params=params, alpha=0.1)
    
    model.fit(X, y)
    assert hasattr(model, "centers_"), "The estimator must expose the centers after fitting."
    assert hasattr(model, "pipeline_"), "The internal pipeline should be instantiated."
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred)), "Predictions must be finite numbers."


def test_rbf_estimator_unfitted_error():
    """
    """
    params = {'partitions': np.array([3, 3])}
    model = RBFEstimator(kernel=None, clustering_func=fuzzymeans, clustering_params=params)
    with pytest.raises(Exception): # NotFittedError
        model.predict(np.random.rand(5, 2))


def test_rbf_estimator_score():
    """
    """
    X, y = make_regression(n_samples=100, n_features=3, random_state=42)
    
    sigma_strat = partial(compute_adaptive_sigma, n_neighbors=2)
    kernel = GaussianKernel(sigma_strategy=sigma_strat)
    params = {'partitions': np.array([3, 3, 3])}
    model = RBFEstimator(kernel=kernel, clustering_func=fuzzymeans, clustering_params=params)
    
    model.fit(X, y)
    r2_score = model.score(X, y)
    
    assert 0 <= r2_score <= 1.0, f"The r^2 value should be within a reasonable range; the result was : {r2_score}"


def test_rbf_estimator_sine_curve_fit():
    np.random.seed(42)
    t = np.linspace(0, 1, 200).reshape(-1, 1) # 2D array (n_samples, 1)
    y_clean = np.sin(2 * np.pi * 4 * t)
    y_noisy = y_clean + np.random.normal(0, 0.05, t.shape)

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = x_scaler.fit_transform(t)
    
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = y_scaler.fit_transform(y_noisy)
    
    # y_scaled from (200, 1) to (200,)
    y_scaled = y_scaled.ravel() 

    params = {'partitions': np.array([25])}  # adjust this
    sigma_strat = partial(compute_adaptive_sigma, n_neighbors=3)
    kernel = GaussianKernel(sigma_strategy=sigma_strat)
    model = RBFEstimator(kernel=kernel, clustering_func=fuzzymeans, alpha=0.01, clustering_params=params)

    model.fit(X_scaled, y_scaled)

    y_pred_scaled = model.predict(X_scaled) 
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    mse = mean_squared_error(y_noisy, y_pred)
    assert mse < 0.05