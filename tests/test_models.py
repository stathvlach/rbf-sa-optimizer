import numpy as np
import pytest
from pathlib import Path
from functools import partial
from sklearn.utils.estimator_checks import check_is_fitted
from rbf_sa_optimizer.models import RBFModel
from rbf_sa_optimizer.utils import fuzzymeans, compute_adaptive_sigma
from rbf_sa_optimizer.kernels import GaussianKernel
from rbf_sa_optimizer.optimizers import SARBFOptimizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def test_rbf_model_full_run():
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    
    clustering = partial(fuzzymeans, partitions=[3, 3])
    adaptieve_sigma = partial(compute_adaptive_sigma, n_neighbors=3)
    kernel = GaussianKernel(sigma_strategy=adaptieve_sigma)
    
    model = RBFModel(
        clustering_func=clustering,
        kernel_func=kernel,
        alpha=0.1
    )
    
    try:
        model.fit(X, y)
    except Exception as e:
        pytest.fail(f"Model fit failed with error: {e}")
    
    assert hasattr(model, "centers_"), "Model must have centers"
    assert hasattr(model, "weights_"), "Model must have weights"

    n_centers, n_features = model.centers_.shape
    assert n_centers > 0, "Minimum of one center"
    assert n_features == 2, f"Expected 2 features, got {n_features}"
    
    X_test = np.random.rand(5, 2)
    y_pred = model.predict(X_test)
    
    assert y_pred.shape == (5,), "Wrong prediction shape"
    assert not np.isnan(y_pred).any(), "NaN values appeared in prediction"
    
    try:
        check_is_fitted(model)
    except Exception:
        pytest.fail("check_is_fitted failed on trained model")

def test_sine():
    data_path = Path("data/sine_1sec_4cycles_sigma005.csv")

    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    print("Raw data shape:", data.shape)

    # Columns: t, y_clean, y_noisy
    t = data[:, 0:1]           # time
    y_clean = data[:, 1:2]     # clean sine
    y_noisy = data[:, 2:3]     # noisy sine (regression target)

    # For training: X = time, y = noisy signal
    X = t
    y = y_noisy

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Normalize input (X) and output (y) to [-1, 1]
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = x_scaler.fit_transform(X).reshape(-1, 1)

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = y_scaler.fit_transform(y).ravel()

    clustering = partial(fuzzymeans, partitions=[25])
    adaptieve_sigma = partial(compute_adaptive_sigma, n_neighbors=3)
    kernel = GaussianKernel(sigma_strategy=adaptieve_sigma)
    
    model = RBFModel(
        clustering_func=clustering,
        kernel_func=kernel,
        alpha=0.1
    )
    
    try:
        model.fit(X_scaled, y_scaled)
    except Exception as e:
        pytest.fail(f"Model fit failed with error: {e}")

    # Predict the sine signal using the trained model
    y_pred_scaled = model.predict(X_scaled)

    # Convert predictions back to the original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    print("y_pred shape:", y_pred.shape)
    
    # Compute error metrics in the original (unscaled) domain
    mse = mean_squared_error(y_noisy, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_noisy, y_pred)

    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")

def test_auto_nonlinear_surface():
    # Load sine dataset from CSV
    data_path = Path("data/smooth_nonlinear_surface.csv")

    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    print("Raw data shape:", data.shape)

    # Columns: t, y_clean, y_noisy
    X = data[:, 0:2]           
    y = data[:, 2:3] 

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Normalize input (X) and output (y) to [-1, 1]
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = x_scaler.fit_transform(X)

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = y_scaler.fit_transform(y).ravel()

    clustering = partial(fuzzymeans, partitions=[12, 12])
    adaptieve_sigma = partial(compute_adaptive_sigma, n_neighbors=7)
    kernel = GaussianKernel(sigma_strategy=adaptieve_sigma)
    
    model = RBFModel(
        clustering_func=clustering,
        kernel_func=kernel,
        alpha=0.1
    )
    
    try:
        model.fit(X_scaled, y_scaled)
    except Exception as e:
        pytest.fail(f"Model fit failed with error: {e}")

    # Predict the sine signal using the trained model
    y_pred_scaled = model.predict(X_scaled)

    # Convert predictions back to the original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    print("y_pred shape:", y_pred.shape)
    
    # Compute error metrics in the original (unscaled) domain
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)

    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")


def test_auto_mpg():
    # Load sine dataset from CSV
    data_path = Path("data/auto_mpg.csv")

    data = np.genfromtxt(
        data_path, 
        delimiter=",", 
        skip_header=1, 
        usecols=(0, 1, 2, 3, 4, 5, 6, 7),
        missing_values="?",
        filling_values=0
    )

    y = data[:, 0:1] # MPG
    X = data[:, 1:8] # Όλα τα υπόλοιπα features

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Normalize input (X) and output (y) to [-1, 1]
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = x_scaler.fit_transform(X)

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = y_scaler.fit_transform(y).ravel()

    clustering = partial(fuzzymeans, partitions=[8, 8, 8, 8, 8, 8, 8])
    adaptieve_sigma = partial(compute_adaptive_sigma, n_neighbors=12)
    kernel = GaussianKernel(sigma_strategy=adaptieve_sigma)
    
    model = RBFModel(
        clustering_func=clustering,
        kernel_func=kernel,
        alpha=0.1
    )
    
    try:
        model.fit(X_scaled, y_scaled)
    except Exception as e:
        pytest.fail(f"Model fit failed with error: {e}")

    # Predict the sine signal using the trained model
    y_pred_scaled = model.predict(X_scaled)

    # Convert predictions back to the original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    print("y_pred shape:", y_pred.shape)
    
    # Compute error metrics in the original (unscaled) domain
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)

    print(f"MSE  : {mse:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")

def test_auto_mpg_sa():
    # Load sine dataset from CSV
    data_path = Path("data/auto_mpg.csv")

    data = np.genfromtxt(
        data_path, 
        delimiter=",", 
        skip_header=1, 
        usecols=(0, 1, 2, 3, 4, 5, 6, 7),
        missing_values="?",
        filling_values=0
    )

    y = data[:, 0:1] # MPG
    X = data[:, 1:8] # Όλα τα υπόλοιπα features

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Normalize input (X) and output (y) to [-1, 1]
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = x_scaler.fit_transform(X)

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = y_scaler.fit_transform(y).ravel()

    sa_optimizer = SARBFOptimizer()

    sa_optimizer.fit(X_scaled, y_scaled)

    print("Best mse score:", sa_optimizer.best_score_)