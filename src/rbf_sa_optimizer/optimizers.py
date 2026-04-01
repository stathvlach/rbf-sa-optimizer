import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils.validation import check_X_y, check_array
from .models import RBFModel
from .kernels import GaussianKernel
from .utils import fuzzymeans, compute_adaptive_sigma
from functools import partial
from sklearn.metrics import mean_squared_error

class SARBFOptimizer(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        k=1.0,
        t_start=1.0,
        t_stop=0.001,
        t_scale=1.0,
        generations=120,
        partition_min=2,
        partition_max=20,
        neighbor_min=0.1,
        neighbor_max=0.9,
        is_melting=True,
        alpha=0.01
    ):
        self.k = k
        self.t_start = t_start
        self.t_stop = t_stop
        self.t_scale = t_scale
        self.generations = generations
        self.partition_min = partition_min
        self.partition_max = partition_max
        self.neighbor_min = neighbor_min
        self.neighbor_max = neighbor_max
        self.is_melting = is_melting
        self.alpha = alpha

    def _create_schedules(self):
            n = np.arange(1, self.generations + 1)
            
            if self.generations <= 1:
                self.t_schedule_ = np.array([self.t_start])
                g = np.array([0.1])
            else:
                ratio = self.t_stop / self.t_start
                factor = 10 ** (np.log10(ratio) / (self.generations - 1))
                self.t_schedule_ = self.t_start * (factor ** (n - 1))
                
                g = 0.1 * (((-2 * np.linspace(self.t_start, self.t_stop, self.generations) + 
                            (self.t_stop + self.t_start)) * self.t_scale) / (self.t_stop - self.t_start))
            
            if self.is_melting:
                self.partition_schedule_ = self._melting_logic(g, "partition")
                self.neighbor_schedule_ = self._melting_logic(g, "neighbor")
            else:
                self.partition_schedule_ = np.ones(self.generations) * 30
                self.neighbor_schedule_ = np.ones(self.generations) * 0.5

    def _melting_logic(self, g, mode):
        if mode == "partition":
            return self.partition_max * 0.2 * np.exp(-g) + 1
        else:
            return 0.3 * np.exp(-g) + 0.01

    def _generate_neighbor(self, current_s, m):
        new_s = current_s.copy()
        n_features = len(current_s) - 1
        
        p_noise = np.random.randn(n_features) * self.partition_schedule_[m]
        new_s[:-1] = np.clip(np.round(new_s[:-1] + p_noise), 
                             self.partition_min, self.partition_max)
        
        n_noise = np.random.randn() * self.neighbor_schedule_[m]
        new_s[-1] = np.clip(new_s[-1] + n_noise, 
                            self.neighbor_min, self.neighbor_max)
        return new_s

    def _evaluate_solution(self, s, X, y):
        parts = s[:-1].astype(int)
        neighbor_ratio = s[-1]
        
        clustering = partial(fuzzymeans, partitions=parts)
        centers = clustering(X)

        n_centers = centers.shape[0]
        n_neighbors = int(np.round(neighbor_ratio * n_centers))
        n_neighbors = max(2, min(n_neighbors, n_centers - 1))

        adaptieve_sigma = partial(compute_adaptive_sigma, n_neighbors=n_neighbors)
        kernel = GaussianKernel(sigma_strategy=adaptieve_sigma)
        
        model = RBFModel(
            clustering_func=clustering,
            kernel_func=kernel,
            alpha=0.01
        )
        
        return model.fit(X, y)    

    def fit(self, X, y, X_val=None, y_val=None):
        """
        """
        X, y = check_X_y(X, y)
        if X_val is not None:
            X_val, y_val = check_array(X_val), np.asarray(y_val)
        else:
            X_val, y_val = X, y

        self.io_dimension_ = X.shape[1]
        
        self._create_schedules()
        
        self.best_params_ = None
        self.best_score_ = float('inf')
        self.best_estimator_ = None
        
        curr_s = np.zeros(self.io_dimension_ + 1)
        curr_s[:-1] = np.random.randint(self.partition_min, self.partition_max + 1, self.io_dimension_)
        curr_s[-1] = np.random.uniform(self.neighbor_min, self.neighbor_max)
        
        initial_est = self._evaluate_solution(curr_s, X, y)
        curr_err = mean_squared_error(y_val, initial_est.predict(X_val))
        
        self.best_score_ = curr_err
        self.best_estimator_ = initial_est
        self.best_params_ = curr_s.copy()
        
        y_norm = np.mean(np.abs(y)) if np.mean(np.abs(y)) != 0 else 1.0

        for m in range(self.generations):
            cand_s = self._generate_neighbor(curr_s, m)
            
            try:
                cand_est = self._evaluate_solution(cand_s, X, y)
                cand_err = mean_squared_error(y_val, cand_est.predict(X_val))
            except Exception:
                continue
            
            diff = cand_err - curr_err
            
            # Boltzmann Distribution - exponential_term = -(ΔE / y_norm) / (k * T)
            prob = np.exp(-(diff / y_norm) / (self.k * self.t_schedule_[m]))
            
            # Metropolis citerion
            if diff < 0 or np.random.rand() <= prob:
                curr_s = cand_s
                curr_err = cand_err
                
                if curr_err < self.best_score_:
                    self.best_score_ = curr_err
                    self.best_estimator_ = cand_est
                    self.best_params_ = curr_s.copy()
        
        return self
