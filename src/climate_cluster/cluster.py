import logging
import numpy as np
from typing import Tuple, Dict, Any
from scipy.stats import chi2
from scipy.linalg import det, inv
import warnings


class GMMCluster:
    """
    Gaussian Mixture Model clustering with EM algorithm implementation
    Supports confidence-based classification and AIC model selection
    """

    def __init__(self, n_components: int, max_iter: int = 100, tol: float = 1e-6, 
                 random_state: int = 42, n_init: int = 10):
        """
        Initialize GMM clustering

        Args:
            n_components: Number of mixture components (K)
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance (epsilon in equation 9)
            random_state: Random seed for reproducibility
            n_init: Number of random initializations
        """
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

        # Model parameters
        self.pi = None          # Mixture weights (K,)
        self.mu = None          # Means (K, d)
        self.Sigma = None       # Covariances (K, d, d)

        # Training results
        self.is_fitted = False
        self.log_likelihood_history = []
        self.final_log_likelihood = None
        self.n_features = None
        self.n_samples = None

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize GMM parameters randomly

        Args:
            X: Input data (N, d)
        """
        N = X.shape[0]
        np.random.seed(self.random_state)

        # Initialize mixture weights uniformly
        self.pi = np.ones(self.K) / self.K

        # Initialize means by randomly selecting K data points
        indices = np.random.choice(N, self.K, replace=False)
        self.mu = X[indices].copy()

        # Initialize covariances as identity matrices scaled by data variance
        data_cov = np.cov(X.T)
        self.Sigma = np.array([data_cov for _ in range(self.K)])

    def _multivariate_gaussian(self, X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """
        Calculate multivariate gaussian probability density

        Args:
            X: Data points (N, d)
            mu: Mean vector (d,)
            Sigma: Covariance matrix (d, d)

        Returns:
            Probability densities (N,)
        """
        d = X.shape[1]

        # Handle numerical stability
        try:
            Sigma_inv = inv(Sigma)
            Sigma_det = det(Sigma)
        except np.linalg.LinAlgError:
            # Add small regularization for numerical stability
            Sigma_reg = Sigma + 1e-6 * np.eye(d)
            Sigma_inv = inv(Sigma_reg)
            Sigma_det = det(Sigma_reg)

        if Sigma_det <= 0:
            warnings.warn("Non-positive definite covariance matrix detected")
            Sigma_det = 1e-6

        # Calculate Mahalanobis distance
        diff = X - mu
        mahal_dist = np.sum((diff @ Sigma_inv) * diff, axis=1)

        # Calculate probability density
        normalization = 1.0 / np.sqrt((2 * np.pi) ** d * Sigma_det)
        prob = normalization * np.exp(-0.5 * mahal_dist)

        return prob

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: Calculate posterior probabilities gamma_nk (equation 7)

        Args:
            X: Input data (N, d)

        Returns:
            Posterior probabilities (N, K)
        """
        N = X.shape[0]
        gamma = np.zeros((N, self.K))

        # Calculate likelihood for each component
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * self._multivariate_gaussian(X, self.mu[k], self.Sigma[k])

        # Normalize to get posterior probabilities
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-10)  # Avoid division by zero
        gamma = gamma / gamma_sum

        return gamma

    def _m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """
        M-step: Update parameters (equation 8)

        Args:
            X: Input data (N, d)
            gamma: Posterior probabilities (N, K)
        """
        N, d = X.shape

        # Effective number of points assigned to each component
        N_k = np.sum(gamma, axis=0)
        N_k = np.maximum(N_k, 1e-10)  # Avoid division by zero

        # Update mixture weights
        self.pi = N_k / N

        # Update means
        self.mu = (gamma.T @ X) / N_k.reshape(-1, 1)

        # Update covariances
        for k in range(self.K):
            diff = X - self.mu[k]
            weighted_diff = gamma[:, k].reshape(-1, 1) * diff
            self.Sigma[k] = (weighted_diff.T @ diff) / N_k[k]

            # Add regularization for numerical stability
            self.Sigma[k] += 1e-6 * np.eye(d)

    def _calculate_log_likelihood(self, X: np.ndarray) -> float:
        """
        Calculate log likelihood of data

        Args:
            X: Input data (N, d)

        Returns:
            Log likelihood value
        """
        N = X.shape[0]
        log_likelihood = 0

        for n in range(N):
            likelihood = 0
            for k in range(self.K):
                likelihood += self.pi[k] * self._multivariate_gaussian(
                    X[n:n+1], self.mu[k], self.Sigma[k]
                )[0]

            if likelihood > 0:
                log_likelihood += np.log(likelihood)
            else:
                log_likelihood += -1e10  # Handle numerical issues

        return log_likelihood

    def fit(self, X: np.ndarray) -> 'GMMCluster':
        """
        Fit GMM using EM algorithm with multiple random initializations

        Args:
            X: Input data (N, d)

        Returns:
            Self for method chaining
        """
        self.n_samples, self.n_features = X.shape
        best_log_likelihood = -np.inf
        best_params = None

        for init in range(self.n_init):
            # Random initialization
            self.random_state += init  # Different seed for each initialization
            self._initialize_parameters(X)

            log_likelihood_history = []
            prev_log_likelihood = -np.inf

            # EM iterations
            for iteration in range(self.max_iter):
                # E-step
                gamma = self._e_step(X)

                # M-step
                self._m_step(X, gamma)

                # Calculate log likelihood
                current_log_likelihood = self._calculate_log_likelihood(X)
                log_likelihood_history.append(current_log_likelihood)

                # Check convergence (equation 9)
                if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                    print(f"Converged at iteration {iteration + 1} (init {init + 1})")
                    break

                prev_log_likelihood = current_log_likelihood

            # Keep best initialization
            if current_log_likelihood > best_log_likelihood:
                best_log_likelihood = current_log_likelihood
                best_params = {
                    'pi': self.pi.copy(),
                    'mu': self.mu.copy(),
                    'Sigma': self.Sigma.copy(),
                    'log_likelihood_history': log_likelihood_history
                }

        # Set best parameters
        self.pi = best_params['pi']
        self.mu = best_params['mu']
        self.Sigma = best_params['Sigma']
        self.log_likelihood_history = best_params['log_likelihood_history']
        self.final_log_likelihood = best_log_likelihood
        self.is_fitted = True

        print(f"Final log likelihood: {self.final_log_likelihood:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities (soft clustering)

        Args:
            X: Input data (N, d)

        Returns:
            Posterior probabilities (N, K)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self._e_step(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels (hard clustering)

        Args:
            X: Input data (N, d)

        Returns:
            Cluster labels (N,)
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_with_confidence(self, X: np.ndarray, confidence_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence threshold

        Args:
            X: Input data (N, d)
            confidence_threshold: Minimum confidence for classification (eta)

        Returns:
            (labels, confident_mask): Labels and mask indicating confident predictions
        """
        probabilities = self.predict_proba(X)
        labels = np.argmax(probabilities, axis=1)
        max_probabilities = np.max(probabilities, axis=1)
        confident_mask = max_probabilities >= confidence_threshold

        return labels, confident_mask

    def mahalanobis_distance(self, X: np.ndarray, cluster_id: int) -> np.ndarray:
        """
        Calculate Mahalanobis distance to specific cluster (equation 10)

        Args:
            X: Input data (N, d)
            cluster_id: Target cluster index

        Returns:
            Mahalanobis distances (N,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating distances")

        if cluster_id >= self.K:
            raise ValueError(f"Cluster ID must be < {self.K}")

        diff = X - self.mu[cluster_id]
        Sigma_inv = inv(self.Sigma[cluster_id])

        # Calculate squared Mahalanobis distance
        distances_squared = np.sum((diff @ Sigma_inv) * diff, axis=1)

        # Return distance (not squared)
        return np.sqrt(distances_squared)

    def get_chi_square_threshold(self, alpha: float = 0.05) -> float:
        """
        Calculate chi-square threshold for outlier detection (equation 11)

        Args:
            alpha: Significance level

        Returns:
            Chi-square threshold (tau)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating threshold")

        # Degrees of freedom equals number of features
        dof = self.n_features
        chi_square_value = chi2.ppf(1 - alpha, dof)

        return np.sqrt(chi_square_value)

    def calculate_aic(self) -> float:
        """
        Calculate Akaike Information Criterion (equation 12)

        Returns:
            AIC value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating AIC")

        # Number of parameters: K-1 mixture weights + K*d means + K*d*(d+1)/2 covariance parameters
        n_params = (self.K - 1) + self.K * self.n_features + self.K * self.n_features * (self.n_features + 1) // 2

        aic = -2 * self.final_log_likelihood + 2 * n_params

        return aic

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information

        Returns:
            Dictionary containing model parameters and statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting info")

        return {
            'n_components': self.K,
            'n_features': self.n_features,
            'n_samples': self.n_samples,
            'mixture_weights': self.pi,
            'means': self.mu,
            'covariances': self.Sigma,
            'log_likelihood': self.final_log_likelihood,
            'aic': self.calculate_aic(),
            'converged_iterations': len(self.log_likelihood_history)
        }


def select_optimal_k_with_aic(X: np.ndarray, k_range: range, **gmm_params) -> Tuple[int, Dict[int, float]]:
    """
    Select optimal number of components using AIC criterion

    Args:
        X: Input data (N, d)
        k_range: Range of K values to test
        **gmm_params: Additional parameters for GMMCluster

    Returns:
        (optimal_k, aic_scores): Best K and AIC scores for all tested K values
    """
    aic_scores = {}

    for k in k_range:
        print(f"Testing K = {k}...")
        gmm = GMMCluster(n_components=k, **gmm_params)
        gmm.fit(X)
        aic_scores[k] = gmm.calculate_aic()
        print(f"K = {k}, AIC = {aic_scores[k]:.4f}")

    optimal_k = min(aic_scores.keys(), key=lambda k: aic_scores[k])
    print(f"Optimal K = {optimal_k} (AIC = {aic_scores[optimal_k]:.4f})")

    return optimal_k, aic_scores