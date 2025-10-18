import os
import logging
import warnings
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.stats import chi2
from scipy.linalg import det, inv
import pandas as pd

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

logger = logging.getLogger(__name__)

class ClusterResult:
    """
    Cluster result class
    """
    def __init__(self, labels: np.ndarray, confident_mask: np.ndarray, quality_metrics: Optional[Dict[str, Any]] = None):
        self.labels = labels
        self.confident_mask = confident_mask
        self.quality_metrics = quality_metrics

    def save(self, output_path: str):
        """
        Save cluster result to output directory as csv and print quality metrics
        """
        # Save clustering labels and confidence mask
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame({'labels': self.labels, "confident_mask": self.confident_mask}).to_csv(output_path, index=True)
        logger.info(f"Cluster results saved to: {output_path}")

        # Print quality metrics if available
        if self.quality_metrics is not None:
            logger.info("=" * 60)
            logger.info("CLUSTER QUALITY METRICS")
            logger.info("=" * 60)

            # Print cluster sizes
            if 'cluster_sizes' in self.quality_metrics:
                sizes = self.quality_metrics['cluster_sizes']
                logger.info(f"Cluster sizes: {sizes}")
                for k, size in enumerate(sizes):
                    logger.info(f"  Cluster {k}: {size} samples")

            # Print covariance determinants (compactness indicator)
            if 'covariance_determinants' in self.quality_metrics:
                dets = self.quality_metrics['covariance_determinants']
                logger.info(f"\nCovariance determinants (lower = more compact):")
                for k, det_val in enumerate(dets):
                    logger.info(f"  Cluster {k}: {det_val:.6f}")

            # Print mean Mahalanobis distances (within-cluster dispersion)
            if 'mean_mahalanobis_distances' in self.quality_metrics:
                dists = self.quality_metrics['mean_mahalanobis_distances']
                logger.info(f"\nMean Mahalanobis distances (lower = more compact):")
                for k, dist in enumerate(dists):
                    logger.info(f"  Cluster {k}: {dist:.4f}")

            # Print variance traces
            if 'cluster_std_traces' in self.quality_metrics:
                traces = self.quality_metrics['cluster_std_traces']
                logger.info(f"\nCovariance traces (total variance):")
                for k, trace in enumerate(traces):
                    logger.info(f"  Cluster {k}: {trace:.4f}")

            # Print separation metrics if available
            if 'cluster_separation' in self.quality_metrics:
                sep = self.quality_metrics['cluster_separation']
                logger.info(f"\nMinimum cluster separation ratio: {sep:.4f}")

            logger.info("=" * 60)

            # Save quality metrics to a separate file
            metrics_path = output_path.replace('.csv', '_quality_metrics.csv')
            metrics_df = pd.DataFrame({
                'cluster_id': range(len(self.quality_metrics.get('cluster_sizes', []))),
                'cluster_size': self.quality_metrics.get('cluster_sizes', []),
                'covariance_determinant': self.quality_metrics.get('covariance_determinants', []),
                'mean_mahalanobis_distance': self.quality_metrics.get('mean_mahalanobis_distances', []),
                'covariance_trace': self.quality_metrics.get('cluster_std_traces', [])
            })
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Quality metrics saved to: {metrics_path}")

    def exclude(self, data: np.ndarray) -> np.ndarray:
        """
        Exclude data that is not confident

        Args:
            data: Input data (N, d)

        Returns:
            Excluded data (N, d)
        """
        return data[~self.confident_mask] # return data that is not confident

class GMMCluster:
    """
    Gaussian Mixture Model clustering with EM algorithm implementation
    Supports confidence-based classification and AIC model selection
    """

    def __init__(self, n_components: int, confidence: float, max_iter: int = 100, tol: float = 1e-6,
                 random_state: int = 42, n_init: int = 10, 
                 max_covariance_det: Optional[float] = None,
                 min_cluster_separation: Optional[float] = None,
                 max_mean_mahalanobis: Optional[float] = None):
        """
        Initialize GMM clustering

        Args:
            n_components: Number of mixture components (K)
            confidence: Confidence threshold for classification
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance (epsilon in equation 9)
            random_state: Random seed for reproducibility
            n_init: Number of random initializations
            max_covariance_det: Maximum covariance determinant threshold for cluster compactness.
                               Lower values require more compact clusters. None means no check.
            min_cluster_separation: Minimum cluster separation ratio (center distance / avg std).
                                   Higher values require greater inter-cluster distances. None means no check.
            max_mean_mahalanobis: Maximum average Mahalanobis distance within each cluster.
                                 Lower values require more compact clusters. None means no check.
        """
        self.k = n_components
        self.confidence = confidence
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

        # Strictness parameters
        self.max_covariance_det = max_covariance_det
        self.min_cluster_separation = min_cluster_separation
        self.max_mean_mahalanobis = max_mean_mahalanobis

        # Model parameters
        self.pi = None          # Mixture weights (K,)
        self.mu = None          # Means (K, d)
        self.sigma = None       # Covariances (K, d, d)

        # Training results
        self.is_fitted = False
        self.log_likelihood_history = []
        self.final_log_likelihood = None
        self.n_features = None
        self.n_samples = None
        self.cluster_quality_metrics = None

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize GMM parameters randomly

        Args:
            X: Input data (N, d)
        """
        N = X.shape[0]
        np.random.seed(self.random_state)

        # Initialize mixture weights uniformly
        self.pi = np.ones(self.k) / self.k

        # Initialize means by randomly selecting K data points
        indices = np.random.choice(N, self.k, replace=False)
        self.mu = X[indices].copy()

        # Initialize covariances as identity matrices scaled by data variance
        data_cov = np.cov(X.T)
        self.sigma = np.array([data_cov for _ in range(self.k)])

    def _multivariate_gaussian(self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate multivariate gaussian probability density

        Args:
            X: Data points (N, d)
            mu: Mean vector (d,)
            sigma: Covariance matrix (d, d)

        Returns:
            Probability densities (N,)
        """
        d = X.shape[1]

        # Handle numerical stability
        try:
            sigma_inv = inv(sigma)
            Sigma_det = det(sigma)
        except np.linalg.LinAlgError:
            # Add small regularization for numerical stability
            Sigma_reg = sigma + 1e-6 * np.eye(d)
            sigma_inv = inv(Sigma_reg)
            Sigma_det = det(Sigma_reg)

        if Sigma_det <= 0:
            warnings.warn("Non-positive definite covariance matrix detected")
            Sigma_det = 1e-6

        # Calculate Mahalanobis distance
        diff = X - mu
        mahal_dist = np.sum((diff @ sigma_inv) * diff, axis=1)

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
        gamma = np.zeros((N, self.k))

        # Calculate likelihood for each component
        for k in range(self.k):
            gamma[:, k] = self.pi[k] * self._multivariate_gaussian(X, self.mu[k], self.sigma[k])

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
        for k in range(self.k):
            diff = X - self.mu[k]
            weighted_diff = gamma[:, k].reshape(-1, 1) * diff
            self.sigma[k] = (weighted_diff.T @ diff) / N_k[k]

            # Add regularization for numerical stability
            self.sigma[k] += 1e-6 * np.eye(d)

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
            for k in range(self.k):
                likelihood += self.pi[k] * self._multivariate_gaussian(
                    X[n:n+1], self.mu[k], self.sigma[k]
                )[0]

            if likelihood > 0:
                log_likelihood += np.log(likelihood)
            else:
                log_likelihood += -1e10  # Handle numerical issues

        return log_likelihood

    def _calculate_cluster_compactness(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Calculate compactness metrics for each cluster

        Args:
            X: Input data (N, d)

        Returns:
            Dictionary containing various compactness metrics
        """
        # Get cluster assignments for each sample
        gamma = self._e_step(X)
        labels = np.argmax(gamma, axis=1)

        metrics = {
            'covariance_determinants': [],
            'mean_mahalanobis_distances': [],
            'cluster_sizes': [],
            'cluster_std_traces': []
        }

        for k in range(self.k):
            # Covariance matrix determinant (smaller means more compact)
            cov_det = det(self.sigma[k])
            metrics['covariance_determinants'].append(cov_det)

            # Get samples belonging to this cluster
            cluster_mask = labels == k
            cluster_points = X[cluster_mask]
            metrics['cluster_sizes'].append(len(cluster_points))

            if len(cluster_points) > 0:
                # Calculate average Mahalanobis distance to cluster center
                distances = self.mahalanobis_distance(cluster_points, k)
                mean_dist = np.mean(distances)
                metrics['mean_mahalanobis_distances'].append(mean_dist)

                # Covariance matrix trace (sum of diagonal elements, represents total variance)
                std_trace = np.trace(self.sigma[k])
                metrics['cluster_std_traces'].append(std_trace)
            else:
                metrics['mean_mahalanobis_distances'].append(np.inf)
                metrics['cluster_std_traces'].append(np.inf)

        return metrics

    def _calculate_cluster_separation(self) -> float:
        """
        Calculate inter-cluster separation (minimum center distance to average std ratio)

        Returns:
            Cluster separation ratio, higher values mean greater inter-cluster distances
        """
        min_separation = np.inf

        for i in range(self.k):
            for j in range(i + 1, self.k):
                # Calculate Euclidean distance between centers
                center_dist = np.linalg.norm(self.mu[i] - self.mu[j])

                # Calculate average standard deviation (using square root of covariance trace)
                avg_std = (np.sqrt(np.trace(self.sigma[i])) + np.sqrt(np.trace(self.sigma[j]))) / 2

                # Separation ratio = distance / average std
                separation = center_dist / (avg_std + 1e-10)

                if separation < min_separation:
                    min_separation = separation

        return min_separation

    def _validate_cluster_quality(self, X: np.ndarray) -> Tuple[bool, str]:
        """
        Validate cluster quality against strictness requirements

        Args:
            X: Input data (N, d)

        Returns:
            (is_valid, message): Whether validation passed and detailed information
        """
        metrics = self._calculate_cluster_compactness(X)

        # Add cluster separation to metrics
        if self.min_cluster_separation is not None:
            separation = self._calculate_cluster_separation()
            metrics['cluster_separation'] = separation

        self.cluster_quality_metrics = metrics

        # Check 1: Covariance determinant (compactness)
        if self.max_covariance_det is not None:
            for k, cov_det in enumerate(metrics['covariance_determinants']):
                if cov_det > self.max_covariance_det:
                    msg = (f"Cluster {k} covariance determinant {cov_det:.4f} exceeds threshold {self.max_covariance_det:.4f}, "
                          f"cluster is not compact enough")
                    logger.warning(msg)
                    return False, msg

        # Check 2: Mean Mahalanobis distance within clusters
        if self.max_mean_mahalanobis is not None:
            for k, mean_dist in enumerate(metrics['mean_mahalanobis_distances']):
                if mean_dist > self.max_mean_mahalanobis:
                    msg = (f"Cluster {k} mean Mahalanobis distance {mean_dist:.4f} exceeds threshold {self.max_mean_mahalanobis:.4f}, "
                          f"within-cluster dispersion is too high")
                    logger.warning(msg)
                    return False, msg

        # Check 3: Inter-cluster separation
        if self.min_cluster_separation is not None:
            separation = metrics['cluster_separation']
            if separation < self.min_cluster_separation:
                msg = (f"Cluster separation {separation:.4f} is below threshold {self.min_cluster_separation:.4f}, "
                      f"inter-cluster distance is not sufficient")
                logger.warning(msg)
                return False, msg

        logger.info("Cluster quality validation passed")
        logger.info(f"  Covariance determinants: {metrics['covariance_determinants']}")
        logger.info(f"  Mean Mahalanobis distances: {metrics['mean_mahalanobis_distances']}")
        if self.min_cluster_separation is not None:
            logger.info(f"  Cluster separation: {separation:.4f}")

        return True, "Cluster quality meets all requirements"

    def fit(self, X: np.ndarray, validate_quality: bool = True) -> 'GMMCluster':
        """
        Fit GMM using EM algorithm with multiple random initializations

        Args:
            X: Input data (N, d)
            validate_quality: Whether to validate cluster quality after training

        Returns:
            Self for method chaining

        Raises:
            ValueError: If cluster quality does not meet strictness requirements
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
                    logger.info(f"Converged at iteration {iteration + 1} (init {init + 1})")
                    break

                prev_log_likelihood = current_log_likelihood

            # Keep best initialization
            if current_log_likelihood > best_log_likelihood:
                best_log_likelihood = current_log_likelihood
                best_params = {
                    'pi': self.pi.copy(),
                    'mu': self.mu.copy(),
                    'sigma': self.sigma.copy(),
                    'log_likelihood_history': log_likelihood_history
                }

        # Set best parameters
        self.pi = best_params['pi']
        self.mu = best_params['mu']
        self.sigma = best_params['sigma']
        self.log_likelihood_history = best_params['log_likelihood_history']
        self.final_log_likelihood = best_log_likelihood
        self.is_fitted = True

        logger.info(f"Final log likelihood: {self.final_log_likelihood:.4f}")

        # Validate cluster quality
        if validate_quality:
            is_valid, message = self._validate_cluster_quality(X)
            if not is_valid:
                self.is_fitted = False  # Mark as not fitted
                raise ValueError(f"Cluster quality does not meet requirements: {message}")
        else:
            # Calculate quality metrics even if not validating (for reporting)
            self.cluster_quality_metrics = self._calculate_cluster_compactness(X)
            if self.min_cluster_separation is not None:
                self.cluster_quality_metrics['cluster_separation'] = self._calculate_cluster_separation()

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

    def predict(self, X: np.ndarray) -> ClusterResult:
        """
        Predict cluster labels (hard clustering)

        Args:
            X: Input data (N, d)

        Returns:
            ClusterResult with labels, confidence mask, and quality metrics
        """
        probabilities = self.predict_proba(X)
        labels = np.argmax(probabilities, axis=1)
        confident_mask = np.max(probabilities, axis=1) >= self.confidence
        return ClusterResult(labels, confident_mask, self.cluster_quality_metrics)

    def predict_with_confidence(self, X: np.ndarray) -> ClusterResult:
        """
        Predict with confidence threshold

        Args:
            X: Input data (N, d)

        Returns:
            ClusterResult with labels, confidence mask, and quality metrics
        """
        probabilities = self.predict_proba(X)
        labels = np.argmax(probabilities, axis=1)
        max_probabilities = np.max(probabilities, axis=1)
        confident_mask = max_probabilities >= self.confidence

        return ClusterResult(labels, confident_mask, self.cluster_quality_metrics)

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

        if cluster_id >= self.k:
            raise ValueError(f"Cluster ID must be < {self.k}")

        # Check for NaN/Inf in input data
        if not np.all(np.isfinite(X)):
            raise ValueError("Input data contains NaN or Inf values")

        # Check for NaN/Inf in cluster parameters
        if not np.all(np.isfinite(self.mu[cluster_id])):
            raise ValueError(f"Cluster {cluster_id} mean contains NaN or Inf values")
        if not np.all(np.isfinite(self.sigma[cluster_id])):
            raise ValueError(f"Cluster {cluster_id} covariance contains NaN or Inf values")

        diff = X - self.mu[cluster_id]

        try:
            sigma_inv = inv(self.sigma[cluster_id])
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Cannot invert covariance matrix for cluster {cluster_id}: {e}")

        # Calculate squared Mahalanobis distance
        distances_squared = np.sum((diff @ sigma_inv) * diff, axis=1)

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
        n_params = (self.k - 1) + self.k * self.n_features + self.k * self.n_features * (self.n_features + 1) // 2

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

        info = {
            'n_components': self.k,
            'n_features': self.n_features,
            'n_samples': self.n_samples,
            'mixture_weights': self.pi,
            'means': self.mu,
            'covariances': self.sigma,
            'log_likelihood': self.final_log_likelihood,
            'aic': self.calculate_aic(),
            'converged_iterations': len(self.log_likelihood_history)
        }

        # Add cluster quality metrics if calculated
        if self.cluster_quality_metrics is not None:
            info['cluster_quality_metrics'] = self.cluster_quality_metrics

        return info

def select_optimal_k_with_aic(X: np.ndarray, k_range: range, use_mpi: bool = False, **gmm_params) -> Tuple[int, Dict[int, float]]:
    """
    Select optimal number of components using AIC criterion

    Args:
        X: Input data (N, d)
        k_range: Range of K values to test
        use_mpi: Whether to use MPI parallelization. Defaults to False.
        **gmm_params: Additional parameters for GMMCluster

    Returns:
        (optimal_k, aic_scores): Best K and AIC scores for all tested K values
    """
    if use_mpi and MPI_AVAILABLE:
        return _select_optimal_k_with_aic_mpi(X, k_range, **gmm_params)
    else:
        if use_mpi and not MPI_AVAILABLE:
            logger.warning("MPI not available, falling back to serial execution")
        return _select_optimal_k_with_aic_serial(X, k_range, **gmm_params)


def _select_optimal_k_with_aic_serial(X: np.ndarray, k_range: range, **gmm_params) -> Tuple[int, Dict[int, float]]:
    """
    Serial version of optimal K selection using AIC criterion

    Args:
        X: Input data (N, d)
        k_range: Range of K values to test
        **gmm_params: Additional parameters for GMMCluster

    Returns:
        (optimal_k, aic_scores): Best K and AIC scores for all tested K values
    """
    aic_scores = {}

    for k in k_range:
        logger.info(f"Testing K = {k}...")
        gmm = GMMCluster(n_components=k, **gmm_params)
        gmm.fit(X)
        aic_scores[k] = gmm.calculate_aic()
        logger.info(f"K = {k}, AIC = {aic_scores[k]:.4f}")

    optimal_k = min(aic_scores.keys(), key=lambda k: aic_scores[k])
    logger.info(f"Optimal K = {optimal_k} (AIC = {aic_scores[optimal_k]:.4f})")

    return optimal_k, aic_scores


def _select_optimal_k_with_aic_mpi(X: np.ndarray, k_range: range, **gmm_params) -> Tuple[int, Dict[int, float]]:
    """
    MPI parallel version of optimal K selection using AIC criterion

    Args:
        X: Input data (N, d)
        k_range: Range of K values to test
        **gmm_params: Additional parameters for GMMCluster

    Returns:
        (optimal_k, aic_scores): Best K and AIC scores for all tested K values

    Notes:
        - Each MPI process will compute AIC for a subset of K values
        - Results are gathered to rank 0 for final selection
        - All processes will receive the same optimal_k and aic_scores
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Convert range to list for easier distribution
    k_list = list(k_range)

    # Distribute K values across processes
    # Each process gets approximately len(k_list)/size values
    k_per_process = len(k_list) // size
    remainder = len(k_list) % size

    # Calculate start and end indices for this process
    if rank < remainder:
        start_idx = rank * (k_per_process + 1)
        end_idx = start_idx + k_per_process + 1
    else:
        start_idx = rank * k_per_process + remainder
        end_idx = start_idx + k_per_process

    local_k_values = k_list[start_idx:end_idx]

    if rank == 0:
        logger.info(f"MPI parallel execution with {size} processes")
        logger.info(f"Total K values to test: {len(k_list)}")

    # Each process computes AIC for its assigned K values
    local_aic_scores = {}
    for k in local_k_values:
        logger.info(f"[Rank {rank}] Testing K = {k}...")
        try:
            gmm = GMMCluster(n_components=k, **gmm_params)
            gmm.fit(X)
            aic = gmm.calculate_aic()
            local_aic_scores[k] = aic
            logger.info(f"[Rank {rank}] K = {k}, AIC = {aic:.4f}")
        except Exception as e:
            logger.error(f"[Rank {rank}] Error computing K = {k}: {e}")
            local_aic_scores[k] = np.inf  # Use infinity for failed computations

    # Gather all results to rank 0
    all_aic_scores = comm.gather(local_aic_scores, root=0)

    # Rank 0 combines results and finds optimal K
    if rank == 0:
        # Merge all dictionaries
        aic_scores = {}
        for scores_dict in all_aic_scores:
            aic_scores.update(scores_dict)

        # Find optimal K
        valid_scores = {k: v for k, v in aic_scores.items() if np.isfinite(v)}
        if not valid_scores:
            raise ValueError("All K values resulted in invalid AIC scores")

        optimal_k = min(valid_scores.keys(), key=lambda k: valid_scores[k])
        logger.info("=" * 60)
        logger.info(f"Optimal K = {optimal_k} (AIC = {aic_scores[optimal_k]:.4f})")
        logger.info("=" * 60)

        result = (optimal_k, aic_scores)
    else:
        result = None

    # Broadcast result to all processes
    result = comm.bcast(result, root=0)

    return result
