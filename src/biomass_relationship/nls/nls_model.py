"""
Apply NLS model to explore the relationship between nitrogen deposition and biomass

Model assumption: nitrogen_addition -> PCA_components -> biomass
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ModelFitResult:
    """NLS model fitting result"""
    params: np.ndarray  # 模型参数
    params_std: np.ndarray  # parameter standard error
    r2: float  # R² coefficient of determination
    rmse: float  # root mean squared error
    mae: float  # mean absolute error
    aic: float  # Akaike information criterion
    bic: float  # Bayesian information criterion
    residuals: np.ndarray  # residuals
    predictions: np.ndarray  # predictions
    convergence: bool  # convergence
    message: str  # fitting information
    param_names: List[str]  # parameter names


class NLSModel:
    """
    Base class for non-linear least squares models

    All specific models should inherit this class and implement the model_func method
    """

    def __init__(self, name: str = "NLS Model"):
        """
        Initialize NLS model

        Args:
            name: model name
        """
        self.name = name
        self.params = None
        self.params_std = None
        self.fit_result = None

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        """
        Model function - subclasses must implement this method

        Args:
            X: input feature matrix [nitrogen_add, PC1, PC2, ..., PCn]
            *params: model parameters

        Returns:
            predicted biomass values
        """
        raise NotImplementedError("Subclasses must implement the model_func method")

    def get_param_names(self) -> List[str]:
        """Get parameter names - subclasses should override this method"""
        return [f"param_{i}" for i in range(len(self.params))] if self.params is not None else []

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get initial parameter estimates - subclasses can override this method to provide better initial values

        Args:
            X: input feature matrix
            y: target variable (some models need it for initialization)

        Returns:
            initial parameter array
        """
        # Default implementation: use simple initial values
        n_params = self._get_n_params(X)
        # Use y to avoid unused parameter warning, but default implementation does not need it
        _ = y  
        return np.ones(n_params)

    def _get_n_params(self, X: np.ndarray) -> int:
        """Get number of parameters - subclasses should override this method"""
        return X.shape[1] + 1  # Default: intercept + one coefficient for each feature

    def _check_overfitting_risk(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Check overfitting risk based on parameter-to-sample ratio

        Args:
            X: input feature matrix
            y: target variable

        Returns:
            (risk_level, param_ratio) where risk_level is 'high', 'moderate', 'low', or 'minimal'
        """
        n_samples = len(y)
        n_params = self._get_n_params(X)
        param_ratio = n_params / n_samples

        logger.info(f"  Overfitting risk assessment:")
        logger.info(f"    Parameters: {n_params}, Samples: {n_samples}, Ratio: {param_ratio:.3f}")

        return risk_level, param_ratio

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            method: str = 'trf',
            max_nfev: int = 10000,
            alpha_l1: float = 0.0,
            alpha_l2: float = 0.0,
            check_overfitting: bool = True) -> ModelFitResult:
        """
        Fit model

        Args:
            X: input feature matrix [nitrogen_add, PC1, PC2, ..., PCn]
               shape: (n_samples, n_features)
            y: target variable (biomass)
               shape: (n_samples,)
            method: optimization method ('trf', 'dogbox', 'lm')
            max_nfev: maximum number of function evaluations
            alpha_l1: L1 regularization parameter (Lasso), default 0.0
                      Promotes sparsity in parameters
            alpha_l2: L2 regularization parameter (Ridge), default 0.0
                      Promotes small parameter values
            check_overfitting: whether to check overfitting risk, default True

        Returns:
            ModelFitResult object
        """
        logger.info(f"Starting model fitting: {self.name}")
        logger.info(f"  Sample size: {len(y)}, Features: {X.shape[1]}")

        # Check overfitting risk
        if check_overfitting:
            risk_level, _ = self._check_overfitting_risk(X, y)

            # Suggest regularization if not provided
            if (risk_level in ['high', 'moderate']) and (alpha_l1 == 0.0 and alpha_l2 == 0.0):
                logger.warning("  WARNING: Overfitting risk detected but no regularization applied!")
                logger.warning("  Suggestion: Use alpha_l1 > 0 (e.g., 0.01-0.1) or alpha_l2 > 0")

        # Get initial parameters
        p0 = self.get_initial_params(X, y)
        logger.info(f"  Initial parameters: {p0}")

        # Log regularization settings
        if alpha_l1 > 0 or alpha_l2 > 0:
            logger.info(f"  Regularization: L1 (α={alpha_l1}), L2 (α={alpha_l2})")

        try:
            # Use curve_fit or least_squares for non-linear least squares fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # If regularization is used, we need to use least_squares directly
                if alpha_l1 > 0 or alpha_l2 > 0:
                    from scipy.optimize import least_squares

                    def residual_func(params):
                        """Compute residuals with regularization"""
                        # Model residuals
                        y_pred = self.model_func(X.T, *params)
                        residuals = y - y_pred

                        # Add L2 (Ridge) penalty to residuals
                        if alpha_l2 > 0:
                            l2_penalty = np.sqrt(alpha_l2) * params
                            residuals = np.concatenate([residuals, l2_penalty])

                        # Add L1 (Lasso) penalty via soft thresholding in loss
                        # For L1, we modify the optimization but keep residuals for RSS
                        return residuals

                    # For L1 regularization, we use soft_l1 loss which approximates L1
                    loss_type = 'soft_l1' if alpha_l1 > 0 else 'linear'

                    result = least_squares(
                        fun=residual_func,
                        x0=p0,
                        method=method,
                        max_nfev=max_nfev,
                        loss=loss_type,
                        f_scale=alpha_l1 if alpha_l1 > 0 else 1.0
                    )

                    popt = result.x

                    # Compute covariance matrix approximation
                    try:
                        # Jacobian at solution
                        J = result.jac
                        # Covariance matrix: (J^T J)^-1 * var(residuals)
                        y_pred_final = self.model_func(X.T, *popt)
                        residuals_final = y - y_pred_final
                        var_residuals = np.var(residuals_final, ddof=len(popt))
                        pcov = np.linalg.inv(J.T @ J) * var_residuals
                    except (np.linalg.LinAlgError, ValueError):
                        pcov = np.full((len(popt), len(popt)), np.inf)

                else:
                    # Standard curve_fit without regularization
                    def model_wrapper(x, *p):
                        return self.model_func(x, *p)

                    popt, pcov = curve_fit(
                        f=model_wrapper,
                        xdata=X.T,  # curve_fit expects xdata to be (n_features, n_samples)
                        ydata=y,
                        p0=p0,
                        method=method,
                        maxfev=max_nfev,
                        full_output=False
                    )

            self.params = popt

            # Calculate parameter standard errors
            try:
                perr = np.sqrt(np.diag(pcov))
                self.params_std = perr
            except (ValueError, RuntimeError):
                self.params_std = np.full_like(popt, np.nan)
                logger.warning("Unable to compute parameter standard errors")

            # Calculate predicted values and residuals
            y_pred = self.predict(X)
            residuals = y - y_pred

            # Calculate evaluation metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            # Calculate information criteria
            n = len(y)
            k = len(popt)
            rss = np.sum(residuals**2)

            # Handle edge case where RSS is zero or very small
            epsilon = 1e-10
            if rss <= epsilon:
                logger.warning(f"RSS ({rss:.2e}) is zero or very small, possibly indicating perfect fit or numerical issues")
                logger.warning("Using adjusted RSS for AIC/BIC calculation to avoid log(0)")
                rss_adjusted = epsilon
            else:
                rss_adjusted = rss

            aic = n * np.log(rss_adjusted / n) + 2 * k
            bic = n * np.log(rss_adjusted / n) + k * np.log(n)

            # Create result object
            self.fit_result = ModelFitResult(
                params=popt,
                params_std=self.params_std,
                r2=r2,
                rmse=rmse,
                mae=mae,
                aic=aic,
                bic=bic,
                residuals=residuals,
                predictions=y_pred,
                convergence=True,
                message="Fitting successful",
                param_names=self.get_param_names()
            )

            logger.info(f"Model fitting completed: {self.name}")
            logger.info(f"  R² = {r2:.4f}")
            logger.info(f"  RMSE = {rmse:.4f}")
            logger.info(f"  AIC = {aic:.2f}")

            return self.fit_result

        except (RuntimeError, ValueError) as e:
            logger.error(f"Model fitting failed: {self.name} - {str(e)}")

            # Return failed result
            self.fit_result = ModelFitResult(
                params=p0,
                params_std=np.full_like(p0, np.nan),
                r2=np.nan,
                rmse=np.nan,
                mae=np.nan,
                aic=np.inf,
                bic=np.inf,
                residuals=np.full(len(y), np.nan),
                predictions=np.full(len(y), np.nan),
                convergence=False,
                message=f"Fitting failed: {str(e)}",
                param_names=self.get_param_names()
            )

            return self.fit_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict

        Args:
            X: input feature matrix

        Returns:
            predicted values array
        """
        if self.params is None:
            raise ValueError("Model not fitted yet, please call fit method first")

        return self.model_func(X.T, *self.params)

    def summary(self) -> str:
        """Generate model summary report"""
        if self.fit_result is None:
            return f"Model {self.name} not fitted yet"

        lines = []
        lines.append("=" * 70)
        lines.append(f"Model: {self.name}")
        lines.append("=" * 70)
        lines.append("")

        # Fitting statistics
        lines.append("Fitting Statistics:")
        lines.append(f"  Convergence: {'Success' if self.fit_result.convergence else 'Failed'}")
        if not np.isnan(self.fit_result.r2):
            lines.append(f"  R² = {self.fit_result.r2:.4f}")
            lines.append(f"  RMSE = {self.fit_result.rmse:.4f}")
            lines.append(f"  MAE = {self.fit_result.mae:.4f}")
            lines.append(f"  AIC = {self.fit_result.aic:.2f}")
            lines.append(f"  BIC = {self.fit_result.bic:.2f}")
        lines.append("")

        # Parameter estimates
        lines.append("Parameter estimates:")
        lines.append(f"{'Parameter':<20} {'Estimated value':<15} {'Standard error':<15}")
        lines.append("-" * 50)
        for name, val, std in zip(self.fit_result.param_names, 
                                   self.fit_result.params, 
                                   self.fit_result.params_std):
            lines.append(f"{name:<20} {val:>14.6f} {std:>14.6f}")

        lines.append("=" * 70)

        return "\n".join(lines)
