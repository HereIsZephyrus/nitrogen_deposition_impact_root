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

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            method: str = 'trf',
            max_nfev: int = 10000) -> ModelFitResult:
        """
        Fit model

        Args:
            X: input feature matrix [nitrogen_add, PC1, PC2, ..., PCn]
               shape: (n_samples, n_features)
            y: target variable (biomass)
               shape: (n_samples,)
            method: optimization method ('trf', 'dogbox', 'lm')
            max_nfev: maximum number of function evaluations

        Returns:
            ModelFitResult object
        """
        logger.info(f"Starting model fitting: {self.name}")
        logger.info(f"  Sample size: {len(y)}, Features: {X.shape[1]}")

        # Get initial parameters
        p0 = self.get_initial_params(X, y)
        logger.info(f"  Initial parameters: {p0}")

        try:
            # Use curve_fit for non-linear least squares fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

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
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)

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


class LinearModel(NLSModel):
    """
    Linear model (OLS): simple linear regression as baseline reference

    Model form:
    biomass = β0 + β_N * N + Σ(βi * PCi)

    Where:
    - β0: intercept
    - β_N: coefficient for nitrogen addition
    - βi: coefficients for PCA components
    - N: nitrogen addition
    - PCi: i-th principal component

    This is the simplest model using ordinary least squares (OLS) as a baseline for comparison.
    """

    def __init__(self):
        super().__init__("Linear Model (OLS)")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        """
        X: shape (n_features, n_samples)
           X[0, :] = nitrogen_add
           X[1:, :] = PC components
        """
        nitrogen = X[0, :]
        pcs = X[1:, :]

        # Parameter allocation
        # params = [β0, β_N, β1, ..., βn]
        beta0 = params[0]
        beta_n = params[1]
        betas = np.array(params[2:])

        # Calculate predicted values
        # biomass = β0 + β_N * N + Σ(βi * PCi)
        pred = beta0 + beta_n * nitrogen
        pred += np.sum(betas[:, np.newaxis] * pcs, axis=0)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1  # Subtract nitrogen column
        return 2 + n_pcs  # β0 + β_N + βs

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_pcs = len(self.params) - 2

        names = ['β0_intercept', 'β_N']
        names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Use simple linear regression to get initial parameter estimates"""
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))

        # Use mean of y as intercept initial value
        p0[0] = np.mean(y)
        # Initial value for nitrogen coefficient
        p0[1] = 0.0
        # Initial values for PC coefficients
        p0[2:] = 0.0

        return p0


class AdditiveModel(NLSModel):
    """
    Additive model: nitrogen deposition through the additive effect of PCA components on biomass

    Model form:
    biomass = β0 + Σ(βi * PCi) + Σ(γi * N * PCi)

    Where:
    - β0: intercept
    - βi: main effect of PCA components
    - γi: interaction effect of nitrogen deposition and PCA components
    - N: nitrogen addition
    - PCi: i-th principal component
    """

    def __init__(self):
        super().__init__("Additive Interaction Model")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        """
        X: shape (n_features, n_samples)
           X[0, :] = nitrogen_add
           X[1:, :] = PC components
        """
        nitrogen = X[0, :]
        pcs = X[1:, :]
        n_pcs = pcs.shape[0]

        # Parameter allocation
        # params = [β0, β1, ..., βn, γ1, ..., γn]
        beta0 = params[0]
        betas = np.array(params[1:n_pcs+1])
        gammas = np.array(params[n_pcs+1:])

        # Calculate predicted values
        # biomass = β0 + Σ(βi * PCi) + Σ(γi * N * PCi)
        pred = beta0
        pred += np.sum(betas[:, np.newaxis] * pcs, axis=0)
        pred += np.sum(gammas[:, np.newaxis] * nitrogen[np.newaxis, :] * pcs, axis=0)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1  # Subtract nitrogen column
        return 1 + n_pcs + n_pcs  # β0 + βs + γs

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_total = len(self.params)
        n_pcs = (n_total - 1) // 2

        names = ['β0_intercept']
        names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
        names += [f'γ{i+1}_N×PC{i+1}' for i in range(n_pcs)]
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))
        p0[0] = np.mean(y)  # Initial value of intercept is the mean of y
        p0[1:n_pcs+1] = 0.1  # Initial value of main effect
        p0[n_pcs+1:] = 0.01  # Initial value of interaction effect (smaller)
        return p0


class MultiplicativeModel(NLSModel):
    """
    Multiplicative model: nitrogen deposition through the multiplicative effect of PCA components on biomass

    Model form:
    biomass = β0 * exp(Σ(βi * PCi) + Σ(γi * N * PCi))

    Or simplified as:
    biomass = β0 * Π(exp(βi * PCi)) * Π(exp(γi * N * PCi))
    """

    def __init__(self):
        super().__init__("Multiplicative Interaction Model")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        nitrogen = X[0, :]
        pcs = X[1:, :]

        beta0 = params[0]
        n_pcs = pcs.shape[0]
        betas = np.array(params[1:n_pcs+1])
        gammas = np.array(params[n_pcs+1:])

        # Predicted values: biomass = β0 * exp(Σ(βi * PCi) + Σ(γi * N * PCi))
        exponent = np.sum(betas[:, np.newaxis] * pcs, axis=0)
        exponent += np.sum(gammas[:, np.newaxis] * nitrogen[np.newaxis, :] * pcs, axis=0)

        pred = beta0 * np.exp(exponent)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1
        return 1 + n_pcs + n_pcs

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_total = len(self.params)
        n_pcs = (n_total - 1) // 2

        names = ['β0_scale']
        names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
        names += [f'γ{i+1}_N×PC{i+1}' for i in range(n_pcs)]
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))
        p0[0] = np.median(y)  # Scale parameter
        p0[1:n_pcs+1] = 0.01
        p0[n_pcs+1:] = 0.001
        return p0


class MichaelisMentenModel(NLSModel):
    """
    Michaelis-Menten model: saturation response model

    Model form:
    biomass = (Vmax * f(PC)) / (Km + N) + baseline

    Where:
    - Vmax: maximum response
    - Km: half-saturation constant
    - f(PC): linear combination of PCA components
    - baseline: baseline biomass
    """

    def __init__(self):
        super().__init__("Michaelis-Menten Saturation Model")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        nitrogen = X[0, :]
        pcs = X[1:, :]

        # Parameters: [Vmax, Km, baseline, β1, ..., βn]
        Vmax = params[0]
        Km = params[1]
        baseline = params[2]
        betas = np.array(params[3:])

        # f(PC) = Σ(βi * PCi) = linear combination of PCA components
        f_pc = np.sum(betas[:, np.newaxis] * pcs, axis=0)

        # Predicted values: biomass = (Vmax * f(PC)) / (Km + N) + baseline
        pred = (Vmax * f_pc) / (Km + nitrogen + 1e-10) + baseline

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1
        return 3 + n_pcs  # Vmax, Km, baseline, βs = parameters

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_pcs = len(self.params) - 3

        names = ['Vmax', 'Km', 'baseline']
        names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))
        p0[0] = np.max(y) - np.min(y)  # Vmax
        p0[1] = np.mean(X[:, 0])  # Km
        p0[2] = np.min(y)  # baseline
        p0[3:] = 0.1  # βs
        return p0


class ExponentialModel(NLSModel):
    """
    Exponential model: exponential response model

    Model form:
    biomass = β0 + Σ(αi * PCi) * exp(β * N)

    Or:
    biomass = β0 * exp(Σ(βi * PCi) + γ * N)
    """

    def __init__(self, variant: str = 'v1'):
        """
        Args:
            variant: 'v1' or 'v2'
                v1: Predicted values: biomass = β0 + Σ(αi * PCi) * exp(β * N)
                v2: Predicted values: biomass = β0 * exp(Σ(βi * PCi) + γ * N)
        """
        self.variant = variant
        name = f"Exponential-{variant} Model"
        super().__init__(name)

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        nitrogen = X[0, :]
        pcs = X[1:, :]
        n_pcs = pcs.shape[0]

        if self.variant == 'v1':
            # biomass = β0 + Σ(αi * PCi) * exp(β * N)
            beta0 = params[0]
            beta_n = params[1]
            alphas = np.array(params[2:])

            pc_effect = np.sum(alphas[:, np.newaxis] * pcs, axis=0)
            pred = beta0 + pc_effect * np.exp(beta_n * nitrogen)

        else:  # v2
            # biomass = β0 * exp(Σ(βi * PCi) + γ * N)
            beta0 = params[0]
            betas = np.array(params[1:n_pcs+1])
            gamma = params[-1]

            exponent = np.sum(betas[:, np.newaxis] * pcs, axis=0) + gamma * nitrogen
            pred = beta0 * np.exp(exponent)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1
        if self.variant == 'v1':
            return 2 + n_pcs  # β0, β_n, αs
        else:
            return 2 + n_pcs  # β0, βs, γ

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_pcs = len(self.params) - 2

        if self.variant == 'v1':
            names = ['β0_intercept', 'β_N']
            names += [f'α{i+1}_PC{i+1}' for i in range(n_pcs)]
        else:
            names = ['β0_scale']
            names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
            names.append('γ_N')
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))

        if self.variant == 'v1':
            p0[0] = np.mean(y)
            p0[1] = 0.01
            p0[2:] = 0.1
        else:
            p0[0] = np.median(y)
            p0[1:-1] = 0.01
            p0[-1] = 0.001

        return p0


def fit_nls_models(nitrogen_add: np.ndarray,
                   pca_components: np.ndarray,
                   biomass: np.ndarray,
                   models: Optional[List[NLSModel]] = None) -> Dict[str, ModelFitResult]:
    """
    Fit multiple NLS models

    Args:
        nitrogen_add: nitrogen addition array, shape (n_samples,)
        pca_components: PCA components after dimensionality reduction, shape (n_samples, n_components)
        biomass: biomass array, shape (n_samples,)
        models: list of models to fit, if None, use all default models

    Returns:
        dictionary, key is model name, value is ModelFitResult
    """
    logger.info("=" * 70)
    logger.info("Starting to fit multiple NLS models")
    logger.info("=" * 70)
    logger.info(f"Sample size: {len(biomass)}")
    logger.info(f"PCA components: {pca_components.shape[1]}")
    logger.info(f"Nitrogen addition range: [{nitrogen_add.min():.2f}, {nitrogen_add.max():.2f}]")
    logger.info(f"Biomass range: [{biomass.min():.2f}, {biomass.max():.2f}]")

    # Build input matrix X = [nitrogen_add, PC1, PC2, ..., PCn]
    X = np.column_stack([nitrogen_add, pca_components])

    # If no models are specified, use all default models
    if models is None:
        models = [
            LinearModel(),
            AdditiveModel(),
            MultiplicativeModel(),
            MichaelisMentenModel(),
            ExponentialModel('v1'),
            ExponentialModel('v2')
        ]

    results = {}

    for model in models:
        logger.info(f"\nFitting model: {model.name}")
        try:
            result = model.fit(X, biomass)
            results[model.name] = result

            if result.convergence:
                logger.info(f"✓ {model.name} fitting successful")
                logger.info(f"  R² = {result.r2:.4f}, RMSE = {result.rmse:.4f}")
            else:
                logger.warning(f"✗ {model.name} fitting failed: {result.message}")
        except Exception as e:
            logger.error(f"✗ {model.name} fitting error: {str(e)}")
            continue

    logger.info("\n" + "=" * 70)
    logger.info(f"Total {len(results)} models fitted")
    logger.info("=" * 70)

    return results


def compare_models(results: Dict[str, ModelFitResult]) -> pd.DataFrame:
    """
    Compare the fitting results of multiple models

    Args:
        results: dictionary of model fitting results

    Returns:
        DataFrame of comparison results
    """
    comparison_data = []

    for model_name, result in results.items():
        if result.convergence:
            comparison_data.append({
                'Model': model_name,
                'R²': result.r2,
                'RMSE': result.rmse,
                'MAE': result.mae,
                'AIC': result.aic,
                'BIC': result.bic,
                'Number of parameters': len(result.params),
                'Convergence': 'Yes'
            })
        else:
            comparison_data.append({
                'Model': model_name,
                'R²': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'AIC': np.inf,
                'BIC': np.inf,
                'Number of parameters': len(result.params),
                'Convergence': 'No'
            })

    df = pd.DataFrame(comparison_data)

    # Sort by AIC (lower is better)
    df = df.sort_values('AIC', ascending=True)

    logger.info("\nModel comparison results:")
    logger.info("\n" + df.to_string(index=False))

    return df
