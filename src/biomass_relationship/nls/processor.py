import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from .nls_model import NLSModel, ModelFitResult
from .linear_model import LinearModel
from .additive_model import AdditiveModel
from .multiplicative_model import MultiplicativeModel
from .menten_model import MichaelisMentenModel
from .exponential_model import ExponentialModel

logger = logging.getLogger(__name__)

def fit_nls_models(nitrogen_add: np.ndarray,
                   pca_components: np.ndarray,
                   biomass: np.ndarray,
                   models: Optional[List[NLSModel]] = None,
                   alpha_l1: float = 0.0,
                   alpha_l2: float = 0.0,
                   check_overfitting: bool = True) -> Dict[str, ModelFitResult]:
    """
    Fit multiple NLS models

    Args:
        nitrogen_add: nitrogen addition array, shape (n_samples,)
        pca_components: PCA components after dimensionality reduction, shape (n_samples, n_components)
        biomass: biomass array, shape (n_samples,)
        models: list of models to fit, if None, use all default models
        alpha_l1: L1 regularization parameter (Lasso), default 0.0
                  Recommended: 0.01-0.1 for overfitting issues
        alpha_l2: L2 regularization parameter (Ridge), default 0.0
                  Recommended: 0.01-0.1 for overfitting issues
        check_overfitting: whether to check overfitting risk, default True

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
            result = model.fit(
                X,
                biomass,
                alpha_l1=alpha_l1,
                alpha_l2=alpha_l2,
                check_overfitting=check_overfitting
            )
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
