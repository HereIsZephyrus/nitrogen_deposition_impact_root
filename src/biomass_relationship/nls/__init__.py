"""
NLS module
Used to explore how nitrogen deposition affects biomass through PCA-reduced variables
"""

from .nls_model import (
    NLSModel,
    LinearModel,
    AdditiveModel,
    MultiplicativeModel,
    MichaelisMentenModel,
    ExponentialModel,
    fit_nls_models,
    compare_models,
    ModelFitResult
)

from .analyzer import (
    NLSAnalyzer,
    quick_nls_analysis
)

__all__ = [
    'NLSModel',
    'LinearModel',
    'AdditiveModel',
    'MultiplicativeModel',
    'MichaelisMentenModel',
    'ExponentialModel',
    'fit_nls_models',
    'compare_models',
    'ModelFitResult',
    'NLSAnalyzer',
    'quick_nls_analysis'
]

