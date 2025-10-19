"""
非线性最小二乘(NLS)模块
用于探索氮沉降如何通过影响PCA降维后的变量来影响生物量
"""

from .nls_model import (
    NLSModel,
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

