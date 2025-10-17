"""
生物量关系统计分析框架

基于XGBoost.ipynb构建的模块化统计框架，提供以下功能：

1. XGBoost模型训练和评估 (decision_tree.xgboost)
2. LightGBM模型训练和评估 (decision_tree.lightgbm) 
3. SHAP可解释性分析 (shap)
4. 模型比较和优化分析

使用示例:
    from src.biomass_relationship.decision_tree.xgboost import XGBoostAnalyzer
    from src.biomass_relationship.decision_tree.lightgbm import LightGBMAnalyzer
    from src.biomass_relationship.shap import SHAPAnalyzer

    # 训练XGBoost模型
    xgb_analyzer = XGBoostAnalyzer()
    xgb_analyzer.load_data("path/to/data.csv")
    results = xgb_analyzer.train_with_ratio(zero_ratio=19, use_log=True)

    # SHAP分析
    shap_analyzer = SHAPAnalyzer()
    shap_analyzer.analyze_model(xgb_analyzer.model, results['test_x'])
"""

from .decision_tree.xgboost import XGBoostAnalyzer
from .decision_tree.lightgbm import LightGBMAnalyzer
from .shap import SHAPAnalyzer, compare_models_shap

__all__ = [
    'XGBoostAnalyzer',
    'LightGBMAnalyzer',
    'SHAPAnalyzer',
    'compare_models_shap'
]

__version__ = '1.0.0'
