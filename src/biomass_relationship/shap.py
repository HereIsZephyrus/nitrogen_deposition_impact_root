import logging
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """SHAP Model Interpreter"""

    def __init__(self, model=None):
        """
        Initialize SHAP analyzer

        Args:
            model: Trained model (XGBoost or LightGBM)
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.test_data = None

    def set_model(self, model):
        """Set model to analyze"""
        self.model = model
        self.explainer = None  # Reset explainer

    def create_explainer(self, X_test: pd.DataFrame, explainer_type: str = 'auto'):
        """
        Create SHAP explainer

        Args:
            X_test: Test data features
            explainer_type: Explainer type ('tree', 'linear', 'kernel', 'auto')
        """
        if self.model is None:
            raise ValueError("Please set the model first")

        self.test_data = X_test

        if explainer_type == 'auto':
            # Automatically select explainer type
            model_name = type(self.model).__name__.lower()
            if 'xgb' in model_name or 'lgb' in model_name or 'lightgbm' in model_name:
                explainer_type = 'tree'
            else:
                explainer_type = 'kernel'

        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_test)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, X_test.sample(min(100, len(X_test))))
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")

        logger.info(f"Explainer created: {explainer_type}")

    def calculate_shap_values(self, X_test: Optional[pd.DataFrame] = None,
                            max_samples: int = 1000):
        """
        Calculate SHAP values

        Args:
            X_test: Test data (if None, use previously set data)
            max_samples: Maximum number of samples (for faster computation)
        """
        if self.explainer is None:
            raise ValueError("Please create the explainer first")

        if X_test is not None:
            self.test_data = X_test

        if self.test_data is None:
            raise ValueError("No test data")

        # If data is too large, randomly sample
        if len(self.test_data) > max_samples:
            sample_data = self.test_data.sample(n=max_samples, random_state=42)
            logger.info(f"Data is too large, randomly sampling {max_samples} samples for SHAP analysis")
        else:
            sample_data = self.test_data

        logger.info("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(sample_data)
        self.test_data = sample_data  # 更新为采样后的数据
        logger.info("SHAP values calculation completed")

        return self.shap_values

    def summary_plot(self, plot_type: str = 'dot', max_display: int = 20,
                    figsize: tuple = (10, 6), save_path: Optional[str] = None):
        """
        Plot SHAP summary plot

        Args:
            plot_type: Plot type ('dot', 'bar', 'violin')
            max_display: Maximum number of features to display
            figsize: Plot size
            save_path: Save path
        """
        if self.shap_values is None:
            raise ValueError("Please calculate SHAP values first")

        plt.figure(figsize=figsize)

        if plot_type == 'dot':
            shap.summary_plot(self.shap_values, self.test_data,
                            max_display=max_display, show=False)
        elif plot_type == 'bar':
            shap.summary_plot(self.shap_values, self.test_data,
                            plot_type="bar", max_display=max_display, show=False)
        elif plot_type == 'violin':
            shap.summary_plot(self.shap_values, self.test_data,
                            plot_type="violin", max_display=max_display, show=False)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        plt.title(f'SHAP Feature importance summary plot ({plot_type})')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")

        plt.show()

    def waterfall_plot(self, sample_idx: int = 0, save_path: Optional[str] = None):
        """
        Plot waterfall plot for a single sample

        Args:
            sample_idx: Sample index
            save_path: Save path
        """
        if self.shap_values is None:
            raise ValueError("Please calculate SHAP values first")

        if sample_idx >= len(self.shap_values):
            raise ValueError(f"Sample index out of range, maximum index is {len(self.shap_values)-1}")

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(values=self.shap_values[sample_idx],
                           base_values=self.explainer.expected_value,
                           data=self.test_data.iloc[sample_idx].values,
                           feature_names=self.test_data.columns.tolist()),
            show=False
        )

        plt.title(f'SHAP waterfall plot for sample {sample_idx}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")

        plt.show()

    def force_plot(self, sample_idx: int = 0, matplotlib: bool = True):
        """
        Plot force plot for a single sample

        Args:
            sample_idx: Sample index
            matplotlib: Whether to use matplotlib to display
        """
        if self.shap_values is None:
            raise ValueError("Please calculate SHAP values first")

        if sample_idx >= len(self.shap_values):
            raise ValueError(f"Sample index out of range, maximum index is {len(self.shap_values)-1}")

        if matplotlib:
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[sample_idx],
                self.test_data.iloc[sample_idx],
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP force plot for sample {sample_idx}')
            plt.tight_layout()
            plt.show()
        else:
            return shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[sample_idx],
                self.test_data.iloc[sample_idx]
            )

    def dependence_plot(self, feature_name: str, interaction_feature: str = 'auto',
                       figsize: tuple = (8, 6), save_path: Optional[str] = None):
        """
        Plot feature dependence plot

        Args:
            feature_name: Feature name
            interaction_feature: Interaction feature name
            figsize: Plot size
            save_path: Save path
        """
        if self.shap_values is None:
            raise ValueError("Please calculate SHAP values first")

        if feature_name not in self.test_data.columns:
            raise ValueError(f"Feature '{feature_name}' does not exist")

        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature_name, self.shap_values, self.test_data,
            interaction_index=interaction_feature, show=False
        )

        plt.title(f'Feature {feature_name} dependence plot')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")

        plt.show()

    def get_feature_importance(self, importance_type: str = 'mean_abs') -> Dict[str, float]:
        """
        Get feature importance based on SHAP values

        Args:
            importance_type: Importance calculation type ('mean_abs', 'mean', 'sum_abs', 'sum')

        Returns:
            Feature importance dictionary
        """
        if self.shap_values is None:
            raise ValueError("Please calculate SHAP values first")

        shap_df = pd.DataFrame(self.shap_values, columns=self.test_data.columns)

        if importance_type == 'mean_abs':
            importance = shap_df.abs().mean()
        elif importance_type == 'mean':
            importance = shap_df.mean()
        elif importance_type == 'sum_abs':
            importance = shap_df.abs().sum()
        elif importance_type == 'sum':
            importance = shap_df.sum()
        else:
            raise ValueError(f"Unsupported importance calculation type: {importance_type}")

        # Sort by importance
        importance_dict = importance.sort_values(ascending=False).to_dict()

        return importance_dict

    def plot_feature_importance(self, importance_type: str = 'mean_abs',
                              top_n: int = 20, figsize: tuple = (10, 8),
                              save_path: Optional[str] = None):
        """
        Plot feature importance plot

        Args:
            importance_type: Importance calculation type
            top_n: Display top N important features
            figsize: Plot size
            save_path: Save path
        """
        importance_dict = self.get_feature_importance(importance_type)

        # Get top N features
        top_features = list(importance_dict.keys())[:top_n]
        top_values = [importance_dict[feat] for feat in top_features]

        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_values)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel(f'SHAP importance ({importance_type})')
        plt.title(f'SHAP feature importance (top {top_n} features)')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")

        plt.show()

    def export_shap_values(self, file_path: str):
        """
        Export SHAP values to CSV file

        Args:
            file_path: Output file path
        """
        if self.shap_values is None:
            raise ValueError("Please calculate SHAP values first")

        shap_df = pd.DataFrame(self.shap_values, columns=self.test_data.columns)
        shap_df.to_csv(file_path, index=False)
        logger.info(f"SHAP values exported to: {file_path}")

    def analyze_model(self, model, X_test: pd.DataFrame,
                     max_samples: int = 1000,
                     create_plots: bool = True,
                     save_dir: Optional[str] = None):
        """
        Complete model analysis process

        Args:
            model: Model to analyze
            X_test: Test data
            max_samples: Maximum number of samples
            create_plots: Whether to create visualizations
            save_dir: Save directory for plots
        """
        logger.info("Starting SHAP analysis...")

        # Set model and create explainer
        self.set_model(model)
        self.create_explainer(X_test)

        # Calculate SHAP values
        self.calculate_shap_values(max_samples=max_samples)

        if create_plots:
            logger.info("Creating visualizations...")

            # Summary plot
            save_path = f"{save_dir}/shap_summary_dot.png" if save_dir else None
            self.summary_plot(plot_type='dot', save_path=save_path)

            save_path = f"{save_dir}/shap_summary_bar.png" if save_dir else None
            self.summary_plot(plot_type='bar', save_path=save_path)

            # Feature importance plot
            save_path = f"{save_dir}/shap_feature_importance.png" if save_dir else None
            self.plot_feature_importance(save_path=save_path)

            # Single sample waterfall plot
            save_path = f"{save_dir}/shap_waterfall_sample0.png" if save_dir else None
            self.waterfall_plot(sample_idx=0, save_path=save_path)

        # Return feature importance
        importance = self.get_feature_importance()
        logger.info("SHAP analysis completed!")

        return importance


def compare_models_shap(models: Dict[str, Any], X_test: pd.DataFrame,
                       model_names: Optional[list] = None,
                       max_samples: int = 1000):
    """
    Compare SHAP analysis results of multiple models

    Args:
        models: Model dictionary {model_name: model}
        X_test: Test data
        model_names: Model name list
        max_samples: Maximum number of samples

    Returns:
        Feature importance dictionary for each model
    """
    if model_names is None:
        model_names = list(models.keys())

    importance_results = {}

    for model_name, model in models.items():
        if model_name not in model_names:
            continue

        logger.info(f"\nAnalyzing model: {model_name}")
        logger.info("=" * 50)

        analyzer = SHAPAnalyzer(model)
        analyzer.create_explainer(X_test)
        analyzer.calculate_shap_values(max_samples=max_samples)

        importance = analyzer.get_feature_importance()
        importance_results[model_name] = importance

        # Display top 10 important features
        logger.info(f"\n{model_name} top 10 important features:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            logger.info(f"{i+1:2d}. {feat}: {imp:.4f}")

    return importance_results
