"""
Main interface for NLS analysis
Used in processor.py for advanced analysis functions
"""

import logging
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .nls_model import (
    AdditiveModel,
    MultiplicativeModel,
    MichaelisMentenModel,
    ExponentialModel,
    ModelFitResult,
    fit_nls_models,
    compare_models
)
from .visualization import (
    plot_model_comparison,
    plot_residuals,
    plot_parameter_importance,
    plot_nitrogen_response_curve,
    generate_summary_table
)

logger = logging.getLogger(__name__)


class NLSAnalyzer:
    """
    NLS analyzer: explore how nitrogen deposition affects biomass through PCA variables
    """

    def __init__(self, output_dir: str, climate_group: Optional[int] = None):
        """
        Initialize NLS analyzer

        Args:
            output_dir: output directory
            climate_group: climate group number (optional, for group analysis)
        """
        self.output_dir = output_dir
        self.climate_group = climate_group
        self.results = {}
        self.best_model = None

        # Create subdirectory
        self.nls_dir = os.path.join(output_dir, 'nls_results')
        if climate_group is not None:
            self.nls_dir = os.path.join(self.nls_dir, f'group_{climate_group}')
        os.makedirs(self.nls_dir, exist_ok=True)

        logger.info(f"Initialized NLS analyzer, output directory: {self.nls_dir}")

    def analyze(self,
                nitrogen_add: np.ndarray,
                pca_components: np.ndarray,
                biomass: np.ndarray,
                models: Optional[List[str]] = None) -> Dict[str, ModelFitResult]:
        """
        Execute NLS analysis

        Args:
            nitrogen_add: nitrogen addition array, shape (n_samples,)
            pca_components: PCA-reduced data, shape (n_samples, n_components)
            biomass: biomass array, shape (n_samples,)
            models: list of models to use, optional values:
                   'additive', 'multiplicative', 'michaelis_menten', 
                   'exponential_v1', 'exponential_v2'
                   if None, all models will be used

        Returns:
            model fitting results dictionary
        """
        logger.info("=" * 70)
        logger.info("Starting NLS analysis")
        if self.climate_group is not None:
            logger.info(f"Climate group: {self.climate_group}")
        logger.info("=" * 70)

        # Data validation
        if len(nitrogen_add) != len(biomass) or len(pca_components) != len(biomass):
            raise ValueError("Input data length mismatch")

        if len(nitrogen_add) < 10:
            logger.warning(f"Sample size is too small (n={len(nitrogen_add)}), results may be unreliable")

        # Select models
        model_objs = []
        if models is None:
            model_objs = [
                AdditiveModel(),
                MultiplicativeModel(),
                MichaelisMentenModel(),
                ExponentialModel('v1'),
                ExponentialModel('v2')
            ]
        else:
            model_map = {
                'additive': AdditiveModel(),
                'multiplicative': MultiplicativeModel(),
                'michaelis_menten': MichaelisMentenModel(),
                'exponential_v1': ExponentialModel('v1'),
                'exponential_v2': ExponentialModel('v2')
            }
            for model_name in models:
                if model_name in model_map:
                    model_objs.append(model_map[model_name])
                else:
                    logger.warning(f"Unknown model: {model_name}, skipped")

        # Fit models
        self.results = fit_nls_models(
            nitrogen_add=nitrogen_add,
            pca_components=pca_components,
            biomass=biomass,
            models=model_objs
        )

        # Select best model (based on AIC)
        best_aic = np.inf
        for model_name, result in self.results.items():
            if result.convergence and result.aic < best_aic:
                best_aic = result.aic
                self.best_model = model_name

        if self.best_model:
            logger.info(f"\nBest model (based on AIC): {self.best_model}")
            logger.info(f"  AIC = {self.results[self.best_model].aic:.2f}")
            logger.info(f"  R² = {self.results[self.best_model].r2:.4f}")

        return self.results

    def save_results(self) -> None:
        """Save analysis results"""
        if not self.results:
            logger.warning("No results to save")
            return

        logger.info("Saving NLS analysis results...")

        # 1. Save model comparison table
        comparison_df = compare_models(self.results)
        comparison_path = os.path.join(self.nls_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        logger.info(f"  Model comparison table saved: {comparison_path}")

        # 2. Save detailed summary table
        summary_path = os.path.join(self.nls_dir, 'model_summary.csv')
        generate_summary_table(self.results, save_path=summary_path)
        logger.info(f"  Detailed summary table saved: {summary_path}")

        # 3. Save detailed report for each model
        for model_name, result in self.results.items():
            if result.convergence:
                # Save parameter estimates
                param_df = pd.DataFrame({
                    'Parameter names': result.param_names,
                    'Estimated values': result.params,
                    'Standard errors': result.params_std
                })
                param_path = os.path.join(
                    self.nls_dir, 
                    f'parameters_{model_name.replace(" ", "_")}.csv'
                )
                param_df.to_csv(param_path, index=False, encoding='utf-8-sig')

                # Save residual data
                residual_df = pd.DataFrame({
                    'Predicted values': result.predictions,
                    'Residuals': result.residuals
                })
                residual_path = os.path.join(
                    self.nls_dir,
                    f'residuals_{model_name.replace(" ", "_")}.csv'
                )
                residual_df.to_csv(residual_path, index=False, encoding='utf-8-sig')

        logger.info("All results saved")

    def create_visualizations(self,
                            nitrogen_add: np.ndarray,
                            pca_components: np.ndarray,
                            biomass: np.ndarray) -> None:
        """
        Create visualizations

        Args:
            nitrogen_add: nitrogen addition array
            pca_components: PCA components array
            biomass: biomass array
        """
        if not self.results:
            logger.warning("No visualizations to create")
            return

        logger.info("Creating visualizations...")

        # 1. Model comparison plot
        try:
            fig_path = os.path.join(self.nls_dir, 'model_comparison.png')
            plot_model_comparison(self.results, save_path=fig_path)
            logger.info("  Model comparison plot saved")
        except (IOError, RuntimeError, ValueError) as e:
            logger.error(f"  Failed to create model comparison plot: {e}")

        # 2. Create diagnostic plots for each successfully fitted model
        for model_name, result in self.results.items():
            if not result.convergence:
                continue

            try:
                # Residual diagnostic plot
                fig_path = os.path.join(
                    self.nls_dir,
                    f'residuals_{model_name.replace(" ", "_")}.png'
                )
                plot_residuals(result, model_name, save_path=fig_path)

                # Parameter importance plot
                fig_path = os.path.join(
                    self.nls_dir,
                    f'parameters_{model_name.replace(" ", "_")}.png'
                )
                plot_parameter_importance(result, model_name, save_path=fig_path)

            except (IOError, RuntimeError, ValueError) as e:
                logger.error(f"  Failed to create plot for {model_name}: {e}")

        # 3. Nitrogen response curve (only for best model and representative PCA values)
        if self.best_model and self.best_model in self.results:
            try:
                self._plot_nitrogen_response(
                    nitrogen_add, pca_components, biomass
                )
            except (IOError, RuntimeError, ValueError, KeyError) as e:
                logger.error(f"  Failed to create nitrogen response curve: {e}")

        logger.info("All visualizations created")

    def _plot_nitrogen_response(self,
                               nitrogen_add: np.ndarray,
                               pca_components: np.ndarray,
                               biomass: np.ndarray) -> None:
        """
        Plot nitrogen response curve (internal method)
        """
        # Create nitrogen deposition range
        n_min, n_max = nitrogen_add.min(), nitrogen_add.max()
        nitrogen_range = np.linspace(n_min, n_max, 100)

        # Use the mean of PCA components as a fixed value
        pca_mean = np.mean(pca_components, axis=0)

        # Generate predictions for each successfully fitted model
        predictions_dict = {}

        for model_name, result in self.results.items():
            if not result.convergence:
                continue

            try:
                # Build input matrix
                X_pred = np.column_stack([
                    nitrogen_range,
                    np.tile(pca_mean, (len(nitrogen_range), 1))
                ])

                # Predict using model parameters
                # 需要从results中获取模型对象
                # Here we simplify the process: directly use the stored prediction function

                # Since we don't store model objects, we need to recreate the models
                model_map = {
                    'Additive Interaction': AdditiveModel(),
                    'Multiplicative': MultiplicativeModel(),
                    'Michaelis-Menten': MichaelisMentenModel(),
                    'Exponential-v1': ExponentialModel('v1'),
                    'Exponential-v2': ExponentialModel('v2')
                }

                if model_name in model_map:
                    model = model_map[model_name]
                    model.params = result.params
                    y_pred = model.predict(X_pred)
                    predictions_dict[model_name] = y_pred

            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"  Failed to generate prediction curve for {model_name}: {e}")

        if predictions_dict:
            fig_path = os.path.join(self.nls_dir, 'nitrogen_response_curve.png')
            plot_nitrogen_response_curve(
                nitrogen_range=nitrogen_range,
                biomass_predictions=predictions_dict,
                actual_nitrogen=nitrogen_add,
                actual_biomass=biomass,
                save_path=fig_path
            )
            logger.info("  Nitrogen response curve saved")

    def get_summary(self) -> str:
        """
        Generate text summary

        Returns:
            Summary text
        """
        if not self.results:
            return "Analysis not executed"

        lines = []
        lines.append("=" * 70)
        lines.append("NLS analysis summary")
        if self.climate_group is not None:
            lines.append(f"Climate group: {self.climate_group}")
        lines.append("=" * 70)
        lines.append("")

        # Model performance
        lines.append("Model performance:")
        lines.append(f"{'Model name':<35} {'R²':<10} {'AIC':<10} {'Convergence':<10}")
        lines.append("-" * 70)

        for model_name, result in sorted(self.results.items(), 
                                        key=lambda x: x[1].aic):
            if result.convergence:
                lines.append(
                    f"{model_name:<35} "
                    f"{result.r2:<10.4f} "
                    f"{result.aic:<10.2f} "
                    f"{'Yes':<10}"
                )
            else:
                lines.append(
                    f"{model_name:<35} "
                    f"{'N/A':<10} "
                    f"{'N/A':<10} "
                    f"{'No':<10}"
                )

        lines.append("")

        # Best model details
        if self.best_model:
            lines.append(f"Best model: {self.best_model}")
            lines.append("-" * 70)
            result = self.results[self.best_model]
            lines.append(f"R² = {result.r2:.4f}")
            lines.append(f"RMSE = {result.rmse:.4f}")
            lines.append(f"AIC = {result.aic:.2f}")
            lines.append("")
            lines.append("Parameter estimates:")
            for name, val, std in zip(result.param_names, 
                                     result.params, 
                                     result.params_std):
                lines.append(f"  {name}: {val:.6f} (± {std:.6f})")

        lines.append("=" * 70)

        summary = "\n".join(lines)

        # 保存文本摘要
        summary_path = os.path.join(self.nls_dir, 'analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Text summary saved: {summary_path}")

        return summary

    def run_complete_analysis(self,
                             nitrogen_add: np.ndarray,
                             pca_components: np.ndarray,
                             biomass: np.ndarray,
                             models: Optional[List[str]] = None,
                             create_plots: bool = True) -> Dict[str, ModelFitResult]:
        """
        Execute complete NLS analysis process

        Args:
            nitrogen_add: nitrogen addition array
            pca_components: PCA components array
            biomass: biomass array
            models: list of models to use (optional)
            create_plots: whether to create visualizations

        Returns:
            model fitting results dictionary
        """
        # Execute analysis
        results = self.analyze(nitrogen_add, pca_components, biomass, models)

        # Save results
        self.save_results()

        # Create visualizations
        if create_plots:
            self.create_visualizations(nitrogen_add, pca_components, biomass)

        # Generate summary
        summary = self.get_summary()
        logger.info("\n" + summary)

        return results


def quick_nls_analysis(nitrogen_add: np.ndarray,
                      pca_components: np.ndarray,
                      biomass: np.ndarray,
                      output_dir: str,
                      climate_group: Optional[int] = None) -> NLSAnalyzer:
    """
    Quick NLS analysis (convenience function)

    Args:
        nitrogen_add: nitrogen addition array
        pca_components: PCA components array
        biomass: biomass array
        output_dir: output directory
        climate_group: climate group number (optional)

    Returns:
        NLSAnalyzer对象
    """
    analyzer = NLSAnalyzer(output_dir, climate_group)
    analyzer.run_complete_analysis(
        nitrogen_add=nitrogen_add,
        pca_components=pca_components,
        biomass=biomass,
        create_plots=True
    )
    return analyzer

