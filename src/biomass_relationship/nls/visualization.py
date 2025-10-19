"""
NLS模型可视化工具
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from .nls_model import ModelFitResult

logger = logging.getLogger(__name__)

# Set font support for Chinese
matplotlib.rcParams['font.sans-serif'] = ['Unifont', 'DejaVu Sans']  # Support Chinese
matplotlib.rcParams['axes.unicode_minus'] = False  # Solve the problem of displaying negative signs


def plot_model_comparison(results: Dict[str, ModelFitResult],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot model comparison figure

    Args:
        results: dictionary of model fitting results
        save_path: save path
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('NLS model comparison', fontsize=16, fontweight='bold')

    # Prepare data
    model_names = []
    r2_scores = []
    rmse_scores = []
    aic_scores = []
    bic_scores = []

    for name, result in results.items():
        if result.convergence:
            model_names.append(name)
            r2_scores.append(result.r2)
            rmse_scores.append(result.rmse)
            aic_scores.append(result.aic)
            bic_scores.append(result.bic)

    x_pos = np.arange(len(model_names))

    # R² scores
    axes[0, 0].bar(x_pos, r2_scores, alpha=0.7, color='steelblue')
    axes[0, 0].set_ylabel('R²', fontsize=12)
    axes[0, 0].set_title('Coefficient of determination (R²)', fontsize=12)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='0.7 threshold')
    axes[0, 0].legend()

    # RMSE scores
    axes[0, 1].bar(x_pos, rmse_scores, alpha=0.7, color='coral')
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('Root mean squared error (RMSE)', fontsize=12)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # AIC scores
    axes[1, 0].bar(x_pos, aic_scores, alpha=0.7, color='lightgreen')
    axes[1, 0].set_ylabel('AIC', fontsize=12)
    axes[1, 0].set_title('Akaike information criterion (AIC, lower is better)', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # BIC scores
    axes[1, 1].bar(x_pos, bic_scores, alpha=0.7, color='plum')
    axes[1, 1].set_ylabel('BIC', fontsize=12)
    axes[1, 1].set_title('Bayesian information criterion (BIC, lower is better)', fontsize=12)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison figure saved to: {save_path}")

    return fig


def plot_residuals(result: ModelFitResult,
                   model_name: str,
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    draw residual diagnostic figure

    Args:
        result: model fitting result
        model_name: model name
        save_path: save path
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name} - Residual diagnostic', fontsize=16, fontweight='bold')

    predictions = result.predictions
    residuals = result.residuals

    # 1. Residual vs predicted value
    axes[0, 0].scatter(predictions, residuals, alpha=0.6, s=50)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted value', fontsize=12)
    axes[0, 0].set_ylabel('Residual', fontsize=12)
    axes[0, 0].set_title('Residual vs predicted value', fontsize=12)
    axes[0, 0].grid(alpha=0.3)

    # 2. Q-Q plot (normality test)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q plot (normality)', fontsize=12)
    axes[0, 1].grid(alpha=0.3)

    # 3. Residual histogram
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residual', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Residual distribution', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Observed value vs predicted value
    actual = predictions + residuals
    axes[1, 1].scatter(actual, predictions, alpha=0.6, s=50)
    # Add ideal fit line (y=x)
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal fit')
    axes[1, 1].set_xlabel('Observed value', fontsize=12)
    axes[1, 1].set_ylabel('Predicted value', fontsize=12)
    axes[1, 1].set_title(f'Observed value vs predicted value (R²={result.r2:.4f})', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residual diagnostic figure saved to: {save_path}")

    return fig


def plot_parameter_importance(result: ModelFitResult,
                              model_name: str,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot parameter importance figure

    Args:
        result: model fitting result
        model_name: model name
        save_path: save path
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate the standardized importance of parameters (absolute value)
    param_importance = np.abs(result.params)

    # Sort
    sorted_idx = np.argsort(param_importance)[::-1]
    sorted_importance = param_importance[sorted_idx]
    sorted_names = [result.param_names[i] for i in sorted_idx]

    # Plot bar chart
    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_importance, alpha=0.7)

    # Use different colors for positive and negative parameters
    for i, (idx, bar) in enumerate(zip(sorted_idx, bars)):
        if result.params[idx] >= 0:
            bar.set_color('steelblue')
        else:
            bar.set_color('coral')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('Parameter absolute value', fontsize=12)
    ax.set_title(f'{model_name} - Parameter importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Positive parameters'),
        Patch(facecolor='coral', label='Negative parameters')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Parameter importance figure saved to: {save_path}")

    return fig


def plot_nitrogen_response_curve(nitrogen_range: np.ndarray,
                                 biomass_predictions: Dict[str, np.ndarray],
                                 actual_nitrogen: Optional[np.ndarray] = None,
                                 actual_biomass: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot nitrogen response curve

    Args:
        nitrogen_range: nitrogen range
        biomass_predictions: biomass predictions for each model, dictionary format {model_name: predictions}
        actual_nitrogen: actual observed nitrogen values (optional)
        actual_biomass: actual observed biomass values (optional)
        save_path: save path
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual observed points
    if actual_nitrogen is not None and actual_biomass is not None:
        ax.scatter(actual_nitrogen, actual_biomass, s=50, alpha=0.6,
                  color='black', label='观测值', zorder=5)

    # Plot predictions for each model
    # Use colormap to generate colors
    cmap = plt.get_cmap('Set3')
    colors = [cmap(i) for i in np.linspace(0, 1, len(biomass_predictions))]

    for (model_name, predictions), color in zip(biomass_predictions.items(), colors):
        ax.plot(nitrogen_range, predictions, linewidth=2.5,
               label=model_name, color=color, alpha=0.8)

    ax.set_xlabel('Nitrogen Addition', fontsize=13)
    ax.set_ylabel('Biomass', fontsize=13)
    ax.set_title('Nitrogen response curve', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Nitrogen response curve saved to: {save_path}")

    return fig


def plot_interaction_effects(nitrogen_values: np.ndarray,
                            pc_values: np.ndarray,
                            biomass_grid: np.ndarray,
                            pc_idx: int,
                            model_name: str,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot nitrogen response heatmap with PCA components

    Args:
        nitrogen_values: nitrogen values array
        pc_values: PCA components values array
        biomass_grid: biomass predictions grid (shape: len(nitrogen_values) x len(pc_values))
        pc_idx: PCA component index
        model_name: model name
        save_path: save path
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.contourf(nitrogen_values, pc_values, biomass_grid.T,
                     levels=20, cmap='RdYlGn', alpha=0.8)

    # Add contours
    contours = ax.contour(nitrogen_values, pc_values, biomass_grid.T,
                         levels=10, colors='black', linewidths=0.5, alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    # Add color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Biomass', fontsize=12)

    ax.set_xlabel('Nitrogen Addition', fontsize=12)
    ax.set_ylabel(f'PC{pc_idx+1} value', fontsize=12)
    ax.set_title(f'{model_name} - Nitrogen response with PC{pc_idx+1} interaction effect',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Interaction effect figure saved to: {save_path}")

    return fig


def generate_summary_table(results: Dict[str, ModelFitResult],
                           save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate model comparison summary table

    Args:
        results: model fitting results dictionary
        save_path: save path (CSV format)

    Returns:
        summary DataFrame
    """
    summary_data = []

    for model_name, result in results.items():
        row = {
            'Model name': model_name,
            'Convergence status': 'Success' if result.convergence else 'Failed',
            'R²': result.r2 if result.convergence else np.nan,
            'RMSE': result.rmse if result.convergence else np.nan,
            'MAE': result.mae if result.convergence else np.nan,
            'AIC': result.aic if result.convergence else np.inf,
            'BIC': result.bic if result.convergence else np.inf,
            'Number of parameters': len(result.params)
        }

        # Add each parameter value
        if result.convergence:
            for param_name, param_val in zip(result.param_names, result.params):
                row[f'{param_name}'] = param_val

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Sort by AIC
    df = df.sort_values('AIC', ascending=True)

    if save_path:
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"Summary table saved to: {save_path}")

    return df

