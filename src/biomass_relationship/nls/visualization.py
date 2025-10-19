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

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_model_comparison(results: Dict[str, ModelFitResult],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    绘制模型比较图

    Args:
        results: 模型拟合结果字典
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        matplotlib Figure对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('NLS模型比较', fontsize=16, fontweight='bold')

    # 准备数据
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

    # R² 分数
    axes[0, 0].bar(x_pos, r2_scores, alpha=0.7, color='steelblue')
    axes[0, 0].set_ylabel('R²', fontsize=12)
    axes[0, 0].set_title('决定系数 (R²)', fontsize=12)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='0.7阈值')
    axes[0, 0].legend()

    # RMSE
    axes[0, 1].bar(x_pos, rmse_scores, alpha=0.7, color='coral')
    axes[0, 1].set_ylabel('RMSE', fontsize=12)
    axes[0, 1].set_title('均方根误差 (RMSE)', fontsize=12)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # AIC
    axes[1, 0].bar(x_pos, aic_scores, alpha=0.7, color='lightgreen')
    axes[1, 0].set_ylabel('AIC', fontsize=12)
    axes[1, 0].set_title('赤池信息准则 (AIC, 越小越好)', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # BIC
    axes[1, 1].bar(x_pos, bic_scores, alpha=0.7, color='plum')
    axes[1, 1].set_ylabel('BIC', fontsize=12)
    axes[1, 1].set_title('贝叶斯信息准则 (BIC, 越小越好)', fontsize=12)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"模型比较图已保存至: {save_path}")

    return fig


def plot_residuals(result: ModelFitResult,
                   model_name: str,
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    绘制残差诊断图

    Args:
        result: 模型拟合结果
        model_name: 模型名称
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        matplotlib Figure对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name} - 残差诊断', fontsize=16, fontweight='bold')

    predictions = result.predictions
    residuals = result.residuals

    # 1. 残差 vs 预测值
    axes[0, 0].scatter(predictions, residuals, alpha=0.6, s=50)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('预测值', fontsize=12)
    axes[0, 0].set_ylabel('残差', fontsize=12)
    axes[0, 0].set_title('残差 vs 预测值', fontsize=12)
    axes[0, 0].grid(alpha=0.3)

    # 2. Q-Q图（正态性检验）
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q图 (正态性)', fontsize=12)
    axes[0, 1].grid(alpha=0.3)

    # 3. 残差直方图
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('残差', fontsize=12)
    axes[1, 0].set_ylabel('频数', fontsize=12)
    axes[1, 0].set_title('残差分布', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. 观测值 vs 预测值
    # 添加实际值（从预测值和残差反算）
    actual = predictions + residuals
    axes[1, 1].scatter(actual, predictions, alpha=0.6, s=50)
    # 添加理想拟合线 (y=x)
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想拟合')
    axes[1, 1].set_xlabel('观测值', fontsize=12)
    axes[1, 1].set_ylabel('预测值', fontsize=12)
    axes[1, 1].set_title(f'观测值 vs 预测值 (R²={result.r2:.4f})', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"残差诊断图已保存至: {save_path}")

    return fig


def plot_parameter_importance(result: ModelFitResult,
                              model_name: str,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    绘制参数重要性图

    Args:
        result: 模型拟合结果
        model_name: 模型名称
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 计算参数的标准化重要性（绝对值）
    param_importance = np.abs(result.params)

    # 排序
    sorted_idx = np.argsort(param_importance)[::-1]
    sorted_importance = param_importance[sorted_idx]
    sorted_names = [result.param_names[i] for i in sorted_idx]

    # 绘制条形图
    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_importance, alpha=0.7)

    # 为正负参数使用不同颜色
    for i, (idx, bar) in enumerate(zip(sorted_idx, bars)):
        if result.params[idx] >= 0:
            bar.set_color('steelblue')
        else:
            bar.set_color('coral')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel('参数绝对值', fontsize=12)
    ax.set_title(f'{model_name} - 参数重要性', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='正参数'),
        Patch(facecolor='coral', label='负参数')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"参数重要性图已保存至: {save_path}")

    return fig


def plot_nitrogen_response_curve(nitrogen_range: np.ndarray,
                                 biomass_predictions: Dict[str, np.ndarray],
                                 actual_nitrogen: Optional[np.ndarray] = None,
                                 actual_biomass: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    绘制氮沉降响应曲线

    Args:
        nitrogen_range: 氮沉降范围
        biomass_predictions: 各模型的生物量预测，字典格式 {model_name: predictions}
        actual_nitrogen: 实际观测的氮沉降值（可选）
        actual_biomass: 实际观测的生物量值（可选）
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制实际观测点
    if actual_nitrogen is not None and actual_biomass is not None:
        ax.scatter(actual_nitrogen, actual_biomass, s=50, alpha=0.6,
                  color='black', label='观测值', zorder=5)

    # 绘制各模型的预测曲线
    # 使用colormap生成颜色
    cmap = plt.get_cmap('Set3')
    colors = [cmap(i) for i in np.linspace(0, 1, len(biomass_predictions))]

    for (model_name, predictions), color in zip(biomass_predictions.items(), colors):
        ax.plot(nitrogen_range, predictions, linewidth=2.5,
               label=model_name, color=color, alpha=0.8)

    ax.set_xlabel('氮添加量 (Nitrogen Addition)', fontsize=13)
    ax.set_ylabel('生物量 (Biomass)', fontsize=13)
    ax.set_title('氮沉降对生物量的响应曲线', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"氮响应曲线已保存至: {save_path}")

    return fig


def plot_interaction_effects(nitrogen_values: np.ndarray,
                            pc_values: np.ndarray,
                            biomass_grid: np.ndarray,
                            pc_idx: int,
                            model_name: str,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    绘制氮沉降与PCA分量的交互效应热图

    Args:
        nitrogen_values: 氮沉降值数组
        pc_values: PCA分量值数组
        biomass_grid: 生物量预测网格 (shape: len(nitrogen_values) x len(pc_values))
        pc_idx: PCA分量索引
        model_name: 模型名称
        save_path: 保存路径
        figsize: 图形大小

    Returns:
        matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热图
    im = ax.contourf(nitrogen_values, pc_values, biomass_grid.T,
                     levels=20, cmap='RdYlGn', alpha=0.8)

    # 添加等高线
    contours = ax.contour(nitrogen_values, pc_values, biomass_grid.T,
                         levels=10, colors='black', linewidths=0.5, alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('生物量 (Biomass)', fontsize=12)

    ax.set_xlabel('氮添加量 (Nitrogen Addition)', fontsize=12)
    ax.set_ylabel(f'PC{pc_idx+1} 值', fontsize=12)
    ax.set_title(f'{model_name} - 氮沉降与PC{pc_idx+1}的交互效应',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"交互效应图已保存至: {save_path}")

    return fig


def generate_summary_table(results: Dict[str, ModelFitResult],
                           save_path: Optional[str] = None) -> pd.DataFrame:
    """
    生成模型对比摘要表

    Args:
        results: 模型拟合结果字典
        save_path: 保存路径（CSV格式）

    Returns:
        摘要DataFrame
    """
    summary_data = []

    for model_name, result in results.items():
        row = {
            '模型名称': model_name,
            '收敛状态': '成功' if result.convergence else '失败',
            'R²': result.r2 if result.convergence else np.nan,
            'RMSE': result.rmse if result.convergence else np.nan,
            'MAE': result.mae if result.convergence else np.nan,
            'AIC': result.aic if result.convergence else np.inf,
            'BIC': result.bic if result.convergence else np.inf,
            '参数数量': len(result.params)
        }

        # 添加各个参数值
        if result.convergence:
            for param_name, param_val in zip(result.param_names, result.params):
                row[f'{param_name}'] = param_val

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # 按AIC排序
    df = df.sort_values('AIC', ascending=True)

    if save_path:
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"摘要表已保存至: {save_path}")

    return df

