"""
NLS分析主接口
用于processor.py中调用的高级分析函数
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
    NLS分析器：探索氮沉降如何通过PCA变量影响生物量
    """

    def __init__(self, output_dir: str, climate_group: Optional[int] = None):
        """
        初始化NLS分析器

        Args:
            output_dir: 输出目录
            climate_group: 气候组编号（可选，用于分组分析）
        """
        self.output_dir = output_dir
        self.climate_group = climate_group
        self.results = {}
        self.best_model = None

        # 创建子目录
        self.nls_dir = os.path.join(output_dir, 'nls_results')
        if climate_group is not None:
            self.nls_dir = os.path.join(self.nls_dir, f'group_{climate_group}')
        os.makedirs(self.nls_dir, exist_ok=True)

        logger.info(f"初始化NLS分析器，输出目录: {self.nls_dir}")

    def analyze(self,
                nitrogen_add: np.ndarray,
                pca_components: np.ndarray,
                biomass: np.ndarray,
                models: Optional[List[str]] = None) -> Dict[str, ModelFitResult]:
        """
        执行NLS分析

        Args:
            nitrogen_add: 氮添加量数组, shape (n_samples,)
            pca_components: PCA降维后的数据, shape (n_samples, n_components)
            biomass: 生物量数组, shape (n_samples,)
            models: 要使用的模型列表，可选值：
                   'additive', 'multiplicative', 'michaelis_menten', 
                   'exponential_v1', 'exponential_v2'
                   如果为None，则使用所有模型

        Returns:
            模型拟合结果字典
        """
        logger.info("=" * 70)
        logger.info("开始NLS分析")
        if self.climate_group is not None:
            logger.info(f"气候组: {self.climate_group}")
        logger.info("=" * 70)

        # 数据验证
        if len(nitrogen_add) != len(biomass) or len(pca_components) != len(biomass):
            raise ValueError("输入数据长度不匹配")

        if len(nitrogen_add) < 10:
            logger.warning(f"样本数量较少 (n={len(nitrogen_add)})，结果可能不可靠")

        # 选择模型
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
                    logger.warning(f"未知模型: {model_name}，已跳过")

        # 拟合模型
        self.results = fit_nls_models(
            nitrogen_add=nitrogen_add,
            pca_components=pca_components,
            biomass=biomass,
            models=model_objs
        )

        # 选择最佳模型（基于AIC）
        best_aic = np.inf
        for model_name, result in self.results.items():
            if result.convergence and result.aic < best_aic:
                best_aic = result.aic
                self.best_model = model_name

        if self.best_model:
            logger.info(f"\n最佳模型（基于AIC）: {self.best_model}")
            logger.info(f"  AIC = {self.results[self.best_model].aic:.2f}")
            logger.info(f"  R² = {self.results[self.best_model].r2:.4f}")

        return self.results

    def save_results(self) -> None:
        """保存分析结果"""
        if not self.results:
            logger.warning("没有可保存的结果")
            return

        logger.info("保存NLS分析结果...")

        # 1. 保存模型比较表
        comparison_df = compare_models(self.results)
        comparison_path = os.path.join(self.nls_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        logger.info(f"  模型比较表已保存: {comparison_path}")

        # 2. 保存详细摘要表
        summary_path = os.path.join(self.nls_dir, 'model_summary.csv')
        generate_summary_table(self.results, save_path=summary_path)
        logger.info(f"  详细摘要表已保存: {summary_path}")

        # 3. 保存各模型的详细报告
        for model_name, result in self.results.items():
            if result.convergence:
                # 保存参数估计
                param_df = pd.DataFrame({
                    '参数名称': result.param_names,
                    '估计值': result.params,
                    '标准误': result.params_std
                })
                param_path = os.path.join(
                    self.nls_dir, 
                    f'parameters_{model_name.replace(" ", "_")}.csv'
                )
                param_df.to_csv(param_path, index=False, encoding='utf-8-sig')

                # 保存残差数据
                residual_df = pd.DataFrame({
                    '预测值': result.predictions,
                    '残差': result.residuals
                })
                residual_path = os.path.join(
                    self.nls_dir,
                    f'residuals_{model_name.replace(" ", "_")}.csv'
                )
                residual_df.to_csv(residual_path, index=False, encoding='utf-8-sig')

        logger.info("所有结果已保存完毕")

    def create_visualizations(self,
                            nitrogen_add: np.ndarray,
                            pca_components: np.ndarray,
                            biomass: np.ndarray) -> None:
        """
        创建可视化图表

        Args:
            nitrogen_add: 氮添加量数组
            pca_components: PCA分量数组
            biomass: 生物量数组
        """
        if not self.results:
            logger.warning("没有可视化的结果")
            return

        logger.info("创建可视化图表...")

        # 1. 模型比较图
        try:
            fig_path = os.path.join(self.nls_dir, 'model_comparison.png')
            plot_model_comparison(self.results, save_path=fig_path)
            logger.info("  模型比较图已保存")
        except (IOError, RuntimeError, ValueError) as e:
            logger.error(f"  创建模型比较图失败: {e}")

        # 2. 为每个成功拟合的模型创建诊断图
        for model_name, result in self.results.items():
            if not result.convergence:
                continue

            try:
                # 残差诊断图
                fig_path = os.path.join(
                    self.nls_dir,
                    f'residuals_{model_name.replace(" ", "_")}.png'
                )
                plot_residuals(result, model_name, save_path=fig_path)

                # 参数重要性图
                fig_path = os.path.join(
                    self.nls_dir,
                    f'parameters_{model_name.replace(" ", "_")}.png'
                )
                plot_parameter_importance(result, model_name, save_path=fig_path)

            except (IOError, RuntimeError, ValueError) as e:
                logger.error(f"  创建{model_name}的图表失败: {e}")

        # 3. 氮响应曲线（仅针对最佳模型和有代表性的PCA值）
        if self.best_model and self.best_model in self.results:
            try:
                self._plot_nitrogen_response(
                    nitrogen_add, pca_components, biomass
                )
            except (IOError, RuntimeError, ValueError, KeyError) as e:
                logger.error(f"  创建氮响应曲线失败: {e}")

        logger.info("所有可视化图表已创建完毕")

    def _plot_nitrogen_response(self,
                               nitrogen_add: np.ndarray,
                               pca_components: np.ndarray,
                               biomass: np.ndarray) -> None:
        """
        绘制氮响应曲线（内部方法）
        """
        # 创建氮沉降范围
        n_min, n_max = nitrogen_add.min(), nitrogen_add.max()
        nitrogen_range = np.linspace(n_min, n_max, 100)

        # 使用PCA分量的均值作为固定值
        pca_mean = np.mean(pca_components, axis=0)

        # 为每个成功的模型生成预测
        predictions_dict = {}

        for model_name, result in self.results.items():
            if not result.convergence:
                continue

            try:
                # 构建输入矩阵
                X_pred = np.column_stack([
                    nitrogen_range,
                    np.tile(pca_mean, (len(nitrogen_range), 1))
                ])

                # 使用模型参数预测
                # 需要从results中获取模型对象
                # 这里简化处理：直接使用存储的预测函数

                # 由于我们没有存储模型对象，这里需要重新创建模型
                model_map = {
                    '加法交互模型 (Additive Interaction)': AdditiveModel(),
                    '乘法交互模型 (Multiplicative)': MultiplicativeModel(),
                    'Michaelis-Menten饱和模型': MichaelisMentenModel(),
                    '指数模型 (Exponential-v1)': ExponentialModel('v1'),
                    '指数模型 (Exponential-v2)': ExponentialModel('v2')
                }

                if model_name in model_map:
                    model = model_map[model_name]
                    model.params = result.params
                    y_pred = model.predict(X_pred)
                    predictions_dict[model_name] = y_pred

            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"  无法为{model_name}生成预测曲线: {e}")

        if predictions_dict:
            fig_path = os.path.join(self.nls_dir, 'nitrogen_response_curve.png')
            plot_nitrogen_response_curve(
                nitrogen_range=nitrogen_range,
                biomass_predictions=predictions_dict,
                actual_nitrogen=nitrogen_add,
                actual_biomass=biomass,
                save_path=fig_path
            )
            logger.info("  氮响应曲线已保存")

    def get_summary(self) -> str:
        """
        生成文本摘要

        Returns:
            摘要文本
        """
        if not self.results:
            return "尚未执行分析"

        lines = []
        lines.append("=" * 70)
        lines.append("NLS分析摘要")
        if self.climate_group is not None:
            lines.append(f"气候组: {self.climate_group}")
        lines.append("=" * 70)
        lines.append("")

        # 模型性能
        lines.append("模型性能:")
        lines.append(f"{'模型名称':<35} {'R²':<10} {'AIC':<10} {'收敛':<10}")
        lines.append("-" * 70)

        for model_name, result in sorted(self.results.items(), 
                                        key=lambda x: x[1].aic):
            if result.convergence:
                lines.append(
                    f"{model_name:<35} "
                    f"{result.r2:<10.4f} "
                    f"{result.aic:<10.2f} "
                    f"{'是':<10}"
                )
            else:
                lines.append(
                    f"{model_name:<35} "
                    f"{'N/A':<10} "
                    f"{'N/A':<10} "
                    f"{'否':<10}"
                )

        lines.append("")

        # 最佳模型详情
        if self.best_model:
            lines.append(f"最佳模型: {self.best_model}")
            lines.append("-" * 70)
            result = self.results[self.best_model]
            lines.append(f"R² = {result.r2:.4f}")
            lines.append(f"RMSE = {result.rmse:.4f}")
            lines.append(f"AIC = {result.aic:.2f}")
            lines.append("")
            lines.append("参数估计:")
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
        logger.info(f"文本摘要已保存: {summary_path}")

        return summary

    def run_complete_analysis(self,
                             nitrogen_add: np.ndarray,
                             pca_components: np.ndarray,
                             biomass: np.ndarray,
                             models: Optional[List[str]] = None,
                             create_plots: bool = True) -> Dict[str, ModelFitResult]:
        """
        执行完整的NLS分析流程

        Args:
            nitrogen_add: 氮添加量数组
            pca_components: PCA分量数组
            biomass: 生物量数组
            models: 要使用的模型列表（可选）
            create_plots: 是否创建可视化图表

        Returns:
            模型拟合结果字典
        """
        # 执行分析
        results = self.analyze(nitrogen_add, pca_components, biomass, models)

        # 保存结果
        self.save_results()

        # 创建可视化
        if create_plots:
            self.create_visualizations(nitrogen_add, pca_components, biomass)

        # 生成摘要
        summary = self.get_summary()
        logger.info("\n" + summary)

        return results


def quick_nls_analysis(nitrogen_add: np.ndarray,
                      pca_components: np.ndarray,
                      biomass: np.ndarray,
                      output_dir: str,
                      climate_group: Optional[int] = None) -> NLSAnalyzer:
    """
    快速NLS分析（便捷函数）

    Args:
        nitrogen_add: 氮添加量数组
        pca_components: PCA分量数组
        biomass: 生物量数组
        output_dir: 输出目录
        climate_group: 气候组编号（可选）

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

