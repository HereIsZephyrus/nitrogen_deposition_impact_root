"""
非线性最小二乘(NLS)模型实现
探索氮沉降如何通过影响PCA降维后的变量来影响生物量

模型假设：nitrogen_addition -> PCA_components -> biomass
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
    """NLS模型拟合结果"""
    params: np.ndarray  # 模型参数
    params_std: np.ndarray  # 参数标准误
    r2: float  # R²决定系数
    rmse: float  # 均方根误差
    mae: float  # 平均绝对误差
    aic: float  # 赤池信息准则
    bic: float  # 贝叶斯信息准则
    residuals: np.ndarray  # 残差
    predictions: np.ndarray  # 预测值
    convergence: bool  # 是否收敛
    message: str  # 拟合信息
    param_names: List[str]  # 参数名称


class NLSModel:
    """
    非线性最小二乘模型基类

    所有具体模型都应继承此类并实现model_func方法
    """

    def __init__(self, name: str = "NLS Model"):
        """
        初始化NLS模型

        Args:
            name: 模型名称
        """
        self.name = name
        self.params = None
        self.params_std = None
        self.fit_result = None

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        """
        模型函数 - 子类必须实现

        Args:
            X: 输入特征矩阵 [nitrogen_add, PC1, PC2, ..., PCn]
            *params: 模型参数

        Returns:
            预测的biomass值
        """
        raise NotImplementedError("子类必须实现model_func方法")

    def get_param_names(self) -> List[str]:
        """获取参数名称 - 子类应重写"""
        return [f"param_{i}" for i in range(len(self.params))] if self.params is not None else []

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        获取初始参数估计 - 子类可以重写以提供更好的初始值

        Args:
            X: 输入特征矩阵
            y: 目标变量（某些模型需要用于初始化）

        Returns:
            初始参数数组
        """
        # 默认实现：使用简单的初始值
        n_params = self._get_n_params(X)
        # 使用y来避免未使用参数警告，但默认实现不需要它
        _ = y  
        return np.ones(n_params)

    def _get_n_params(self, X: np.ndarray) -> int:
        """获取参数数量 - 子类应重写"""
        return X.shape[1] + 1  # 默认：截距 + 每个特征一个系数

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            method: str = 'trf',
            max_nfev: int = 10000) -> ModelFitResult:
        """
        拟合模型

        Args:
            X: 输入特征矩阵 [nitrogen_add, PC1, PC2, ..., PCn]
               shape: (n_samples, n_features)
            y: 目标变量 (biomass)
               shape: (n_samples,)
            method: 优化方法 ('trf', 'dogbox', 'lm')
            max_nfev: 最大函数评估次数

        Returns:
            ModelFitResult对象
        """
        logger.info(f"开始拟合模型: {self.name}")
        logger.info(f"  样本数: {len(y)}, 特征数: {X.shape[1]}")

        # 获取初始参数
        p0 = self.get_initial_params(X, y)
        logger.info(f"  初始参数: {p0}")

        try:
            # 使用curve_fit进行非线性最小二乘拟合
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                def model_wrapper(x, *p):
                    return self.model_func(x, *p)

                popt, pcov = curve_fit(
                    f=model_wrapper,
                    xdata=X.T,  # curve_fit期望xdata为(n_features, n_samples)
                    ydata=y,
                    p0=p0,
                    method=method,
                    maxfev=max_nfev,
                    full_output=False
                )

            self.params = popt

            # 计算参数标准误
            try:
                perr = np.sqrt(np.diag(pcov))
                self.params_std = perr
            except (ValueError, RuntimeError):
                self.params_std = np.full_like(popt, np.nan)
                logger.warning("无法计算参数标准误")

            # 计算预测值和残差
            y_pred = self.predict(X)
            residuals = y - y_pred

            # 计算评估指标
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            # 计算信息准则
            n = len(y)
            k = len(popt)
            rss = np.sum(residuals**2)
            aic = n * np.log(rss / n) + 2 * k
            bic = n * np.log(rss / n) + k * np.log(n)

            # 创建结果对象
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
                message="拟合成功",
                param_names=self.get_param_names()
            )

            logger.info(f"模型拟合完成: {self.name}")
            logger.info(f"  R² = {r2:.4f}")
            logger.info(f"  RMSE = {rmse:.4f}")
            logger.info(f"  AIC = {aic:.2f}")

            return self.fit_result

        except (RuntimeError, ValueError) as e:
            logger.error(f"模型拟合失败: {self.name} - {str(e)}")

            # 返回失败结果
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
                message=f"拟合失败: {str(e)}",
                param_names=self.get_param_names()
            )

            return self.fit_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 输入特征矩阵

        Returns:
            预测值数组
        """
        if self.params is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        return self.model_func(X.T, *self.params)

    def summary(self) -> str:
        """生成模型摘要报告"""
        if self.fit_result is None:
            return f"模型 {self.name} 尚未拟合"

        lines = []
        lines.append("=" * 70)
        lines.append(f"模型: {self.name}")
        lines.append("=" * 70)
        lines.append("")

        # 拟合统计
        lines.append("拟合统计:")
        lines.append(f"  收敛状态: {'成功' if self.fit_result.convergence else '失败'}")
        if not np.isnan(self.fit_result.r2):
            lines.append(f"  R² = {self.fit_result.r2:.4f}")
            lines.append(f"  RMSE = {self.fit_result.rmse:.4f}")
            lines.append(f"  MAE = {self.fit_result.mae:.4f}")
            lines.append(f"  AIC = {self.fit_result.aic:.2f}")
            lines.append(f"  BIC = {self.fit_result.bic:.2f}")
        lines.append("")

        # 参数估计
        lines.append("参数估计:")
        lines.append(f"{'参数':<20} {'估计值':<15} {'标准误':<15}")
        lines.append("-" * 50)
        for name, val, std in zip(self.fit_result.param_names, 
                                   self.fit_result.params, 
                                   self.fit_result.params_std):
            lines.append(f"{name:<20} {val:>14.6f} {std:>14.6f}")

        lines.append("=" * 70)

        return "\n".join(lines)


class AdditiveModel(NLSModel):
    """
    加法模型: 氮沉降通过改变PCA分量的加法效应影响biomass

    模型形式:
    biomass = β0 + Σ(βi * PCi) + Σ(γi * N * PCi)

    其中:
    - β0: 截距
    - βi: PCA分量的主效应
    - γi: 氮沉降与PCA分量的交互效应
    - N: 氮添加量
    - PCi: 第i个主成分
    """

    def __init__(self):
        super().__init__("加法交互模型 (Additive Interaction)")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        """
        X: shape (n_features, n_samples)
           X[0, :] = nitrogen_add
           X[1:, :] = PC components
        """
        nitrogen = X[0, :]
        pcs = X[1:, :]
        n_pcs = pcs.shape[0]

        # 参数分配
        # params = [β0, β1, ..., βn, γ1, ..., γn]
        beta0 = params[0]
        betas = np.array(params[1:n_pcs+1])
        gammas = np.array(params[n_pcs+1:])

        # 计算预测值
        # biomass = β0 + Σ(βi * PCi) + Σ(γi * N * PCi)
        pred = beta0
        pred += np.sum(betas[:, np.newaxis] * pcs, axis=0)
        pred += np.sum(gammas[:, np.newaxis] * nitrogen[np.newaxis, :] * pcs, axis=0)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1  # 减去nitrogen列
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
        p0[0] = np.mean(y)  # 截距初始值为y的均值
        p0[1:n_pcs+1] = 0.1  # 主效应初始值
        p0[n_pcs+1:] = 0.01  # 交互效应初始值（较小）
        return p0


class MultiplicativeModel(NLSModel):
    """
    乘法模型: 氮沉降通过改变PCA分量的乘法效应影响biomass

    模型形式:
    biomass = β0 * exp(Σ(βi * PCi) + Σ(γi * N * PCi))

    或简化为:
    biomass = β0 * Π(exp(βi * PCi)) * Π(exp(γi * N * PCi))
    """

    def __init__(self):
        super().__init__("乘法交互模型 (Multiplicative)")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        nitrogen = X[0, :]
        pcs = X[1:, :]

        beta0 = params[0]
        n_pcs = pcs.shape[0]
        betas = np.array(params[1:n_pcs+1])
        gammas = np.array(params[n_pcs+1:])

        # biomass = β0 * exp(Σ(βi * PCi) + Σ(γi * N * PCi))
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
        p0[0] = np.median(y)  # 尺度参数
        p0[1:n_pcs+1] = 0.01
        p0[n_pcs+1:] = 0.001
        return p0


class MichaelisMentenModel(NLSModel):
    """
    Michaelis-Menten模型: 饱和响应模型

    模型形式:
    biomass = (Vmax * f(PC)) / (Km + N) + baseline

    其中:
    - Vmax: 最大响应
    - Km: 半饱和常数
    - f(PC): PCA分量的线性组合
    - baseline: 基线biomass
    """

    def __init__(self):
        super().__init__("Michaelis-Menten饱和模型")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        nitrogen = X[0, :]
        pcs = X[1:, :]

        # params = [Vmax, Km, baseline, β1, ..., βn]
        Vmax = params[0]
        Km = params[1]
        baseline = params[2]
        betas = np.array(params[3:])

        # f(PC) = Σ(βi * PCi)
        f_pc = np.sum(betas[:, np.newaxis] * pcs, axis=0)

        # biomass = (Vmax * f(PC)) / (Km + N) + baseline
        pred = (Vmax * f_pc) / (Km + nitrogen + 1e-10) + baseline

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1
        return 3 + n_pcs  # Vmax, Km, baseline, βs

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
    指数模型: 指数响应模型

    模型形式:
    biomass = β0 + Σ(αi * PCi) * exp(β * N)

    或:
    biomass = β0 * exp(Σ(βi * PCi) + γ * N)
    """

    def __init__(self, variant: str = 'v1'):
        """
        Args:
            variant: 'v1' 或 'v2'
                v1: biomass = β0 + Σ(αi * PCi) * exp(β * N)
                v2: biomass = β0 * exp(Σ(βi * PCi) + γ * N)
        """
        self.variant = variant
        name = f"指数模型 (Exponential-{variant})"
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
    拟合多个NLS模型

    Args:
        nitrogen_add: 氮添加量数组, shape (n_samples,)
        pca_components: PCA降维后的数据, shape (n_samples, n_components)
        biomass: 生物量数组, shape (n_samples,)
        models: 要拟合的模型列表，如果为None则使用所有默认模型

    Returns:
        字典，键为模型名称，值为ModelFitResult
    """
    logger.info("=" * 70)
    logger.info("开始拟合多个NLS模型")
    logger.info("=" * 70)
    logger.info(f"样本数: {len(biomass)}")
    logger.info(f"PCA分量数: {pca_components.shape[1]}")
    logger.info(f"氮添加量范围: [{nitrogen_add.min():.2f}, {nitrogen_add.max():.2f}]")
    logger.info(f"生物量范围: [{biomass.min():.2f}, {biomass.max():.2f}]")

    # 构建输入矩阵 X = [nitrogen_add, PC1, PC2, ..., PCn]
    X = np.column_stack([nitrogen_add, pca_components])

    # 如果未指定模型，使用所有默认模型
    if models is None:
        models = [
            AdditiveModel(),
            MultiplicativeModel(),
            MichaelisMentenModel(),
            ExponentialModel('v1'),
            ExponentialModel('v2')
        ]

    results = {}

    for model in models:
        logger.info(f"\n拟合模型: {model.name}")
        try:
            result = model.fit(X, biomass)
            results[model.name] = result

            if result.convergence:
                logger.info(f"✓ {model.name} 拟合成功")
                logger.info(f"  R² = {result.r2:.4f}, RMSE = {result.rmse:.4f}")
            else:
                logger.warning(f"✗ {model.name} 拟合失败: {result.message}")
        except Exception as e:
            logger.error(f"✗ {model.name} 拟合出错: {str(e)}")
            continue

    logger.info("\n" + "=" * 70)
    logger.info(f"共拟合 {len(results)} 个模型")
    logger.info("=" * 70)

    return results


def compare_models(results: Dict[str, ModelFitResult]) -> pd.DataFrame:
    """
    比较多个模型的拟合结果

    Args:
        results: 模型拟合结果字典

    Returns:
        比较结果DataFrame
    """
    comparison_data = []

    for model_name, result in results.items():
        if result.convergence:
            comparison_data.append({
                '模型': model_name,
                'R²': result.r2,
                'RMSE': result.rmse,
                'MAE': result.mae,
                'AIC': result.aic,
                'BIC': result.bic,
                '参数数量': len(result.params),
                '收敛': '是'
            })
        else:
            comparison_data.append({
                '模型': model_name,
                'R²': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'AIC': np.inf,
                'BIC': np.inf,
                '参数数量': len(result.params),
                '收敛': '否'
            })

    df = pd.DataFrame(comparison_data)

    # 按AIC排序（越小越好）
    df = df.sort_values('AIC', ascending=True)

    logger.info("\n模型比较结果:")
    logger.info("\n" + df.to_string(index=False))

    return df


# 优化警告类
class OptimizationWarning(UserWarning):
    """优化过程警告"""
    pass

