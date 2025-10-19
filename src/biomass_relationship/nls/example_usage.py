"""
NLS模块使用示例
演示如何使用NLS模块分析氮沉降对生物量的影响
"""

import numpy as np
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.biomass_relationship.nls import (
    NLSAnalyzer,
    quick_nls_analysis,
    AdditiveModel
)


def example_basic_usage():
    """基本使用示例"""
    print("=" * 70)
    print("示例 1: 基本使用")
    print("=" * 70)

    # 模拟数据
    np.random.seed(42)
    n_samples = 100

    # 氮添加量
    nitrogen_add = np.random.uniform(0, 50, n_samples)

    # PCA分量 (假设有3个主成分)
    pca_components = np.random.randn(n_samples, 3)

    # 生物量 (模拟一个加法交互效应)
    biomass = (
        10 +  # 基线
        2 * pca_components[:, 0] +  # PC1效应
        1.5 * pca_components[:, 1] +  # PC2效应
        0.5 * nitrogen_add * pca_components[:, 0] +  # N与PC1交互
        np.random.normal(0, 2, n_samples)  # 噪声
    )

    # 创建输出目录
    output_dir = './example_output'
    os.makedirs(output_dir, exist_ok=True)

    # 使用快速分析函数
    analyzer = quick_nls_analysis(
        nitrogen_add=nitrogen_add,
        pca_components=pca_components,
        biomass=biomass,
        output_dir=output_dir
    )

    print("\n分析完成！结果已保存至:", output_dir)
    print("\n最佳模型:", analyzer.best_model)
    if analyzer.best_model:
        result = analyzer.results[analyzer.best_model]
        print(f"R² = {result.r2:.4f}")
        print(f"RMSE = {result.rmse:.4f}")


def example_advanced_usage():
    """高级使用示例：手动控制分析流程"""
    print("\n" + "=" * 70)
    print("示例 2: 高级使用 - 手动控制")
    print("=" * 70)

    # 模拟数据
    np.random.seed(123)
    n_samples = 80

    nitrogen_add = np.random.uniform(0, 40, n_samples)
    pca_components = np.random.randn(n_samples, 5)

    # 模拟Michaelis-Menten响应
    Vmax = 20
    Km = 15
    biomass = (Vmax * pca_components[:, 0]) / (Km + nitrogen_add) + 5
    biomass += np.random.normal(0, 1, n_samples)

    output_dir = './example_output_advanced'
    os.makedirs(output_dir, exist_ok=True)

    # 创建分析器
    analyzer = NLSAnalyzer(output_dir=output_dir, climate_group=1)

    # 仅拟合指定的模型
    analyzer.analyze(
        nitrogen_add=nitrogen_add,
        pca_components=pca_components,
        biomass=biomass,
        models=['additive', 'michaelis_menten', 'exponential_v1']
    )

    # 保存结果
    analyzer.save_results()

    # 创建可视化（可选）
    analyzer.create_visualizations(nitrogen_add, pca_components, biomass)

    # 生成摘要
    summary = analyzer.get_summary()
    print("\n", summary)


def example_single_model():
    """示例3：拟合单个模型"""
    print("\n" + "=" * 70)
    print("示例 3: 拟合单个模型")
    print("=" * 70)

    # 模拟数据
    np.random.seed(456)
    n_samples = 60

    nitrogen_add = np.random.uniform(5, 45, n_samples)
    pca_components = np.random.randn(n_samples, 4)

    # 简单线性关系
    biomass = 8 + 1.2 * pca_components[:, 0] + 0.3 * nitrogen_add
    biomass += np.random.normal(0, 1.5, n_samples)

    # 构建输入矩阵
    X = np.column_stack([nitrogen_add, pca_components])

    # 创建并拟合单个模型
    model = AdditiveModel()
    model.fit(X, biomass)

    # 打印结果
    print(model.summary())

    # 预测
    y_pred = model.predict(X)
    print(f"\n预测值范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")


def example_group_analysis():
    """示例4：分组分析（不同气候组）"""
    print("\n" + "=" * 70)
    print("示例 4: 分组分析")
    print("=" * 70)

    np.random.seed(789)

    # 模拟两个不同气候组的数据
    output_dir = './example_output_groups'

    for group_id in [1, 2]:
        print(f"\n分析气候组 {group_id}...")

        n_samples = 50
        nitrogen_add = np.random.uniform(0, 50, n_samples)
        pca_components = np.random.randn(n_samples, 3)

        # 不同组有不同的响应模式
        if group_id == 1:
            # 组1: 加法效应
            biomass = 12 + 2 * pca_components[:, 0] + 0.5 * nitrogen_add
        else:
            # 组2: 饱和效应
            biomass = (25 * pca_components[:, 0]) / (20 + nitrogen_add) + 8

        biomass += np.random.normal(0, 2, n_samples)

        # 为每组创建独立分析
        analyzer = NLSAnalyzer(
            output_dir=output_dir,
            climate_group=group_id
        )

        analyzer.run_complete_analysis(
            nitrogen_add=nitrogen_add,
            pca_components=pca_components,
            biomass=biomass,
            create_plots=True
        )

        print(f"组 {group_id} 最佳模型: {analyzer.best_model}")


if __name__ == "__main__":
    print("NLS模块使用示例\n")

    try:
        # 运行各个示例
        example_basic_usage()
        example_advanced_usage()
        example_single_model()
        example_group_analysis()

        print("\n" + "=" * 70)
        print("所有示例运行完成！")
        print("=" * 70)

    except (RuntimeError, ValueError, IOError) as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

