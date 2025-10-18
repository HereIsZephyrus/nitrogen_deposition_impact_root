# GMM Cluster Strictness Parameters Guide

## Overview
这三个参数用于控制聚类的严格程度，确保每个类内部紧凑且类间分离良好。

## Parameters

### 1. `max_covariance_det` (Maximum Covariance Determinant)

**含义**: 协方差矩阵行列式的最大值，衡量聚类的"体积"

**数学意义**: 
- 行列式越小，聚类在多维空间中占据的体积越小，表示越紧凑
- 对于d维数据，det(Σ) ≈ (体积)²

**设置建议**:
- **标准化数据** (均值0，方差1):
  - 非常严格: `0.01 - 0.1` (适用于2-3维)
  - 中等严格: `0.1 - 1.0` (适用于3-5维)  
  - 宽松: `1.0 - 10.0` (适用于5+维)

- **未标准化数据**: 需要根据数据尺度调整，建议先标准化

**注意**: 
- 维度越高，行列式值增长越快，阈值需要相应增大
- 对于d维标准化数据，参考值约为: `0.1 ^ d` 到 `1.0 ^ d`

---

### 2. `min_cluster_separation` (Minimum Cluster Separation Ratio)

**含义**: 类间中心距离与平均标准差的比值

**计算公式**:
```
separation = ||μᵢ - μⱼ|| / ((σᵢ + σⱼ) / 2)
```

**物理意义**:
- 衡量两个聚类中心的距离是其尺寸的多少倍
- 类似于"信噪比"的概念

**设置建议**:
- **非常严格**: `> 3.0` - 要求聚类之间有明显分离，几乎无重叠
- **严格**: `2.0 - 3.0` - 聚类有清晰的边界
- **中等**: `1.0 - 2.0` - 聚类有一定重叠但仍可区分
- **宽松**: `< 1.0` - 允许较大重叠

**推荐起始值**: 
- 标准化数据: `1.5 - 2.0`
- 未标准化数据: 需要实验调整

---

### 3. `max_mean_mahalanobis` (Maximum Mean Mahalanobis Distance)

**含义**: 聚类内样本到中心的平均马氏距离上限

**数学意义**:
- 马氏距离考虑了数据的协方差结构
- 对于标准正态分布，期望马氏距离约为 `√d`（d为维度）

**设置建议**:
- **标准化数据**:
  - 非常紧凑: `0.5 * √d` 到 `0.8 * √d`
  - 中等紧凑: `0.8 * √d` 到 `1.2 * √d`
  - 宽松: `1.2 * √d` 到 `2.0 * √d`

**维度参考表** (标准化数据):
| 维度 d | √d | 严格 | 中等 | 宽松 |
|--------|-------|------|------|------|
| 2      | 1.41  | 1.0  | 1.5  | 2.5  |
| 3      | 1.73  | 1.2  | 1.8  | 3.0  |
| 4      | 2.00  | 1.4  | 2.0  | 3.5  |
| 5      | 2.24  | 1.5  | 2.2  | 4.0  |
| 10     | 3.16  | 2.0  | 3.2  | 5.5  |

---

## Recommended Settings by Use Case

### Case 1: Climate Zones (High Distinctness Required)
```python
# 假设: 5维标准化气候数据
GMMCluster(
    n_components=5,
    confidence=0.95,
    max_covariance_det=1.0,           # 相对严格的体积限制
    min_cluster_separation=2.0,        # 要求明显分离
    max_mean_mahalanobis=2.5          # √5 ≈ 2.24, 略宽松
)
```

### Case 2: Exploratory Analysis (Moderate Strictness)
```python
# 假设: 3维标准化数据
GMMCluster(
    n_components=3,
    confidence=0.90,
    max_covariance_det=0.5,           # 中等紧凑度
    min_cluster_separation=1.5,        # 中等分离度
    max_mean_mahalanobis=2.0          # √3 ≈ 1.73, 略宽松
)
```

### Case 3: Strict Quality Control
```python
# 假设: 4维标准化数据
GMMCluster(
    n_components=4,
    confidence=0.95,
    max_covariance_det=0.1,           # 非常严格
    min_cluster_separation=3.0,        # 非常严格
    max_mean_mahalanobis=1.5          # √4 = 2.0, 很严格
)
```

---

## Practical Workflow

### Step 1: Start Without Constraints
```python
# 先不设置约束，观察自然聚类质量
gmm = GMMCluster(n_components=k, confidence=0.9)
gmm.fit(data)
result = gmm.predict(data)
result.save('output.csv')  # 查看质量指标
```

### Step 2: Analyze Quality Metrics
查看输出的质量指标，了解数据的自然聚类特性：
- 协方差行列式的范围
- 类间分离度
- 平均马氏距离

### Step 3: Set Appropriate Thresholds
基于Step 2的观察，设置略严格于自然值的阈值：
```python
# 假设观察到:
# - covariance_det: [0.5, 0.8, 1.2]
# - separation: 1.8
# - mean_mahalanobis: [2.5, 2.8, 2.3]

# 设置略严格的阈值:
gmm = GMMCluster(
    n_components=k,
    confidence=0.9,
    max_covariance_det=1.0,      # 略低于最大观察值1.2
    min_cluster_separation=2.0,   # 略高于观察值1.8
    max_mean_mahalanobis=2.5     # 略低于最大观察值2.8
)
```

### Step 4: Iterate
如果聚类失败（抛出异常），逐步放宽约束直到找到合理平衡点。

---

## Important Notes

1. **数据必须标准化**: 如果数据未标准化，这些阈值将无意义
2. **维度影响**: 维度越高，所有指标都会增大
3. **样本量影响**: 样本量小时，估计的协方差矩阵可能不稳定
4. **Trade-off**: 约束越严格，可能导致聚类失败或需要更少的聚类数

## Debugging Tips

如果遇到 `ValueError: Cluster quality does not meet requirements`:

1. 检查哪个约束失败（查看日志）
2. 逐个放宽约束参数
3. 考虑减少聚类数 K
4. 检查数据是否已正确标准化
5. 检查是否有离群点影响聚类质量

