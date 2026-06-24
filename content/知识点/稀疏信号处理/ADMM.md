# ADMM (Alternating Direction Method of Multipliers)

## 1. 学习定位

> [!info] 📚 拓展知识点说明
> **定位**：本知识点是本科课程的拓展内容，适合学有余力的同学深入学习。
> **一句话理解**：交替方向乘子法，将大问题拆分为小问题交替求解。
> **前置要求**：凸优化基础、近端算子概念。
> **与本科课程的关系**：与展开网络密切相关——ADMM-Net即基于此算法展开。

**前置知识**：
- **凸优化基础**：凸集、凸函数、拉格朗日对偶
- **近端算子**：近端映射、近端算子求解子问题

**相关知识点**：
- [[ISTA与FISTA]] — 迭代优化算法
- [[压缩感知]] — CS重构
- [[../优化方法/ADMM算法]] — 优化方法视角
- [[../优化方法/近端算子]] — 子问题求解
- [[../深度学习/展开网络]] — ADMM-Net

> **与 [[../优化方法/ADMM算法]] 的关系：** 本文从稀疏信号处理视角介绍ADMM，侧重压缩感知重建、稀疏正则化等应用场景；优化方法下的 [[../优化方法/ADMM算法]] 则从通用优化理论视角介绍ADMM的数学框架、收敛性分析和分裂策略。两者内容有重叠但侧重不同，建议对照阅读。

## 2. 通俗入口

想象你是一个项目经理，要完成一项大工程：一边要画图纸（优化 x），一边要买材料（优化 z），但两者必须协调一致（满足约束 Ax+Bz=c）。你不会同时做两件事，而是交替推进——先画好图纸交给采购，采购更新进度后反馈给你，你再调整图纸。这个"交替协调"的过程就是 ADMM 的核心思想。

**为什么需要 ADMM？** 当目标函数可以拆分成两部分（一个管 x，一个管 z），但它们被一个等式约束绑在一起时，直接求解很困难。ADMM 通过引入辅助变量和对偶变量，把大问题拆成两个小问题交替求解，每个子问题都更容易处理。

**关键词联想**：
- "交替" → 轮流优化 x 和 z
- "方向" → 对偶变量引导搜索方向
- "乘子" → 拉格朗日乘子（惩罚约束违反）

## 3. 严谨核心

### 定义与概念

ADMM（交替方向乘子法）是一种将对偶上升法的可分解性与乘子法的收敛性相结合的优化算法，特别适合求解具有可分离结构的凸优化问题。

ADMM的历史可追溯到20世纪70年代，由Glowinski、Marrocco以及Gabay、Mercier独立提出。近年来在大规模分布式优化和稀疏信号处理中得到广泛应用。

核心思想：通过引入辅助变量将原问题分解为多个子问题，交替求解各子问题并更新对偶变量（乘子）。

关键概念：
- **增广拉格朗日函数**：在拉格朗日函数中加入二次罚项，改善收敛性
- **对偶变量（乘子）**：对应等式约束的拉格朗日乘子
- **可分离结构**：目标函数可分解为仅依赖不同变量的独立部分
- **罚参数** $\rho$：控制二次罚项的权重，影响收敛速度

### 核心公式

标准ADMM求解的问题形式：

$$\min_{\mathbf{x}, \mathbf{z}} f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}$$

增广拉格朗日函数：

$$L_\rho(\mathbf{x}, \mathbf{z}, \boldsymbol{\mu}) = f(\mathbf{x}) + g(\mathbf{z}) + \boldsymbol{\mu}^T(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}\|_2^2$$

ADMM迭代格式（三步更新）：

$$\mathbf{x}^{(k+1)} = \arg\min_{\mathbf{x}} L_\rho(\mathbf{x}, \mathbf{z}^{(k)}, \boldsymbol{\mu}^{(k)})$$

$$\mathbf{z}^{(k+1)} = \arg\min_{\mathbf{z}} L_\rho(\mathbf{x}^{(k+1)}, \mathbf{z}, \boldsymbol{\mu}^{(k)})$$

$$\boldsymbol{\mu}^{(k+1)} = \boldsymbol{\mu}^{(k)} + \rho(\mathbf{A}\mathbf{x}^{(k+1)} + \mathbf{B}\mathbf{z}^{(k+1)} - \mathbf{c})$$

缩放形式（令 $\mathbf{u} = \boldsymbol{\mu}/\rho$ 为缩放对偶变量）：

$$\mathbf{x}^{(k+1)} = \arg\min_{\mathbf{x}} \left(f(\mathbf{x}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z}^{(k)} - \mathbf{c} + \mathbf{u}^{(k)}\|_2^2\right)$$

$$\mathbf{z}^{(k+1)} = \arg\min_{\mathbf{z}} \left(g(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x}^{(k+1)} + \mathbf{B}\mathbf{z} - \mathbf{c} + \mathbf{u}^{(k)}\|_2^2\right)$$

$$\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} + \mathbf{A}\mathbf{x}^{(k+1)} + \mathbf{B}\mathbf{z}^{(k+1)} - \mathbf{c}$$

收敛准则（原始残差和对偶残差）：

$$r^{(k)} = \mathbf{A}\mathbf{x}^{(k)} + \mathbf{B}\mathbf{z}^{(k)} - \mathbf{c}, \quad s^{(k)} = \rho \mathbf{A}^T \mathbf{B}(\mathbf{z}^{(k)} - \mathbf{z}^{(k-1)})$$

### 典型应用

- **全变分（TV）去噪/去模糊**：$\min \frac{1}{2}\|\mathbf{y}-\mathbf{H}\mathbf{x}\|_2^2 + \lambda\|\nabla\mathbf{x}\|_1$
- **LASSO与基追踪**：$\ell_1$ 正则化回归问题
- **分布式优化**：大规模机器学习中的分布式ADMM
- **图像复原**：去噪、去模糊、超分辨率
- **矩阵分解**：低秩矩阵恢复、主成分追踪
- **稀疏信号恢复**：结合 $\ell_1$ 正则项的信号重建
- **统计学习**：LASSO、Group LASSO、Elastic Net等

### ADMM 与其他优化方法对比

| 方法 | 可分解性 | 收敛速度 | 参数调节 | 适用问题规模 | 典型应用 |
| ------|---------|---------|---------|------------|---------|
| ADMM | ✓（可分离子问题） | 收敛但较慢 | 需调 ρ | 大规模分布式 | TV去噪、LASSO、分布式优化 |
| 梯度下降 | ✗ | 线性 | 步长/学习率 | 中小规模 | 通用光滑优化 |
| ISTA/FISTA | 部分 | O(1/k)~O(1/k²) | Lipschitz常数 | 中大规模 | ℓ₁ 正则化问题 |
| 原始对偶法 | ✓ | 线性 | 步长参数 | 大规模 | 变分不等式、鞍点问题 |
| 内点法 | ✗ | 超线性 | 少 | 小规模（精确） | 小规模凸优化 |

## 4. 方法流程

### ADMM 迭代流程图

**Step 1：问题建模**
将原问题改写为 ADMM 标准形式：
$$\min f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}$$
- 识别可分离结构，引入辅助变量 z
- 例如 LASSO：令 z=x，约束 x=z

**Step 2：构造增广拉格朗日函数**
$$L_\rho(\mathbf{x}, \mathbf{z}, \boldsymbol{\mu}) = f(\mathbf{x}) + g(\mathbf{z}) + \boldsymbol{\mu}^T(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}\|_2^2$$

**Step 3：交替迭代（核心循环）**
1. **x-更新**：固定 z, μ，求解关于 x 的子问题（通常为最小二乘或近端算子）
2. **z-更新**：固定 x, μ，求解关于 z 的子问题（通常涉及近端算子如软阈值）
3. **μ-更新**：对偶变量梯度上升 $\boldsymbol{\mu} \leftarrow \boldsymbol{\mu} + \rho(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c})$

**Step 4：收敛判断**
- 原始残差 $\|r^{(k)}\| < \epsilon_{\text{pri}}$
- 对偶残差 $\|s^{(k)}\| < \epsilon_{\text{dual}}$
- 两者同时满足时停止

**参数选择建议**：
- ρ 通常从 1.0 开始，可根据残差比值自适应调整
- ρ 增大 → 侧重约束满足，但子问题更难求解
- ρ 减小 → 子问题易解，但收敛可能变慢

![[assets/admm_iteration_ai.png]]

*图：ADMM 迭代过程：交替优化原始变量、辅助变量和对偶变量的收敛示意。*

## 5. Python 最小实验


**运行前准备**：本代码需要以下Python库，请确保已安装：
- 本地运行：`pip install numpy matplotlib`
- Pyodide环境：先运行下面的安装代码块

```python
# Pyodide环境：安装依赖库
import micropip
await micropip.install(["numpy", "matplotlib"])
```

```python
# === 环境预加载 ===
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ADMM求解LASSO问题演示
# min 0.5*||Ax - b||^2 + lam*||z||_1, s.t. x = z
np.random.seed(42)
m, n = 50, 100
A = np.random.randn(m, n) / np.sqrt(m)
x_true = np.zeros(n)
x_true[[10, 30, 50, 70]] = [3, -2, 1.5, -1]
b = A @ x_true + 0.1 * np.random.randn(m)
lam = 0.5
rho = 1.0  # 罚参数

def soft_thresh(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)

# ADMM迭代
x = np.zeros(n)
z = np.zeros(n)
u = np.zeros(n)  # 缩放对偶变量
max_iter = 100
primal_res, dual_res, obj_vals = [], [], []

for k in range(max_iter):
    # x更新：最小二乘问题
    x = np.linalg.solve(A.T @ A + rho * np.eye(n), A.T @ b + rho * (z - u))
    # z更新：软阈值
    z_old = z.copy()
    z = soft_thresh(x + u, lam / rho)
    # u更新：对偶变量
    u = u + x - z
    # 记录残差和目标值
    r = x - z  # 原始残差
    s = rho * (z - z_old)  # 对偶残差
    primal_res.append(np.linalg.norm(r))
    dual_res.append(np.linalg.norm(s))
    obj_vals.append(0.5 * np.linalg.norm(b - A @ x)**2 + lam * np.linalg.norm(z, 1))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].semilogy(primal_res, 'b-', label='Primal Residual', linewidth=2)
axes[0].semilogy(dual_res, 'r--', label='Dual Residual', linewidth=2)
axes[0].set_title('ADMM残差收敛', fontsize=13)
axes[0].set_xlabel('Iterations')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(obj_vals, 'g-', linewidth=2)
axes[1].set_title('目标函数值', fontsize=13)
axes[1].set_xlabel('Iterations')
axes[1].grid(True, alpha=0.3)
axes[2].stem(range(n), x, linefmt='r-', markerfmt='ro', basefmt='k-', label='ADMM恢复')
axes[2].stem(range(n), x_true, linefmt='b-', markerfmt='b^', basefmt='k-', label='True Value')
axes[2].set_title('Signal Recovery Comparison', fontsize=13)
axes[2].set_xlabel('索引')
axes[2].legend(fontsize=10)
plt.suptitle('ADMM求解LASSO问题', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
```

## 6. 常见误区

| 误区 | 正确理解 |
|------|----------|
| ADMM 对所有凸问题都收敛很快 | ADMM 保证收敛，但速度通常是 O(1/k)，比内点法慢；优势在于可分解性和分布式能力 |
| ρ 越大越好 | ρ 过大会使子问题病态（条件数增大），应根据残差比值自适应调整 |
| ADMM 只能处理等式约束 | 不等式约束可通过引入松弛变量转化为等式约束 |
| x-更新和 z-更新必须精确求解 | 实际中常使用近似求解（如少量迭代的内点法），称为 inexact ADMM |
| ADMM 收敛意味着原始残差和对偶残差都趋于零 | 是的，但实际中常设置容忍度而非精确为零，且残差下降速度可能不同步 |
| ADMM 与对偶上升法完全相同 | ADMM 在对偶上升基础上加入了二次罚项（增广拉格朗日），改善了收敛性 |

## 7. 启发问题与创新拓展

### 思考题
1. ADMM 中 x-更新和 z-更新的顺序能否交换？交换后收敛性是否改变？
2. 如何选择罚参数 ρ？如果 ρ 自适应调整，对收敛速度有何影响？
3. ADMM 与 ISTA/FISTA 都能求解 ℓ₁ 正则化问题，各自的优劣是什么？
4. 如何将 ADMM 推广到三个或更多可分离变量的情形（multi-block ADMM）？

### 创新拓展方向
- **ADMM-Net / 展开网络**：将 ADMM 迭代展开为神经网络层，每层参数可学习，结合模型驱动与数据驱动
- **分布式 ADMM**：大规模联邦学习、边缘计算中的分布式优化
- **非凸 ADMM**：将 ADMM 应用于非凸问题（如低秩矩阵恢复），收敛性分析仍是开放问题
- **随机 ADMM**：每次迭代只用部分数据，适合超大规模问题
- **自适应 ρ 策略**：根据原始/对偶残差比值动态调整罚参数

## 参考文献

1. Boyd, S., Parikh, N., Chu, E., Peleato, B. & Eckstein, J. "Distributed optimization and statistical learning via the alternating direction method of multipliers." *Foundations and Trends in Machine Learning*, 3(1): 1-122, 2011.
2. Boyd, S. & Vandenberghe, L. *Convex Optimization*. Cambridge University Press, 2004.
3. Gabay, D. & Mercier, B. "A dual algorithm for the solution of nonlinear variational problems via finite element approximation." *Computers & Mathematics with Applications*, 2(1): 17-40, 1976.
4. Glowinski, R. & Marrocco, A. "Sur l'approximation par éléments finis d'ordre un, et la résolution, par pénalisation-dualité, d'une classe de problèmes de Dirichlet non linéaires." *Revue Française d'Automatique, Informatique et Recherche Opérationnelle*, 9(R2): 41-76, 1975.
