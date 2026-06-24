# ISTA与FISTA (Iterative Shrinkage-Thresholding Algorithm)

> [!info] 📚 拓展知识点说明
> **定位**：本知识点是本科课程的拓展内容，适合学有余力的同学深入学习。
> **一句话理解**：迭代收缩阈值算法，一步步迭代逼近稀疏解。
> **前置要求**：凸优化基础、近端算子概念。
> **与本科课程的关系**：与展开网络密切相关——LISTA就是ISTA的展开版本。

## 1. 学习定位

- **建议周次**：W10-W11
- **学习深度**：必须掌握（ISTA/FISTA是稀疏信号恢复的核心算法，也是展开网络的基础）
- **关联实验**：实验6（稀疏信号恢复）、实验7（展开网络设计）
- **关联项目**：实现LISTA展开网络并与ISTA对比、图像去模糊的FISTA实现
- **前置知识**：[[../优化方法/凸优化基础]]（凸性、最优性条件）、[[../优化方法/近端算子]]（软阈值算子）
- **后续延伸**：[[../深度学习/展开网络]]（LISTA）、[[ADMM]]（另一种分裂算法）

## 2. 通俗入口

**一个真实问题**：你有一张被模糊+噪声污染的照片，想恢复清晰图像。这个问题可以建模为"在图像先验（稀疏性）约束下拟合观测数据"。ISTA就是解决这个问题的经典迭代算法。

**生活类比**：ISTA就像在迷宫中摸索前进——每一步先沿最陡的方向走一步（梯度下降），然后"修剪"掉不重要的分量（软阈值）。FISTA则像一个聪明的探索者——它记住了上一步的方向，利用"惯性"加速前进，比ISTA更快到达终点。

**学这个能解决什么**：压缩感知信号恢复、图像去噪/去模糊、LASSO回归、稀疏编码——凡是涉及 $l_1$ 正则化的优化问题，ISTA/FISTA都是首选算法之一。

## 3. 严谨核心

**数学对象**：复合目标函数 $F(\mathbf{x}) = f(\mathbf{x}) + g(\mathbf{x})$，其中 $f$ 光滑凸、$g$ 非光滑凸

**基本假设**：
- $f(\mathbf{x})$ 是凸函数，梯度 $\nabla f$ 具有 Lipschitz 常数 $L$
- $g(\mathbf{x})$ 是闭真凸函数，其近端算子有闭式解（如 $l_1$ 范数→软阈值）

ISTA和FISTA是求解稀疏优化问题的迭代算法，属于近端梯度法（Proximal Gradient Method）的具体实现，用于解决含 $\ell_1$ 正则化的优化问题。

核心问题形式：

$$\min_{\mathbf{x}} F(\mathbf{x}) = f(\mathbf{x}) + g(\mathbf{x})$$

其中 $f(\mathbf{x})$ 是光滑凸函数（如数据保真项 $\frac{1}{2}\|\mathbf{y}-\mathbf{A}\mathbf{x}\|_2^2$），$g(\mathbf{x})$ 是非光滑凸函数（如 $\lambda\|\mathbf{x}\|_1$）。

**核心公式：**

### ISTA

迭代格式：

$$\mathbf{x}^{(k+1)} = \mathcal{S}_{\lambda/L}\left(\mathbf{x}^{(k)} - \frac{1}{L}\nabla f(\mathbf{x}^{(k)})\right)$$

其中 $L$ 是 $\nabla f$ 的Lipschitz常数，$\mathcal{S}_\tau$ 为软阈值算子：

$$[\mathcal{S}_\tau(\mathbf{z})]_i = \text{sign}(z_i) \cdot \max(|z_i| - \tau, 0)$$

对于 $f(\mathbf{x}) = \frac{1}{2}\|\mathbf{y}-\mathbf{A}\mathbf{x}\|_2^2$，有 $L = \|\mathbf{A}^T\mathbf{A}\|_2$（最大特征值）。

收敛速率：$F(\mathbf{x}^{(k)}) - F(\mathbf{x}^*) \leq \frac{L\|\mathbf{x}^{(0)}-\mathbf{x}^*\|_2^2}{2k}$

### FISTA

引入动量变量的加速格式：

$$\mathbf{y}^{(k)} = \mathbf{x}^{(k)} + \frac{t_{k-1}-1}{t_k}(\mathbf{x}^{(k)} - \mathbf{x}^{(k-1)})$$

$$\mathbf{x}^{(k+1)} = \mathcal{S}_{\lambda/L}\left(\mathbf{y}^{(k)} - \frac{1}{L}\nabla f(\mathbf{y}^{(k)})\right)$$

$$t_{k+1} = \frac{1+\sqrt{1+4t_k^2}}{2}$$

其中初始值 $t_0 = 1$（或 $t_1 = 1$，依具体实现而定）。

收敛速率：$F(\mathbf{x}^{(k)}) - F(\mathbf{x}^*) \leq \frac{2L\|\mathbf{x}^{(0)}-\mathbf{x}^*\|_2^2}{(k+1)^2}$

**适用条件和边界**：
- 凸+光滑 $f$+近端算子可用：ISTA/FISTA适用且有收敛保证
- 非凸问题：近端梯度法仍可运行，但只有收敛到驻点的保证
- 当 $g$ 的近端算子无闭式解时，每步需要内嵌求解，计算代价大幅上升

## 4. 方法流程

**输入**：矩阵 $\mathbf{A}$，观测向量 $\mathbf{y}$，正则化参数 $\lambda$，最大迭代次数

**处理步骤**：
1. 计算 Lipschitz 常数 $L = \|\mathbf{A}^T\mathbf{A}\|_2$
2. 初始化 $\mathbf{x}^{(0)} = \mathbf{0}$（FISTA还需 $t_0 = 1$）
3. **梯度步**：$\mathbf{z} = \mathbf{x}^{(k)} - \frac{1}{L}\mathbf{A}^T(\mathbf{A}\mathbf{x}^{(k)} - \mathbf{y})$
4. **近端步（软阈值）**：$\mathbf{x}^{(k+1)} = \mathcal{S}_{\lambda/L}(\mathbf{z})$
5. （FISTA额外步骤）计算动量外推点 $\mathbf{y}^{(k+1)}$
6. 检查收敛条件（$\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\| < \epsilon$ 或目标值变化小于阈值）
7. 未收敛则返回步骤3

**输出如何解释**：
- $\mathbf{x}^*$：稀疏恢复信号，非零分量对应信号的主要成分
- 目标函数值序列 $F(\mathbf{x}^{(k)})$：反映收敛过程，FISTA下降更快
- 支撑集 $\text{supp}(\mathbf{x}^*)$：恢复出的非零分量位置

**常见参数如何选择**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| $\lambda$ | 与噪声水平相关 | 噪声大→$\lambda$ 大（更稀疏） |
| $L$ | $\|\mathbf{A}^T\mathbf{A}\|_2$ | 保证收敛的上界，过大导致步长过小 |
| 最大迭代 | 200~1000 | 视精度要求而定 |

## ISTA 与 FISTA 迭代流程对比

## ISTA 与 FISTA 性能对比

| 特性 | ISTA | FISTA |
| ------|------|-------|
| 收敛速率 | O(1/k) | O(1/k²) |
| 每步计算量 | 1次梯度 + 1次软阈值 | 1次梯度 + 1次软阈值 + 动量更新 |
| 内存需求 | 1个变量 | 2个变量（x⁽ᵏ⁾, x⁽ᵏ⁻¹⁾） |
| 单调收敛 | ✓（目标值单调下降） | ✗（非单调，需重启技巧） |
| 步长选择 | L = ‖AᵀA‖₂ | L = ‖AᵀA‖₂ |
| 适用场景 | 需要稳定收敛 | 需要快速收敛 |

**典型应用：**
- **压缩感知信号恢复**：从欠采样测量中恢复稀疏信号
- **图像去噪与去模糊**：$\ell_1$ 正则化的图像复原问题
- **稀疏编码**：字典学习中的稀疏编码步骤
- **LASSO回归**：统计学习中的变量选择问题
- **矩阵补全**：推荐系统中的缺失数据填充（需结合其他技巧）
- **CT/MRI重建**：医学图像的稀疏重建

![[assets/ista_fista_comparison_ai.png]]

*图：ISTA 与 FISTA 收敛速度对比：FISTA 的加速梯度策略带来显著的收敛提升。*

## 5. Python最小实验

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

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ISTA与FISTAConvergenceVelocity对比演示
np.random.seed(42)
m, n = 64, 128  # 欠定System m < n
A = np.random.randn(m, n) / np.sqrt(m)
L = np.max(np.linalg.eigvalsh(A.T @ A))  # Lipschitz常数
x_true = np.zeros(n)
x_true[[10, 30, 50, 70]] = [3, -2, 1.5, -1]  # SparseSignal
y = A @ x_true + 0.1 * np.random.randn(m)  # 含噪Observation
lam = 0.5  # Regularization参数

# 软阈值算子
def soft_thresh(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)

# ISTA算法
def ista(A, y, lam, L, max_iter=200):
    x = np.zeros(A.shape[1])
    costs = []
    for _ in range(max_iter):
        grad = A.T @ (A @ x - y)
        x = soft_thresh(x - grad / L, lam / L)
        cost = 0.5 * np.linalg.norm(y - A @ x)**2 + lam * np.linalg.norm(x, 1)
        costs.append(cost)
    return x, costs

# FISTA算法
def fista(A, y, lam, L, max_iter=200):
    x = np.zeros(A.shape[1])
    z = x.copy()
    t = 1.0
    costs = []
    for _ in range(max_iter):
        grad = A.T @ (A @ z - y)
        x_new = soft_thresh(z - grad / L, lam / L)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + (t - 1) / t_new * (x_new - x)
        x, t = x_new, t_new
        cost = 0.5 * np.linalg.norm(y - A @ x)**2 + lam * np.linalg.norm(x, 1)
        costs.append(cost)
    return x, costs

_, cost_ista = ista(A, y, lam, L)
_, cost_fista = fista(A, y, lam, L)
opt = min(min(cost_ista), min(cost_fista))

plt.figure(figsize=(10, 5))
plt.semilogy(np.array(cost_ista) - opt, label='ISTA (O(1/k))', linewidth=2)
plt.semilogy(np.array(cost_fista) - opt, label='FISTA (O(1/k²))', linewidth=2)
plt.xlabel('Iteration', fontsize=13)
plt.ylabel('目标Function值误差 (log)', fontsize=13)
plt.title('ISTA与FISTAConvergenceVelocity对比', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**结果解释**：
- 对数尺度下，FISTA的误差曲线斜率约为ISTA的两倍——对应 $O(1/k^2)$ vs $O(1/k)$ 的理论收敛速率差异
- 在前50次迭代中FISTA的优势尤其明显，适合需要快速得到"够好"解的场景
- 两种算法最终收敛到相同的目标值（同为凸问题的全局最优），区别只在收敛速度

## 6. 常见误区

1. **FISTA一定比ISTA好**：FISTA是非单调的——目标函数值可能偶尔上升。在需要严格单调收敛的场景（如作为外层循环的子问题求解器），ISTA可能更稳定。

2. **步长 $1/L$ 一定是最优的**：$1/L$ 保证收敛但不一定最快。实践中可用线搜索（backtracking line search）自适应选择步长，常能得到更好的实际性能。

3. **ISTA只能处理 $l_1$ 正则化**：任何近端算子有闭式解的正则化项都可以用ISTA/FISTA处理——$l_2$ 范数、核范数、指示函数等。

4. **收敛速率 $O(1/k^2)$ 是最优的**：对于一阶方法最小化光滑凸函数，Nesterov证明了 $O(1/k^2)$ 是最优下界。但对于复合问题（光滑+非光滑），最优速率取决于具体问题结构。

5. **$l_1$ 正则化的解一定是稀疏的**：$l_1$ 正则化倾向于产生稀疏解，但不保证。当感知矩阵 $\mathbf{A}$ 的列高度相关时，$l_1$ 正则化的解可能不稀疏（"群组效应"）。

## 7. 启发问题与创新拓展

**一个反例或边界问题**：
考虑 $f(\mathbf{x}) = \frac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2$ 其中 $\mathbf{A}$ 的条件数 $\kappa = 10^6$。此时 $L = \lambda_{\max}(\mathbf{A}^T\mathbf{A})$ 很大，导致步长 $1/L$ 极小，ISTA每步只前进一点点。而使用预条件（preconditioning）或坐标下降法可以大幅改善这种情况。

**一个跨学科迁移场景**：
在天文学中，射电干涉仪测量的是天空亮度的傅里叶系数（欠采样）。将天空亮度建模为小波域稀疏信号，用FISTA从欠采样的uv覆盖中重建天空图像。这就是"压缩感知射电天文学"的核心方法。

**一个可做成课程项目的小想法**：
"LISTA：把ISTA展开成神经网络"——将ISTA的 $K$ 次迭代"展开"为 $K$ 层神经网络，每层的软阈值参数 $\lambda_k$ 和权重矩阵 $\mathbf{W}_k$ 设为可学习参数。用训练数据学习这些参数，比较LISTA与原始ISTA在收敛速度和恢复精度上的差异。这是"深度展开网络"的入门实践。

## 参考文献

1. Beck, A. & Teboulle, M. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." *SIAM Journal on Imaging Sciences*, 2(1): 183-202, 2009.
2. Daubechies, I., Defrise, M. & De Mol, C. "An iterative thresholding algorithm for linear inverse problems with a sparsity constraint." *Communications on Pure and Applied Mathematics*, 57(11): 1413-1457, 2004.
3. Nesterov, Y. "A method of solving a convex programming problem with convergence rate $O(1/k^2)$." *Soviet Mathematics Doklady*, 27(2): 372-376, 1983.
4. Parikh, N. & Boyd, S. "Proximal algorithms." *Foundations and Trends in Optimization*, 1(3): 127-239, 2014.

---

## 相关知识点

- [[稀疏表示与字典学习]] — 稀疏编码
- [[压缩感知]] — 信号重构
- [[../优化方法/近端算子]] — 近端梯度法
- [[../深度学习/展开网络]] — LISTA网络
- [[ADMM]] — 另一种分裂算法
