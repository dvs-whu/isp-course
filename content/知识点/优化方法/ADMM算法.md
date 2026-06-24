# ADMM算法

> [!info] 📚 拓展知识点说明
> **定位**：本知识点是本科课程的拓展内容，适合学有余力的同学深入学习。
> **一句话理解**：把大问题拆成小问题交替求解，像左右手轮流干活。
> **前置要求**：凸优化基础、近端算子。
> **与本科课程的关系**：展开网络中的 ADMM-Net，是 ADMM 算法在深度展开网络中的典型应用。

## 1. 学习定位

- **建议周次**：W9-W10
- **学习深度**：会用即可（掌握ADMM的迭代格式和参数调节，理解收敛性结论即可）
- **关联实验**：实验6（稀疏信号恢复）、实验8（展开网络设计）
- **关联项目**：ADMM求解TV去噪、分布式ADMM实现联邦学习参数聚合
- **前置知识**：[[凸优化基础]]（拉格朗日对偶、KKT条件）、[[近端算子]]
- **后续延伸**：[[../稀疏信号处理/ADMM]]（信号处理视角）、[[../深度学习/展开网络]]（ADMM-Net）

## 2. 通俗入口

**一个真实问题**：一个大工程要同时完成地基建设和设备安装，但这两个团队的工序互相依赖。解决方案：A团队先干自己的部分，B团队再根据A的成果干自己的部分，然后A根据B的反馈调整……如此交替，直到双方的工作成果协调一致。

**生活类比**：ADMM就像两个人合作搬一张大桌子过窄门。一个人负责控制桌子的前后位置（x子问题），另一个人负责控制左右角度（z子问题），他们通过喊话协调（对偶变量更新），最终找到一个桌子既不碰墙又在门框内通过的位置。

**学这个能解决什么**：大规模优化问题往往无法一次性求解，ADMM将其分解为多个小规模子问题交替求解，天然适合分布式计算和并行处理。

## 3. 严谨核心

**数学对象**：可分离的凸优化问题，变量 $\mathbf{x} \in \mathbb{R}^n, \mathbf{z} \in \mathbb{R}^m$，对偶变量 $\mathbf{y} \in \mathbb{R}^p$

**基本假设**：
- $f, g$ 是闭、真、凸函数
- 增广拉格朗日函数 $L_0$ 有鞍点（保证收敛）

ADMM（Alternating Direction Method of Multipliers，交替方向乘子法）是一种求解带结构的凸优化问题的分解算法，结合了对偶分解（Dual Decomposition）的可分解性和增广拉格朗日方法（Augmented Lagrangian）的良好收敛性。

**适用问题形式：**
$$\begin{aligned} \min_{\mathbf{x}, \mathbf{z}} \quad & f(\mathbf{x}) + g(\mathbf{z}) \\ \text{s.t.} \quad & A\mathbf{x} + B\mathbf{z} = \mathbf{c} \end{aligned}$$

**核心思想：**
- 将原问题分解为两个（或多个）较易求解的子问题
- 交替更新 $\mathbf{x}$ 和 $\mathbf{z}$，而非同时更新
- 通过对偶变量（乘子）的更新来协调子问题间的约束

**核心公式：**

**增广拉格朗日函数：**
$$L_\rho(\mathbf{x}, \mathbf{z}, \mathbf{y}) = f(\mathbf{x}) + g(\mathbf{z}) + \mathbf{y}^T(A\mathbf{x} + B\mathbf{z} - \mathbf{c}) + \frac{\rho}{2}\|A\mathbf{x} + B\mathbf{z} - \mathbf{c}\|_2^2$$
其中 $\mathbf{y}$ 为对偶变量（乘子），$\rho > 0$ 为罚参数。

**ADMM迭代格式：**
$$\begin{cases} \mathbf{x}^{k+1} = \arg\min_{\mathbf{x}} L_\rho(\mathbf{x}, \mathbf{z}^k, \mathbf{y}^k) \\ \mathbf{z}^{k+1} = \arg\min_{\mathbf{z}} L_\rho(\mathbf{x}^{k+1}, \mathbf{z}, \mathbf{y}^k) \\ \mathbf{y}^{k+1} = \mathbf{y}^k + \rho(A\mathbf{x}^{k+1} + B\mathbf{z}^{k+1} - \mathbf{c}) \end{cases}$$

**缩放形式（Scaled Form）：**

引入缩放对偶变量 $\mathbf{u} = \mathbf{y}/\rho$，残差 $\mathbf{r} = A\mathbf{x} + B\mathbf{z} - \mathbf{c}$：
$$\begin{cases} \mathbf{x}^{k+1} = \arg\min_{\mathbf{x}} \left\{ f(\mathbf{x}) + \frac{\rho}{2}\|A\mathbf{x} + B\mathbf{z}^k - \mathbf{c} + \mathbf{u}^k\|_2^2 \right\} \\ \mathbf{z}^{k+1} = \arg\min_{\mathbf{z}} \left\{ g(\mathbf{z}) + \frac{\rho}{2}\|A\mathbf{x}^{k+1} + B\mathbf{z} - \mathbf{c} + \mathbf{u}^k\|_2^2 \right\} \\ \mathbf{u}^{k+1} = \mathbf{u}^k + A\mathbf{x}^{k+1} + B\mathbf{z}^{k+1} - \mathbf{c} \end{cases}$$

**停止准则：**

原始残差：$\|\mathbf{r}^k\|_2 = \|A\mathbf{x}^k + B\mathbf{z}^k - \mathbf{c}\|_2$
对偶残差：$\|\mathbf{s}^k\|_2 = \|\rho A^T B(\mathbf{z}^k - \mathbf{z}^{k-1})\|_2$

停止条件：
$$\|\mathbf{r}^k\|_2 \leq \epsilon^{\text{pri}} \quad \text{and} \quad \|\mathbf{s}^k\|_2 \leq \epsilon^{\text{dual}}$$

**收敛性：**

对凸问题，ADMM在以下条件下收敛：
1. $f, g$ 是闭、真、凸函数
2. $L_0$ 有鞍点

收敛速率：$O(1/k)$

**适用条件和边界**：
- 凸+可分离结构：ADMM适用且收敛
- 非凸问题：ADMM可应用但无全局收敛保证，实践中常能收敛到合理解
- 不可分离问题：标准ADMM不直接适用，需要变量分裂或线性化技巧

## 4. 方法流程

**输入**：目标函数 $f(\mathbf{x}) + g(\mathbf{z})$、等式约束 $A\mathbf{x} + B\mathbf{z} = \mathbf{c}$

**处理步骤**：
1. 初始化 $\mathbf{z}^0 = \mathbf{0}$, $\mathbf{u}^0 = \mathbf{0}$，选择罚参数 $\rho > 0$
2. **x-更新**：求解关于 $\mathbf{x}$ 的子问题（通常是二次规划或线性方程组）
3. **z-更新**：求解关于 $\mathbf{z}$ 的子问题（通常是近端算子，如软阈值）
4. **对偶变量更新**：$\mathbf{u}^{k+1} = \mathbf{u}^k + A\mathbf{x}^{k+1} + B\mathbf{z}^{k+1} - \mathbf{c}$
5. 检查原始残差和对偶残差是否同时低于阈值
6. 未收敛则返回步骤2

**输出如何解释**：
- $\mathbf{x}^*, \mathbf{z}^*$：满足约束的最优解
- 对偶变量 $\mathbf{y}^*$：反映了约束的拉格朗日乘子，可解释为约束的"价格"
- 残差的历史记录：可用来诊断收敛质量

**常见参数如何选择**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 罚参数 $\rho$ | 0.1~10 | 影响收敛速度，过大导致子问题病态，过小导致收敛慢 |
| 原始容差 $\epsilon^{\text{pri}}$ | $\sqrt{n}\epsilon^{\text{abs}} + \epsilon^{\text{rel}}\|\mathbf{r}^0\|$ | 参考Boyd综述中的自适应准则 |
| 对偶容差 $\epsilon^{\text{dual}}$ | $\sqrt{n}\epsilon^{\text{abs}} + \epsilon^{\text{rel}}\|\rho A^T B \mathbf{u}^0\|$ | 同上 |
| $\rho$ 自适应调整 | $\rho^{k+1} = \tau \rho^k$ | 当 $\|\mathbf{r}^k\| > \mu \|\mathbf{s}^k\|$ 时增大，反之减小 |

**ADMM与其他分裂方法对比：**

| 方法 | 问题形式 | 子问题耦合 | 收敛速率 | 收敛条件 | 分布式 | 典型应用 |
| ------|---------|-----------|---------|---------|--------|---------|
| 对偶分解 | min f+g s.t. x=z | 对偶变量 | 慢 | 严格凸+可分 | ✅ | 分布式优化 |
| 增广拉格朗日 | min f+g s.t. x=z | 罚项耦合 | 较快 | 凸 | ❌ | 约束优化 |
| ADMM | min f(x)+g(z) s.t. Ax+Bz=c | 交替更新 | O(1/k) | 凸+闭+真 | ✅ | 通用分裂优化 |
| Douglas-Rachford | 单调包含问题 | 反射+平均 | O(1/k) | 单调算子 | ✅ | 非光滑优化 |
| Primal-Dual Hybrid | min f+g∘A | 原对偶交替 | O(1/k) | 凸 | ✅ | 图像处理 |
| 近端梯度法 | min f(x)+g(x) | 梯度+近端 | O(1/k²)加速 | f光滑,g非光滑 | ❌ | LASSO |

## ADMM迭代流程图

## ADMM分解策略示意图

![[assets/admm_framework_ai.png]]

*图：ADMM 算法框架：将大问题拆分为子问题，通过对偶分解实现分布式优化。*

**典型应用：**
- **LASSO / 基追踪**：$\min \frac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + \lambda\|\mathbf{x}\|_1$
- **全变分图像去噪**：$\min \frac{1}{2}\|\mathbf{y} - \mathbf{x}\|_2^2 + \lambda\|\nabla \mathbf{x}\|_1$
- **分布式优化**：多智能体系统的共识优化（Consensus ADMM）
- **矩阵补全**：$\min \|\mathbf{X}\|_* \text{ s.t. } P_\Omega(\mathbf{X}) = P_\Omega(\mathbf{M})$
- **总变分图像重建**
- **模型预测控制**：在线优化的ADMM快速求解
- **分布式机器学习**：联邦学习中的参数聚合

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

# ADMMAlgorithmConvergence性演示：求解LASSO问题
# min 0.5*||Ax - b||^2 + lam*||z||_1, s.t. x = z
np.random.seed(42)
m, n = 80, 160
A = np.random.randn(m, n) / np.sqrt(m)
x_true = np.zeros(n)
x_true[[5, 25, 50, 75, 100, 120]] = [3, -2, 1.5, -1, 2, -0.5]
b = A @ x_true + 0.05 * np.random.randn(m)
lam = 0.3

def soft_thresh(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0)

# ADMMIterative（缩放形式）
def run_admm(rho, max_iter=150):
    x = np.zeros(n); z = np.zeros(n); u = np.zeros(n)
    primal_res, dual_res, obj = [], [], []
    for _ in range(max_iter):
        x = np.linalg.solve(A.T @ A + rho * np.eye(n), A.T @ b + rho * (z - u))
        z_old = z.copy()
        z = soft_thresh(x + u, lam / rho)
        u = u + x - z
        primal_res.append(np.linalg.norm(x - z))
        dual_res.append(rho * np.linalg.norm(z - z_old))
        obj.append(0.5 * np.linalg.norm(b - A @ x)**2 + lam * np.linalg.norm(z, 1))
    return primal_res, dual_res, obj

# 不同罚Parameter对比
rhos = [0.5, 1.0, 5.0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for rho, color in zip(rhos, colors):
    pr, dr, ob = run_admm(rho)
    axes[0].semilogy(pr, color=color, linewidth=2, label=f'ρ={rho}')
    axes[1].semilogy(dr, color=color, linewidth=2, label=f'ρ={rho}')
    axes[2].semilogy(ob, color=color, linewidth=2, label=f'ρ={rho}')

for ax, title, ylabel in zip(axes,
    ['Primal残差Convergence', 'Dual残差Convergence', '目标Function值Convergence'],
    ['Primal残差', 'Dual残差', '目标Function值']):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel + ' (log)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.suptitle('ADMMAlgorithm：不同罚Parameterρ的Convergence行为', fontsize=15)
plt.tight_layout()
plt.show()
```

**结果解释**：
- 三列分别展示原始残差、对偶残差和目标函数值的收敛曲线
- $\rho$ 较小时收敛慢但子问题易解，$\rho$ 较大时子问题可能病态但残差下降快
- 实践中 $\rho = 1.0$ 常作为不错的默认起点，也可使用自适应策略动态调整

## 6. 常见误区

1. **ADMM总是比梯度下降快**：ADMM的收敛速率是 $O(1/k)$，并不比梯度下降快。ADMM的优势在于可分解性和分布式适用性，而非单机速度。

2. **$\rho$ 越大收敛越快**：过大的 $\rho$ 使子问题中的二次项权重过大，导致子问题病态（条件数大），反而降低子问题求解精度，拖累整体收敛。

3. **ADMM只能用于凸问题**：ADMM的迭代格式对非凸问题也可定义，实践中在许多非凸问题（如深度学习、矩阵分解）上表现良好，但缺乏全局收敛保证。

4. **x和z必须同时更新**：ADMM的核心就是交替更新（先x后z），不是同时更新。同时更新变成了增广拉格朗日方法，失去了ADMM的分解优势。

5. **ADMM的残差一定单调下降**：ADMM的原始残差和对偶残差不一定单调下降，目标函数值也不一定单调下降。收敛是渐近的。

## 7. 启发问题与创新拓展

**一个反例或边界问题**：
考虑 $f(\mathbf{x}) = \|\mathbf{x}\|_1$，$g(\mathbf{z}) = 0$，约束 $\mathbf{x} = \mathbf{z}$。这是一个LASSO问题，ADMM中z-子问题退化为软阈值操作。但若将约束改为 $\mathbf{x} = \mathbf{z}$，$f$ 改为非凸的 $\|\mathbf{x}\|_0$（$l_0$ 范数），ADMM仍可运行但可能收敛到非全局最优。这说明ADMM对非凸推广需谨慎。

**一个跨学科迁移场景**：
在电力系统中，多个发电厂需要在满足各自约束的同时最小化总发电成本。这天然具有分布式结构：每个电厂的变量是本地的，但全网供需平衡构成耦合约束。Consensus ADMM可以将全网优化分解为各电厂的本地子问题，通过邻居间通信达成全局最优——这是智能电网的核心优化框架。

**一个可做成课程项目的小想法**：
"ADMM vs ISTA：稀疏信号恢复的算法竞赛"——在相同的压缩感知问题上，比较ADMM和ISTA/FISTA的收敛速度、重建精度和参数敏感性。探究：在什么问题结构下ADMM优于ISTA？什么情况下反过来？用实验数据回答"选算法"的实际问题。

## 参考文献

- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers." *Foundations and Trends in Machine Learning*, 3(1):1-122, 2011. ✅ 确定（ADMM经典综述，Boyd课题组）
- Boyd, S. & Vandenberghe, L. *Convex Optimization*. Cambridge University Press, 2004. ✅ 确定（Augmented Lagrangian与DualMethod基础）
- Glowinski, R. & Marrocco, A. "Sur l'approximation, par éléments finis d'ordre un, et la résolution, par pénalisation-dualité, d'une classe de problèmes de Dirichlet non linéaires." *Revue Française d'Automatique, Informatique et Recherche Opérationnelle*, 9(R2):41-76, 1975. ✅ 确定（ADMM早期论文之一）
- Gabay, D. & Mercier, B. "A Dual Algorithm for the Solution of Nonlinear Variational Problems via Finite Element Approximation." *Computers & Mathematics with Applications*, 2(1):17-40, 1976. ✅ 确定（ADMM早期论文之一）
- Eckstein, J. & Bertsekas, D. P. "On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators." *Mathematical Programming*, 55:293-318, 1992. ✅ 确定

---

## 相关知识点

- [[凸优化基础]] — 优化框架
- [[近端算子]] — 子问题求解
- [[../稀疏信号处理/ADMM]] — 信号处理视角
- [[../稀疏信号处理/压缩感知]] — CS优化求解
- [[../深度学习/展开网络]] — ADMM-Net
