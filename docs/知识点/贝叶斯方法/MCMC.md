# MCMC（马尔可夫链蒙特卡洛）

## 1. 学习定位

> [!info] 📚 拓展知识点说明
> **定位**：本知识点是本科课程的拓展内容，适合学有余力的同学深入学习。
> **一句话理解**：用随机采样来近似复杂分布，像在黑暗中摸索概率地形。
> **前置要求**：贝叶斯推断、概率论
> **与本科课程的关系**：随机信号基础中的采样与统计

**前置知识**：
- [[贝叶斯推断]]：后验分布、贝叶斯定理
- 概率论：马尔可夫链、平稳分布、条件分布

**与本科课程的关联**：
- [[../统计信号处理/随机信号基础]] — 随机过程与采样理论是 MCMC 的统计基础
- [[../图像信号处理/图像重建]] — MCMC 方法广泛用于贝叶斯图像重建中的后验采样

**相关知识点**：
- [[贝叶斯推断]] — 后验推断
- [[变分推断]] — 近似推断对比
- [[概率图模型]] — 图模型采样

## 2. 通俗入口

想象你被蒙上眼睛放在一片山地上，要找出最低的山谷。你看不见全貌，但能感受脚下的坡度。你的策略是：随机迈一步，如果脚下更低了就走过去，如果更高了就大概率留在原地（偶尔也走过去以避免卡在小坑里）。走足够多步后，你停留时间最长的地方大概率就是最低的山谷。

这就是 MCMC 的核心思想：
- **蒙特卡洛**：通过随机采样来估计分布（用停留时间估计概率）
- **马尔可夫链**：下一步只依赖当前位置（无记忆性）
- **目标**：构造一条链，使其平稳分布恰好是我们想要的后验分布

**为什么需要 MCMC？** 贝叶斯推断中的后验分布 $p(\theta|\mathcal{D})$ 往往无法解析计算（积分没有闭式解），特别是高维情况下数值积分也不可行。MCMC 提供了一种通用的数值方法：只要能计算后验的"未归一化密度"（即先验×似然），就能从中采样。

## 3. 严谨核心

### 定义与概念

MCMC 是一类通过构造马尔可夫链来从复杂概率分布中采样的数值方法。当后验分布无法解析求解、且维度较高导致数值积分不可行时，MCMC 提供了近似推断的有效手段。其基本思想是：构造一条马尔可夫链，使其平稳分布等于目标分布，当链收敛后，样本可近似视为目标分布的独立样本。

主要算法：
- **Metropolis-Hastings (MH)**：通用的接受-拒绝采样框架
- **Gibbs 采样**：逐维条件采样，是 MH 的特例
- **Hamiltonian Monte Carlo (HMC)**：引入辅助动量变量，利用梯度信息提高高维采样效率

### 核心公式

**Metropolis-Hastings 接受概率**：从提议分布 $q(\theta'|\theta_t)$ 生成候选样本 $\theta'$，以概率接受：

$$
\alpha = \min\left(1,\; \frac{p(\theta'|\mathcal{D})\,q(\theta_t|\theta')}{p(\theta_t|\mathcal{D})\,q(\theta'|\theta_t)}\right)
$$

**Gibbs 采样**：对多维参数 $\theta = (\theta_1, \ldots, \theta_d)$，依次从满条件分布采样：

$$
\theta_i^{(t+1)} \sim p\!\left(\theta_i \mid \theta_{1}^{(t+1)},\ldots,\theta_{i-1}^{(t+1)},\;\theta_{i+1}^{(t)},\ldots,\theta_{d}^{(t)},\;\mathcal{D}\right)
$$

**HMC 哈密顿动力学**（蛙跳积分器）：

$$
p \leftarrow p - \frac{\epsilon}{2}\nabla_\theta U(\theta), \quad
\theta \leftarrow \theta + \epsilon\, p, \quad
p \leftarrow p - \frac{\epsilon}{2}\nabla_\theta U(\theta)
$$

其中 $U(\theta) = -\log p(\theta|\mathcal{D})$ 为势能函数。

### MCMC算法对比表

| 算法 | 提议机制 | 需要梯度 | 高维效率 | 调参难度 | 混合速度 | 自相关 | 典型应用 |
| ------|---------|---------|---------|---------|---------|--------|---------|
| Random Walk MH | 高斯随机游走 | ❌ | 低 | 中(步长) | 慢 | 高 | 低维通用 |
| Independent MH | 全局提议分布 | ❌ | 低 | 高(分布选择) | 中 | 中 | 已知形状后验 |
| Gibbs | 条件分布采样 | ❌ | 中 | 低 | 中 | 中 | 分层贝叶斯模型 |
| Collapsed Gibbs | 边缘化+条件 | ❌ | 中-高 | 低 | 较快 | 较低 | LDA主题模型 |
| HMC | 哈密顿动力学 | ✅ | 高 | 中(ε, L) | 快 | 低 | 连续高维后验 |
| NUTS | 自适应HMC | ✅ | 高 | 低(自动) | 快 | 低 | Stan默认采样器 |
| SGLD | 随机梯度Langevin | ✅(mini-batch) | 很高 | 中 | 快 | 中 | 大规模贝叶斯ML |

### 典型应用

- 贝叶斯后验分布的数值近似
- 贝叶斯信号处理中的参数估计（如频率估计、时延估计）
- 物理学中的统计力学模拟
- 贝叶斯模型选择与贝叶斯因子估计

## 4. 方法流程

### MCMC算法分类图

**Metropolis-Hastings 算法流程**：

**Step 1：初始化**
- 选择初始值 $\theta_0$
- 选择提议分布 $q(\theta'|\theta)$（如高斯随机游走 $q(\theta'|\theta) = \mathcal{N}(\theta, \sigma^2 I)$）

**Step 2：迭代采样**（对 $t = 0, 1, 2, \ldots$）
1. 从提议分布采样候选：$\theta' \sim q(\theta'|\theta_t)$
2. 计算接受概率：$\alpha = \min\left(1, \frac{p(\theta'|\mathcal{D})\,q(\theta_t|\theta')}{p(\theta_t|\mathcal{D})\,q(\theta'|\theta_t)}\right)$
3. 以概率 $\alpha$ 接受：$\theta_{t+1} = \theta'$，否则 $\theta_{t+1} = \theta_t$

**Step 3：去除 burn-in**
- 丢弃前 $B$ 个样本（burn-in 期），保留后续样本

**Step 4：后处理**
- 稀疏化（thinning）：每隔 $k$ 步取一个样本，降低自相关
- 计算后验统计量：均值、方差、可信区间

**收敛诊断**：
- Trace plot：样本轨迹应无明显趋势
- Gelman-Rubin $\hat{R}$：多条链的方差比，接近 1 表示收敛
- 有效样本量（ESS）：衡量独立信息量

**参数调优建议**：
- Random Walk MH：步长 σ 过大→接受率低，过小→移动慢；目标接受率约 23%~44%
- HMC：步长 ε 和蛙跳步数 L 需要平衡；NUTS 自动调参
- Gibbs：需要知道满条件分布的解析形式

![[assets/mcmc_methods_ai.png]]

*图：MCMC 采样方法体系：从 Metropolis-Hastings 到 HMC 的演进与权衡。*

## 5. Python 最小实验

**运行前准备**：本代码需要以下Python库，请确保已安装：
- 本地运行：`pip install numpy matplotlib scipy`
- Pyodide环境：先运行下面的安装代码块

```python
# Pyodide环境：安装依赖库
import micropip
await micropip.install(["numpy", "matplotlib", "scipy"])
```

```python
# === 环境预加载 ===
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Metropolis-Hastings采样演示
# 目标分布：混合高斯 0.3*N(-2, 0.5^2) + 0.7*N(3, 1^2)
def target_pdf(x):
    return 0.3 * stats.norm.pdf(x, -2, 0.5) + 0.7 * stats.norm.pdf(x, 3, 1.0)

# MH采样器
np.random.seed(42)
n_samples = 10000
samples = np.zeros(n_samples)
x0 = 0.0
proposal_std = 2.0  # 提议分布标准差
n_accept = 0

for i in range(n_samples):
    x_curr = x0 if i == 0 else samples[i - 1]
    # 从提议分布q(x'|x)采样（高斯随机游走）
    x_prop = x_curr + proposal_std * np.random.randn()
    # 计算接受概率
    alpha = min(1.0, target_pdf(x_prop) / (target_pdf(x_curr) + 1e-300))
    # 接受/拒绝
    if np.random.rand() < alpha:
        samples[i] = x_prop
        n_accept += 1
    else:
        samples[i] = x_curr

print(f"接受率: {n_accept / n_samples:.2%}")
# 去除burn-in期
samples = samples[2000:]

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 样本轨迹（trace plot）
axes[0].plot(samples[:500], 'b-', alpha=0.7, linewidth=0.8)
axes[0].set_title('Sampling Trace (First 500 Samples)', fontsize=13)
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('样本值')
axes[0].grid(True, alpha=0.3)
# 直方图与目标分布对比
x_range = np.linspace(-5, 7, 300)
axes[1].hist(samples, bins=80, density=True, alpha=0.6, color='steelblue', label='MCMC直方图')
axes[1].plot(x_range, target_pdf(x_range), 'r-', linewidth=2.5, label='目标分布')
axes[1].set_title('样本分布 vs 目标分布', fontsize=13)
axes[1].set_xlabel('x')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
# 自相关函数
max_lag = 100
acf = np.correlate(samples - samples.mean(), samples - samples.mean(), mode='full')
acf = acf[len(acf) // 2:len(acf) // 2 + max_lag] / acf[len(acf) // 2]
axes[2].plot(acf, 'g-', linewidth=2)
axes[2].set_title('自相关函数', fontsize=13)
axes[2].set_xlabel('滞后')
axes[2].grid(True, alpha=0.3)
plt.suptitle('Metropolis-Hastings MCMC Sampling', fontsize=15)
plt.tight_layout()
plt.show()
```

## 6. 常见误区

| 误区 | 正确理解 |
|------|----------|
| MCMC 样本是独立的 | MCMC 样本是相关的（马尔可夫链），需要用有效样本量（ESS）而非样本数衡量信息量 |
| 样本越多越好 | 过多样本可能因自相关而冗余；关键是 ESS 足够大，且链已收敛 |
| burn-in 不重要 | burn-in 期的样本未收敛到平稳分布，必须丢弃；否则估计有偏 |
| 接受率越高越好 | 接受率过高说明步长太小、移动太慢；过低说明步长太大、经常拒绝；需平衡 |
| MCMC 一定能找到全局最优 | MCMC 采样的是整个后验分布，不是优化；但在多峰分布中可能被困在某个峰 |
| HMC 总比 Random Walk MH 好 | HMC 需要梯度信息，不适用于离散参数；且调参更复杂（步长、蛙跳步数） |
| 收敛诊断 $\hat{R} \approx 1$ 就万事大吉 | $\hat{R}$ 只检测"多链间一致性"，不能发现所有类型的不收敛（如链移动太慢） |

## 7. 启发问题与创新拓展

### 思考题
1. 为什么 MCMC 的平稳分布恰好等于目标分布？细致平衡条件（detailed balance）起什么作用？
2. 在高维空间中，Random Walk MH 为什么效率很低？HMC 如何利用梯度信息改善采样？
3. Gibbs 采样和 MH 的关系是什么？为什么 Gibbs 是 MH 的特例（接受率恒为 1）？
4. 如何判断 MCMC 链已经收敛？除了 $\hat{R}$ 还有哪些诊断方法？

### 创新拓展方向
- **NUTS（No-U-Turn Sampler）**：自适应 HMC，自动选择蛙跳步数，Stan 的默认采样器
- **SGLD / SGHMC**：随机梯度 Langevin 动力学，将 mini-batch 引入 MCMC，适合大规模贝叶斯机器学习
- **并行回火（Parallel Tempering）**：多条不同"温度"的链交换信息，帮助跳出多峰分布
- **MCMC vs 变分推断**：精度-速度权衡，何时选择哪种方法
- **概率编程中的 MCMC**：PyMC、Stan、TensorFlow Probability 如何自动实现 MCMC
- **Stein 变分梯度下降（SVGD）**：结合粒子方法和核方法的新型采样算法

## 参考文献

1. Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. Chapter 11.
2. Murphy, K. P. *Machine Learning: A Probabilistic Perspective*. MIT Press, 2012. Chapter 24.
3. Neal, R. M. "MCMC using Hamiltonian dynamics." *Handbook of Markov Chain Monte Carlo*, 2011. （需核实：原始出处与年份）
