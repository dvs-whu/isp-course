# DOA 估计

## 一、学习定位

> [!info] 📚 拓展知识点说明
> **定位**：本知识点是本科课程的拓展内容，适合学有余力的同学深入学习。
> **一句话理解**：估计信号从哪个方向来，像"空间谱分析"。
> **前置要求**：阵列信号模型、线性代数
> **与本科课程的关系**：功率谱估计在空间域的推广

## 二、通俗入口

### 直觉类比

你在一间黑暗的屋子里，想知道房间里有几个火把、分别在哪个方向。你闭上眼睛转头感受不同方向的光线强度——这就是"波束扫描"做 DOA 估计。而 MUSIC 算法更聪明：它不仅感受光线，还分析"哪个方向完全没有光"（噪声子空间），从而精确锁定火把的方向——这就是子空间方法的超分辨能力。

### DOA 估计方法分类

> 子空间类方法通过协方差矩阵的特征分解分离信号与噪声子空间，突破 Rayleigh 分辨限。

![[assets/doa_estimation_ai.png]]

*图：DOA 估计方法：从波束扫描到 MUSIC/ESPRIT 的子空间高分辨测向。*

## 三、严谨核心

### 定义与概念

**DOA（Direction of Arrival）估计** 是利用传感器阵列接收信号估计来波方向的技术，是阵列信号处理的核心问题之一。

经典方法按思路可分为：
- **波束扫描类**：CBF（常规波束形成），分辨率受阵列孔径限制（Rayleigh 限）
- **子空间类**：利用协方差矩阵的特征分解，将信号子空间与噪声子空间分离，突破 Rayleigh 限
  - **MUSIC（Multiple Signal Classification）**：Schmidt 1986 年提出的经典算法
  - **ESPRIT**：利用旋转不变性，无需谱搜索
- **最大似然类**：理论最优但计算复杂度高

协方差矩阵的特征分解：

$$R_{xx} = U_s \Sigma_s U_s^H + U_n \Sigma_n U_n^H$$

其中 $U_s$ 张成信号子空间，$U_n$ 张成噪声子空间。

### 核心公式

MUSIC 空间谱：

$$P_{\text{MUSIC}}(\theta) = \frac{1}{\mathbf{a}^H(\theta) U_n U_n^H \mathbf{a}(\theta)}$$

DOA 估计值：

$$\hat{\theta} = \arg\max_\theta P_{\text{MUSIC}}(\theta)$$

MUSIC 算法步骤：
1. 估计协方差矩阵 $\hat{R} = \frac{1}{L} \sum_{l=1}^{L} \mathbf{x}(l) \mathbf{x}^H(l)$
2. 特征分解，确定信号源数 $D$，取噪声子空间 $U_n$
3. 构造 MUSIC 谱并搜索峰值

### 典型应用

- 雷达目标方位估计
- 无线通信中的到达角估计（AoA）
- 声纳中的目标定位
- 射电天文中的信号源定位
- 车载雷达（自动驾驶）

## 四、方法流程

### DOA 估计方法对比

| 方法 | 分辨力 | 是否需要谱搜索 | 计算复杂度 | 阵列要求 | 优缺点 |
| ------|--------|--------------|-----------|---------|-------|
| CBF | Rayleigh 限 | 是 | $O(M^2)$ | 任意 | 简单稳健，分辨力低 |
| MUSIC | 超分辨 | 是 | $O(M^3 + N_\theta M^2)$ | 已知阵列流形 | 高分辨，需已知源数 |
| ESPRIT | 超分辨 | 否 | $O(M^3)$ | 需旋转不变结构 | 无需搜索，阵列受限 |
| Root-MUSIC | 超分辨 | 否（多项式求根） | $O(M^3)$ | ULA | 高精度，仅限 ULA |
| ML | 理论最优 | 是（多维优化） | 高 | 任意 | 性能最优，计算量大 |

### MUSIC 算法完整流程

1. **数据采集**：收集 $L$ 个快拍的阵列接收数据 $\mathbf{x}(1), \dots, \mathbf{x}(L)$
2. **协方差矩阵估计**：$\hat{R} = \frac{1}{L}\sum_{l=1}^{L}\mathbf{x}(l)\mathbf{x}^H(l)$
3. **特征分解**：对 $\hat{R}$ 做特征值分解，得到 $M$ 个特征值和对应特征向量
4. **确定信号源数 $D$**：利用 AIC/MDL 准则或观察特征值间隙
5. **提取噪声子空间**：取最小的 $M-D$ 个特征向量组成 $U_n$
6. **构造空间谱**：遍历感兴趣的角度范围，计算 $P_{\text{MUSIC}}(\theta)$
7. **峰值搜索**：谱峰对应的 $\theta$ 即为 DOA 估计值

## 五、Python 最小实验

**运行前准备**：本代码需要以下Python库，请确保已安装：
- 本地运行：`pip install numpy matplotlib`
- Pyodide环境：先运行下面的安装代码块

```python
# Pyodide环境：安装依赖库
import micropip
await micropip.install(["numpy", "matplotlib"])
```

```python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# === 实验：MUSIC 算法 DOA 估计 ===
M = 12   # 阵元数
L = 200  # 快拍数
d_lam = 0.5  # 半波长间距
np.random.seed(42)

# 信号方向
theta_true = np.array([10, 25, -20])  # 三个信号源
D = len(theta_true)

# 导向矢量函数
def steer_vec(theta_deg):
    m = np.arange(M)
    return np.exp(-1j * 2 * np.pi * d_lam * m * np.sin(np.radians(theta_deg)))

# 生成阵列信号
A_true = np.array([steer_vec(th) for th in theta_true]).T
S = np.sqrt(0.5) * (np.random.randn(D, L) + 1j * np.random.randn(D, L))
sigma_n = 0.1
N = sigma_n * (np.random.randn(M, L) + 1j * np.random.randn(M, L))
X = A_true @ S + N

# 估计协方差矩阵
R_hat = (X @ X.conj().T) / L

# 特征分解
eigenvalues, eigenvectors = np.linalg.eigh(R_hat)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 噪声子空间
Un = eigenvectors[:, D:]

# MUSIC 空间谱
theta_scan = np.linspace(-90, 90, 361)
P_music = np.zeros(len(theta_scan))
for i, th in enumerate(theta_scan):
    a = steer_vec(th)
    P_music[i] = 1.0 / np.abs(a.conj() @ Un @ Un.conj() @ a)
P_music_dB = 10 * np.log10(P_music / np.max(P_music))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# 左图：特征值分布
axes[0].stem(range(M), eigenvalues, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].axvline(D - 0.5, color='r', linestyle='--', linewidth=2, label=f'Source Count D={D}')
axes[0].set_title('协方差矩阵特征值', fontsize=13)
axes[0].set_xlabel('特征值索引')
axes[0].set_ylabel('特征值大小')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 右图：MUSIC谱
axes[1].plot(theta_scan, P_music_dB, 'b-', linewidth=2)
for th in theta_true:
    axes[1].axvline(th, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_title('MUSIC空间谱', fontsize=13)
axes[1].set_xlabel('角度 (度)')
axes[1].set_ylabel('空间谱 (dB)')
axes[1].set_ylim([-40, 3])
axes[1].grid(True, alpha=0.3)
for th in theta_true:
    axes[1].annotate(f'θ={th}°', xy=(th, 0), fontsize=10, color='red',
                     ha='center', va='bottom')
plt.suptitle(f'MUSIC算法DOA估计 (M={M}, D={D})', fontsize=15)
plt.tight_layout()
plt.show()
```

**实验要点**：左图观察特征值的"阶梯"结构——前 $D$ 个大特征值对应信号，其余对应噪声；右图三个锐利峰值精确对应真实方向 10°、25°、-20°，验证 MUSIC 的超分辨能力。

## 六、常见误区

### 误区 1：MUSIC 总能找到正确的方向
**正解**：MUSIC 需要预先知道信号源数 $D$。如果 $D$ 估计错误，结果会严重偏差。实际中常用 AIC 或 MDL 准则估计源数，但低信噪比下可能失败。

### 误区 2：MUSIC 可以分辨任意接近的两个信号
**正解**：虽然 MUSIC 突破了 Rayleigh 限，但仍受信噪比和快拍数约束。两个信号角度差太小或 SNR 太低时，MUSIC 谱的两个峰会合并，无法分辨。分辨力有对应的统计下界（如 Cramér-Rao 界）。

### 误区 3：ESPRIT 不需要阵列结构信息
**正解**：ESPRIT 需要阵列具有"旋转不变性"（即存在两个位移子阵），这要求阵列结构满足特定条件（如 ULA 或特定双子阵配置）。并非任意阵列都能用 ESPRIT。

### 误区 4：协方差矩阵的特征分解不需要太多快拍
**正解**：快拍数 $L$ 至少需要 $O(M)$ 量级才能保证子空间估计质量。当 $L < M$ 时，协方差矩阵甚至不满秩，子空间方法可能失效。

## 七、启发问题与创新拓展

### 思考题
1. MUSIC 和 ESPRIT 的核心区别是什么？在什么场景下应该选择哪种方法？
2. 如果信号源是相干的（多径传播导致），MUSIC 的性能会如何退化？空间平滑技术如何解决这个问题？
3. Cramér-Rao 下界（CRB）如何定量描述 DOA 估计的理论精度极限？

### 创新拓展方向
- **稀疏 DOA 估计**：利用压缩感知理论，用少量快拍实现高分辨 DOA
- **深度学习 DOA**：用神经网络直接从协方差矩阵学习 DOA，如 DeepMUSIC、RBFNet
- **二维 DOA 估计**：方位角 + 俯仰角联合估计，需要平面阵或特殊阵列结构
- **宽带 DOA**：多频段联合估计，利用频率分集提升性能
- **动态 DOA 跟踪**：目标运动场景下的在线 DOA 估计与跟踪算法

### 跨领域联系
- **傅里叶分析**：常规波束形成法做 DOA 估计等价于空间域的傅里叶谱分析，CBF 空间谱与时域功率谱完全类比
- **统计信号处理/功率谱估计**：MUSIC 通过协方差矩阵特征分解实现空间超分辨，是功率谱估计中子空间方法（如 Pisarenko）在空域的推广
- **数字信号处理**：Root-MUSIC 将谱搜索转化为多项式求根，利用了 ULA 导向矢量的 Vandermonde 结构

---

## 前置知识

- [[阵列信号模型]]（导向矢量、阵列流形）
- 线性代数（特征值分解、子空间理论）

## 参考文献

- R. O. Schmidt, "Multiple emitter location and signal parameter estimation," IEEE Trans. AP, vol. 34, no. 3, pp. 276–280, Mar. 1986.
- H. L. Van Trees, *Optimum Array Processing*, Wiley, 2002.
- R. Roy and T. Kailath, "ESPRIT—estimation of signal parameters via rotational invariance techniques," IEEE TASSP, 1989.（需核实期刊缩写）

---

## 相关知识点

- [[阵列信号模型]] — 导向矢量
- [[波束形成]] — 波束扫描法
- [[../统计信号处理/功率谱估计]] — 空间谱估计
- [[../稀疏信号处理/压缩感知]] — 稀疏DOA
- [[MIMO雷达]] — MIMO DOA
- [[../深度学习/深度学习基础]] — DeepMUSIC

---

> **课程关联**：本知识点为研究生/高年级拓展内容。项目相关：[[/Users/leiyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/智能信号处理/课程/项目库/项目二/项目二_跨学科项目选题指南|项目二]]选做方向。
