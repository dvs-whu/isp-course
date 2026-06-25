# MIMO 雷达

## 一、学习定位

> [!info] 📚 拓展知识点说明
> **定位**：本知识点是本科课程的拓展内容，适合学有余力的同学深入学习。
> **一句话理解**：多天线同时收发，获得更高的空间分辨率。
> **前置要求**：阵列信号模型、波束形成
> **与本科课程的关系**：信号与系统基础、波束形成

## 二、通俗入口

### 直觉类比

传统相控阵雷达像一只眼睛看世界——只有一套视角。MIMO 雷达像同时用多只眼睛（多个发射天线发射不同编码信号）看世界，然后在接收端把各"视角"的信息叠加融合。结果是：用 $M_T$ 个发射 + $M_R$ 个接收天线，等效获得了 $M_T \times M_R$ 个虚拟天线的观测能力——这就是"虚拟阵列扩展"的魔力。

### MIMO 雷达虚拟阵列原理

> $M_T=3$ 个发射天线与 $M_R=4$ 个接收天线经匹配滤波后等效形成 $M_T \times M_R = 12$ 个虚拟阵元。

![[assets/mimo_radar_ai.png]]

*图：MIMO 雷达原理：发射端波形分集 + 接收端联合处理，等效虚拟阵列扩展孔径。*

## 三、严谨核心

### 定义与概念

**MIMO 雷达（Multiple-Input Multiple-Output Radar）** 利用多个发射天线发射正交（或低相关）波形、多个天线接收回波，形成远大于物理阵元数的虚拟孔径，从而提升角度分辨率和参数估计性能。

MIMO 雷达的两种主要构型：
- **统计 MIMO 雷达**：发射/接收天线间距较大，利用空间分集对抗目标 RCS 衰落
- **相干 MIMO 雷达（共址）**：天线间距小，利用波形分集形成虚拟阵列扩展孔径

共址 MIMO 雷达的核心思想：$M_T$ 个发射阵元、$M_R$ 个接收阵元，通过匹配滤波可分离各发射波形，等效形成 $M_T \times M_R$ 个虚拟阵元，虚拟阵列孔径显著增大。

接收信号经匹配滤波后的虚拟阵列模型：

$$\mathbf{y}(t) = (\mathbf{a}_R(\theta) \otimes \mathbf{a}_T(\theta)) s(t) + \mathbf{n}(t)$$

### 核心公式

虚拟导向矢量（Kronecker 积形式）：

$$\mathbf{b}(\theta) = \mathbf{a}_R(\theta) \otimes \mathbf{a}_T(\theta)$$

匹配滤波后的信号模型：

$$\mathbf{y} = \mathbf{b}(\theta) s + \mathbf{n}$$

虚拟阵列的等效阵元数：

$$M_{\text{virtual}} \leq M_T \times M_R$$

最大可分辨目标数：

$$D_{\max} = M_T \times M_R - 1$$

### MIMO 雷达构型对比

| 特性 | 相干 MIMO（共址） | 统计 MIMO |
| ------|-----------------|-----------|
| 天线间距 | 半波长量级 | 远大于波长 |
| 核心优势 | 波形分集 → 虚拟孔径扩展 | 空间分集 → 对抗 RCS 衰落 |
| 等效阵元数 | $M_T \times M_R$ | $M_T \times M_R$（独立视角） |
| 角度分辨率 | 显著提升 | 取决于空间分集增益 |
| 适用场景 | 高分辨率雷达、汽车雷达 | 雷达网、分布式雷达 |
| 信号处理 | 匹配滤波 + 虚拟阵列 DOA | 分布式检测与融合 |

### 典型应用

- 雷达目标检测与角度-速度联合估计
- 空时自适应处理（STAP）
- 汽车雷达（高分辨率角度估计）
- 无线通信中的大规模 MIMO

## 四、方法流程

### MIMO 雷达信号处理流程

1. **波形设计**：设计 $M_T$ 个正交（或低互相关）的发射波形 $s_1(t), \dots, s_{M_T}(t)$
2. **发射**：各天线同时发射各自的编码波形
3. **接收**：$M_R$ 个接收天线采集回波信号
4. **匹配滤波**：用各发射波形的副本对接收信号做匹配滤波，分离出 $M_T \times M_R$ 个通道
5. **虚拟阵列构造**：匹配滤波输出排列为虚拟阵列向量 $\mathbf{y}$
6. **参数估计**：在虚拟阵列上应用 DOA 估计算法（MUSIC、ESPRIT 等）
7. **检测与跟踪**：对估计结果做目标检测和轨迹跟踪

### 波形设计要点

| 设计目标 | 要求 | 典型方法 |
|---------|------|---------|
| 正交性 | 互相关峰值低 | 正交频分（OFDM）、相位编码 |
| 脉冲压缩比 | 自相关主瓣窄 | 线性调频（LFM）、伪随机码 |
| PAPR | 峰均比低 | 恒包络波形设计 |
| 多普勒容忍度 | 对速度不敏感 | 优化码字选择 |

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

# === 实验：MIMO 雷达虚拟阵列 vs 物理阵列 ===
d_lam = 0.5  # 半波长间距
MT = 3   # 发射阵元数
MR = 4   # 接收阵元数

tx_pos = np.arange(MT) * d_lam  # 发射阵元位置
rx_pos = np.arange(MR) * d_lam  # 接收阵元位置

# 虚拟阵列位置 = 所有 (tx_pos[i] + rx_pos[j]) 组合
virtual_pos = []
for r in rx_pos:
    for t in tx_pos:
        virtual_pos.append(r + t)
virtual_pos = np.array(virtual_pos)
virtual_pos_unique = np.sort(np.unique(virtual_pos))

print(f"发射阵元数: {MT}, 接收阵元数: {MR}")
print(f"虚拟阵元数(含重复): {len(virtual_pos)}")
print(f"虚拟阵元位置(唯一): {virtual_pos_unique}")

# 计算方向图
theta_range = np.linspace(-90, 90, 361)
theta_rad = np.radians(theta_range)

def array_pattern(pos, theta_scan):
    pattern = np.zeros(len(theta_scan))
    for i, th in enumerate(theta_scan):
        sv = np.exp(-1j * 2 * np.pi * pos * np.sin(th) / d_lam)
        pattern[i] = np.abs(np.sum(sv)) / len(pos)
    return pattern

AF_phys = array_pattern(rx_pos, theta_rad)
AF_virtual = array_pattern(virtual_pos_unique, theta_rad)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 左图：阵列位置
axes[0].scatter(tx_pos, np.ones(MT), s=120, c='red', marker='s', label=f'发射({MT}个)', zorder=5)
axes[0].scatter(rx_pos, np.zeros(MR), s=120, c='blue', marker='o', label=f'接收({MR}个)', zorder=5)
axes[0].scatter(virtual_pos_unique, -np.ones(len(virtual_pos_unique)), s=120,
                c='green', marker='^', label=f'虚拟({len(virtual_pos_unique)}个)', zorder=5)
axes[0].set_title('阵列几何位置', fontsize=13)
axes[0].set_xlabel('位置 (λ)')
axes[0].legend(fontsize=10)
axes[0].set_yticks([-1, 0, 1])
axes[0].set_yticklabels(['虚拟', '接收', '发射'])
axes[0].grid(True, alpha=0.3)

# 中图：方向图对比
axes[1].plot(theta_range, 20*np.log10(AF_phys + 1e-10), 'b-', linewidth=2, label=f'接收阵({MR}元)')
axes[1].plot(theta_range, 20*np.log10(AF_virtual + 1e-10), 'g-', linewidth=2, label=f'虚拟阵({len(virtual_pos_unique)}元)')
axes[1].set_title('方向图对比', fontsize=13)
axes[1].set_xlabel('角度 (度)')
axes[1].set_ylabel('Gain (dB)')
axes[1].set_ylim([-40, 0])
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# 右图：孔径对比
positions = [('发射阵', tx_pos, 'red'), ('接收阵', rx_pos, 'blue'), ('虚拟阵', virtual_pos_unique, 'green')]
for i, (name, pos, color) in enumerate(positions):
    axes[2].barh(i, pos[-1] - pos[0], left=pos[0], height=0.3, color=color, alpha=0.7)
    axes[2].scatter(pos, [i]*len(pos), c=color, s=60, zorder=5)
axes[2].set_title('阵列孔径对比', fontsize=13)
axes[2].set_xlabel('位置 (λ)')
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(['发射阵', '接收阵', '虚拟阵'])
axes[2].grid(True, alpha=0.3)

plt.suptitle(f'MIMO雷达虚拟阵列 (MT={MT}, MR={MR})', fontsize=15)
plt.tight_layout()
plt.show()
```

**实验要点**：$M_T=3, M_R=4$ 只用了 7 个物理阵元，却获得了 12 个虚拟阵元。中图可明显看到虚拟阵列的主瓣比物理接收阵列窄得多（分辨率更高），这就是 MIMO 的核心优势。

## 六、常见误区

### 误区 1：MIMO 雷达的虚拟阵元数一定是 $M_T \times M_R$
**正解**：这是上限。当发射和接收阵列的间距配置不当时，多个虚拟阵元位置可能重叠，实际唯一阵元数会少于 $M_T \times M_R$。设计时需要仔细选择发射/接收阵元间距以最大化虚拟孔径。

### 误区 2：MIMO 雷达只适合共址场景
**正解**：统计 MIMO（天线间距大）利用空间分集对抗目标 RCS 闪烁，在分布式雷达网中有独特优势。两种构型解决不同的问题，不能混为一谈。

### 误区 3：正交波形越好越容易实现
**正解**：完美的正交波形在有限带宽和有限时宽下无法同时满足自相关尖锐 + 互相关为零。实际中需要在正交性、脉冲压缩性能、PAPR 之间权衡，波形设计本身是一个活跃的研究课题。

### 误区 4：MIMO 雷达处理只是"多发多收"
**正解**：关键不在于多发多收本身，而在于波形分集带来的匹配滤波分离能力。如果发射相同波形，就退化为普通相控阵，失去了虚拟阵列扩展的优势。

## 七、启发问题与创新拓展

### 思考题
1. 如果发射阵列和接收阵列都是 ULA，间距都为 $d$，虚拟阵列的结构是什么？为什么有些虚拟阵元位置会重叠？
2. MIMO 雷达的 Cramér-Rao 界与等效虚拟阵列的物理阵列有什么关系？虚拟阵列能否突破物理阵列的 CRB？
3. OFDM 信号用作 MIMO 雷达波形有什么优势和挑战？

### 创新拓展方向
- **OFDM MIMO 雷达**：利用 OFDM 的子载波正交性实现波形分集，同时具备通信-雷达一体化潜力
- **大规模 MIMO 雷达**：借鉴 5G massive MIMO 的经验，用数百个天线实现极高分辨率
- **认知 MIMO 雷达**：根据环境反馈自适应调整波形和处理策略
- **MIMO-SAR**：MIMO 技术与合成孔径雷达结合，提升方位分辨率
- **稀疏 MIMO 阵列设计**：互质/嵌套配置用更少物理阵元获得更多虚拟自由度

### 跨领域联系
- **信号与系统基础**：MIMO 雷达的匹配滤波是经典时域匹配滤波在波形分集场景下的应用，正交波形设计依赖信号的互相关特性
- **线性代数**：虚拟阵列的 Kronecker 积结构将发射与接收导向矢量合并，扩展了阵列自由度
- **雷达原理**：MIMO 雷达是传统相控阵雷达的波形分集推广，虚拟阵列概念将发射与接收阵列的孔径乘积化

---

## 前置知识

- [[阵列信号模型]]（导向矢量、协方差矩阵）
- [[波束形成]]（发射波束形成、空间滤波概念）
- [[DOA估计]]（MUSIC、ESPRIT 等子空间方法）

## 参考文献

- H. L. Van Trees, *Optimum Array Processing*, Wiley, 2002.
- E. Fishler et al., "MIMO radar: An idea whose time has come," Proc. IEEE Radar Conf., 2004.（需核实会议名）
- J. Li and P. Stoica, "MIMO radar with colocated antennas," IEEE SPM, 2007.（需核实期刊名）

---

## 相关知识点

- [[阵列信号模型]] — 阵列扩展
- [[波束形成]] — 发射波束形成
- [[DOA估计]] — MIMO DOA
- [[../稀疏信号处理/压缩感知]] — 稀疏MIMO

---

> **课程关联**：本知识点为研究生/高年级拓展内容。项目相关：[[/Users/leiyu/Library/Mobile Documents/iCloud~md~obsidian/Documents/智能信号处理/课程/项目库/项目二/项目二_跨学科项目选题指南|项目二]]选做方向。
