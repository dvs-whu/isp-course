# C2：NMR波谱自动解析

## 一、项目背景

核磁共振（NMR）波谱是化学家鉴定分子结构的"眼睛"。通过测量原子核在磁场中的共振频率，NMR能够揭示分子中每个原子的化学环境。传统的NMR谱图解析依赖化学家的经验——逐个峰识别、归属、积分。本项目将用信号处理技术自动化这一过程。

**交叉学科**：信号处理 × 结构化学

### NMR信号处理链

```
样品 → 射频脉冲激发 → FID信号采集 → FFT → 频谱 → 峰检测 → 结构鉴定
         (物理过程)     (时域信号)    (信号处理)  (频域)   (自动分析)
```

FID（自由感应衰减）是NMR的原始信号——一个随时间衰减的振荡信号。通过FFT将FID从时域转换到频域，就得到我们熟悉的NMR频谱。

---

## 二、信号处理建模

### 2.1 FID信号模型

FID信号可以建模为多个衰减正弦波的叠加：

$$
s(t) = \sum_{k=1}^{K} A_k \sin(2\pi f_k t + \phi_k) e^{-t/T_{2,k}^*} + n(t)
$$

其中：
- $A_k$：第k个峰的振幅
- $f_k$：第k个峰的共振频率（对应化学位移）
- $\phi_k$：相位
- $T_{2,k}^*$：横向弛豫时间（决定线宽）
- $n(t)$：噪声

### 2.2 处理流程

```
FID信号
  ↓
加窗函数（指数/高斯窗）← 减少频谱泄漏，提高信噪比
  ↓
FFT（快速傅里叶变换）← 时域→频域
  ↓
相位校正（零阶+一阶）← 消除相位畸变
  ↓
基线校正 ← 消除基线漂移
  ↓
峰检测与积分 ← 自动识别峰位置和面积
  ↓
化学位移归属
```

### 2.3 关键参数

| 参数 | 典型值 | 含义 |
|------|--------|------|
| 采样频率 | 10-100 kHz | NMR采样率 |
| FID长度 | 16K-64K点 | 采样点数 |
| 谱宽 | 10-20 ppm | 化学位移范围 |
| 窗函数 | 指数窗（LB=0.3Hz） | 线宽增强 |

---

## 三、数据来源

### 3.1 BMRB（Biological Magnetic Resonance Bank）

- **URL**：https://bmrb.io/
- **内容**：生物大分子NMR数据，含FID原始数据
- **格式**：NMR-STAR文本格式
- **下载**：可通过REST API或直接下载

### 3.2 SDBS（Spectral Database for Organic Compounds）

- **URL**：https://sdbs.db.aist.go.jp/
- **内容**：有机化合物NMR、IR、MS谱图
- **格式**：JCAMP-DX文本格式
- **下载**：网页直接查看和下载

### 3.3 自行生成合成数据

由于NMR原始数据格式较复杂，建议先用合成FID数据练手：

```python
import numpy as np

def generate_fid(peaks, n_points=4096, sw=4000, noise_level=0.5):
    """生成合成FID信号
    peaks: [(频率Hz, 振幅, 线宽Hz, 相位rad), ...]
    """
    dt = 1.0 / sw
    t = np.arange(n_points) * dt
    fid = np.zeros(n_points)
    
    for freq, amp, lw, phase in peaks:
        T2star = 1.0 / (np.pi * lw)
        fid += amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-t / T2star)
    
    fid += noise_level * np.random.randn(n_points)
    return t, fid
```

---

## 四、数据处理方法

### Step 1：加窗函数

```python
def apply_window(fid, window='exponential', lb=0.3, sw=4000):
    """对FID加窗函数
    lb: 线宽增强参数 (Hz)
    """
    n = len(fid)
    dt = 1.0 / sw
    t = np.arange(n) * dt
    
    if window == 'exponential':
        # 指数窗：提高信噪比，但增加线宽
        w = np.exp(-np.pi * lb * t)
    elif window == 'gaussian':
        # 高斯窗：平衡分辨率和信噪比
        w = np.exp(-((t - t[-1]/2) ** 2) / (2 * (t[-1]/4) ** 2))
    else:
        w = np.ones(n)
    
    return fid * w
```

### Step 2：FFT

```python
def fid_to_spectrum(fid, sw=4000):
    """FID经FFT得到频谱"""
    spectrum = np.fft.fft(fid)
    spectrum = np.fft.fftshift(spectrum)
    n = len(fid)
    freq = np.linspace(-sw/2, sw/2, n)
    return freq, spectrum
```

### Step 3：相位校正

```python
def phase_correction(spectrum, p0=0, p1=0):
    """零阶和一阶相位校正
    p0: 零阶相位 (rad)
    p1: 一阶相位 (rad)
    """
    n = len(spectrum)
    phases = p0 + p1 * np.linspace(-1, 1, n)
    return spectrum * np.exp(-1j * phases)
```

### Step 4：峰检测

```python
from scipy.signal import find_peaks

def detect_peaks(spectrum_real, prominence=0.1, distance=50):
    """检测频谱中的峰"""
    # 归一化
    spec_norm = spectrum_real / np.max(np.abs(spectrum_real))
    
    peaks, properties = find_peaks(spec_norm, 
                                    prominence=prominence,
                                    distance=distance)
    return peaks, properties
```

### 完整处理流程

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. 生成合成FID（模拟3个峰）
peaks_config = [
    (500, 1.0, 2.0, 0),      # 峰1: 500Hz, 振幅1, 线宽2Hz
    (800, 0.6, 3.0, np.pi/4), # 峰2: 800Hz, 振幅0.6
    (1200, 0.8, 2.5, 0),      # 峰3: 1200Hz, 振幅0.8
]
t, fid = generate_fid(peaks_config, n_points=4096, sw=4000, noise_level=0.3)

# 2. 加指数窗
fid_windowed = apply_window(fid, window='exponential', lb=0.5, sw=4000)

# 3. FFT
freq, spectrum = fid_to_spectrum(fid_windowed, sw=4000)

# 4. 相位校正
spectrum_corrected = phase_correction(spectrum, p0=0.2, p1=0.05)

# 5. 取实部
spectrum_real = np.real(spectrum_corrected)

# 6. 峰检测
peak_indices, props = detect_peaks(spectrum_real, prominence=0.1, distance=100)

# 7. 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(t, fid)
axes[0,0].set_xlabel('时间 (s)')
axes[0,0].set_title('原始FID信号')

axes[0,1].plot(t, fid_windowed)
axes[0,1].set_xlabel('时间 (s)')
axes[0,1].set_title('加窗后FID')

axes[1,0].plot(freq, np.abs(spectrum))
axes[1,0].set_xlabel('频率 (Hz)')
axes[1,0].set_title('FFT幅度谱')

axes[1,1].plot(freq, spectrum_real)
axes[1,1].plot(freq[peak_indices], spectrum_real[peak_indices], 'rv', markersize=10)
axes[1,1].set_xlabel('频率 (Hz)')
axes[1,1].set_title('相位校正后频谱（红色标记检测到的峰）')

plt.tight_layout()
plt.savefig('nmr_analysis.png', dpi=150)
plt.show()

print(f"检测到 {len(peak_indices)} 个峰:")
for i, idx in enumerate(peak_indices):
    print(f"  峰{i+1}: 频率={freq[idx]:.1f}Hz, 强度={spectrum_real[idx]:.2f}")
```

---

## 五、涉及知识点

| 知识点 | 在本项目中的应用 | 链接 |
|--------|----------------|------|
| 傅里叶分析 | FID→频谱的核心变换 | [[../知识点/经典信号处理/傅里叶分析]] |
| 窗函数 | 减少频谱泄漏，提高信噪比 | [[../知识点/经典信号处理/数字滤波器设计]] |
| 频谱泄漏 | 理解为什么需要加窗 | [[../知识点/经典信号处理/傅里叶分析]] |
| 采样与重建 | 理解采样频率和谱宽的关系 | [[../知识点/经典信号处理/采样与重建]] |
| 信号与系统基础 | 时域/频域转换的基本概念 | [[../知识点/经典信号处理/信号与系统基础]] |
| 滤波器设计 | 噪声滤除 | [[../知识点/经典信号处理/数字滤波器设计]] |

---

## 六、扩展方向

### 进阶挑战
1. **2D NMR**：COSY、HSQC等二维谱的处理
2. **自动归属**：用深度学习自动归属峰到原子
3. **定量分析**：通过峰面积计算各组分浓度
4. **多维数据处理**：NMR数据的张量分解

### 参考文献
1. Claridge T D W. *High-Resolution NMR Techniques in Organic Chemistry*. Elsevier, 2016.
2. Keeler J. *Understanding NMR Spectroscopy*. Wiley, 2010.
3. Ernst R R, Bodenhausen G, Wokaun A. *Principles of Nuclear Magnetic Resonance*. Oxford, 1990.

---

*项目说明书 C2 · 智能信号处理课程*


---

> **课程关联**：本项目属于[[欢迎|智能信号处理课程]]项目二（组队/AI方法）。选题指南详见[[项目二_跨学科项目选题指南]]。评分标准详见[[考核评分标准]]。