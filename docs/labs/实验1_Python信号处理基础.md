# 实验1：Python信号处理基础

## 一、实验信息

| 项目 | 内容 |
|------|------|
| 实验时长 | 2学时（90分钟） |
| 实验类型 | 基础验证型 |
| 适用课程 | 智能信号处理 |
| 面向对象 | 大二本科生 |

---

## 二、实验目的

1. **熟悉Python信号处理开发环境**：掌握NumPy、Matplotlib、SciPy三大核心库的基本使用方法。
2. **掌握基本信号的生成与可视化**：学会用代码生成正弦波、复合信号及含噪信号，并绘制时域波形图。
3. **理解傅里叶变换的基本原理**：通过编程实现FFT变换，观察信号在频域中的表示，建立时域—频域对应关系的直觉。

---

## 三、实验环境

| 软件/库 | 推荐版本 |
|---------|---------|
| Python | 3.9+ |
| NumPy | 1.24+ |
| Matplotlib | 3.7+ |
| SciPy | 1.10+ |

**安装命令**（如尚未安装）：

```bash
pip install numpy matplotlib scipy
```

---

## 四、实验原理

### 4.1 正弦信号

一个频率为 $f$ Hz 的正弦信号可以表示为：

$$x(t) = A \sin(2\pi f t + \phi)$$

其中 $A$ 为振幅，$f$ 为频率，$\phi$ 为初始相位。对连续信号进行采样（采样率 $f_s$），得到离散序列：

$$x[n] = A \sin\left(2\pi \frac{f}{f_s} n + \phi\right), \quad n = 0, 1, 2, \ldots, N-1$$

### 4.2 傅里叶变换与FFT

离散傅里叶变换（DFT）将长度为 $N$ 的时域序列变换为频域表示：

$$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j2\pi kn/N}, \quad k = 0, 1, \ldots, N-1$$

快速傅里叶变换（FFT）是DFT的高效算法，时间复杂度从 $O(N^2)$ 降为 $O(N \log N)$。频率分辨率为：

$$\Delta f = \frac{f_s}{N}$$

### 4.3 信噪比（SNR）

信号与噪声的强度之比，常用分贝表示：

$$\text{SNR (dB)} = 10 \log_{10} \frac{P_{\text{signal}}}{P_{\text{noise}}}$$

---

## 五、实验步骤

### 步骤1：环境验证

运行以下代码，确认环境正常：

```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import signal

# 设置中文字体
font_path = '/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf'
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

print(f"NumPy 版本: {np.__version__}")
print(f"SciPy 版本: {scipy.__version__}")
print(f"Matplotlib 版本: {matplotlib.__version__}")
print("环境验证通过！")
```

### 步骤2：生成单频正弦信号

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体
font_prop = fm.FontProperties(fname='/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf')

# --- 参数设置 ---
fs = 1000          # 采样率 1000 Hz
T = 1.0            # 信号时长 1 秒
f = 5              # 信号频率 5 Hz
A = 1.0            # 振幅

# --- 生成时间轴和信号 ---
t = np.arange(0, T, 1/fs)    # 时间向量 [0, 0.001, 0.002, ..., 0.999]
x = A * np.sin(2 * np.pi * f * t)

# --- 绘制波形 ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t[:200], x[:200], 'b-', linewidth=1.5)  # 只显示前200个采样点
ax.set_xlabel('时间 (s)', fontproperties=font_prop)
ax.set_ylabel('幅值', fontproperties=font_prop)
ax.set_title('正弦信号 x(t) = sin(2π×5×t)', fontproperties=font_prop)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sine_wave.png', dpi=150)
plt.show()

print(f"采样点数: {len(t)}")
print(f"时间范围: {t[0]:.3f} ~ {t[-1]:.3f} s")
```

**预期输出**：一条频率为5Hz的正弦波，显示前0.2秒的波形。

### 步骤3：生成复合信号

实际场景中信号通常包含多个频率分量。我们生成一个包含3个频率的复合信号：

```python
# --- 复合信号参数 ---
fs = 1000
T = 1.0
t = np.arange(0, T, 1/fs)

# 三个分量：50Hz + 120Hz + 300Hz
f1, A1 = 50,  1.0
f2, A2 = 120, 0.5
f3, A3 = 300, 0.3

x_clean = A1 * np.sin(2*np.pi*f1*t) + \
          A2 * np.sin(2*np.pi*f2*t) + \
          A3 * np.sin(2*np.pi*f3*t)

# --- 绘图 ---
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# 时域波形
axes[0].plot(t[:200], x_clean[:200], 'b-', linewidth=1)
axes[0].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[0].set_ylabel('幅值', fontproperties=font_prop)
axes[0].set_title('复合信号时域波形（50Hz + 120Hz + 300Hz）', fontproperties=font_prop)
axes[0].grid(True, alpha=0.3)

# 各分量
axes[1].plot(t[:200], A1*np.sin(2*np.pi*f1*t[:200]), label=f'{f1}Hz', alpha=0.8)
axes[1].plot(t[:200], A2*np.sin(2*np.pi*f2*t[:200]), label=f'{f2}Hz', alpha=0.8)
axes[1].plot(t[:200], A3*np.sin(2*np.pi*f3*t[:200]), label=f'{f3}Hz', alpha=0.8)
axes[1].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[1].set_ylabel('幅值', fontproperties=font_prop)
axes[1].set_title('各频率分量', fontproperties=font_prop)
axes[1].legend(prop=font_prop)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('composite_signal.png', dpi=150)
plt.show()

print(f"复合信号峰值: {np.max(np.abs(x_clean)):.3f}")
```

### 步骤4：添加高斯白噪声

```python
# --- 添加噪声 ---
np.random.seed(42)  # 固定随机种子，保证可复现
noise_level = 0.5
noise = noise_level * np.random.randn(len(t))
x_noisy = x_clean + noise

# 计算实际SNR
signal_power = np.mean(x_clean**2)
noise_power = np.mean(noise**2)
snr_db = 10 * np.log10(signal_power / noise_power)
print(f"信噪比 SNR = {snr_db:.1f} dB")

# --- 对比绘图 ---
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(t[:300], x_clean[:300], 'b-', linewidth=1, label='纯净信号')
axes[0].set_title('纯净复合信号', fontproperties=font_prop)
axes[0].set_ylabel('幅值', fontproperties=font_prop)
axes[0].legend(prop=font_prop)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t[:300], x_noisy[:300], 'r-', linewidth=0.8, label='含噪信号')
axes[1].set_title(f'含噪信号 (SNR = {snr_db:.1f} dB)', fontproperties=font_prop)
axes[1].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[1].set_ylabel('幅值', fontproperties=font_prop)
axes[1].legend(prop=font_prop)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noisy_signal.png', dpi=150)
plt.show()
```

### 步骤5：FFT频谱分析

```python
# --- 对含噪信号做FFT ---
N = len(x_noisy)
X = np.fft.fft(x_noisy)           # FFT变换
freqs = np.fft.fftfreq(N, 1/fs)   # 对应频率轴

# 取单边频谱（正频率部分）
X_magnitude = 2.0 / N * np.abs(X[:N//2])
freqs_positive = freqs[:N//2]

# --- 绘制频谱 ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 含噪信号频谱
axes[0].plot(freqs_positive, X_magnitude, 'r-', linewidth=0.8)
axes[0].set_title('含噪信号频谱', fontproperties=font_prop)
axes[0].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[0].set_ylabel('幅值', fontproperties=font_prop)
axes[0].set_xlim(0, 500)
axes[0].grid(True, alpha=0.3)

# 纯净信号频谱（对比）
X_clean = np.fft.fft(x_clean)
X_clean_mag = 2.0 / N * np.abs(X_clean[:N//2])

axes[1].plot(freqs_positive, X_clean_mag, 'b-', linewidth=1.0)
axes[1].set_title('纯净信号频谱', fontproperties=font_prop)
axes[1].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[1].set_ylabel('幅值', fontproperties=font_prop)
axes[1].set_xlim(0, 500)
axes[1].grid(True, alpha=0.3)

# 标注峰值频率
peak_indices = np.argsort(X_clean_mag)[-3:][::-1]
for idx in peak_indices:
    if X_clean_mag[idx] > 0.05:
        axes[1].annotate(f'{freqs_positive[idx]:.0f} Hz',
                         xy=(freqs_positive[idx], X_clean_mag[idx]),
                         xytext=(freqs_positive[idx]+20, X_clean_mag[idx]+0.05),
                         fontproperties=font_prop,
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=12, color='red')

plt.tight_layout()
plt.savefig('frequency_spectrum.png', dpi=150)
plt.show()

print("频谱峰值频率:")
for idx in peak_indices:
    print(f"  f = {freqs_positive[idx]:.1f} Hz, 幅值 = {X_clean_mag[idx]:.4f}")
```

**预期输出**：频谱图中应清晰看到50Hz、120Hz、300Hz三个峰值。

### 步骤6（拓展）：读取音频文件并分析

```python
from scipy.io import wavfile

# --- 生成一段测试音频（因为没有外部文件） ---
fs_audio = 44100  # CD质量采样率
duration = 2.0    # 2秒
t_audio = np.arange(0, duration, 1/fs_audio)

# 模拟一段简单音频：A4音（440Hz）+ E5音（659Hz）
audio_signal = 0.5 * np.sin(2*np.pi*440*t_audio) + \
               0.3 * np.sin(2*np.pi*659*t_audio)

# 归一化到16-bit范围并保存
audio_int16 = np.int16(audio_signal / np.max(np.abs(audio_signal)) * 32767)
wavfile.write('test_audio.wav', fs_audio, audio_int16)
print(f"已生成测试音频: test_audio.wav")

# --- 读取并分析 ---
fs_read, audio_data = wavfile.read('test_audio.wav')
print(f"采样率: {fs_read} Hz")
print(f"采样点数: {len(audio_data)}")
print(f"时长: {len(audio_data)/fs_read:.2f} 秒")

# FFT分析
N_audio = len(audio_data)
X_audio = np.fft.fft(audio_data.astype(float))
freqs_audio = np.fft.fftfreq(N_audio, 1/fs_read)

# 只取正频率，限制到2000Hz
mask = (freqs_audio > 0) & (freqs_audio < 2000)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# 波形
t_plot = np.arange(len(audio_data)) / fs_read
axes[0].plot(t_plot[:2000], audio_data[:2000], 'b-', linewidth=0.5)
axes[0].set_title('音频波形（前2000个采样点）', fontproperties=font_prop)
axes[0].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[0].set_ylabel('幅值', fontproperties=font_prop)
axes[0].grid(True, alpha=0.3)

# 频谱
axes[1].plot(freqs_audio[mask], 2.0/N_audio * np.abs(X_audio[mask]), 'r-', linewidth=0.8)
axes[1].set_title('音频频谱 (0-2000 Hz)', fontproperties=font_prop)
axes[1].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[1].set_ylabel('幅值', fontproperties=font_prop)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('audio_spectrum.png', dpi=150)
plt.show()
```

---

## 六、实验报告要求

请将以下内容整理到实验报告中：

1. **截图与图表**：将步骤2~6中生成的所有图表粘贴到报告中，并给出简要说明。
2. **代码记录**：记录每一步的核心代码（可精简，但关键部分不可省略）。
3. **回答问题**：
   - 在步骤5的频谱图中，三个峰值分别对应什么频率？与信号生成时的参数是否一致？
   - 噪声对频谱有什么影响？低频和高频部分的表现有何不同？
   - 如果将采样率从1000Hz改为500Hz，300Hz分量在频谱中还能正确显示吗？为什么？

---

## 七、思考题

1. **频谱泄漏**：将信号长度 $T$ 从1.0秒改为0.95秒（即非整数周期），观察频谱峰值是否仍然尖锐？试解释原因（提示：考虑窗函数与频谱泄漏的关系）。

2. **奈奎斯特采样定理**：如果要检测一个频率为800Hz的信号分量，采样率最低应设为多少？如果采样率设为1000Hz，会出现什么现象？

3. **FFT点数的影响**：将FFT长度 $N$ 从默认的信号长度改为 $2^{10}=1024$ 点（使用 `np.fft.fft(x, n=1024)`），观察频谱分辨率的变化。频率分辨率 $\Delta f$ 与 $N$ 和 $f_s$ 的关系是什么？

---

> **实验提示**：如在使用中文字体时遇到问题，可将 `font_prop` 参数暂时去掉，改用英文标注。核心实验内容不受影响。


---

> **课程关联**：本实验安排在[[欢迎|智能信号处理课程]]W1。涉及知识点：[[知识点/经典信号处理/信号与系统基础|信号与系统基础]]、[[知识点/经典信号处理/傅里叶分析|傅里叶分析]]、[[知识点/经典信号处理/采样与重建|采样与重建]]。关联项目：项目一准备。
> 详见[[13周教学计划|13周教学计划]] | [[课程关系图谱]]
