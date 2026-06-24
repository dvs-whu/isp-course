# 实验2：FFT频谱分析与滤波器设计

## 一、实验信息

| 项目 | 内容 |
|------|------|
| 实验时长 | 2学时（90分钟） |
| 实验类型 | 综合设计型 |
| 适用课程 | 智能信号处理 |
| 面向对象 | 大二本科生 |

---

## 二、实验目的

1. **深入理解FFT频谱分析**：掌握FFT的计算过程、频率分辨率的概念，能够正确解读频谱图。
2. **掌握FIR与IIR滤波器的设计方法**：学会使用Python的`scipy.signal`模块设计低通、高通、带通滤波器，并理解截止频率、滤波器阶数等关键参数的影响。
3. **实现完整的信号滤波流程**：从信号加载、频谱分析、滤波器设计到滤波效果验证，完成端到端的信号处理实验。

---

## 三、实验环境

| 软件/库 | 推荐版本 |
|---------|---------|
| Python | 3.9+ |
| NumPy | 1.24+ |
| SciPy | 1.10+ |
| Matplotlib | 3.7+ |

```bash
pip install numpy scipy matplotlib
```

---

## 四、实验原理

### 4.1 DFT与FFT

离散傅里叶变换（DFT）：

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$$

频率分辨率：

$$\Delta f = \frac{f_s}{N}$$

其中 $f_s$ 为采样率，$N$ 为FFT点数。频率分辨率决定了我们能区分的最小频率间隔。

### 4.2 FIR滤波器

有限脉冲响应（FIR）滤波器的输出为：

$$y[n] = \sum_{k=0}^{M} b_k \, x[n-k]$$

其中 $M$ 为滤波器阶数。FIR滤波器的优点是**线性相位**和**绝对稳定**。常用设计方法包括窗函数法和等波纹法。

### 4.3 IIR滤波器

无限脉冲响应（IIR）滤波器：

$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$

IIR滤波器阶数低、计算效率高，但可能存在**非线性相位**和**稳定性问题**。常用类型有Butterworth、Chebyshev、Elliptic等。

### 4.4 滤波器性能指标

| 指标 | 含义 |
|------|------|
| 截止频率 $f_c$ | 通带与过渡带的边界频率（通常取-3dB点） |
| 通带纹波 | 通带内幅度响应的波动范围 |
| 阻带衰减 | 阻带内信号被抑制的程度 |
| 过渡带宽度 | 从通带到阻带的频率过渡区间 |

---

## 五、实验步骤

### 步骤1：生成测试信号并分析频谱

我们生成一个包含多个频率分量并叠加噪声的测试信号，模拟实际场景：

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import signal

# 中文字体设置
font_prop = fm.FontProperties(fname='/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf')

# --- 生成测试信号 ---
fs = 1000           # 采样率 1000 Hz
T = 2.0             # 时长 2 秒
t = np.arange(0, T, 1/fs)
N = len(t)

# 信号分量
x1 = 1.0 * np.sin(2*np.pi*50*t)     # 50Hz 有用信号
x2 = 0.5 * np.sin(2*np.pi*120*t)    # 120Hz 有用信号
x3 = 0.3 * np.sin(2*np.pi*200*t)    # 200Hz 干扰信号
x4 = 0.8 * np.sin(2*np.pi*350*t)    # 350Hz 干扰信号

np.random.seed(0)
noise = 0.5 * np.random.randn(N)
x_clean = x1 + x2
x_interf = x3 + x4
x_noisy = x_clean + x_interf + noise

print(f"采样率: {fs} Hz, 采样点数: {N}, 频率分辨率: {fs/N:.2f} Hz")

# --- FFT分析 ---
X_mag = 2.0/N * np.abs(np.fft.rfft(x_noisy))
freqs = np.fft.rfftfreq(N, 1/fs)

X_clean_mag = 2.0/N * np.abs(np.fft.rfft(x_clean))

# --- 绘图 ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(t[:400], x_noisy[:400], 'b-', linewidth=0.8)
axes[0].set_title('含噪+干扰信号时域波形', fontproperties=font_prop)
axes[0].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[0].set_ylabel('幅值', fontproperties=font_prop)
axes[0].grid(True, alpha=0.3)

axes[1].plot(freqs, X_mag, 'r-', linewidth=0.8, label='含噪+干扰')
axes[1].set_title('含噪+干扰信号频谱', fontproperties=font_prop)
axes[1].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[1].set_ylabel('幅值', fontproperties=font_prop)
axes[1].set_xlim(0, 500)
axes[1].legend(prop=font_prop)
axes[1].grid(True, alpha=0.3)

axes[2].plot(freqs, X_clean_mag, 'b-', linewidth=1, label='纯净信号')
axes[2].set_title('纯净信号频谱（参考）', fontproperties=font_prop)
axes[2].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[2].set_ylabel('幅值', fontproperties=font_prop)
axes[2].set_xlim(0, 500)
axes[2].legend(prop=font_prop)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step1_spectrum_analysis.png', dpi=150)
plt.show()
```

**分析要点**：频谱图中应看到50Hz、120Hz（有用信号）和200Hz、350Hz（干扰）四个峰值。我们的目标是用低通滤波器保留200Hz以下的信号，滤除200Hz以上的干扰。

### 步骤2：设计FIR低通滤波器（窗函数法）

```python
# --- FIR滤波器设计（窗函数法） ---
fc = 150            # 截止频率 150 Hz
num_taps = 101      # 滤波器阶数（必须为奇数以便线性相位）

# 方法1：使用firwin
b_fir = signal.firwin(num_taps, fc, fs=fs, window='hamming')
a_fir = [1.0]  # FIR滤波器的a系数为1

print(f"FIR滤波器阶数: {num_taps-1}")
print(f"截止频率: {fc} Hz")
print(f"系数个数: {len(b_fir)}")

# --- 频率响应 ---
w_fir, h_fir = signal.freqz(b_fir, a_fir, worN=2048, fs=fs)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# 幅度响应
axes[0].plot(w_fir, 20*np.log10(np.abs(h_fir) + 1e-10), 'b-', linewidth=1.5)
axes[0].axvline(fc, color='r', linestyle='--', label=f'截止频率 {fc}Hz')
axes[0].axhline(-3, color='g', linestyle=':', alpha=0.5, label='-3 dB')
axes[0].set_title('FIR低通滤波器 — 幅度响应', fontproperties=font_prop)
axes[0].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[0].set_ylabel('增益 (dB)', fontproperties=font_prop)
axes[0].set_xlim(0, 500)
axes[0].set_ylim(-80, 5)
axes[0].legend(prop=font_prop)
axes[0].grid(True, alpha=0.3)

# 相位响应
phase_fir = np.unwrap(np.angle(h_fir))
axes[1].plot(w_fir, np.degrees(phase_fir), 'r-', linewidth=1.5)
axes[1].set_title('FIR低通滤波器 — 相位响应', fontproperties=font_prop)
axes[1].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[1].set_ylabel('相位 (度)', fontproperties=font_prop)
axes[1].set_xlim(0, 500)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step2_fir_response.png', dpi=150)
plt.show()
```

### 步骤3：设计IIR低通滤波器（Butterworth）

```python
# --- IIR滤波器设计（Butterworth） ---
order_iir = 6       # 滤波器阶数
fc_iir = 150        # 截止频率

b_iir, a_iir = signal.butter(order_iir, fc_iir, btype='low', fs=fs)

print(f"Butterworth滤波器阶数: {order_iir}")

# 频率响应
w_iir, h_iir = signal.freqz(b_iir, a_iir, worN=2048, fs=fs)

# --- 对比FIR和IIR ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(w_fir, 20*np.log10(np.abs(h_fir)+1e-10), 'b-', linewidth=1.5, label=f'FIR (N={num_taps-1})')
axes[0].plot(w_iir, 20*np.log10(np.abs(h_iir)+1e-10), 'r-', linewidth=1.5, label=f'IIR Butterworth (N={order_iir})')
axes[0].axvline(fc, color='gray', linestyle='--', alpha=0.5)
axes[0].axhline(-3, color='gray', linestyle=':', alpha=0.5)
axes[0].set_title('FIR vs IIR 低通滤波器幅度响应对比', fontproperties=font_prop)
axes[0].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[0].set_ylabel('增益 (dB)', fontproperties=font_prop)
axes[0].set_xlim(0, 500)
axes[0].set_ylim(-80, 5)
axes[0].legend(prop=font_prop)
axes[0].grid(True, alpha=0.3)

# 相位对比
phase_iir = np.unwrap(np.angle(h_iir))
axes[1].plot(w_fir, np.degrees(phase_fir), 'b-', linewidth=1.5, label='FIR')
axes[1].plot(w_iir, np.degrees(phase_iir), 'r-', linewidth=1.5, label='IIR')
axes[1].set_title('FIR vs IIR 相位响应对比', fontproperties=font_prop)
axes[1].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[1].set_ylabel('相位 (度)', fontproperties=font_prop)
axes[1].set_xlim(0, 500)
axes[1].legend(prop=font_prop)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step3_fir_vs_iir.png', dpi=150)
plt.show()
```

### 步骤4：应用滤波器并对比效果

```python
# --- 应用滤波 ---
# FIR滤波（零相位滤波，避免相位延迟）
x_filtered_fir = signal.filtfilt(b_fir, a_fir, x_noisy)

# IIR滤波
x_filtered_iir = signal.filtfilt(b_iir, a_iir, x_noisy)

# --- 滤波后频谱 ---
X_filt_fir = 2.0/N * np.abs(np.fft.rfft(x_filtered_fir))
X_filt_iir = 2.0/N * np.abs(np.fft.rfft(x_filtered_iir))

# --- 综合对比图 ---
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# 时域对比
t_show = slice(0, 400)
axes[0].plot(t[t_show], x_clean[t_show], 'b-', linewidth=1.2, label='纯净信号')
axes[0].plot(t[t_show], x_filtered_fir[t_show], 'r--', linewidth=1, alpha=0.8, label='FIR滤波后')
axes[0].set_title('FIR滤波效果（时域）', fontproperties=font_prop)
axes[0].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[0].set_ylabel('幅值', fontproperties=font_prop)
axes[0].legend(prop=font_prop)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t[t_show], x_clean[t_show], 'b-', linewidth=1.2, label='纯净信号')
axes[1].plot(t[t_show], x_filtered_iir[t_show], 'g--', linewidth=1, alpha=0.8, label='IIR滤波后')
axes[1].set_title('IIR滤波效果（时域）', fontproperties=font_prop)
axes[1].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[1].set_ylabel('幅值', fontproperties=font_prop)
axes[1].legend(prop=font_prop)
axes[1].grid(True, alpha=0.3)

# 频域对比
axes[2].plot(freqs, X_mag, 'gray', linewidth=0.5, label='滤波前')
axes[2].plot(freqs, X_filt_fir, 'r-', linewidth=1.2, label='FIR滤波后')
axes[2].plot(freqs, X_clean_mag, 'b--', linewidth=1, label='纯净信号')
axes[2].set_title('FIR滤波频谱对比', fontproperties=font_prop)
axes[2].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[2].set_ylabel('幅值', fontproperties=font_prop)
axes[2].set_xlim(0, 500)
axes[2].legend(prop=font_prop)
axes[2].grid(True, alpha=0.3)

axes[3].plot(freqs, X_mag, 'gray', linewidth=0.5, label='滤波前')
axes[3].plot(freqs, X_filt_iir, 'g-', linewidth=1.2, label='IIR滤波后')
axes[3].plot(freqs, X_clean_mag, 'b--', linewidth=1, label='纯净信号')
axes[3].set_title('IIR滤波频谱对比', fontproperties=font_prop)
axes[3].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[3].set_ylabel('幅值', fontproperties=font_prop)
axes[3].set_xlim(0, 500)
axes[3].legend(prop=font_prop)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step4_filter_comparison.png', dpi=150)
plt.show()

# --- 定量评估 ---
error_fir = np.mean((x_filtered_fir - x_clean)**2)
error_iir = np.mean((x_filtered_iir - x_clean)**2)
print(f"FIR滤波后 MSE = {error_fir:.6f}")
print(f"IIR滤波后 MSE = {error_iir:.6f}")
```

### 步骤5：带通滤波器设计（拓展）

```python
# --- 带通滤波器：只保留80~160Hz ---
low_cut = 80
high_cut = 160

# FIR带通
b_bp_fir = signal.firwin(num_taps, [low_cut, high_cut], pass_zero=False, fs=fs)

# IIR带通
b_bp_iir, a_bp_iir = signal.butter(4, [low_cut, high_cut], btype='band', fs=fs)

# 频率响应
w_bp, h_bp_fir = signal.freqz(b_bp_fir, [1], worN=2048, fs=fs)
_, h_bp_iir = signal.freqz(b_bp_iir, a_bp_iir, worN=2048, fs=fs)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(w_bp, 20*np.log10(np.abs(h_bp_fir)+1e-10), 'b-', linewidth=1.5, label='FIR带通')
ax.plot(w_bp, 20*np.log10(np.abs(h_bp_iir)+1e-10), 'r-', linewidth=1.5, label='IIR带通')
ax.axvline(low_cut, color='gray', linestyle='--', alpha=0.5)
ax.axvline(high_cut, color='gray', linestyle='--', alpha=0.5)
ax.set_title(f'带通滤波器 ({low_cut}-{high_cut} Hz)', fontproperties=font_prop)
ax.set_xlabel('频率 (Hz)', fontproperties=font_prop)
ax.set_ylabel('增益 (dB)', fontproperties=font_prop)
ax.set_xlim(0, 500)
ax.set_ylim(-80, 5)
ax.legend(prop=font_prop)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('step5_bandpass.png', dpi=150)
plt.show()

# 应用带通滤波
x_bp = signal.filtfilt(b_bp_fir, [1], x_noisy)
X_bp_mag = 2.0/N * np.abs(np.fft.rfft(x_bp))

print("带通滤波后，120Hz分量应被保留，其他频率被抑制。")
```

### 步骤6：保存和读取WAV文件（完整流程）

```python
from scipy.io import wavfile

# --- 保存滤波前后信号为WAV文件 ---
# 归一化到16-bit
def normalize_to_int16(x):
    return np.int16(x / (np.max(np.abs(x)) + 1e-10) * 32767 * 0.9)

# 使用较高的采样率以适合音频
fs_audio = 8000
T_audio = 2.0
t_audio = np.arange(0, T_audio, 1/fs_audio)

# 模拟音频信号（440Hz基频 + 谐波 + 高频噪声）
audio_clean = 0.6*np.sin(2*np.pi*440*t_audio) + \
              0.3*np.sin(2*np.pi*880*t_audio) + \
              0.1*np.sin(2*np.pi*1320*t_audio)

np.random.seed(123)
hf_noise = 0.4*np.sin(2*np.pi*3500*t_audio) + \
           0.3*np.sin(2*np.pi*4200*t_audio) + \
           0.3*np.random.randn(len(t_audio))

audio_noisy = audio_clean + hf_noise

# 设计低通滤波器，截止频率2000Hz
b_audio, a_audio = signal.butter(8, 2000, btype='low', fs=fs_audio)
audio_filtered = signal.filtfilt(b_audio, a_audio, audio_noisy)

# 保存
wavfile.write('audio_noisy.wav', fs_audio, normalize_to_int16(audio_noisy))
wavfile.write('audio_filtered.wav', fs_audio, normalize_to_int16(audio_filtered))
wavfile.write('audio_clean.wav', fs_audio, normalize_to_int16(audio_clean))

print("已保存: audio_noisy.wav, audio_filtered.wav, audio_clean.wav")

# --- 频谱对比 ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
N_audio = len(t_audio)
f_audio = np.fft.rfftfreq(N_audio, 1/fs_audio)

for ax, data, title in zip(axes,
    [audio_noisy, audio_filtered, audio_clean],
    ['含噪音频频谱', '滤波后音频频谱', '纯净音频频谱']):
    mag = 2.0/N_audio * np.abs(np.fft.rfft(data))
    ax.plot(f_audio, mag, linewidth=0.8)
    ax.set_title(title, fontproperties=font_prop)
    ax.set_xlabel('频率 (Hz)', fontproperties=font_prop)
    ax.set_ylabel('幅值', fontproperties=font_prop)
    ax.set_xlim(0, 5000)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step6_audio_filtering.png', dpi=150)
plt.show()
```

---

## 六、实验报告要求

1. **图表汇总**：将步骤1~6生成的所有频谱图、滤波器响应图整理到报告中。
2. **FIR vs IIR对比分析**：
   - 从幅度响应、相位响应、滤波后MSE三个方面对比两种滤波器。
   - 讨论各自的优缺点及适用场景。
3. **回答问题**：
   - 频率分辨率由哪些因素决定？如果要区分49Hz和51Hz的信号，FFT长度至少应为多少？
   - 为什么使用`filtfilt`而不是`lfilter`？两者在输出信号上有何区别？
   - FIR滤波器阶数增大对滤波效果有何影响？试将阶数改为31和201，对比结果。

---

## 七、思考题

1. **滤波器阶数与计算复杂度**：FIR滤波器的计算复杂度为 $O(M \cdot N)$（$M$为阶数，$N$为信号长度）。对于一个10秒、采样率44100Hz的音频信号，如果要求过渡带宽度为100Hz，FIR滤波器大约需要多少阶？这会带来多大的计算量？

2. **零相位滤波的代价**：`filtfilt`通过前后两次滤波实现零相位，但会导致滤波器的有效阶数加倍。这在实时系统（如在线音频处理）中是否可行？如果不可以，有什么替代方案？

3. **Chebyshev与Butterworth**：用`signal.cheby1`设计相同截止频率和阶数的Chebyshev I型滤波器，与Butterworth对比。Chebyshev滤波器在通带内的纹波和过渡带陡峭度有何不同？在什么应用场景中Chebyshev更优？

---

> **实验提示**：如果对滤波器设计原理感到困惑，建议先从简单的2阶Butterworth滤波器开始，逐步提高阶数，观察频率响应的变化趋势。


---

> **课程关联**：本实验安排在[[欢迎|智能信号处理课程]]W4。涉及知识点：[[知识点/经典信号处理/傅里叶分析|傅里叶分析]]、[[知识点/经典信号处理/数字滤波器设计|数字滤波器设计]]、[[知识点/经典信号处理/Z变换|Z变换]]。关联项目：[[课程/项目库/项目一/项目说明书_迷你项目A_心电信号|项目一A]]、[[课程/项目库/项目一/项目说明书_迷你项目B_音频降噪|项目一B]]。
> 详见[[13周教学计划|13周教学计划]] | [[课程关系图谱]]
