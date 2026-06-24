# 迷你项目 B：音频信号降噪与采样实验

> **课程**：智能信号处理 · 武汉大学人工智能学院
> **适用年级**：大二本科生
> **项目性质**：个人独立完成
> **关键约束**：仅使用经典信号处理方法（**禁止深度学习**）
> **时间线**：第 2 周启动 → 第 6 周期中检查 → 第 12 周提交终稿

---

## 一、项目概述

本项目围绕**数字音频**展开，从采样定理出发，依次完成"采样—量化—加噪—降噪—对比评估"的完整信号处理流水线。你将亲手感受欠采样导致的混叠、量化带来的颗粒噪声，并分别使用 **FIR（有限脉冲响应）** 和 **IIR（无限脉冲响应）** 两种经典滤波器对含噪语音进行降噪，最终从频域、零极点图、信噪比等维度进行系统对比。

**核心问题**：给定一段被白噪声和 50 Hz 工频干扰污染的语音，如何用经典滤波方法恢复可懂度？

---

## 二、学习目标

完成本项目后，你应该能够：

1. **理解并验证奈奎斯特采样定理**：通过对同一段音频分别以 44100 / 16000 / 8000 / 4000 Hz 采样，直观体会混叠失真与信息损失。
2. **理解量化与信噪比的关系**：将 16 bit 音频逐步量化为 8 bit、4 bit，计算量化噪声功率并聆听差异。
3. **掌握 FIR 滤波器的窗函数设计法**：使用 `scipy.signal.firwin` 设计低通 FIR 滤波器，理解阶数、截止频率、窗类型对频率响应的影响。
4. **掌握 IIR Butterworth 滤波器设计**：使用 `scipy.signal.butter` + `scipy.signal.sosfilt` 完成 IIR 滤波，理解极点位置与滤波器稳定性的关系。
5. **具备 Z 域分析能力**：能绘制并解读零极点图（pole-zero plot）、幅频/相频响应曲线，并据此评价滤波器性能。

---

## 三、数据准备

### 3.1 获取音频文件

你需要 **至少一段干净语音**（建议 5–15 秒，单声道，16 bit，44.1 kHz）。来源可选：

| 方式 | 说明 |
|------|------|
| 自行录制 | 用手机录音 App 导出 WAV，注意选安静环境 |
| 公共数据集 | [LibriSpeech](https://www.openslr.org/12) 的 `test-clean` 子集（英文）；或自行寻找中文语料 |
| 合成信号 | 用 `numpy` 生成正弦波 + 谐波模拟"语音"（见下方模板） |

**推荐文件名**：`clean_speech.wav`

### 3.2 Python 读写 WAV 的基本用法

```python
import numpy as np
from scipy.io import wavfile

# ---------- 读取 ----------
fs, data = wavfile.read("clean_speech.wav")
print(f"采样率: {fs} Hz, 数据类型: {data.dtype}, 时长: {len(data)/fs:.2f} s")

# 若为双声道，取单声道
if data.ndim == 2:
    data = data[:, 0]

# 归一化到 [-1, 1]
data_float = data.astype(np.float64) / np.iinfo(data.dtype).max

# ---------- 写入 ----------
# 先还原到 int16
data_int16 = np.clip(data_float * 32767, -32768, 32767).astype(np.int16)
wavfile.write("output.wav", fs, data_int16)
```

### 3.3 Jupyter 中播放音频

```python
from IPython.display import Audio, display

display(Audio(data_float, rate=fs))
```

### 3.4 如果没有现成音频

下面的代码可生成一段"模拟语音"（多谐波合成）：

```python
fs = 44100
duration = 5  # 秒
t = np.linspace(0, duration, fs * duration, endpoint=False)

# 基频 200 Hz + 3 次谐波，模拟元音 /a/
signal = (0.5 * np.sin(2 * np.pi * 200 * t) +
          0.3 * np.sin(2 * np.pi * 400 * t) +
          0.15 * np.sin(2 * np.pi * 600 * t) +
          0.05 * np.sin(2 * np.pi * 800 * t))

# 加一点幅度包络使其更自然
envelope = np.exp(-0.3 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
signal = signal * envelope
signal = signal / np.max(np.abs(signal))  # 归一化

wavfile.write("clean_speech.wav", fs,
              (signal * 32767).astype(np.int16))
```

---

## 四、实验步骤与代码模板

> **说明**：以下每个步骤对应一个 Jupyter Notebook Cell。带 `# TODO` 标记的部分需要你自行补全。所有函数参数仅为参考，请根据你的实际音频调整。

---

### Step 1：加载音频并降采样——感受采样定理

**目标**：将 44.1 kHz 音频分别降采样到 16 kHz、8 kHz、4 kHz，对比波形与听感。

```python
# ==================== Step 1: 降采样对比 ====================
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
from IPython.display import Audio, display

# 加载原始音频
fs_orig, data_raw = wavfile.read("clean_speech.wav")
if data_raw.ndim == 2:
    data_raw = data_raw[:, 0]
data_orig = data_raw.astype(np.float64) / np.iinfo(data_raw.dtype).max

print(f"原始采样率: {fs_orig} Hz, 时长: {len(data_orig)/fs_orig:.2f} s")

# --- 播放原始音频 ---
print("▶ 原始音频 (44100 Hz):")
display(Audio(data_orig, rate=fs_orig))

# TODO: 使用 scipy.signal.resample 将 data_orig 降采样到以下目标采样率
target_rates = [16000, 8000, 4000]
downsampled = {}

for rate in target_rates:
    # TODO: 计算降采样后的样本数
    n_samples = # TODO: 填写公式，提示 len(data_orig) * rate / fs_orig

    # TODO: 使用 resample 函数降采样
    downsampled[rate] = # TODO

    print(f"\n▶ 降采样后 ({rate} Hz):")
    display(Audio(downsampled[rate], rate=rate))

# --- 绘制波形对比 ---
fig, axes = plt.subplots(len(target_rates) + 1, 1, figsize=(12, 8))
t_orig = np.arange(len(data_orig)) / fs_orig
axes[0].plot(t_orig, data_orig, linewidth=0.3)
axes[0].set_title(f"原始 ({fs_orig} Hz)")
axes[0].set_ylabel("振幅")

for i, rate in enumerate(target_rates):
    t_ds = np.arange(len(downsampled[rate])) / rate
    axes[i + 1].plot(t_ds, downsampled[rate], linewidth=0.3)
    axes[i + 1].set_title(f"降采样 ({rate} Hz)")
    axes[i + 1].set_ylabel("振幅")

axes[-1].set_xlabel("时间 (s)")
plt.tight_layout()
plt.savefig("step1_downsample_waveforms.png", dpi=150)
plt.show()

# --- 频谱对比 ---
fig, axes = plt.subplots(len(target_rates) + 1, 1, figsize=(12, 8))

def plot_spectrum(ax, signal, fs, title):
    """绘制单边幅度谱"""
    N = len(signal)
    freq = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(np.fft.rfft(signal)) / N
    ax.plot(freq, magnitude, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("|X(f)|")
    ax.set_xlim(0, fs / 2)

plot_spectrum(axes[0], data_orig, fs_orig, f"原始频谱 ({fs_orig} Hz)")
for i, rate in enumerate(target_rates):
    # TODO: 对降采样信号绘制频谱
    # TODO: 调用 plot_spectrum 函数
    pass  # 删除此行，填写你的代码

plt.tight_layout()
plt.savefig("step1_downsample_spectra.png", dpi=150)
plt.show()
```

**思考题**：
1. 当采样率降到 4000 Hz 时，你听到了什么异常？这与奈奎斯特定理有什么关系？
2. 原始信号的有效带宽大约是多少？你是如何判断的？

---

### Step 2：降低量化位深——感受量化噪声

**目标**：将 16 bit 音频量化为 8 bit、4 bit，听辨量化噪声。

```python
# ==================== Step 2: 量化位深对比 ====================

def quantize(signal_float, bits):
    """
    将 [-1, 1] 浮点信号量化为指定位深的整数，再还原为浮点。
    返回量化后的浮点信号及量化噪声。
    """
    levels = 2 ** bits  # TODO: 计算量化级数
    # TODO: 将浮点信号映射到整数区间 [-levels//2, levels//2 - 1]
    quantized_int = np.round(signal_float * (levels / 2 - 1))  # TODO: 填写正确的缩放
    # TODO: 还原到 [-1, 1]
    quantized_float = quantized_int / (levels / 2 - 1)  # TODO: 填写正确的缩放
    noise = signal_float - quantized_float
    return quantized_float, noise

bits_list = [16, 8, 4]
quantized_signals = {}

for bits in bits_list:
    q_sig, noise = quantize(data_orig, bits)
    quantized_signals[bits] = (q_sig, noise)

    # 计算 SQNR (Signal-to-Quantization-Noise Ratio)
    sig_power = np.mean(data_orig ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        sqnr_db = 10 * np.log10(sig_power / noise_power)
    else:
        sqnr_db = float('inf')
    print(f"{bits} bit: SQNR = {sqnr_db:.1f} dB")

    print(f"▶ {bits} bit 量化:")
    display(Audio(q_sig, rate=fs_orig))

# --- 波形 & 噪声对比 ---
fig, axes = plt.subplots(len(bits_list), 2, figsize=(14, 8))
t = np.arange(len(data_orig)) / fs_orig

for i, bits in enumerate(bits_list):
    q_sig, noise = quantized_signals[bits]
    axes[i, 0].plot(t, q_sig, linewidth=0.3)
    axes[i, 0].set_title(f"{bits} bit 量化波形")
    axes[i, 0].set_ylabel("振幅")

    # TODO: 绘制量化噪声的波形（右列）
    axes[i, 1].plot(t, noise, linewidth=0.3, color='red')
    axes[i, 1].set_title(f"{bits} bit 量化噪声")
    axes[i, 1].set_ylabel("振幅")

axes[-1, 0].set_xlabel("时间 (s)")
axes[-1, 1].set_xlabel("时间 (s)")
plt.tight_layout()
plt.savefig("step2_quantization.png", dpi=150)
plt.show()
```

**思考题**：
1. 理论上 N bit 均匀量化的 SQNR 约为 `6.02N + 1.76 dB`，你的实测值与理论值相差多少？分析原因。
2. 4 bit 量化后的噪声听起来像什么？它与白噪声有何不同？

---

### Step 3：添加噪声——构造含噪信号

**目标**：向干净语音添加高斯白噪声和 50 Hz 工频干扰，并分析噪声的频域特征。

```python
# ==================== Step 3: 添加噪声 ====================
np.random.seed(42)  # 保证可复现

# --- 3a: 添加高斯白噪声 ---
snr_db = 5  # 信噪比 (dB)，可调
sig_power = np.mean(data_orig ** 2)

# TODO: 根据目标 SNR 计算噪声功率
noise_power_target = sig_power / (10 ** (snr_db / 10))  # TODO: 填写公式

# TODO: 生成高斯白噪声
white_noise = np.random.randn(len(data_orig)) * np.sqrt(noise_power_target)  # TODO

# --- 3b: 添加 50 Hz 工频干扰 ---
powerline_freq = 50
amplitude_powerline = 0.05  # 工频幅度，可调
t = np.arange(len(data_orig)) / fs_orig

# TODO: 生成 50 Hz 正弦干扰
powerline_noise = amplitude_powerline * np.sin(2 * np.pi * powerline_freq * t)  # TODO

# --- 合成含噪信号 ---
noisy_signal = data_orig + white_noise + powerline_noise
noisy_signal = np.clip(noisy_signal, -1, 1)  # 防止溢出

print("▶ 含噪音频:")
display(Audio(noisy_signal, rate=fs_orig))

# 计算实际 SNR
actual_noise = noisy_signal - data_orig
actual_snr = 10 * np.log10(np.mean(data_orig ** 2) / np.mean(actual_noise ** 2))
print(f"实际 SNR = {actual_snr:.1f} dB")

# --- 频谱分析 ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# TODO: 绘制干净信号频谱
plot_spectrum(axes[0], data_orig, fs_orig, "干净信号频谱")

# TODO: 绘制含噪信号频谱
plot_spectrum(axes[1], noisy_signal, fs_orig, "含噪信号频谱")

# TODO: 绘制噪声频谱
plot_spectrum(axes[2], noisy_signal - data_orig, fs_orig, "噪声频谱")

plt.tight_layout()
plt.savefig("step3_noisy_spectra.png", dpi=150)
plt.show()

# 保存含噪音频
noisy_int16 = np.clip(noisy_signal * 32767, -32768, 32767).astype(np.int16)
wavfile.write("noisy_speech.wav", fs_orig, noisy_int16)
```

**思考题**：
1. 在频谱图中，50 Hz 工频干扰表现为怎样的特征？
2. 如果 SNR 降到 -5 dB（噪声功率大于信号功率），语音还能听懂吗？

---

### Step 4：FIR 低通滤波器设计与降噪

**目标**：使用窗函数法设计 FIR 低通滤波器，对含噪信号进行滤波。

```python
# ==================== Step 4: FIR 低通滤波 ====================
from scipy.signal import firwin, freqz, lfilter

# --- 设计 FIR 滤波器参数 ---
cutoff = 3400     # 截止频率 (Hz)，语音信号通常 300-3400 Hz
num_taps = 101    # 滤波器阶数（抽头数），必须为奇数
window = 'hamming'  # 窗函数类型，可选 'hamming', 'hanning', 'blackman', 'kaiser'

# TODO: 使用 firwin 设计 FIR 低通滤波器
# 提示: firwin(num_taps, cutoff, fs=fs_orig, window=window)
b_fir = firwin(num_taps, cutoff, fs=fs_orig, window=window)  # TODO

# --- 查看滤波器频率响应 ---
w, h = freqz(b_fir, worN=2048, fs=fs_orig)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# 幅频响应
axes[0].plot(w, 20 * np.log10(np.abs(h) + 1e-12))
axes[0].set_title("FIR 低通滤波器 — 幅频响应")
axes[0].set_xlabel("频率 (Hz)")
axes[0].set_ylabel("增益 (dB)")
axes[0].axvline(cutoff, color='r', linestyle='--', label=f'截止频率 {cutoff} Hz')
axes[0].set_ylim(-80, 5)
axes[0].legend()
axes[0].grid(True)

# 相频响应
axes[1].plot(w, np.unwrap(np.angle(h)))
axes[1].set_title("FIR 低通滤波器 — 相频响应")
axes[1].set_xlabel("频率 (Hz)")
axes[1].set_ylabel("相位 (rad)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("step4_fir_response.png", dpi=150)
plt.show()

# --- 应用滤波器 ---
# TODO: 使用 lfilter 对含噪信号进行滤波
filtered_fir = lfilter(b_fir, [1.0], noisy_signal)  # TODO

print("▶ FIR 滤波后:")
display(Audio(filtered_fir, rate=fs_orig))

# --- 滤波前后对比 ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
t_plot = np.arange(int(fs_orig * 0.05)) / fs_orig  # 只画前 50ms 看细节

# TODO: 绘制含噪信号波形（前 50ms）
axes[0].plot(t_plot, noisy_signal[:len(t_plot)], linewidth=0.8)
axes[0].set_title("含噪信号")

# TODO: 绘制 FIR 滤波后波形（前 50ms）
axes[1].plot(t_plot, filtered_fir[:len(t_plot)], linewidth=0.8, color='green')
axes[1].set_title("FIR 滤波后")

# TODO: 绘制干净信号波形（前 50ms）
axes[2].plot(t_plot, data_orig[:len(t_plot)], linewidth=0.8, color='orange')
axes[2].set_title("原始干净信号")

for ax in axes:
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("振幅")
plt.tight_layout()
plt.savefig("step4_fir_waveform_compare.png", dpi=150)
plt.show()

# --- 频谱对比 ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_spectrum(axes[0], noisy_signal, fs_orig, "含噪信号频谱")
# TODO: 绘制 FIR 滤波后信号频谱
plot_spectrum(axes[1], filtered_fir, fs_orig, "FIR 滤波后频谱")  # TODO
plt.tight_layout()
plt.savefig("step4_fir_spectrum_compare.png", dpi=150)
plt.show()

# --- 计算滤波后 SNR ---
residual_noise_fir = filtered_fir - data_orig  # 注意：这里忽略了滤波器延迟
# 更好的做法是对齐后再计算（见 Step 6）
snr_fir = 10 * np.log10(np.mean(data_orig ** 2) / np.mean(residual_noise_fir ** 2))
print(f"FIR 滤波后 SNR = {snr_fir:.1f} dB (未对齐延迟)")
```

**实验扩展**：
- 改变 `num_taps`（如 31、101、201），观察过渡带宽的变化。
- 改变 `window`（如 `'hamming'` → `'blackman'` → `'kaiser'`），对比旁瓣衰减。

**思考题**：
1. FIR 滤波器的阶数和延迟是什么关系？线性相位意味着什么？
2. 为什么 FIR 滤波器天然稳定？

---

### Step 5：IIR Butterworth 滤波器设计与降噪

**目标**：设计 IIR Butterworth 低通滤波器，对比其与 FIR 的差异。

```python
# ==================== Step 5: IIR Butterworth 滤波 ====================
from scipy.signal import butter, sosfilt, sosfreqz, zpk2sos

# --- 设计 IIR 滤波器参数 ---
order = 4           # 滤波器阶数
cutoff_iir = 3400   # 截止频率 (Hz)

# TODO: 使用 butter 函数设计 IIR Butterworth 低通滤波器
# 提示: butter(order, cutoff_iir, btype='low', fs=fs_orig, output='sos')
sos = butter(order, cutoff_iir, btype='low', fs=fs_orig, output='sos')  # TODO

# 也获取 zpk（零极点增益）形式用于分析
z, p, k = butter(order, cutoff_iir, btype='low', fs=fs_orig, output='zpk')

# --- 频率响应 ---
w, h = sosfreqz(sos, worN=2048, fs=fs_orig)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(w, 20 * np.log10(np.abs(h) + 1e-12))
axes[0].set_title(f"IIR Butterworth (阶数={order}) — 幅频响应")
axes[0].set_xlabel("频率 (Hz)")
axes[0].set_ylabel("增益 (dB)")
axes[0].axvline(cutoff_iir, color='r', linestyle='--', label=f'截止频率 {cutoff_iir} Hz')
axes[0].set_ylim(-80, 5)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(w, np.unwrap(np.angle(h)))
axes[1].set_title("IIR Butterworth — 相频响应")
axes[1].set_xlabel("频率 (Hz)")
axes[1].set_ylabel("相位 (rad)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("step5_iir_response.png", dpi=150)
plt.show()

# --- 零极点图 ---
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
unit_circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax.add_patch(unit_circle)

# TODO: 绘制零点 (z) 和极点 (p)
ax.plot(np.real(z), np.imag(z), 'bo', markersize=8, label='零点')  # TODO
ax.plot(np.real(p), np.imag(p), 'rx', markersize=10, label='极点')  # TODO

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_xlabel("实部")
ax.set_ylabel("虚部")
ax.set_title("IIR Butterworth 零极点图")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("step5_pole_zero.png", dpi=150)
plt.show()

# --- 应用滤波器 ---
# TODO: 使用 sosfilt 对含噪信号进行滤波
# 提示: sosfilt(sos, noisy_signal)
filtered_iir = sosfilt(sos, noisy_signal)  # TODO

print("▶ IIR 滤波后:")
display(Audio(filtered_iir, rate=fs_orig))

# --- 滤波前后频谱对比 ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_spectrum(axes[0], noisy_signal, fs_orig, "含噪信号频谱")
# TODO: 绘制 IIR 滤波后信号频谱
plot_spectrum(axes[1], filtered_iir, fs_orig, "IIR 滤波后频谱")  # TODO
plt.tight_layout()
plt.savefig("step5_iir_spectrum_compare.png", dpi=150)
plt.show()
```

**实验扩展**：
- 改变 `order`（2 → 4 → 6 → 8），观察截止频率附近的滚降陡度。
- 观察极点是否全部在单位圆内（稳定性条件）。

**思考题**：
1. 相同截止频率下，IIR 的阶数比 FIR 低很多就能达到类似效果，为什么？
2. IIR 滤波器的相位响应是线性的吗？这对语音信号意味着什么？

---

### Step 6：FIR vs IIR 系统对比

**目标**：从频率响应、零极点、延迟、SNR 等维度全面对比两种滤波器。

```python
# ==================== Step 6: FIR vs IIR 对比 ====================

# --- 6a: 幅频响应对比 ---
w_fir, h_fir = freqz(b_fir, worN=2048, fs=fs_orig)
w_iir, h_iir = sosfreqz(sos, worN=2048, fs=fs_orig)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-12), label='FIR')
axes[0].plot(w_iir, 20 * np.log10(np.abs(h_iir) + 1e-12), label='IIR')
axes[0].set_title("幅频响应对比")
axes[0].set_xlabel("频率 (Hz)")
axes[0].set_ylabel("增益 (dB)")
axes[0].set_ylim(-80, 5)
axes[0].legend()
axes[0].grid(True)

# 相频响应对比
axes[1].plot(w_fir, np.unwrap(np.angle(h_fir)), label='FIR')
axes[1].plot(w_iir, np.unwrap(np.angle(h_iir)), label='IIR')
axes[1].set_title("相频响应对比")
axes[1].set_xlabel("频率 (Hz)")
axes[1].set_ylabel("相位 (rad)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("step6_freq_response_compare.png", dpi=150)
plt.show()

# --- 6b: 零极点对比 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax in axes:
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(unit_circle)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel("实部")
    ax.set_ylabel("虚部")
    ax.grid(True)

# FIR 零极点（FIR 只有零点，极点全在原点）
# TODO: 用 np.roots 求 FIR 滤波器 b_fir 的零点
z_fir = np.roots(b_fir)  # TODO
p_fir = np.zeros(len(b_fir) - 1)  # FIR 的极点全在 z=0

axes[0].plot(np.real(z_fir), np.imag(z_fir), 'bo', markersize=5, label='零点')
axes[0].plot(np.real(p_fir), np.imag(p_fir), 'rx', markersize=8, label='极点')
axes[0].set_title("FIR 零极点图")
axes[0].legend()

# IIR 零极点
axes[1].plot(np.real(z), np.imag(z), 'bo', markersize=8, label='零点')
axes[1].plot(np.real(p), np.imag(p), 'rx', markersize=10, label='极点')
axes[1].set_title("IIR Butterworth 零极点图")
axes[1].legend()

plt.tight_layout()
plt.savefig("step6_pole_zero_compare.png", dpi=150)
plt.show()

# --- 6c: SNR 对比（对齐延迟后） ---
from scipy.signal.correlate import correlate

def align_and_compute_snr(original, filtered):
    """通过互相关对齐信号后计算 SNR"""
    # TODO: 使用互相关找到最佳延迟
    corr = correlate(filtered, original, mode='full')
    # TODO: 找到最大相关值对应的延迟
    delay = np.argmax(corr) - (len(original) - 1)  # TODO

    # 对齐
    if delay >= 0:
        aligned_orig = original[delay:]
        aligned_filt = filtered[:len(aligned_orig)]
    else:
        aligned_filt = filtered[-delay:]
        aligned_orig = original[:len(aligned_filt)]

    # 截取相同长度
    min_len = min(len(aligned_orig), len(aligned_filt))
    aligned_orig = aligned_orig[:min_len]
    aligned_filt = aligned_filt[:min_len]

    # TODO: 计算残余噪声和 SNR
    residual = aligned_filt - aligned_orig
    snr = 10 * np.log10(np.mean(aligned_orig ** 2) / np.mean(residual ** 2))
    return snr, delay

snr_noisy, _ = align_and_compute_snr(data_orig, noisy_signal)
snr_fir_aligned, delay_fir = align_and_compute_snr(data_orig, filtered_fir)
snr_iir_aligned, delay_iir = align_and_compute_snr(data_orig, filtered_iir)

print("=" * 50)
print(f"含噪信号 SNR:       {snr_noisy:.1f} dB")
print(f"FIR 滤波后 SNR:     {snr_fir_aligned:.1f} dB (延迟 {delay_fir} 样本)")
print(f"IIR 滤波后 SNR:     {snr_iir_aligned:.1f} dB (延迟 {delay_iir} 样本)")
print("=" * 50)

# --- 6d: 汇总对比表 ---
print("\n┌──────────────┬────────────────┬────────────────┐")
print("│   对比维度   │     FIR        │     IIR        │")
print("├──────────────┼────────────────┼────────────────┤")
print(f"│ 滤波器阶数   │ {num_taps:>6d} (抽头) │ {order:>6d} (阶)    │")
print(f"│ 延迟 (样本)  │ {delay_fir:>6d}         │ {delay_iir:>6d}         │")
print(f"│ 滤波后 SNR   │ {snr_fir_aligned:>6.1f} dB      │ {snr_iir_aligned:>6.1f} dB      │")
print("│ 相位特性     │ 线性相位       │ 非线性相位     │")
print("│ 稳定性       │ 天然稳定       │ 需设计保证     │")
print("│ 计算复杂度   │ 较高 (高阶)    │ 较低 (低阶)    │")
print("└──────────────┴────────────────┴────────────────┘")
```

**思考题**：
1. 在你的实验中，FIR 和 IIR 哪个降噪效果更好？SNR 提升了多少？
2. FIR 的线性相位在语音处理中为什么重要？
3. 如果要在嵌入式系统上实时处理，你会选 FIR 还是 IIR？为什么？

---

## 五、评分细则（共 100 分）

| 评分项 | 分值 | 评分标准 |
|--------|------|----------|
| **1. 代码正确性与可运行性** | **30** | |
| 　代码能无错误运行 | 10 | 所有 Cell 能顺利执行，无 `NameError` 等 |
| 　信号处理逻辑正确 | 10 | 降采样、量化、加噪、滤波的数学实现正确 |
| 　结果合理可解释 | 10 | 频谱、波形、SNR 数值符合预期 |
| **2. 实验完整性** | **25** | |
| 　6 个步骤全部完成 | 15 | 每个 Step 均有代码和输出（图/音频） |
| 　实验扩展（额外尝试） | 10 | 如尝试不同窗函数/阶数/SNR 并记录对比 |
| **3. 分析与思考** | **25** | |
| 　思考题回答 | 15 | 每道思考题均作答，答案有理有据 |
| 　自主发现与讨论 | 10 | 在 Markdown Cell 中写出个人观察和结论 |
| **4. 报告质量** | **15** | |
| 　排版整洁 | 5 | Markdown 格式规范，图表有标题和标签 |
| 　图表清晰 | 5 | 分辨率足够，坐标轴标注完整 |
| 　叙述连贯 | 5 | 逻辑通顺，有引言和总结 |
| **5. 提交规范** | **5** | |
| 　文件命名规范 | 2 | 按要求命名 |
| 　README 说明 | 3 | 包含运行环境和依赖说明 |
| **合计** | **100** | |

**加分项**（最高 +10 分）：
- 设计带通/带阻/高通滤波器并对比（+3）
- 实现频率采样法设计 FIR 滤波器（+3）
- 用 `scipy.signal.group_delay` 分析群延迟（+2）
- 封装为可复用的 Python 模块（+2）

---

## 六、提交要求

### 6.1 文件结构

```
学号_姓名_项目B/
├── README.md                    # 运行环境、依赖、简要说明
├── audio_denoising.ipynb        # 主 Notebook（含全部 6 步）
├── audio_denoising.html         # Notebook 导出的 HTML（方便查看）
├── clean_speech.wav             # 干净音频
├── noisy_speech.wav             # 含噪音频
├── figures/                     # 所有生成的图片
│   ├── step1_downsample_waveforms.png
│   ├── step1_downsample_spectra.png
│   ├── step2_quantization.png
│   ├── step3_noisy_spectra.png
│   ├── step4_fir_*.png
│   ├── step5_iir_*.png
│   └── step6_*.png
└── report.pdf                   # 实验报告（可选，也可写在 Notebook 中）
```

### 6.2 提交方式与截止时间

| 节点 | 时间 | 提交内容 |
|------|------|----------|
| 期中检查 | 第 6 周 | 提交 Step 1–3 的 Notebook（半成品即可），课堂演示 |
| 最终提交 | 第 12 周周五 23:59 | 完整文件夹打包为 `.zip`，上传至课程网站 |

### 6.3 代码规范要求

- 所有变量名使用有意义的英文命名
- 关键步骤添加中文注释
- 每个 Step 用 Markdown Cell 标题分隔
- 图表必须包含标题、坐标轴标签、图例（如适用）

---

## 七、常见问题 (FAQ)

**Q1：我没有现成的 WAV 文件怎么办？**
A1：见 §3.4，可用 `numpy` 合成多谐波信号；也可在 [Freesound](https://freesound.org/) 下载免费音频（需注册）。注意导出为单声道 WAV。

**Q2：`scipy.signal.resample` 和直接抽取有什么区别？**
A2：`resample` 会先做低通抗混叠滤波再抽取，直接抽取（如 `signal[::n]`）会产生混叠。你可以两者都试试，对比效果。

**Q3：FIR 滤波器的抽头数选多少合适？**
A3：一般经验值：过渡带宽越窄，需要的抽头数越多。公式估算：`N ≈ 4 / (Δf / fs)`，其中 `Δf` 是过渡带宽。建议先试 101，再尝试 31 和 201 对比。

**Q4：为什么用 SOS 形式而不是直接用 b, a 系数？**
A4：IIR 滤波器在高阶时，直接用传递函数系数 `b, a` 容易因数值精度问题导致不稳定。SOS（二阶节级联）形式数值稳定性更好。**强烈建议始终使用 `sosfilt` 而非 `lfilter(b, a, x)`**。

**Q5：滤波后信号有延迟怎么办？**
A5：FIR 线性相位滤波器的延迟为 `(N-1)/2` 个样本，可以简单截掉前面的样本。IIR 的延迟需要用互相关估算（见 Step 6）。计算 SNR 时务必先对齐。

**Q6：可以用 `pydub` 或 `librosa` 吗？**
A6：可以用于辅助（如播放音频），但核心信号处理必须用 `numpy` 和 `scipy.signal` 实现，不能用高级封装函数代替滤波设计。

**Q7：50 Hz 工频噪声怎么滤除？**
A7：可以用陷波滤波器（notch filter）：`scipy.signal.iirnotch(50, Q, fs)`，其中 Q 为品质因数。这是 FIR/IIR 的一个有趣扩展应用，鼓励尝试。

---

## 八、参考资源

### 教材与课程

1. 奥本海姆《信号与系统》（第 2 版）— 第 4、5、6 章（采样、Z 变换、滤波器设计）
2. 课堂 PPT：第 3 讲（采样定理）、第 5 讲（FIR/IIR 滤波器设计）

### Python 文档

| 资源 | 链接 |
|------|------|
| `scipy.signal` 参考 | https://docs.scipy.org/doc/scipy/reference/signal.html |
| `scipy.signal.firwin` | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html |
| `scipy.signal.butter` | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html |
| `scipy.signal.freqz` | https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html |
| `scipy.io.wavfile` | https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.html |
| Matplotlib 频谱图示例 | https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html |

### 在线教程

1. **SciPy 信号处理教程**：https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
2. **FIR 滤波器设计入门**：https://www.allaboutcircuits.com/technical-articles/design-of-fir-filters/
3. **Understanding IIR Filters**：https://www.dsprelated.com/freebooks/filters/IIR_Filter_Design.html

### 工具

- **Jupyter Notebook**：推荐通过 `pip install jupyterlab` 安装
- **Audacity**（免费音频编辑软件）：https://www.audacityteam.org/ — 可用于手动验证你的处理结果

---

> 📌 **最后提醒**：本项目的重点不是"滤波器越复杂越好"，而是**理解每个环节的信号处理原理**。即使结果不完美，只要你能清楚地解释"为什么"，就能获得好成绩。祝实验顺利！

---

> **课程关联**：本项目对应[[欢迎|智能信号处理课程]]W1-W5知识模块。涉及知识点：[[知识点/经典信号处理/信号与系统基础|信号与系统基础]]、[[知识点/经典信号处理/采样与重建|采样与重建]]、[[知识点/经典信号处理/傅里叶分析|傅里叶分析]]、[[知识点/经典信号处理/数字滤波器设计|数字滤波器设计]]（FIR/IIR）、[[知识点/经典信号处理/Z变换|Z变换]]。关联实验：[[实验2_FFT频谱分析与滤波器设计|实验2]]。
