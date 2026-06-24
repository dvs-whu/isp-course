# 迷你项目A：心电信号滤波与心率检测

> 课程：智能信号处理 | 武汉大学人工智能学院
> 类型：个人项目 | 仅限经典信号处理方法（FFT、滤波器）

---

## 一、项目概述

心电图（ECG）是临床中最基本的生物电信号之一，其自动分析在远程医疗和可穿戴设备中具有重要应用价值。本项目要求学生基于 MIT-BIH 心律失常数据库，运用经典信号处理方法（频谱分析、数字滤波器设计、峰值检测）完成 ECG 信号的去噪预处理，并实现自动心率（BPM）检测算法。学生将深入理解采样定理、频域分析、IIR/FIR 滤波器设计、信号卷积等核心知识点，并通过 Python 编程实践将其应用于真实生理信号。

---

## 二、学习目标

完成本项目后，你将能够：

1. **信号加载与探索**：使用 `wfdb` 库从 PhysioNet 加载 MIT-BIH 数据集，理解 ECG 信号的物理含义、采样率和数据格式。
2. **频谱分析**：对含噪 ECG 信号进行 FFT 频谱分析，识别信号主频带与噪声频带（工频干扰 50/60Hz、基线漂移 <0.5Hz）。
3. **滤波器设计与实现**：设计并实现带通滤波器（如 0.5–40Hz）去除基线漂移和高频噪声，对比 IIR（如 Butterworth）与 FIR（如窗函数法）的性能差异。
4. **峰值检测与心率计算**：实现 R 波峰值检测算法，基于 R-R 间期计算瞬时心率与平均心率，检测异常心率。
5. **结果可视化与报告**：制作规范的对比图，撰写包含信号处理理论分析的技术报告。

---

## 三、数据准备

### 3.1 MIT-BIH 数据库下载

MIT-BIH Arrhythmia Database 是 PhysioNet 上的经典 ECG 数据集，包含 48 条 30 分钟的双导联记录，采样率 360Hz，11-bit 分辨率。

**下载方式（推荐 Python 直接读取）：**

```bash
pip install wfdb numpy scipy matplotlib
```

**方式一：Python 在线读取（无需下载）**

```python
import wfdb
# 直接从 PhysioNet 在线读取记录 100
record = wfdb.rdrecord('100', sampto=3600, pn_dir='mitdb')
```

**方式二：下载到本地**

```bash
# 下载整个数据库（约 80MB）
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
```

### 3.2 数据加载代码

```python
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# ============================================
# Cell 1: 加载 ECG 数据
# ============================================
# 从 PhysioNet 在线读取记录 100 的前 10 秒数据
# 采样率 360Hz → 10秒 = 3600 个采样点
record = wfdb.rdrecord('100', sampto=3600, pn_dir='mitdb')
signal = record.p_signal[:, 0]  # 取第一导联（MLII）
fs = record.fs                   # 采样率 = 360

print(f"采样率: {fs} Hz")
print(f"信号长度: {len(signal)} 采样点 ({len(signal)/fs:.1f} 秒)")
print(f"信号范围: [{signal.min():.3f}, {signal.max():.3f}] mV")
print(f"信号均值: {signal.mean():.4f} mV")
print(f"信号标准差: {signal.std():.4f} mV")
```

### 3.3 数据探索

```python
# ============================================
# Cell 2: 数据探索 - 可视化 ECG 信号
# ============================================
t = np.arange(len(signal)) / fs  # 时间轴（秒）

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

# 时域波形
axes[0].plot(t, signal, 'b-', linewidth=0.5)
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('幅值 (mV)')
axes[0].set_title('ECG 信号 - 记录 100, 导联 MLII')
axes[0].grid(True, alpha=0.3)

# 频谱分析
N = len(signal)
freq = np.fft.rfftfreq(N, d=1/fs)
spectrum = np.abs(np.fft.rfft(signal))
axes[1].plot(freq, spectrum, 'r-', linewidth=0.5)
axes[1].set_xlabel('频率 (Hz)')
axes[1].set_ylabel('幅值')
axes[1].set_title('ECG 信号频谱')
axes[1].set_xlim([0, 100])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ecg_exploration.png', dpi=150)
plt.show()

# 打印 R 波标注信息（来自数据库）
annotation = wfdb.rdann('100', 'atr', sampto=3600, pn_dir='mitdb')
print(f"\n标注符号类型: {np.unique(annotation.symbol)}")
print(f"前10个 R 波位置: {annotation.sample[:10]}")
print(f"前10个标注: {annotation.symbol[:10]}")
```

---

## 四、任务步骤

### 步骤 1：信号频谱分析与噪声识别

**目标**：对原始 ECG 信号做 FFT，识别有用信号和噪声的频率分布。

```python
# ============================================
# Cell 3: 步骤1 - 频谱分析
# ============================================
def analyze_spectrum(signal, fs, title='频谱分析'):
    """对信号进行FFT频谱分析，返回频率和幅值"""
    N = len(signal)
    freq = np.fft.rfftfreq(N, d=1/fs)
    spectrum = np.abs(np.fft.rfft(signal)) / N * 2  # 归一化
    return freq, spectrum

# 对原始信号做频谱分析
freq, spec = analyze_spectrum(signal, fs)

# --- TODO: your code here ---
# 1. 绘制频谱图，标注以下频带：
#    - 基线漂移噪声区: 0 ~ 0.5 Hz
#    - ECG有用信号区: 0.5 ~ 40 Hz
#    - 肌电噪声区: 40 ~ 100 Hz
#    - 工频干扰: 50 Hz 或 60 Hz
# 2. 使用 plt.axvspan 标注各频带
# 3. 使用 plt.annotate 标注各频带名称
# --- TODO END ---
```

**预期输出**：
- 频谱图上标注出 4 个频带区域（不同颜色）
- 控制台输出 ECG 主要能量集中的频率范围

**常见坑**：
- ⚠️ FFT 结果需除以 N 做归一化，否则幅值不准确
- ⚠️ `rfftfreq` 的 d 参数是采样间隔 `1/fs`，不是 `fs`

---

### 步骤 2：添加模拟噪声

**目标**：向干净信号中添加 50Hz 工频干扰和低频基线漂移，模拟真实采集场景。

```python
# ============================================
# Cell 4: 步骤2 - 添加模拟噪声
# ============================================
t = np.arange(len(signal)) / fs

# --- TODO: your code here ---
# 1. 添加 50Hz 工频干扰 (幅度 0.15 mV)
powerline_noise = 0.15 * np.sin(2 * np.pi * 50 * t)

# 2. 添加基线漂移: 0.3Hz 低频漂移 (幅度 0.3 mV)
baseline_drift = 0.3 * np.sin(2 * np.pi * 0.3 * t)

# 3. 合成含噪信号
noisy_signal = signal + powerline_noise + baseline_drift
# --- TODO END ---

# 绘制对比图: 原始 vs 含噪
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
t_show = t[:1800]  # 显示前5秒
axes[0].plot(t_show, signal[:1800], 'b-', linewidth=0.8)
axes[0].set_title('原始 ECG 信号')
axes[0].set_ylabel('幅值 (mV)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_show, noisy_signal[:1800], 'r-', linewidth=0.8)
axes[1].set_title('含噪 ECG 信号 (50Hz 工频 + 基线漂移)')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('幅值 (mV)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noisy_comparison.png', dpi=150)
plt.show()

print(f"原始信号 SNR: ∞ (纯净信号)")
print(f"添加噪声后标准差变化: {signal.std():.4f} → {noisy_signal.std():.4f}")
```

**预期输出**：
- 上下对比图，含噪信号可明显看到波形基线起伏和毛刺
- 含噪信号的标准差明显增大

**常见坑**：
- ⚠️ 确保噪声数组长度与信号一致
- ⚠️ 50Hz 幅度不宜太大（0.1~0.2mV），否则滤波后效果不明显

---

### 步骤 3：设计带通滤波器

**目标**：设计一个 0.5–40Hz 的带通滤波器，去除基线漂移和高频噪声。

```python
# ============================================
# Cell 5: 步骤3 - 滤波器设计
# ============================================
from scipy import signal as sig

# --- TODO: your code here ---
# === 方法 A: Butterworth IIR 带通滤波器 ===
low_cut = 0.5   # 下截止频率 (Hz)
high_cut = 40    # 上截止频率 (Hz)
order = 4        # 滤波器阶数

# 1. 使用 scipy.signal.butter 设计滤波器
#    提示: 需要先用 sig.butter 设计, 再用 sig.sosfilt 或 sig.filtfilt 滤波
#    注意: 低截止频率很低时，直接用二阶节(SOS)形式更稳定
sos = sig.butter(order, [low_cut, high_cut], btype='bandpass',
                 fs=fs, output='sos')

# 2. 用零相移滤波 (filtfilt) 对含噪信号滤波
filtered_iir = sig.sosfiltfilt(sos, noisy_signal)

# === 方法 B: FIR 带通滤波器（窗函数法）===
# 3. 使用 scipy.signal.firwin 设计 FIR 滤波器
num_taps = 101  # 滤波器抽头数（必须为奇数）
fir_coeff = sig.firwin(num_taps, [low_cut, high_cut], pass_zero=False, fs=fs)

# 4. 用 lfilter 或 filtfilt 滤波
filtered_fir = sig.filtfilt(fir_coeff, [1.0], noisy_signal)
# --- TODO END ---

# 绘制滤波器频率响应
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# IIR 频率响应
w_iir, h_iir = sig.sosfreqz(sos, worN=2048, fs=fs)
axes[0].plot(w_iir, 20*np.log10(np.abs(h_iir)))
axes[0].set_title('Butterworth IIR 带通滤波器频率响应')
axes[0].set_xlabel('频率 (Hz)')
axes[0].set_ylabel('增益 (dB)')
axes[0].set_xlim([0, 80])
axes[0].set_ylim([-80, 5])
axes[0].axvline(0.5, color='r', linestyle='--', alpha=0.5, label='0.5Hz')
axes[0].axvline(40, color='r', linestyle='--', alpha=0.5, label='40Hz')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# FIR 频率响应
w_fir, h_fir = sig.freqz(fir_coeff, worN=2048, fs=fs)
axes[1].plot(w_fir, 20*np.log10(np.abs(h_fir)))
axes[1].set_title('FIR 带通滤波器频率响应')
axes[1].set_xlabel('频率 (Hz)')
axes[1].set_ylabel('增益 (dB)')
axes[1].set_xlim([0, 80])
axes[1].set_ylim([-80, 5])
axes[1].axvline(0.5, color='r', linestyle='--', alpha=0.5)
axes[1].axvline(40, color='r', linestyle='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('filter_response.png', dpi=150)
plt.show()
```

**预期输出**：
- 两幅滤波器频率响应图，0.5Hz 以下和 40Hz 以上增益显著衰减
- IIR 滤波器阶数低但相位非线性；FIR 阶数高但线性相位

**常见坑**：
- ⚠️ 极低截止频率（0.5Hz）用 `butter` 的 ba 形式数值不稳定，**务必用 SOS 形式 + `sosfiltfilt`**
- ⚠️ FIR 的 `num_taps` 越大过渡带越窄，但计算量和延迟越大
- ⚠️ `firwin` 截止频率参数个数应与 `pass_zero` 匹配

---

### 步骤 4：滤波效果对比

**目标**：对比 IIR 和 FIR 滤波效果，分析滤波前后频谱变化。

```python
# ============================================
# Cell 6: 步骤4 - 滤波效果对比
# ============================================

# --- TODO: your code here ---
# 1. 绘制 4 子图对比：原始、含噪、IIR滤波、FIR滤波
t_show = t[:1800]  # 前5秒

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(t_show, signal[:1800], 'b-', linewidth=0.8)
axes[0].set_title('原始 ECG 信号')
axes[0].set_ylabel('幅值 (mV)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_show, noisy_signal[:1800], 'r-', linewidth=0.8)
axes[1].set_title('含噪 ECG 信号')
axes[1].set_ylabel('幅值 (mV)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_show, filtered_iir[:1800], 'g-', linewidth=0.8)
axes[2].set_title('IIR Butterworth 滤波后')
axes[2].set_ylabel('幅值 (mV)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t_show, filtered_fir[:1800], 'm-', linewidth=0.8)
axes[3].set_title('FIR 窗函数法 滤波后')
axes[3].set_xlabel('时间 (s)')
axes[3].set_ylabel('幅值 (mV)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('filter_comparison.png', dpi=150)
plt.show()

# 2. 计算并打印滤波性能指标
# 提示: 计算滤波后信号与原始信号的 MSE 和相关系数
mse_iir = np.mean((filtered_iir - signal) ** 2)
mse_fir = np.mean((filtered_fir - signal) ** 2)
corr_iir = np.corrcoef(filtered_iir, signal)[0, 1]
corr_fir = np.corrcoef(filtered_fir, signal)[0, 1]

print(f"IIR 滤波 - MSE: {mse_iir:.6f}, 相关系数: {corr_iir:.4f}")
print(f"FIR 滤波 - MSE: {mse_fir:.6f}, 相关系数: {corr_fir:.4f}")

# 3. 绘制滤波前后频谱对比
freq_n, spec_n = analyze_spectrum(noisy_signal, fs, '含噪信号频谱')
freq_f, spec_f = analyze_spectrum(filtered_iir, fs, '滤波后信号频谱')
# --- TODO END ---
```

**预期输出**：
- 4 行子图清晰展示去噪效果
- MSE 和相关系数打印，相关系数应 > 0.95
- 滤波后频谱中 50Hz 峰值消失，低频分量被抑制

**常见坑**：
- ⚠️ `filtfilt` 会导致信号边缘失真（瞬态效应），对比时注意裁剪边缘
- ⚠️ 直接将含噪信号频谱与滤波后频谱在同一图中对比更直观

---

### 步骤 5：R 波峰值检测

**目标**：实现 R 波峰值检测算法，标记 ECG 中每个 QRS 波群的 R 峰。

```python
# ============================================
# Cell 7: 步骤5 - R 波峰值检测
# ============================================
from scipy.signal import find_peaks

# 使用滤波后的信号进行峰值检测
ecg_clean = filtered_iir.copy()

# --- TODO: your code here ---
# 1. 设计峰值检测算法
# 方法: 使用 scipy.signal.find_peaks
# 关键参数:
#   - height: 峰值最小高度 (建议: 信号均值 + 0.5*标准差)
#   - distance: 两个峰之间的最小采样点数 (心率<250BPM → 最小间距 > 360*60/250 ≈ 86 点)
#   - prominence: 峰值的最小突出度

# 计算自适应阈值
threshold = np.mean(ecg_clean) + 0.5 * np.std(ecg_clean)
min_distance = int(fs * 60 / 250)  # 心率不超过250BPM

# 执行峰值检测
r_peaks, properties = find_peaks(ecg_clean,
                                  height=threshold,
                                  distance=min_distance,
                                  prominence=0.3)

print(f"检测到 {len(r_peaks)} 个 R 波")
print(f"前10个 R 波位置: {r_peaks[:10]}")
# --- TODO END ---

# 可视化检测结果
fig, ax = plt.subplots(figsize=(14, 4))
t_show = t[:1800]
ax.plot(t_show, ecg_clean[:1800], 'b-', linewidth=0.8, label='滤波后 ECG')

# 标记检测到的 R 波（只显示前5秒内的）
mask = r_peaks < 1800
ax.plot(r_peaks[mask]/fs, ecg_clean[r_peaks[mask]], 'rv',
        markersize=10, label='检测到的 R 波')

# 与数据库标注对比
annot_mask = annotation.sample < 1800
ax.plot(annotation.sample[annot_mask]/fs,
        signal[annotation.sample[annot_mask]], 'g^',
        markersize=8, alpha=0.5, label='数据库标注')

ax.set_xlabel('时间 (s)')
ax.set_ylabel('幅值 (mV)')
ax.set_title('R 波峰值检测结果')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('r_peak_detection.png', dpi=150)
plt.show()
```

**预期输出**：
- ECG 波形上红色三角形标记检测到的 R 峰
- 绿色三角形为数据库标注，两者应基本重合
- 检测到的 R 波数量与数据库标注数量接近

**常见坑**：
- ⚠️ 阈值过低会检测到 T 波（假阳性），阈值过高会漏检（假阴性）
- ⚠️ `distance` 参数很关键，建议设为 `fs * 60 / 250`（对应最大心率 250BPM）
- ⚠️ 峰值检测在信号边界（前几秒）可能不稳定，可忽略边缘区域

---

### 步骤 6：心率计算与分析

**目标**：基于 R-R 间期计算心率，分析心率变异性。

```python
# ============================================
# Cell 8: 步骤6 - 心率计算与分析
# ============================================

# --- TODO: your code here ---
# 1. 计算 R-R 间期（单位：秒）
rr_intervals = np.diff(r_peaks) / fs  # 相邻 R 波的时间差

# 2. 计算瞬时心率 (BPM)
instant_hr = 60.0 / rr_intervals

# 3. 计算统计量
mean_hr = np.mean(instant_hr)
std_hr = np.std(instant_hr)
min_hr = np.min(instant_hr)
max_hr = np.max(instant_hr)

print("=" * 40)
print("心率分析结果")
print("=" * 40)
print(f"检测到 R 波总数: {len(r_peaks)}")
print(f"平均心率: {mean_hr:.1f} BPM")
print(f"心率标准差: {std_hr:.1f} BPM")
print(f"最小心率: {min_hr:.1f} BPM")
print(f"最大心率: {max_hr:.1f} BPM")
print(f"R-R 间期均值: {np.mean(rr_intervals)*1000:.1f} ms")

# 4. 检测异常心率
abnormal_low = np.where(instant_hr < 60)[0]
abnormal_high = np.where(instant_hr > 100)[0]
print(f"\n心动过缓 (<60 BPM): {len(abnormal_low)} 次")
print(f"心动过速 (>100 BPM): {len(abnormal_high)} 次")
# --- TODO END ---

# 可视化心率变化
fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# R-R 间期
axes[0].plot(rr_intervals * 1000, 'b.-', markersize=3)
axes[0].set_ylabel('R-R 间期 (ms)')
axes[0].set_title('R-R 间期序列')
axes[0].grid(True, alpha=0.3)

# 瞬时心率
axes[1].plot(instant_hr, 'r.-', markersize=3)
axes[1].axhline(60, color='gray', linestyle='--', alpha=0.5, label='60 BPM')
axes[1].axhline(100, color='gray', linestyle='--', alpha=0.5, label='100 BPM')
axes[1].set_ylabel('心率 (BPM)')
axes[1].set_title('瞬时心率')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# R-R 间期直方图
axes[2].hist(rr_intervals * 1000, bins=30, color='steelblue', edgecolor='black')
axes[2].set_xlabel('R-R 间期 (ms)')
axes[2].set_ylabel('频次')
axes[2].set_title('R-R 间期分布直方图')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heart_rate_analysis.png', dpi=150)
plt.show()
```

**预期输出**：
- 3 幅图：R-R 间期序列、瞬时心率曲线、R-R 间期直方图
- 控制台打印平均心率（正常成人静息约 60-100 BPM）
- 识别心动过缓/过速的异常段

**常见坑**：
- ⚠️ `np.diff` 返回的数组长度比 `r_peaks` 少 1
- ⚠️ 如果检测到的 R 波包含误检，心率会出现异常跳变，需检查
- ⚠️ 心率直方图应近似正态分布，若出现双峰说明有漏检/误检

---

## 五、评分细则（总分 100 分）

| 评分项 | 分值 | 评分标准 |
|--------|------|----------|
| **代码质量** | **30 分** | |
| 　代码可运行性 | 10 | 所有代码 cell 可直接运行，无报错（-2分/个报错） |
| 　代码规范 | 5 | 变量命名清晰、有适当注释、函数封装合理 |
| 　滤波器实现正确 | 10 | IIR 和 FIR 带通滤波器频率响应符合要求，滤波后信号干净 |
| 　峰值检测准确 | 5 | R 波检测准确率 > 90%（与数据库标注对比） |
| **报告质量** | **30 分** | |
| 　理论分析 | 10 | 清晰解释滤波器设计原理、截止频率选择依据 |
| 　结果展示 | 10 | 图表规范（有标题、轴标签、图例），对比分析充分 |
| 　结论与讨论 | 10 | 对比 IIR/FIR 优劣，讨论滤波器参数选择的影响 |
| **实验完整性** | **20 分** | |
| 　6 步全部完成 | 12 | 每步 2 分，缺失则扣分 |
| 　多种记录测试 | 4 | 至少测试 3 条不同的 MIT-BIH 记录（如 100, 101, 200） |
| 　参数对比实验 | 4 | 改变滤波器阶数/截止频率，分析对结果的影响 |
| **展示** | **10 分** | |
| 　PPT/演示 | 5 | 5 分钟口头展示，逻辑清晰 |
| 　回答问题 | 5 | 能回答滤波器设计和峰值检测相关技术问题 |
| **创新加分** | **10 分** | |
| 　自适应阈值 | 0-3 | 实现自适应阈值的 R 波检测算法（如 Pan-Tompkins 简化版） |
| 　滤波器对比 | 0-3 | 对比 3 种以上滤波器（如 Chebyshev、Elliptic）并分析 |
| 　心率变异性分析 | 0-2 | 计算 HRV 时域指标（SDNN、RMSSD） |
| 　其他创新 | 0-2 | 如实时滤波、GUI 交互、自选额外数据集验证 |

**评分等级**：
- A（90-100）：各部分优秀，有明显创新
- B（80-89）：各部分完整且质量较好
- C（70-79）：基本完成，有少量问题
- D（60-69）：完成度不足，存在明显错误
- F（<60）：大量缺失或代码不可运行

---

## 六、提交要求

### 提交内容
1. **Jupyter Notebook** (`学号_姓名_项目A.ipynb`)：包含所有代码和运行结果
2. **项目报告** (`学号_姓名_项目A_报告.pdf`)：4-8 页，包含：
   - 问题描述与信号处理方案设计
   - 滤波器设计过程与参数选择依据
   - 实验结果与对比分析（附图）
   - 结论与心得体会
3. **PPT** (`学号_姓名_项目A_展示.pptx`)：5 分钟展示用，10-15 页
4. **结果图片**：所有生成的 `.png` 文件

### 提交方式
- 将上述文件打包为 `学号_姓名_项目A.zip`
- 提交至课程学习通平台
- **截止日期**：第 12 周周日 23:59

### 中期检查（第 6 周）
- 提交阶段性 Notebook（至少完成步骤 1-3）
- 现场演示滤波器设计与频谱分析

---

## 七、常见问题 FAQ

**Q1：MIT-BIH 数据下载太慢怎么办？**
A：推荐使用 `wfdb.rdrecord` 在线读取，无需下载整个数据库。如果网络不好，可从课程群下载教师提供的本地数据副本。也可以先下载单条记录测试。

**Q2：滤波后信号出现明显失真/振铃怎么办？**
A：检查以下几点：(1) 截止频率是否设置合理；(2) 滤波器阶数是否过高（IIR 阶数 > 6 可能不稳定）；(3) 是否使用了 `filtfilt`（零相移）而非 `lfilter`；(4) 对于极低截止频率（0.5Hz），务必使用 SOS 形式。

**Q3：R 波检测率低于 90% 怎么改进？**
A：(1) 调整 `find_peaks` 的 `height`、`distance`、`prominence` 参数；(2) 先对信号做差分和平方增强 QRS 波群（简化版 Pan-Tompkins）；(3) 确认滤波后的信号质量是否足够好。

**Q4：必须同时实现 IIR 和 FIR 吗？**
A：是的，两者对比是评分要点之一。至少实现 Butterworth IIR 和窗函数 FIR 各一种，并从频率响应、滤波效果、计算效率三个维度进行对比。

**Q5：可以用 `scipy.signal` 以外的库吗？**
A：可以使用 `numpy`、`scipy`、`matplotlib`、`wfdb`。**禁止使用** `torch`、`tensorflow`、`keras` 等深度学习框架，禁止使用 `sklearn` 的机器学习模型。可用 `sklearn.metrics` 计算评价指标。

**Q6：报告中需要包含哪些公式？**
A：至少包含：(1) DFT/FFT 公式或原理说明；(2) Butterworth 滤波器传递函数；(3) R-R 间期到心率的换算公式 `HR = 60 / RRI`。鼓励包含滤波器设计步骤中的关键公式。

---

## 八、参考资源

### 知识点对应
| 课程知识点 | 本项目对应内容 |
|-----------|---------------|
| [[知识点/经典信号处理/采样与重建|采样定理]]（Nyquist） | ECG 采样率 360Hz，可分析最高 180Hz 频率分量 |
| [[知识点/经典信号处理/傅里叶分析|DFT 与 FFT]] | 步骤 1、4：ECG 频谱分析，噪声频带识别 |
| [[知识点/经典信号处理/数字滤波器设计|窗函数]] | 步骤 3：FIR 滤波器设计中的窗函数选择 |
| [[知识点/经典信号处理/数字滤波器设计|IIR 滤波器设计]] | 步骤 3：Butterworth 带通滤波器设计 |
| [[知识点/经典信号处理/数字滤波器设计|FIR 滤波器设计]] | 步骤 3：窗函数法设计 FIR 带通滤波器 |
| [[知识点/经典信号处理/信号与系统基础|线性卷积与滤波实现]] | 步骤 3-4：信号滤波的实现（filtfilt 内部） |

### 推荐阅读
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) - 数据集官方页面
- [WFDB Python 库文档](https://wfdb.readthedocs.io/) - 数据加载 API
- [SciPy Signal 处理文档](https://docs.scipy.org/doc/scipy/reference/signal.html) - 滤波器设计函数
- [Pan-Tompkins 算法论文](https://doi.org/10.1109/TBME.1985.325532) - 经典 R 波检测方法（拓展阅读）

### 参考代码
```python
# 常用函数速查
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

# FFT
freq = np.fft.rfftfreq(N, d=1/fs)
spectrum = np.fft.rfft(x)

# Butterworth 带通滤波器 (SOS 形式)
sos = sig.butter(order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
y = sig.sosfiltfilt(sos, x)

# FIR 带通滤波器
b = sig.firwin(num_taps, [f_low, f_high], pass_zero=False, fs=fs)
y = sig.filtfilt(b, [1.0], x)

# 峰值检测
peaks, props = sig.find_peaks(x, height=th, distance=d, prominence=p)
```

---

*本项目说明书由课程组编写，如有疑问请联系助教。*
*最后更新：2026 年 6 月*

---

> **课程关联**：本项目对应[[欢迎|智能信号处理课程]]W1-W5知识模块。涉及知识点：[[知识点/经典信号处理/信号与系统基础|信号与系统基础]]、[[知识点/经典信号处理/傅里叶分析|傅里叶分析]]、[[知识点/经典信号处理/数字滤波器设计|数字滤波器设计]]。关联实验：[[实验2_FFT频谱分析与滤波器设计|实验2]]。
