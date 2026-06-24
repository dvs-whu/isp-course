# 实验3：STFT与时频分析

## 一、实验信息

| 项目 | 内容 |
|------|------|
| 实验时长 | 2学时（90分钟） |
| 实验类型 | 综合探究型 |
| 适用课程 | 智能信号处理 |
| 面向对象 | 大二本科生 |

---

## 二、实验目的

1. **理解时频分析的动机与基本思想**：认识到纯时域或纯频域分析对非平稳信号的局限性，理解时频联合分析的必要性。
2. **掌握短时傅里叶变换（STFT）与频谱图**：能够计算STFT、绘制频谱图，并理解窗长、重叠率等参数对时频分辨率的影响。
3. **了解小波变换的基本原理**：通过代码实践，初步体验连续小波变换（CWT），对比STFT与小波变换在时频分析中的差异。

---

## 三、实验环境

| 软件/库 | 推荐版本 |
|---------|---------|
| Python | 3.9+ |
| NumPy | 1.24+ |
| SciPy | 1.10+ |
| Matplotlib | 3.7+ |
| librosa | 0.10+（可选，用于高级频谱图绘制） |

```bash
pip install numpy scipy matplotlib librosa
```

---

## 四、实验原理

### 4.1 为什么需要时频分析？

傅里叶变换告诉我们信号包含哪些频率成分，但丢失了**时间信息**——我们不知道某个频率在何时出现。对于频率随时间变化的**非平稳信号**（如鸟鸣、语音、雷达信号），需要同时获得时间和频率信息。

### 4.2 短时傅里叶变换（STFT）

STFT的核心思想：在信号上滑动一个短窗函数，在每个窗位置做FFT：

$$\text{STFT}\{x(t)\}(\tau, \omega) = \int_{-\infty}^{\infty} x(t) \, w(t - \tau) \, e^{-j\omega t} \, dt$$

其中 $w(t)$ 是窗函数（如汉宁窗），$\tau$ 是窗的位置。

**离散STFT**：

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \, w[n] \, e^{-j2\pi kn/N}$$

其中 $H$ 为帧移（hop size），$N$ 为窗长，$m$ 为帧索引，$k$ 为频率索引。

### 4.3 时频分辨率的权衡——不确定性原理

信号处理中的不确定性原理（类似量子力学）：

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

- **短窗**：时间分辨率好，频率分辨率差
- **长窗**：频率分辨率好，时间分辨率差

这是STFT的根本局限——窗长一旦选定，时频分辨率就固定了。

### 4.4 连续小波变换（CWT）

小波变换使用可伸缩的母小波函数 $\psi(t)$ 替代固定窗：

$$W_x(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \, \psi^*\left(\frac{t-b}{a}\right) dt$$

- $a$：尺度参数（对应频率，$a$ 小→高频，$a$ 大→低频）
- $b$：平移参数（对应时间）

小波变换的优势：**多分辨率分析**——高频部分时间分辨率好，低频部分频率分辨率好。这恰好符合许多自然信号的特点。

### 4.5 常用小波函数

| 小波名称 | 特点 |
|---------|------|
| Morlet | 频率分辨率好，适合频谱分析 |
| Daubechies (db) | 紧支撑，适合突变检测 |
| Mexican Hat | 对称，类似高斯二阶导数 |

---

## 五、实验步骤

### 步骤1：生成啁啾信号（Chirp Signal）

啁啾信号是频率随时间线性变化的信号，是验证时频分析方法的经典测试信号。

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import signal as sig

# 中文字体
font_prop = fm.FontProperties(fname='/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf')

# --- 生成啁啾信号 ---
fs = 1000           # 采样率
T = 2.0             # 时长
t = np.arange(0, T, 1/fs)
f0 = 10             # 起始频率
f1 = 200            # 终止频率

# 线性啁啾：频率从f0线性增加到f1
x_chirp = sig.chirp(t, f0, T, f1, method='linear')
x_chirp += 0.1 * np.random.randn(len(t))  # 加少量噪声

# --- 时域波形 ---
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(t, x_chirp, 'b-', linewidth=0.5)
axes[0].set_title('线性啁啾信号时域波形', fontproperties=font_prop)
axes[0].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[0].set_ylabel('幅值', fontproperties=font_prop)
axes[0].grid(True, alpha=0.3)

# 整体FFT（应看到宽带频谱，无法反映频率变化）
X_mag = 2.0/len(t) * np.abs(np.fft.rfft(x_chirp))
f_axis = np.fft.rfftfreq(len(t), 1/fs)
axes[1].plot(f_axis, X_mag, 'r-', linewidth=0.8)
axes[1].set_title('啁啾信号FFT频谱（时间信息丢失！）', fontproperties=font_prop)
axes[1].set_xlabel('频率 (Hz)', fontproperties=font_prop)
axes[1].set_ylabel('幅值', fontproperties=font_prop)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step1_chirp_signal.png', dpi=150)
plt.show()

print(f"信号长度: {len(t)} 点, 时长: {T} s")
print(f"频率范围: {f0} Hz → {f1} Hz")
print("注意：FFT频谱显示了频率范围，但无法看到频率随时间的变化！")
```

### 步骤2：计算STFT并绘制频谱图

```python
# --- STFT计算 ---
nperseg = 128       # 窗长（每帧128个采样点）
noverlap = 96       # 帧重叠（96个点，重叠率75%）
nfft = 256          # FFT点数

f_stft, t_stft, Zxx = sig.stft(x_chirp, fs=fs, window='hann',
                                 nperseg=nperseg, noverlap=noverlap,
                                 nfft=nfft)

# 幅度谱（dB）
magnitude = np.abs(Zxx)
magnitude_db = 20 * np.log10(magnitude + 1e-10)

# --- 绘制频谱图 ---
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.pcolormesh(t_stft, f_stft, magnitude_db, shading='gouraud',
                   cmap='inferno', vmin=-40, vmax=0)
ax.set_title(f'STF频谱图 (窗长={nperseg}, 重叠={noverlap})', fontproperties=font_prop)
ax.set_xlabel('时间 (s)', fontproperties=font_prop)
ax.set_ylabel('频率 (Hz)', fontproperties=font_prop)
ax.set_ylim(0, 300)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('幅值 (dB)', fontproperties=font_prop)
plt.tight_layout()
plt.savefig('step2_spectrogram.png', dpi=150)
plt.show()

print(f"STFT输出形状: {Zxx.shape}")
print(f"时间帧数: {len(t_stft)}, 频率点数: {len(f_stft)}")
```

**预期输出**：频谱图中应看到一条从低频（左下）到高频（右上）的亮带，直观反映了频率随时间的变化。

### 步骤3：窗长对时频分辨率的影响

```python
# --- 不同窗长对比 ---
window_lengths = [32, 128, 512]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, nperseg in zip(axes, window_lengths):
    noverlap = int(nperseg * 0.75)
    nfft_val = max(nperseg, 256)

    f_s, t_s, Zxx = sig.stft(x_chirp, fs=fs, window='hann',
                               nperseg=nperseg, noverlap=noverlap,
                               nfft=nfft_val)

    mag_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    im = ax.pcolormesh(t_s, f_s, mag_db, shading='gouraud',
                       cmap='inferno', vmin=-40, vmax=0)
    ax.set_title(f'窗长 = {nperseg} 点\n(Δt={nperseg/fs*1000:.1f}ms, Δf≈{fs/nperseg:.1f}Hz)',
                 fontproperties=font_prop)
    ax.set_xlabel('时间 (s)', fontproperties=font_prop)
    ax.set_ylabel('频率 (Hz)', fontproperties=font_prop)
    ax.set_ylim(0, 300)

plt.suptitle('窗长对STFT时频分辨率的影响', fontproperties=font_prop, fontsize=14)
plt.tight_layout()
plt.savefig('step3_window_comparison.png', dpi=150)
plt.show()

# 打印理论分辨率
print("窗长对时频分辨率的影响：")
print(f"{'窗长':>6} | {'时间分辨率':>10} | {'频率分辨率':>10}")
print("-" * 35)
for nperseg in window_lengths:
    dt = nperseg / fs * 1000
    df = fs / nperseg
    print(f"{nperseg:>6} | {dt:>8.1f} ms | {df:>8.1f} Hz")
```

**分析要点**：
- 窗长=32：时间分辨率好（能精确定位频率变化时刻），但频率分辨率差（频带模糊）
- 窗长=512：频率分辨率好（频带清晰），但时间分辨率差（频率变化位置模糊）
- 窗长=128：折中方案

### 步骤4：复杂时频信号的STFT分析

```python
# --- 生成多分量信号 ---
fs = 1000
T = 3.0
t = np.arange(0, T, 1/fs)

# 分量1：0~1s 内 100Hz
comp1 = np.zeros_like(t)
mask1 = t < 1.0
comp1[mask1] = np.sin(2*np.pi*100*t[mask1])

# 分量2：1~2s 内 250Hz
comp2 = np.zeros_like(t)
mask2 = (t >= 1.0) & (t < 2.0)
comp2[mask2] = np.sin(2*np.pi*250*t[mask2])

# 分量3：2~3s 内 频率从150Hz线性变化到350Hz
comp3 = np.zeros_like(t)
mask3 = t >= 2.0
t_seg3 = t[mask3] - 2.0
comp3[mask3] = sig.chirp(t_seg3, 150, 1.0, 350, method='linear')

# 分量4：全程 50Hz 低频背景
comp4 = 0.5 * np.sin(2*np.pi*50*t)

x_complex = comp1 + comp2 + comp3 + comp4 + 0.1*np.random.randn(len(t))

# --- STFT ---
nperseg = 256
noverlap = 224
f_c, t_c, Zxx_c = sig.stft(x_complex, fs=fs, window='hann',
                             nperseg=nperseg, noverlap=noverlap, nfft=512)

# --- 绘图 ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 时域波形
axes[0].plot(t, x_complex, 'b-', linewidth=0.5)
axes[0].set_title('多分量时变信号波形', fontproperties=font_prop)
axes[0].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[0].set_ylabel('幅值', fontproperties=font_prop)
axes[0].grid(True, alpha=0.3)

# 标注各分量的时间区间
for start, end, freq, label in [(0, 1, 100, '100Hz'), (1, 2, 250, '250Hz')]:
    axes[0].annotate(label, xy=((start+end)/2, 1.2), fontsize=11,
                     fontproperties=font_prop, ha='center', color='red')
axes[0].annotate('150→350Hz chirp', xy=(2.5, 1.2), fontsize=11,
                 fontproperties=font_prop, ha='center', color='red')
axes[0].annotate('50Hz背景', xy=(1.5, -1.2), fontsize=11,
                 fontproperties=font_prop, ha='center', color='blue')

# 频谱图
mag_db = 20 * np.log10(np.abs(Zxx_c) + 1e-10)
im = axes[1].pcolormesh(t_c, f_c, mag_db, shading='gouraud',
                         cmap='magma', vmin=-30, vmax=10)
axes[1].set_title('多分量信号STFT频谱图', fontproperties=font_prop)
axes[1].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[1].set_ylabel('频率 (Hz)', fontproperties=font_prop)
axes[1].set_ylim(0, 400)
cbar = fig.colorbar(im, ax=axes[1])
cbar.set_label('幅值 (dB)', fontproperties=font_prop)

plt.tight_layout()
plt.savefig('step4_complex_spectrogram.png', dpi=150)
plt.show()
```

**预期输出**：频谱图中应清晰看到：
- 0~1s区间有100Hz水平亮带
- 1~2s区间有250Hz水平亮带
- 2~3s区间有从150Hz到350Hz的斜线
- 全程底部有50Hz水平亮带

### 步骤5：连续小波变换（CWT）

```python
from scipy.signal import cwt, morlet2

# --- 连续小波变换 ---
# 使用Morlet小波
widths = np.arange(1, 128)  # 尺度参数（对应不同频率）

# 对啁啾信号做CWT（使用步骤1的信号）
# 重新生成啁啾信号
T_cwt = 2.0
t_cwt = np.arange(0, T_cwt, 1/fs)
x_cwt = sig.chirp(t_cwt, 10, T_cwt, 200, method='linear')

# 使用morlet2小波
cwtmatr = cwt(x_cwt, morlet2, widths, dtype=complex)

# 频率轴近似映射
# 对于Morlet小波，频率 ≈ fs / (2π * width) * ω₀
# 这里简化为 fs / width
freqs_cwt = fs / (2 * widths)  # 近似频率

# --- 绘制CWT时频图 ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# CWT结果
cwt_mag = np.abs(cwtmatr)
im = axes[0].pcolormesh(t_cwt, freqs_cwt, cwt_mag, shading='auto',
                         cmap='jet')
axes[0].set_title('连续小波变换（CWT）时频图 — Morlet小波', fontproperties=font_prop)
axes[0].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[0].set_ylabel('近似频率 (Hz)', fontproperties=font_prop)
axes[0].set_ylim(0, 300)
cbar = fig.colorbar(im, ax=axes[0])
cbar.set_label('小波系数幅值', fontproperties=font_prop)

# 对比：STFT
f_s2, t_s2, Zxx_s2 = sig.stft(x_cwt, fs=fs, window='hann',
                                 nperseg=128, noverlap=96, nfft=256)
mag_db2 = 20 * np.log10(np.abs(Zxx_s2) + 1e-10)
im2 = axes[1].pcolormesh(t_s2, f_s2, mag_db2, shading='gouraud',
                          cmap='inferno', vmin=-40, vmax=0)
axes[1].set_title('STFT频谱图（窗长=128）', fontproperties=font_prop)
axes[1].set_xlabel('时间 (s)', fontproperties=font_prop)
axes[1].set_ylabel('频率 (Hz)', fontproperties=font_prop)
axes[1].set_ylim(0, 300)
cbar2 = fig.colorbar(im2, ax=axes[1])
cbar2.set_label('幅值 (dB)', fontproperties=font_prop)

plt.tight_layout()
plt.savefig('step5_cwt_vs_stft.png', dpi=150)
plt.show()
```

### 步骤6（可选）：使用librosa绘制高级频谱图

```python
try:
    import librosa
    import librosa.display

    # --- 使用librosa计算并绘制频谱图 ---
    # 生成一段模拟语音信号
    fs_lr = 22050
    T_lr = 2.0
    t_lr = np.arange(0, T_lr, 1/fs_lr)

    # 模拟语音：基频随时间变化 + 谐波
    f0_voiced = 150 + 50 * np.sin(2*np.pi*0.5*t_lr)  # 基频抖动
    phase = np.cumsum(2*np.pi*f0_voiced/fs_lr)
    audio = 0.5*np.sin(phase) + \
            0.3*np.sin(2*phase) + \
            0.2*np.sin(3*phase) + \
            0.05*np.random.randn(len(t_lr))

    # librosa STFT
    D = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=1024)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 波形
    librosa.display.waveshow(audio, sr=fs_lr, ax=axes[0])
    axes[0].set_title('模拟语音波形', fontproperties=font_prop)
    axes[0].set_xlabel('时间 (s)', fontproperties=font_prop)
    axes[0].set_ylabel('幅值', fontproperties=font_prop)

    # 频谱图
    img = librosa.display.specshow(S_db, sr=fs_lr, hop_length=512,
                                     x_axis='time', y_axis='hz', ax=axes[1],
                                     cmap='magma')
    axes[1].set_ylim(0, 2000)
    axes[1].set_title('librosa频谱图', fontproperties=font_prop)
    axes[1].set_xlabel('时间 (s)', fontproperties=font_prop)
    axes[1].set_ylabel('频率 (Hz)', fontproperties=font_prop)
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig('step6_librosa_spectrogram.png', dpi=150)
    plt.show()

    print("librosa频谱图绘制成功！")

except ImportError:
    print("librosa 未安装，跳过此步骤。")
    print("安装方法: pip install librosa")
```

---

## 六、实验报告要求

1. **图表汇总**：将步骤1~6生成的所有时频图整理到报告中，并标注关键特征。
2. **窗长影响分析**：
   - 用表格总结步骤3中不同窗长的时间分辨率和频率分辨率。
   - 结合不确定性原理，解释为什么不存在"完美"的窗长选择。
3. **回答问题**：
   - STFT频谱图中，频率轴的分辨率由什么决定？时间轴的分辨率由什么决定？
   - 在步骤4的多分量信号中，如果两个事件在时间上很接近但频率差异很大，应选择长窗还是短窗？如果频率差异很小但时间很接近呢？
   - CWT和STFT在时频分辨率上的行为有何本质区别？各自的适用场景是什么？

---

## 七、思考题

1. **Gabor变换与最优窗**：Gabor变换使用高斯窗，它在时频平面上达到不确定性原理的下界。用 `scipy.signal.windows.gaussian` 替代汉宁窗，重新计算步骤2的STFT。高斯窗的频谱图与汉宁窗有何区别？

2. **多分辨率分析的直觉**：小波变换在高频处时间分辨率好、在低频处频率分辨率好。试从音乐信号的角度解释这一特性为何是有利的（提示：考虑钢琴键盘的频率分布——高音区相邻键的频率差远大于低音区）。

3. **时频分布的改进**：STFT和CWT都是**线性**时频表示。Wigner-Ville分布是一种**二次**时频表示，具有更高的时频分辨率，但会产生交叉项干扰。查阅相关资料，简述Wigner-Ville分布的定义及其交叉项问题。

---

> **实验提示**：时频分析是信号处理的核心工具之一。本实验重点在于建立直觉——通过观察频谱图理解窗长对分辨率的影响，以及对比STFT和小波变换的特点。建议反复调整参数运行代码，加深理解。


---

> **课程关联**：本实验安排在[[欢迎|智能信号处理课程]]第9周。涉及知识点：短时傅里叶变换、小波变换、时频分布。关联项目：项目一C,D。
> 详见[[13周教学进度表]] | [[课程关系图谱]]