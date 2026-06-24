# 迷你项目D：语音信号分析与说话人特征提取

> **课程**：智能信号处理 · 武汉大学人工智能学院  
> **适用年级**：大二  
> **项目类型**：个人项目  
> **工具限制**：仅使用经典信号处理方法（STFT、小波变换、EMD、MFCC），**禁止使用深度学习**

---

## 1. 项目概述

语音信号是最典型的一维非平稳信号之一。本项目要求学生采集或获取语音数据，运用短时傅里叶变换（STFT）、连续小波变换（CWT）、经验模态分解（EMD）等时频分析方法对语音信号进行多角度分析，并手动实现梅尔频率倒谱系数（MFCC）提取流程，最终基于 MFCC 特征完成说话人比较任务。

通过本项目，你将：

- 深入理解语音信号的非平稳特性与多尺度结构
- 掌握 STFT、CWT、EMD 三种时频分析方法的原理与差异
- 从零实现 Mel 滤波器组与 MFCC 提取流程
- 利用 MFCC 特征进行说话人相似度度量

### 时间线

| 阶段 | 时间 | 任务 |
|------|------|------|
| 启动 | 第2周 | 阅读说明，采集/下载数据，完成 Step 1 |
| 中期检查 | 第6周 | 完成 Step 1–3，提交中期报告 |
| 最终提交 | 第12周 | 完成 Step 4–6，提交完整报告与代码 |

---

## 2. 学习目标

完成本项目后，你应该能够：

1. **STFT**：理解窗函数对时频分辨率的影响，能解释 Heisenberg 不确定性原理在语音分析中的体现
2. **小波变换**：掌握连续小波变换的多分辨率分析能力，对比 STFT 与 CWT 的优劣
3. **EMD**：理解经验模态分解的过程，能将语音信号分解为若干本征模态函数（IMF）并解释其物理意义
4. **MFCC**：掌握 Mel 频率标度的心理声学依据，能手动实现 Mel 滤波器组、对数能量、DCT 等关键步骤
5. **说话人比较**：理解基于 MFCC 的说话人特征表示，能用欧氏距离和余弦相似度度量说话人差异

---

## 3. 数据准备

### 方案 A：自行录音（推荐）

使用手机或电脑录音软件，录制以下内容：

| 编号 | 内容 | 时长 | 说明 |
|------|------|------|------|
| `speaker_A_1.wav` | 自己朗读一段中文（如"武汉大学人工智能学院"） | 3–5 秒 | 男声/女声均可 |
| `speaker_A_2.wav` | 自己朗读另一段中文 | 3–5 秒 | 同一说话人，不同内容 |
| `speaker_B_1.wav` | 请同学/朋友朗读相同文本 | 3–5 秒 | 不同说话人 |
| `speaker_B_2.wav` | 同一同学朗读另一段中文 | 3–5 秒 | 同一说话人，不同内容 |
| `noise_test.wav` | 在嘈杂环境中录制任意语音 | 3–5 秒 | 用于噪声分析（选做） |

**录音要求**：
- 采样率：16 kHz 或 44.1 kHz（统一即可）
- 格式：WAV（无压缩）
- 环境：尽量安静，避免背景噪声
- 设备：手机录音 App 或电脑麦克风均可

### 方案 B：LibriSpeech 子集下载

如无法自行录音，可使用 LibriSpeech 公开数据集：

```python
# 方式1：使用 librosa 加载示例音频（内置）
import librosa
y, sr = librosa.load(librosa.ex('trumpet'), sr=16000)  # 仅作测试

# 方式2：从 LibriSpeech 下载（需网络）
# 网址：https://www.openslr.org/12
# 下载 test-clean 子集，选取 2 位说话人各 2 段音频
# 文件命名同方案A
```

> **注意**：无论使用哪种方案，请在报告中注明数据来源。

### 文件组织

```
projectD/
├── data/
│   ├── speaker_A_1.wav
│   ├── speaker_A_2.wav
│   ├── speaker_B_1.wav
│   └── speaker_B_2.wav
├── src/
│   ├── step1_waveform.py
│   ├── step2_stft.py
│   ├── step3_cwt.py
│   ├── step4_emd.py
│   ├── step5_mfcc.py
│   └── step6_speaker_compare.py
├── figures/          # 保存所有实验图片
└── report.md         # 实验报告
```

### 依赖安装

```bash
pip install numpy scipy matplotlib librosa PyEMD
```

> `PyEMD` 用于 EMD 分解（包名 `EMD-signal`，`pip install EMD-signal`）。

---

## 4. 实验步骤

### Step 1：加载语音信号并可视化波形

**目标**：学会加载 WAV 文件，观察语音波形的时域特征。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ===== 读取语音文件 =====
sr, signal = wavfile.read('data/speaker_A_1.wav')

# 若为双声道，取单声道
if signal.ndim > 1:
    signal = signal[:, 0]

# 归一化到 [-1, 1]
signal = signal / np.max(np.abs(signal))

# 时间轴
t = np.arange(len(signal)) / sr

# ===== 绘制波形 =====
plt.figure(figsize=(12, 4))
plt.plot(t, signal, linewidth=0.5)
plt.xlabel('时间 (s)')
plt.ylabel('幅值')
plt.title('语音信号波形')
plt.tight_layout()
plt.savefig('figures/step1_waveform.png', dpi=150)
plt.show()

# ===== TODO 1: 标注有声段和无声段 =====
# 提示：计算短时能量，设定阈值，用不同颜色标注
frame_len = int(0.025 * sr)  # 25ms 帧长
hop_len = int(0.010 * sr)    # 10ms 帧移

# TODO: 计算短时能量
# short_time_energy = ...
# for i in range(0, len(signal) - frame_len, hop_len):
#     frame = signal[i:i+frame_len]
#     short_time_energy.append(np.sum(frame**2))

# TODO: 绘制波形并标注有声段（用红色背景高亮）
# plt.figure(figsize=(12, 4))
# plt.plot(t, signal, linewidth=0.5)
# # 用 fill_between 或 axvspan 标注有声段
# ...

# ===== TODO 2: 对比不同说话人的波形差异 =====
# 加载 speaker_B_1.wav，绘制在同一图中对比
# 讨论：不同说话人的波形有何直观差异？（振幅、持续时间、基频等）
```

**思考题**：
1. 语音波形中哪些区域对应有声段（voiced），哪些对应无声段（unvoiced）？
2. 观察波形能否区分不同说话人？为什么？

---

### Step 2：STFT 时频分析与窗长选择

**目标**：理解 STFT 的原理，探究窗函数长度对时频分辨率的影响。

```python
from scipy.signal import stft
import librosa
import librosa.display

# 加载音频（使用 librosa 方便处理）
y, sr = librosa.load('data/speaker_A_1.wav', sr=16000)

# ===== 基础 STFT =====
def compute_stft(signal, sr, n_fft, hop_length, win_length):
    """
    计算 STFT 并返回幅度谱（dB）
    """
    f, t, Zxx = stft(signal, fs=sr, nperseg=win_length,
                     noverlap=win_length - hop_length, nfft=n_fft)
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    return f, t, magnitude_db

# ===== 不同窗长的 STFT 对比 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

window_configs = [
    {'n_fft': 256,  'win_length': 256,  'hop_length': 64,  'title': '短窗 (256点, 16ms)'},
    {'n_fft': 512,  'win_length': 512,  'hop_length': 128, 'title': '中窗 (512点, 32ms)'},
    {'n_fft': 1024, 'win_length': 1024, 'hop_length': 256, 'title': '长窗 (1024点, 64ms)'},
    {'n_fft': 2048, 'win_length': 2048, 'hop_length': 512, 'title': '超长窗 (2048点, 128ms)'},
]

for ax, cfg in zip(axes.flat, window_configs):
    f, t, mag_db = compute_stft(y, sr, cfg['n_fft'],
                                 cfg['hop_length'], cfg['win_length'])
    # TODO: 绘制 spectrogram
    # ax.pcolormesh(t, f, mag_db, shading='gouraud', cmap='magma')
    # ax.set_title(cfg['title'])
    # ax.set_ylabel('频率 (Hz)')
    # ax.set_xlabel('时间 (s)')
    # ax.set_ylim([0, 4000])

plt.suptitle('不同窗长的 STFT 时频图对比', fontsize=14)
plt.tight_layout()
plt.savefig('figures/step2_stft_comparison.png', dpi=150)
plt.show()

# ===== TODO 3: 分析时间分辨率与频率分辨率的权衡 =====
# 问题：窗长增加时，时间和频率分辨率如何变化？
# 请在代码注释或报告中回答：
# - 短窗：时间分辨率___，频率分辨率___
# - 长窗：时间分辨率___，频率分辨率___
# - 原因（Heisenberg不确定性原理）：___

# ===== TODO 4: 尝试不同窗函数 =====
# 比较汉明窗（Hamming）、汉宁窗（Hanning）、矩形窗的效果
# from scipy.signal import get_window
# windows = ['hamming', 'hann', 'boxcar']
# for win_name in windows:
#     win = get_window(win_name, 512)
#     # 重新计算 STFT 并绘制
#     ...
```

**思考题**：
1. 用 1-2 句话解释 Heisenberg 不确定性原理在 STFT 中的体现。
2. 对于语音分析，你认为哪个窗长最合适？为什么？
3. 语音中的辅音（如 /s/, /t/）和元音（如 /a/, /o/）对时频分辨率的需求有何不同？

---

### Step 3：连续小波变换（CWT）分析

**目标**：掌握 CWT 的多分辨率特性，对比 STFT 与 CWT 的分析效果。

```python
import pywt

# 加载音频
y, sr = librosa.load('data/speaker_A_1.wav', sr=16000)

# ===== CWT 分析 =====
# 选择小波基函数
wavelet_name = 'cmor1.5-1.0'  # 复 Morlet 小波（也可尝试 'morl'）

# 定义尺度（对应不同频率）
# 频率与尺度的关系：f = fc / (scale * dt)，fc 为小波中心频率
# 语音频率范围：约 80 Hz ~ 4000 Hz
scales = np.arange(1, 128)

# ===== TODO 5: 执行 CWT =====
# coefficients, frequencies = pywt.cwt(y, scales, wavelet_name, sampling_period=1.0/sr)
#
# # 绘制 CWT 时频图
# plt.figure(figsize=(12, 6))
# plt.pcolormesh(np.arange(len(y))/sr, frequencies,
#                np.abs(coefficients), shading='gouraud', cmap='jet')
# plt.colorbar(label='幅度')
# plt.ylabel('频率 (Hz)')
# plt.xlabel('时间 (s)')
# plt.title('连续小波变换 (CWT) 时频图')
# plt.ylim([0, 4000])
# plt.tight_layout()
# plt.savefig('figures/step3_cwt.png', dpi=150)
# plt.show()

# ===== TODO 6: CWT 与 STFT 对比 =====
# 绘制 STFT（使用 Step 2 中你认为最合适的窗长）和 CWT 的对比图
# 在报告中分析：
# 1. CWT 在低频和高频区域的分辨率表现
# 2. CWT 的"多分辨率"特性如何体现？
# 3. STFT 和 CWT 各自的优势和局限性

# ===== TODO 7: 尝试不同小波基 =====
# 比较 'morl'（实Morlet）、'cmor1.5-1.0'（复Morlet）、'gaus1' 等小波基
# 讨论不同小波基对分析结果的影响
```

**思考题**：
1. CWT 在低频段和高频段的时间-频率分辨率如何变化？这与 STFT 有何本质不同？
2. 语音信号分析中，CWT 相比 STFT 的主要优势是什么？
3. CWT 的计算量通常大于 STFT，这在实际应用中如何权衡？

---

### Step 4：经验模态分解（EMD）

**目标**：将语音信号分解为本征模态函数（IMF），理解信号的多尺度结构。

```python
# pip install EMD-signal
from PyEMD import EMD

# 加载音频
y, sr = librosa.load('data/speaker_A_1.wav', sr=16000)

# ===== EMD 分解 =====
emd = EMD()
IMFs = emd(y)
n_imfs = IMFs.shape[0]

print(f"共分解出 {n_imfs} 个 IMF")

# ===== TODO 8: 绘制 IMF 分解结果 =====
# t = np.arange(len(y)) / sr
# fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(12, 2.5 * (n_imfs + 1)))
# axes[0].plot(t, y, linewidth=0.5)
# axes[0].set_title('原始信号')
# for i in range(n_imfs):
#     axes[i+1].plot(t, IMFs[i], linewidth=0.5)
#     axes[i+1].set_title(f'IMF {i+1}')
# plt.tight_layout()
# plt.savefig('figures/step4_emd_imfs.png', dpi=150)
# plt.show()

# ===== TODO 9: 分析各 IMF 的频率特性 =====
# 对每个 IMF 计算 FFT，绘制频谱
# from scipy.fft import fft, fftfreq
# for i in range(min(n_imfs, 5)):  # 只看前5个IMF
#     N = len(IMFs[i])
#     yf = np.abs(fft(IMFs[i]))[:N//2]
#     xf = fftfreq(N, 1/sr)[:N//2]
#     # 绘制频谱
#     ...

# ===== TODO 10: IMF 重构实验 =====
# 尝试用不同 IMF 子集重构信号：
# (a) 仅用前3个 IMF（高频成分）
# (b) 仅用后3个 IMF（低频成分）
# (c) 去除最高频 IMF（去噪效果）
# 对比听觉效果（使用 soundfile 或 IPython.display.Audio 播放）

# ===== TODO 11: EMD 与 STFT/CWT 的对比 =====
# 在报告中讨论：
# 1. EMD 是自适应分解，不需要预设基函数，这有什么优缺点？
# 2. IMF 的物理意义是什么？每个 IMF 大致对应语音中的什么成分？
# 3. EMD 存在"模态混叠"问题，你在实验中观察到了吗？
```

**思考题**：
1. EMD 分解出的 IMF 数量受什么因素影响？
2. 前几个 IMF 和后几个 IMF 分别对应语音信号中的什么物理成分？
3. EMD 与小波分解的本质区别是什么？

---

### Step 5：手动实现 MFCC 提取

**目标**：理解 MFCC 的完整提取流程，从零实现 Mel 滤波器组。

```python
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile

# 加载音频
sr, signal = wavfile.read('data/speaker_A_1.wav')
if signal.ndim > 1:
    signal = signal[:, 0]
signal = signal / np.max(np.abs(signal))

# ===== 参数设置 =====
N_FFT = 512          # FFT 点数
N_MELS = 26          # Mel 滤波器数量
N_MFCC = 13          # 保留的 MFCC 系数数量
FRAME_LEN = 0.025    # 帧长 25ms
FRAME_SHIFT = 0.010  # 帧移 10ms
PRE_EMPHASIS = 0.97  # 预加重系数

# ===== Step 5.1: 预加重 =====
signal_emphasized = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])

# ===== Step 5.2: 分帧加窗 =====
frame_len_samples = int(FRAME_LEN * sr)
frame_shift_samples = int(FRAME_SHIFT * sr)

def framing(signal, frame_len, frame_shift):
    """将信号分帧"""
    n_frames = 1 + (len(signal) - frame_len) // frame_shift
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        start = i * frame_shift
        frames[i] = signal[start:start + frame_len]
    return frames

frames = framing(signal_emphasized, frame_len_samples, frame_shift_samples)

# 加汉明窗
window = np.hamming(frame_len_samples)
frames = frames * window

# ===== Step 5.3: FFT 与功率谱 =====
mag_frames = np.abs(fft(frames, N_FFT))[:, :N_FFT // 2 + 1]
pow_frames = (mag_frames ** 2) / N_FFT

# ===== TODO 12: 实现 Mel 滤波器组 =====
def hz_to_mel(hz):
    """Hz 转 Mel 频率"""
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    """Mel 频率转 Hz"""
    return 700 * (10 ** (mel / 2595.0) - 1)

def create_mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    """
    创建 Mel 滤波器组

    参数：
        n_mels: Mel 滤波器数量
        n_fft: FFT 点数
        sr: 采样率
        fmin: 最低频率 (Hz)
        fmax: 最高频率 (Hz)，默认为 sr/2

    返回：
        filterbank: shape (n_mels, n_fft//2 + 1) 的滤波器组矩阵
    """
    if fmax is None:
        fmax = sr / 2

    # TODO: 实现 Mel 滤波器组
    # 步骤 1: 在 Mel 尺度上均匀分布 n_mels+2 个点
    # mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    # hz_points = mel_to_hz(mel_points)
    #
    # 步骤 2: 将 Hz 点映射到 FFT bin 索引
    # bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    #
    # 步骤 3: 构建三角滤波器
    # filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    # for m in range(n_mels):
    #     f_left = bin_points[m]
    #     f_center = bin_points[m + 1]
    #     f_right = bin_points[m + 2]
    #     # 上升斜坡
    #     for k in range(f_left, f_center):
    #         if f_center != f_left:
    #             filterbank[m, k] = (k - f_left) / (f_center - f_left)
    #     # 下降斜坡
    #     for k in range(f_center, f_right):
    #         if f_right != f_center:
    #             filterbank[m, k] = (f_right - k) / (f_right - f_center)
    #
    # return filterbank

    # 临时占位，实现后删除
    return np.zeros((n_mels, n_fft // 2 + 1))

# ===== TODO 13: 绘制 Mel 滤波器组 =====
# filterbank = create_mel_filterbank(N_MELS, N_FFT, sr)
# plt.figure(figsize=(10, 6))
# freq_axis = np.linspace(0, sr/2, N_FFT//2 + 1)
# for i in range(N_MELS):
#     plt.plot(freq_axis, filterbank[i])
# plt.xlabel('频率 (Hz)')
# plt.ylabel('权重')
# plt.title('Mel 滤波器组')
# plt.tight_layout()
# plt.savefig('figures/step5_mel_filterbank.png', dpi=150)
# plt.show()

# ===== TODO 14: 提取 MFCC =====
def extract_mfcc(pow_frames, filterbank, n_mfcc):
    """
    从功率谱帧中提取 MFCC

    步骤：
    1. 将功率谱通过 Mel 滤波器组
    2. 取对数
    3. 做 DCT（离散余弦变换）
    """
    # TODO: 实现 MFCC 提取
    # Step 1: Mel 滤波
    # mel_energies = np.dot(pow_frames, filterbank.T)
    #
    # Step 2: 取对数
    # log_mel_energies = np.log(mel_energies + 1e-10)
    #
    # Step 3: DCT（使用 scipy.fft 或手动实现 Type-II DCT）
    # from scipy.fft import dct
    # mfcc = dct(log_mel_energies, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    #
    # return mfcc

    # 临时占位
    return np.zeros((pow_frames.shape[0], n_mfcc))

# ===== TODO 15: 可视化 MFCC =====
# mfcc = extract_mfcc(pow_frames, filterbank, N_MFCC)
# plt.figure(figsize=(12, 6))
# plt.imshow(mfcc.T, aspect='auto', origin='lower', cmap='RdBu_r')
# plt.colorbar(label='MFCC 系数值')
# plt.xlabel('帧索引')
# plt.ylabel('MFCC 系数')
# plt.title('MFCC 特征矩阵')
# plt.tight_layout()
# plt.savefig('figures/step5_mfcc.png', dpi=150)
# plt.show()

# ===== 验证：与 librosa 对比 =====
# mfcc_librosa = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
#                                      n_fft=N_FFT, hop_length=int(FRAME_SHIFT*sr),
#                                      n_mels=N_MELS)
# 请在报告中对比你的实现与 librosa 的结果，分析差异原因
```

**思考题**：
1. 为什么 Mel 频率标度比线性频率更适合语音分析？这与人耳的什么特性相关？
2. 预加重的作用是什么？为什么需要这一步？
3. 为什么取对数？这对后续的说话人识别有什么帮助？
4. DCT 的作用是什么？为什么保留前 13 个系数就够了？

---

### Step 6：基于 MFCC 的说话人比较

**目标**：利用提取的 MFCC 特征，通过距离度量比较不同说话人。

```python
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import librosa

# ===== 加载所有音频并提取 MFCC =====
files = {
    'A_1': 'data/speaker_A_1.wav',
    'A_2': 'data/speaker_A_2.wav',
    'B_1': 'data/speaker_B_1.wav',
    'B_2': 'data/speaker_B_2.wav',
}

def extract_mfcc_librosa(filepath, sr=16000, n_mfcc=13):
    """使用 librosa 提取 MFCC（用于验证）"""
    y, sr = librosa.load(filepath, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# 提取所有文件的 MFCC
mfccs = {}
for name, path in files.items():
    mfccs[name] = extract_mfcc_librosa(path)
    print(f"{name}: MFCC shape = {mfccs[name].shape}")

# ===== TODO 16: 设计说话人特征表示 =====
def speaker_feature(mfcc):
    """
    将一帧序列的 MFCC 转换为说话人级别的特征向量

    常用方法：
    - 均值向量：对时间轴取平均
    - 均值 + 标准差拼接
    - GMM 超向量（本项目不要求）

    TODO: 实现至少两种方法并比较
    """
    # 方法1：均值向量
    # mean_feat = np.mean(mfcc, axis=1)
    # return mean_feat

    # 方法2：均值 + 标准差
    # mean_feat = np.mean(mfcc, axis=1)
    # std_feat = np.std(mfcc, axis=1)
    # return np.concatenate([mean_feat, std_feat])

    # 临时占位
    return np.mean(mfcc, axis=1)

# ===== TODO 17: 计算相似度矩阵 =====
def compute_similarity_matrix(features_dict, metric='euclidean'):
    """
    计算所有说话人之间的相似度/距离矩阵

    参数：
        features_dict: {name: feature_vector} 字典
        metric: 'euclidean' 或 'cosine'

    返回：
        names: 名称列表
        matrix: 距离/相似度矩阵
    """
    names = list(features_dict.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    # TODO: 填充距离/相似度矩阵
    # for i in range(n):
    #     for j in range(n):
    #         if metric == 'euclidean':
    #             matrix[i, j] = euclidean(features_dict[names[i]],
    #                                       features_dict[names[j]])
    #         elif metric == 'cosine':
    #             matrix[i, j] = cosine(features_dict[names[i]],
    #                                    features_dict[names[j]])

    return names, matrix

# ===== TODO 18: 可视化与分析 =====
# speaker_features = {name: speaker_feature(mfcc) for name, mfcc in mfccs.items()}
#
# # 欧氏距离矩阵
# names, euc_matrix = compute_similarity_matrix(speaker_features, 'euclidean')
#
# plt.figure(figsize=(8, 6))
# plt.imshow(euc_matrix, cmap='YlOrRd')
# plt.xticks(range(len(names)), names)
# plt.yticks(range(len(names)), names)
# plt.colorbar(label='欧氏距离')
# plt.title('说话人 MFCC 欧氏距离矩阵')
# for i in range(len(names)):
#     for j in range(len(names)):
#         plt.text(j, i, f'{euc_matrix[i,j]:.2f}', ha='center', va='center')
# plt.tight_layout()
# plt.savefig('figures/step6_distance_matrix.png', dpi=150)
# plt.show()
#
# # 余弦相似度矩阵
# names, cos_matrix = compute_similarity_matrix(speaker_features, 'cosine')
# # 类似绘制余弦相似度矩阵

# ===== TODO 19: 同说话人 vs 不同说话人 距离分布分析 =====
# 计算：
# - 同说话人距离：(A_1, A_2) 和 (B_1, B_2) 的平均距离
# - 不同说话人距离：(A_1, B_1), (A_1, B_2), (A_2, B_1), (A_2, B_2) 的平均距离
# 在报告中讨论：MFCC 能否有效区分不同说话人？

# ===== TODO 20: 不同特征表示方法的比较 =====
# 比较：
# (a) 仅用均值向量
# (b) 均值 + 标准差
# (c) 仅用前 N 个 MFCC 系数 (N=5, 10, 13)
# 哪种方法的区分效果最好？

# ===== 选做：简单说话人识别 =====
# 基于最近邻（1-NN）分类器：
# 对测试音频 A_2，计算与 A_1、B_1 的距离，判断属于哪个说话人
# 报告识别准确率
```

**思考题**：
1. 欧氏距离和余弦相似度在说话人比较中的表现有何差异？哪种更适合？
2. 同一说话人说不同内容的距离，与不同说话人说相同内容的距离，哪个更大？为什么？
3. 如何进一步提升说话人区分能力？（提示：考虑 MFCC 的动态特征，如 delta MFCC）

---

## 5. 评分细则（满分 100 分）

| 评分项 | 分值 | 详细说明 |
|--------|------|----------|
| **Step 1: 波形分析** | 10 分 | |
| - 波形可视化 | 3 分 | 正确加载并绘制波形 |
| - 有声/无声段标注 | 4 分 | 正确计算短时能量并标注 |
| - 多说话人对比 | 3 分 | 清晰对比不同说话人波形差异 |
| **Step 2: STFT 分析** | 15 分 | |
| - 基础 STFT 实现 | 5 分 | 正确计算并绘制 spectrogram |
| - 不同窗长对比 | 5 分 | 至少 3 种窗长的对比实验 |
| - 时频分辨率分析 | 5 分 | 正确解释 Heisenberg 权衡关系 |
| **Step 3: CWT 分析** | 15 分 | |
| - CWT 实现 | 5 分 | 正确执行 CWT 并可视化 |
| - CWT vs STFT 对比 | 5 分 | 清晰对比两种方法的差异 |
| - 不同小波基实验 | 5 分 | 尝试至少 2 种小波基并分析 |
| **Step 4: EMD 分解** | 15 分 | |
| - EMD 分解实现 | 5 分 | 正确分解并绘制 IMF |
| - IMF 频率分析 | 5 分 | 对各 IMF 做频谱分析 |
| - 重构实验 | 5 分 | IMF 重构与去噪实验 |
| **Step 5: MFCC 实现** | 25 分 | |
| - Mel 滤波器组 | 10 分 | 正确实现三角滤波器组并可视化 |
| - MFCC 提取流程 | 10 分 | 完整实现预加重→分帧→FFT→Mel→log→DCT |
| - 与 librosa 对比验证 | 5 分 | 对比结果并分析差异 |
| **Step 6: 说话人比较** | 15 分 | |
| - 特征表示方法 | 5 分 | 实现说话人级特征向量 |
| - 距离/相似度计算 | 5 分 | 正确实现欧氏距离和余弦相似度 |
| - 结果分析 | 5 分 | 清晰展示区分效果并分析 |
| **报告质量** | 5 分 | 结构清晰、图表规范、分析深入 |
| **总计** | **100 分** | |

### 等级划分

| 等级 | 分数段 | 说明 |
|------|--------|------|
| 优秀 | 90–100 | 所有步骤高质量完成，报告分析深入，有创新性实验 |
| 良好 | 80–89 | 所有步骤完成，代码正确，报告分析较充分 |
| 中等 | 70–79 | 大部分步骤完成，个别实现有小错误 |
| 及格 | 60–69 | 基本步骤完成，但代码或分析不够完整 |
| 不及格 | <60 | 未能完成核心步骤或存在抄袭 |

### 加分项（最多 +5 分）

- [ ] 实现 delta 和 delta-delta MFCC 并验证对说话人区分的提升（+2 分）
- [ ] 使用 DTW（动态时间规整）替代简单均值池化（+2 分）
- [ ] 绘制综合对比图，将 STFT/CWT/EMD/MFCC 在同一信号上的分析结果放在一起（+1 分）

---

## 6. 提交要求

### 中期提交（第6周）

- **内容**：Step 1–3 的代码与初步实验结果
- **形式**：PDF 报告（3–5 页）+ 代码文件
- **提交方式**：通过课程平台上传

### 最终提交（第12周）

- **内容**：全部 Step 1–6 的代码与完整实验报告
- **形式**：
  - 代码：`.py` 文件，包含充分注释
  - 报告：PDF 格式，8–15 页
  - 图表：保存在 `figures/` 目录下
- **报告结构建议**：
  1. 项目简介与背景（1 页）
  2. 数据说明（0.5 页）
  3. 实验方法与结果（按 Step 组织，每步 1–2 页）
  4. 总结与讨论（1 页）
  5. 参考文献
- **提交方式**：打包为 `学号_姓名_项目D.zip`，通过课程平台上传

---

## 7. FAQ

**Q1：录音时采样率不统一怎么办？**  
A：统一重采样到 16 kHz 即可，使用 `librosa.load(path, sr=16000)`。

**Q2：可以使用 librosa 直接计算 MFCC 吗？**  
A：Step 5 要求手动实现 MFCC 提取流程（包括 Mel 滤波器组），但 Step 6 的说话人比较可以使用 librosa 的结果作为对照。报告中需对比你的实现与 librosa 的结果。

**Q3：EMD 分解出的 IMF 数量不固定怎么办？**  
A：这是正常的，EMD 是数据驱动的自适应分解，不同信号分解出的 IMF 数量不同。在报告中记录各信号的 IMF 数量并分析原因。

**Q4：没有同学帮忙录音怎么办？**  
A：可以使用 LibriSpeech 数据集，或使用 TTS（文本转语音）工具生成不同"说话人"的语音。

**Q5：PyEMD 安装失败怎么办？**  
A：尝试 `pip install EMD-signal`（PyEMD 的 PyPI 包名）。如仍有问题，可用 `pip install emd`（emd 包）作为替代，或使用 `vmdpy` 实现变分模态分解（VMD）替代 EMD。

**Q6：CWT 计算很慢怎么办？**  
A：可降低音频采样率（如降至 8 kHz）或减少尺度数量。也可以使用 `ssqueezepy` 库的快速 CWT 实现。

**Q7：如何判断 Mel 滤波器组实现是否正确？**  
A：绘制滤波器组图形，检查：(1) 滤波器在低频区域较密集、高频区域较稀疏；(2) 相邻滤波器有 50% 重叠；(3) 与 librosa 的 `librosa.filters.mmel()` 输出对比。

**Q8：报告中的"分析"部分需要写到什么程度？**  
A：每个思考题至少用 2-3 句话回答，结合你的实验结果（如截图、数据）进行分析。避免泛泛而谈，要有具体数据支撑。

---

## 8. 参考资源

### 教材与书籍

1. **《语音信号处理》**（赵力）— 语音处理经典教材
2. **《数字语音处理》**（Thomas F. Quatieri）— 理论深入
3. **《信号与系统》**（Oppenheim）— 基础信号处理理论

### 在线教程

4. [librosa 官方文档](https://librosa.org/doc/latest/) — Python 音频分析库
5. [Speech Processing for ML](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) — MFCC 详解博客
6. [PyEMD 文档](https://pyemd.readthedocs.io/) — EMD 分解 Python 实现
7. [PyWavelets 文档](https://pywavelets.readthedocs.io/) — 小波变换 Python 实现

### 论文

8. Davis, S. B., & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences.* IEEE TASSP. — MFCC 的经典论文
9. Huang, N. E., et al. (1998). *The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis.* Proc. R. Soc. — EMD 的原始论文

### 工具库

| 库名 | 用途 | 安装命令 |
|------|------|----------|
| `numpy` | 数值计算 | `pip install numpy` |
| `scipy` | 科学计算、信号处理 | `pip install scipy` |
| `matplotlib` | 绘图 | `pip install matplotlib` |
| `librosa` | 音频分析 | `pip install librosa` |
| `PyEMD` (EMD-signal) | 经验模态分解 | `pip install EMD-signal` |
| `PyWavelets` | 小波变换 | `pip install PyWavelets` |
| `soundfile` | 音频读写 | `pip install soundfile` |

### 数据集

10. [LibriSpeech](https://www.openslr.org/12) — 英文语音数据集（test-clean 子集约 346 MB）
11. [THCHS-30](https://www.openslr.org/18/) — 中文语音数据集（清华大学开放）

---

> **祝实验顺利！如有问题，请在课程讨论区提问或联系助教。**

---

> **课程关联**：本项目对应[[欢迎|智能信号处理课程]]W2-W6知识模块。涉及知识点：[[知识点/时频分析/短时傅里叶变换|短时傅里叶变换]]、[[知识点/时频分析/小波变换|小波变换]]、[[知识点/时频分析/经验模态分解|经验模态分解]]、[[知识点/语音信号处理/语音信号基础|语音信号基础]]、[[知识点/经典信号处理/傅里叶分析|傅里叶分析]]。关联实验：[[实验6_综合实验|实验6]]选项B。
