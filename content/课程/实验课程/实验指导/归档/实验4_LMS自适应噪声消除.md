# 实验4：LMS自适应噪声消除

## 一、实验目的

1. 理解LMS（Least Mean Squares）自适应滤波算法的基本原理
2. 掌握自适应噪声消除系统的信号处理流程
3. 学习步长参数对算法收敛速度与稳态误差的影响
4. 通过可视化手段分析自适应滤波器的收敛过程

## 二、实验环境

- Python 3.8+
- NumPy、Matplotlib、SciPy
- Jupyter Notebook 或任意 Python IDE

## 三、实验原理

### 3.1 自适应噪声消除框架

自适应噪声消除（ANC）系统包含以下信号：
- **原始信号** $d(n) = s(n) + v_0(n)$：含噪语音，其中 $s(n)$ 为纯净语音，$v_0(n)$ 为噪声分量
- **参考信号** $x(n)$：与噪声 $v_0(n)$ 相关但与 $s(n)$ 不相关的信号
- **自适应滤波器输出** $y(n)$：对噪声的估计 $\hat{v}_0(n)$
- **误差信号** $e(n) = d(n) - y(n)$：期望恢复的纯净信号

### 3.2 LMS算法

LMS算法的核心是利用瞬时梯度估计来更新滤波器权重：

$$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \cdot e(n) \cdot \mathbf{x}(n)$$

其中：
- $\mathbf{w}(n)$ 为第 $n$ 时刻的滤波器权重向量（长度 $M$）
- $\mu$ 为步长参数，控制收敛速度与稳定性
- $e(n) = d(n) - \mathbf{w}^T(n)\mathbf{x}(n)$ 为误差信号
- $\mathbf{x}(n) = [x(n), x(n-1), \ldots, x(n-M+1)]^T$ 为参考信号向量

**收敛条件**：$0 < \mu < \frac{2}{\lambda_{\max}}$，其中 $\lambda_{\max}$ 为参考信号自相关矩阵的最大特征值。实际中常取 $\mu < \frac{2}{M \cdot P_x}$，$P_x$ 为参考信号功率。

### 3.3 性能指标

- **均方误差（MSE）**：$\text{MSE}(n) = E[e^2(n)]$，用瞬时值 $e^2(n)$ 或滑动平均近似
- **噪声抑制比**：$\text{NSR} = 10\log_{10}\frac{\sum v_0^2(n)}{\sum (e(n)-s(n))^2}$（dB）

## 四、实验步骤

### 步骤1：生成实验信号

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
N = 3000  # 采样点数
fs = 8000  # 采样率 8kHz
n = np.arange(N)

# 生成纯净语音信号（多个正弦波叠加模拟）
s = np.sin(2 * np.pi * 300 * n / fs) + 0.5 * np.sin(2 * np.pi * 600 * n / fs)
s[:500] = 0  # 前500点静音
s[1500:2000] = 0  # 中间静音段

# 生成噪声源（有色噪声：白噪声通过低通滤波器）
white_noise = np.random.randn(N + 100)
b, a = butter(4, 0.3)
colored_noise = lfilter(b, a, white_noise)[:N]

# 噪声经过未知路径到达主通道
unknown_path = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
v0 = lfilter(unknown_path, 1, colored_noise)

# 参考信号：噪声源经过另一条路径
ref_path = np.array([0.6, 0.4, 0.2])
x_ref = lfilter(ref_path, 1, colored_noise)

# 含噪信号
d = s + 0.8 * v0

print(f"信号长度: {N} 点, 采样率: {fs} Hz")
print(f"纯净语音范围: [{s.min():.2f}, {s.max():.2f}]")
print(f"噪声范围: [{v0.min():.2f}, {v0.max():.2f}]")
```

### 步骤2：实现LMS自适应滤波器

```python
def lms_filter(d, x, M=32, mu=0.01):
    """
    LMS自适应滤波器
    参数:
        d  : 主通道信号（含噪信号）
        x  : 参考通道信号
        M  : 滤波器阶数
        mu : 步长参数
    返回:
        y  : 滤波器输出（噪声估计）
        e  : 误差信号（去噪结果）
        w  : 最终权重
        mse_history : MSE历史
    """
    N = len(d)
    w = np.zeros(M)
    e = np.zeros(N)
    y = np.zeros(N)
    mse_history = np.zeros(N)

    for i in range(M, N):
        x_vec = x[i:i - M:-1] if i >= M else np.zeros(M)
        if len(x_vec) < M:
            x_vec = np.pad(x_vec, (0, M - len(x_vec)))
        # 参考信号向量
        x_vec = x[i - M + 1:i + 1][::-1]

        # 滤波器输出
        y[i] = np.dot(w, x_vec)

        # 误差信号
        e[i] = d[i] - y[i]

        # 权重更新
        w = w + mu * e[i] * x_vec

        # MSE估计（指数滑动平均）
        mse_history[i] = 0.99 * mse_history[i - 1] + 0.01 * e[i] ** 2

    return y, e, w, mse_history

# 运行LMS滤波
M = 32
mu = 0.005
y_est, e_out, w_final, mse = lms_filter(d, x_ref, M=M, mu=mu)

print(f"滤波器阶数 M={M}, 步长 μ={mu}")
print(f"最终权重前5项: {w_final[:5]}")
```

### 步骤3：结果可视化

```python
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# 纯净语音
axes[0].plot(n / fs, s, 'b', linewidth=0.5)
axes[0].set_ylabel('幅度')
axes[0].set_title('纯净语音信号 s(n)')

# 含噪信号
axes[1].plot(n / fs, d, 'r', linewidth=0.5)
axes[1].set_ylabel('幅度')
axes[1].set_title('含噪信号 d(n) = s(n) + v₀(n)')

# LMS去噪结果
axes[2].plot(n / fs, e_out, 'g', linewidth=0.5)
axes[2].set_ylabel('幅度')
axes[2].set_title('LMS去噪结果 e(n)')

# MSE收敛曲线
axes[3].plot(n / fs, 10 * np.log10(mse + 1e-10), 'k', linewidth=0.8)
axes[3].set_ylabel('MSE (dB)')
axes[3].set_xlabel('时间 (s)')
axes[3].set_title('MSE收敛曲线')

plt.tight_layout()
plt.savefig('lms_noise_cancellation.png', dpi=150)
plt.show()
```

### 步骤4：步长参数对收敛的影响

```python
mu_values = [0.001, 0.005, 0.01, 0.02]
fig, ax = plt.subplots(figsize=(10, 5))

for mu in mu_values:
    _, _, _, mse_h = lms_filter(d, x_ref, M=32, mu=mu)
    ax.plot(10 * np.log10(mse_h + 1e-10), label=f'μ = {mu}')

ax.set_xlabel('迭代次数')
ax.set_ylabel('MSE (dB)')
ax.set_title('不同步长参数的LMS收敛曲线')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lms_convergence_comparison.png', dpi=150)
plt.show()
```

### 步骤5：性能评估

```python
# 计算信噪比改善
def snr(signal, noise_signal):
    """计算信噪比(dB)"""
    noise = signal - noise_signal
    return 10 * np.log10(np.sum(signal ** 2) / (np.sum(noise ** 2) + 1e-10))

snr_before = snr(s, d - s)
snr_after = snr(s, e_out - s)

print(f"处理前信噪比: {snr_before:.2f} dB")
print(f"处理后信噪比: {snr_after:.2f} dB")
print(f"信噪比改善: {snr_after - snr_before:.2f} dB")

# 频谱对比
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

freqs = np.fft.rfftfreq(N, 1 / fs)

S_clean = np.abs(np.fft.rfft(s))
S_noisy = np.abs(np.fft.rfft(d))
S_denoised = np.abs(np.fft.rfft(e_out))

axes[0].plot(freqs, 20 * np.log10(S_clean + 1e-10), 'b', label='纯净语音')
axes[0].plot(freqs, 20 * np.log10(S_noisy + 1e-10), 'r', alpha=0.7, label='含噪信号')
axes[0].set_title('频谱对比：纯净 vs 含噪')
axes[0].legend()
axes[0].set_xlabel('频率 (Hz)')

axes[1].plot(freqs, 20 * np.log10(S_clean + 1e-10), 'b', label='纯净语音')
axes[1].plot(freqs, 20 * np.log10(S_denoised + 1e-10), 'g', alpha=0.7, label='去噪结果')
axes[1].set_title('频谱对比：纯净 vs 去噪')
axes[1].legend()
axes[1].set_xlabel('频率 (Hz)')

plt.tight_layout()
plt.savefig('lms_spectral_comparison.png', dpi=150)
plt.show()
```

## 五、实验报告要求

1. **算法实现**：提交完整的LMS滤波器实现代码，附关键注释
2. **结果展示**：
   - 四面板图（纯净信号、含噪信号、去噪结果、MSE曲线）
   - 不同步长的收敛对比图
   - 频谱对比图
3. **定量分析**：列表展示不同 $\mu$ 和 $M$ 组合下的信噪比改善值
4. **个人分析**（300字以上）：
   - 步长 $\mu$ 对收敛速度和稳态误差的影响规律
   - 滤波器阶数 $M$ 的选择依据
   - LMS算法的优缺点讨论

## 六、思考题

1. 如果参考信号 $x(n)$ 与噪声 $v_0(n)$ 完全不相关，LMS算法还能有效去噪吗？为什么？
2. 试推导归一化LMS（NLMS）算法的权重更新公式，并与标准LMS比较其优势。
3. 在实际应用中（如降噪耳机），自适应滤波器的实时性要求如何影响步长 $\mu$ 和阶数 $M$ 的选择？
4. 若噪声为非平稳信号（统计特性随时间变化），LMS算法需要做哪些改进？


---

> **课程关联**：本实验安排在[[欢迎|智能信号处理课程]]第12周。涉及知识点：自适应滤波、LMS算法。关联项目：项目一E。
> 详见[[13周教学进度表]] | [[课程关系图谱]]