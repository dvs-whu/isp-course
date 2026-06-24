# 迷你项目 E：自适应滤波、最优估计与统计信号分析

> **难度**：★★★★☆（4/5）  
> **课程**：智能信号处理 · 武汉大学人工智能学院  
> **性质**：个人项目  
> **时间线**：第 2 周启动 → 第 6 周期中检查 → 第 12 周终期答辩  

---

## 1. 项目概述

本项目围绕**随机信号分析与最优滤波**这一经典信号处理核心主题，要求同学从零开始：

1. 生成并分析随机信号（白噪声、有色噪声）；
2. 进行功率谱密度（PSD）估计并比较不同方法；
3. 设计 Wiener 滤波器求解去噪问题；
4. 实现 LMS 自适应滤波器并研究步长参数影响；
5. 设计一维 Kalman 滤波器完成目标跟踪；
6. 系统性对比三种滤波方法的性能与适用场景。

本项目仅使用 **numpy、scipy、matplotlib** 三个库，禁止调用现成的滤波器实现（如 `scipy.signal.wiener`），所有核心算法需自行编写。

---

## 2. 学习目标

完成本项目后，你应该能够：

| 编号 | 目标 | 对应步骤 |
|------|------|----------|
| LO-1 | 理解随机过程的基本统计特性（均值、方差、自相关函数） | Step 1 |
| LO-2 | 掌握功率谱密度的物理意义，能比较周期图法与 Welch 法 | Step 2 |
| LO-3 | 推导并实现 Wiener-Hopf 方程，理解最优线性滤波 | Step 3 |
| LO-4 | 实现 LMS 算法，理解收敛条件与步长选择的权衡 | Step 4 |
| LO-5 | 建立 Kalman 滤波的状态空间模型，完成预测-更新循环 | Step 5 |
| LO-6 | 具备工程视角，能根据场景选择合适的滤波方法 | Step 6 |

---

## 3. 数据准备

本项目所有数据均为**合成生成**，无需下载外部数据集。以下代码块在项目开始时运行一次即可。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(42)  # 保证可复现

# ============================================================
# 3.1 基本参数
# ============================================================
N = 2000          # 信号长度（采样点数）
fs = 1000         # 采样率 Hz
t = np.arange(N) / fs  # 时间轴 (0 ~ 1.999 s)

# ============================================================
# 3.2 白噪声 (White Gaussian Noise, WGN)
# ============================================================
white_noise = np.random.randn(N)  # 均值0, 方差1

# ============================================================
# 3.3 有色噪声 —— 通过滤波白噪声生成
# ============================================================
# 低通有色噪声：让白噪声通过一个低通 Butterworth 滤波器
b_lp, a_lp = signal.butter(4, 0.1)  # 4阶, 截止频率 0.1×Nyquist
colored_noise = signal.filtfilt(b_lp, a_lp, white_noise)

# ============================================================
# 3.4 "干净"信号 + 噪声信号（用于去噪任务）
# ============================================================
# 干净信号：两个正弦波叠加
clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# 加噪：SNR ≈ 0 dB
noise_power = np.var(clean_signal)
noise_for_snr = np.sqrt(noise_power) * np.random.randn(N)
noisy_signal = clean_signal + noise_for_snr

# ============================================================
# 3.5 一维目标跟踪数据（用于 Kalman 滤波）
# ============================================================
# 真实轨迹：匀速运动 + 加速度扰动
N_track = 500
dt_track = 1.0        # 时间步长 (s)
true_velocity = 1.0   # 真实速度 m/s
true_position = np.zeros(N_track)
true_velocity_seq = np.zeros(N_track)
true_velocity_seq[0] = true_velocity

for i in range(1, N_track):
    # 随机加速度扰动（过程噪声）
    accel_noise = np.random.randn() * 0.1
    true_velocity_seq[i] = true_velocity_seq[i-1] + accel_noise
    true_position[i] = true_position[i-1] + true_velocity_seq[i-1] * dt_track + 0.5 * accel_noise * dt_track**2

# 观测：位置 + 观测噪声
measurement_noise_std = 2.0
observations = true_position + np.random.randn(N_track) * measurement_noise_std

# 保存所有数据到字典，方便后续使用
project_data = {
    't': t, 'fs': fs, 'N': N,
    'white_noise': white_noise,
    'colored_noise': colored_noise,
    'clean_signal': clean_signal,
    'noisy_signal': noisy_signal,
    'track_true_pos': true_position,
    'track_true_vel': true_velocity_seq,
    'track_obs': observations,
    'track_dt': dt_track,
    'track_meas_std': measurement_noise_std,
}
print("数据准备完成！所有变量已生成。")
```

---

## 4. 任务步骤

### Step 1：随机信号的生成与统计分析（15 分）

**目标**：掌握随机过程的基本描述方法。

**任务**：

1. 对 `white_noise` 和 `colored_noise` 分别绘制时域波形；
2. 计算并打印均值、方差（与理论值对比）；
3. 计算**自相关函数**（lag 从 -100 到 +100），并绘图；
4. 观察并解释：白噪声与有色噪声的自相关函数有何不同？

**代码模板**：

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_autocorrelation(x, max_lag=100):
    """
    计算归一化自相关函数 R_xx[lag] / R_xx[0]
    
    Parameters:
        x: 输入信号 (1D array)
        max_lag: 最大滞后值
    Returns:
        lags: 滞后轴 [-max_lag, ..., 0, ..., max_lag]
        r_xx: 归一化自相关值
    """
    N = len(x)
    x_centered = x - np.mean(x)
    r_xx_full = np.correlate(x_centered, x_centered, mode='full') / (N * np.var(x))
    mid = len(r_xx_full) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    r_xx = r_xx_full[mid - max_lag : mid + max_lag + 1]
    return lags, r_xx

# ---- 对白噪声进行统计分析 ----
print("=" * 50)
print("白噪声统计特性")
print("=" * 50)
mean_white = # TODO: 计算均值
var_white =  # TODO: 计算方差
print(f"均值: {mean_white:.4f}  (理论值: 0)")
print(f"方差: {var_white:.4f}  (理论值: 1)")

# ---- 对有色噪声进行统计分析 ----
print("\n" + "=" * 50)
print("有色噪声统计特性")
print("=" * 50)
mean_colored = # TODO: 计算均值
var_colored =  # TODO: 计算方差
print(f"均值: {mean_colored:.4f}")
print(f"方差: {var_colored:.4f}")

# ---- 绘图 ----
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# TODO: 在 axes[0,0] 绘制白噪声时域波形（取前200个点）
# TODO: 在 axes[0,1] 绘制有色噪声时域波形（取前200个点）
# TODO: 在 axes[1,0] 绘制白噪声自相关函数
# TODO: 在 axes[1,1] 绘制有色噪声自相关函数

# 提示：
# axes[i,j].plot(x_data, y_data)
# axes[i,j].set_title('...')
# axes[i,j].set_xlabel('...')

plt.tight_layout()
plt.savefig('step1_statistical_analysis.png', dpi=150)
plt.show()
```

**思考题**（写入报告）：
- 白噪声的自相关函数在 lag≠0 时趋近于多少？这说明什么？
- 有色噪声的自相关函数衰减速度与低通滤波器截止频率有什么关系？

---

### Step 2：功率谱密度估计（15 分）

**目标**：理解 PSD 的物理意义，掌握周期图法与 Welch 法。

**任务**：

1. 用 **周期图法**（直接法）计算 `white_noise` 和 `colored_noise` 的 PSD；
2. 用 **Welch 法**（`scipy.signal.welch` 可用）计算 PSD；
3. 将两种方法绘制在同一图中进行对比；
4. 分析：Welch 法为何更平滑？偏差-方差权衡是什么？

**代码模板**：

```python
def periodogram_psd(x, fs, nfft=None):
    """
    周期图法计算功率谱密度
    
    Parameters:
        x: 输入信号
        fs: 采样率
        nfft: FFT 点数（默认取信号长度）
    Returns:
        freqs: 频率轴 (正频率部分)
        psd: 功率谱密度
    """
    N = len(x)
    if nfft is None:
        nfft = N
    
    # TODO: 对信号做 FFT
    X = None  # np.fft.fft(...)
    
    # TODO: 取模的平方，归一化
    psd = None  # |X|^2 / (N * fs)
    
    # TODO: 只取正频率部分
    freqs = None  # np.fft.fftfreq(...)
    # 提示：用 np.arange(nfft//2) 截取前半段
    
    return freqs, psd

# ---- 周期图法 ----
freqs_white, psd_white_period = periodogram_psd(project_data['white_noise'], fs)
freqs_colored, psd_colored_period = periodogram_psd(project_data['colored_noise'], fs)

# ---- Welch 法 ----
nperseg = 256  # 每段长度
# TODO: 使用 scipy.signal.welch 计算白噪声的 PSD
freqs_white_welch, psd_white_welch = None, None  # signal.welch(...)

# TODO: 使用 scipy.signal.welch 计算有色噪声的 PSD
freqs_colored_welch, psd_colored_welch = None, None  # signal.welch(...)

# ---- 绘图 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# TODO: 左图 —— 白噪声 PSD（周期图 vs Welch，建议用 semilogy）
# TODO: 右图 —— 有色噪声 PSD（周期图 vs Welch）
# 别忘加 legend、xlabel、ylabel、title

plt.tight_layout()
plt.savefig('step2_psd_estimation.png', dpi=150)
plt.show()
```

**思考题**：
- 白噪声的理论 PSD 是什么形状？你的估计与理论一致吗？
- 改变 `nperseg`（如 64, 256, 1024），Welch 估计的平滑程度如何变化？

---

### Step 3：Wiener 滤波器设计（20 分）

**目标**：推导并实现最优线性滤波器，用于信号去噪。

**背景**：

给定观测信号 $x[n] = s[n] + v[n]$，其中 $s[n]$ 是干净信号，$v[n]$ 是噪声。Wiener 滤波器的目标是找到长度为 $M$ 的 FIR 滤波器 $\mathbf{w} = [w_0, w_1, \ldots, w_{M-1}]^T$，使得输出 $\hat{s}[n] = \mathbf{w}^T \mathbf{x}[n]$ 的均方误差最小。

最优解由 **Wiener-Hopf 方程** 给出：

$$\mathbf{R}_{xx} \mathbf{w}_{opt} = \mathbf{r}_{sx}$$

其中 $\mathbf{R}_{xx}$ 是观测信号的自相关矩阵，$\mathbf{r}_{sx}$ 是干净信号与观测信号的互相关向量。

**任务**：

1. 构造自相关矩阵 $\mathbf{R}_{xx}$ 和互相关向量 $\mathbf{r}_{sx}$；
2. 求解 Wiener-Hopf 方程得到最优滤波器系数；
3. 将滤波器应用于 `noisy_signal`，计算去噪后信号；
4. 计算去噪前后的 SNR 改善量。

**代码模板**：

```python
def build_autocorrelation_matrix(x, M):
    """
    构造 Toeplitz 自相关矩阵 R_xx (M×M)
    
    利用信号的自相关序列构建 Toeplitz 矩阵：
    R_xx[i,j] = r_xx[|i-j|]
    
    Parameters:
        x: 观测信号 (1D array)
        M: 滤波器长度
    Returns:
        R: 自相关矩阵 (M, M)
    """
    N = len(x)
    x_centered = x - np.mean(x)
    
    # TODO: 计算自相关序列 r[k] for k = 0, 1, ..., M-1
    # 提示: r[k] = (1/N) * sum(x[n]*x[n+k]) for valid n
    r = np.zeros(M)
    for k in range(M):
        r[k] = # TODO
    
    # TODO: 用 r 构造 Toeplitz 矩阵
    # 提示: 可以手动构造，也可以用 scipy.linalg.toeplitz
    R = None  # ...
    return R

def build_cross_correlation(s, x, M):
    """
    构造互相关向量 r_sx
    
    r_sx[k] = E[s[n] * x[n+k]] for k = 0, 1, ..., M-1
    
    Parameters:
        s: 干净信号
        x: 观测信号
        M: 滤波器长度
    Returns:
        r: 互相关向量 (M,)
    """
    N = len(s)
    r = np.zeros(M)
    for k in range(M):
        r[k] = # TODO: 计算互相关
    return r

def wiener_filter_design(s, x, M):
    """
    设计 Wiener 滤波器
    
    求解 Wiener-Hopf 方程: R_xx * w = r_sx
    """
    R_xx = build_autocorrelation_matrix(x, M)
    r_sx = build_cross_correlation(s, x, M)
    
    # TODO: 求解线性方程组 R_xx * w = r_sx
    w_opt = None  # np.linalg.solve(...)
    return w_opt

def apply_fir_filter(w, x):
    """
    将 FIR 滤波器 w 应用于信号 x
    """
    # TODO: 计算卷积 y = w * x
    # 提示: np.convolve, 取 'same' 模式保持长度一致
    y = None
    return y

def compute_snr(signal, noise_or_residual):
    """计算信噪比 (dB)"""
    # TODO: SNR = 10 * log10(P_signal / P_noise)
    snr = None
    return snr

# ---- 设计并应用 Wiener 滤波器 ----
M = 32  # 滤波器长度，可调

w_wiener = wiener_filter_design(clean_signal, noisy_signal, M)
print(f"Wiener 滤波器系数 (前10个): {w_wiener[:10]}")

filtered_signal = apply_fir_filter(w_wiener, noisy_signal)

# ---- 计算 SNR ----
snr_before = compute_snr(clean_signal, noisy_signal - clean_signal)
snr_after = compute_snr(clean_signal, filtered_signal - clean_signal)
print(f"去噪前 SNR: {snr_before:.2f} dB")
print(f"去噪后 SNR: {snr_after:.2f} dB")
print(f"SNR 改善: {snr_after - snr_before:.2f} dB")

# ---- 绘图 ----
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
# TODO: axes[0] —— 干净信号
# TODO: axes[1] —— 含噪信号
# TODO: axes[2] —— Wiener 滤波后信号
plt.tight_layout()
plt.savefig('step3_wiener_filter.png', dpi=150)
plt.show()
```

**思考题**：
- 改变滤波器长度 $M$（如 8, 32, 128），去噪效果如何变化？是否存在最优长度？
- 如果不知道干净信号 $s[n]$（实际场景中通常如此），Wiener 滤波器如何设计？

---

### Step 4：LMS 自适应滤波器（20 分）

**目标**：实现 LMS 算法，理解自适应滤波的收敛过程。

**背景**：

LMS（Least Mean Squares）算法无需预先知道信号统计量，通过梯度下降迭代更新滤波器：

$$\mathbf{w}[n+1] = \mathbf{w}[n] + \mu \cdot e[n] \cdot \mathbf{x}[n]$$

其中 $e[n] = d[n] - \mathbf{w}^T[n] \mathbf{x}[n]$ 是误差，$\mu$ 是步长。

收敛条件：$0 < \mu < \frac{2}{\lambda_{\max}} \approx \frac{2}{M \cdot P_x}$

**任务**：

1. 实现标准 LMS 算法；
2. 在相同去噪问题上运行，与 Wiener 滤波器对比；
3. 研究不同步长 $\mu$ 对收敛速度和稳态误差的影响；
4. 绘制学习曲线（MSE 随迭代次数的变化）。

**代码模板**：

```python
def lms_filter(x, d, M, mu):
    """
    LMS 自适应滤波器
    
    Parameters:
        x: 输入信号 (观测信号)
        d: 期望信号 (干净信号)
        M: 滤波器阶数
        mu: 步长
    Returns:
        w: 最终滤波器系数 (M,)
        y: 滤波器输出 (与 x 等长)
        e: 误差序列 (与 x 等长)
        w_history: 滤波器系数历史 (N, M)
    """
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))
    
    for n in range(M, N):
        # 取当前输入向量
        x_vec = # TODO: x[n-M+1 : n+1]（注意顺序）
        
        # 滤波器输出
        y[n] = # TODO: w^T * x_vec
        
        # 误差
        e[n] = # TODO: d[n] - y[n]
        
        # 权重更新
        w = # TODO: w + mu * e[n] * x_vec
        
        w_history[n] = w.copy()
    
    return w, y, e, w_history

# ---- 运行 LMS ----
M_lms = 32  # 与 Wiener 滤波器相同阶数

# 计算推荐步长范围
P_x = np.mean(noisy_signal**2)
mu_max = 2.0 / (M_lms * P_x)
print(f"信号功率: {P_x:.4f}")
print(f"理论最大步长: {mu_max:.6f}")

# 选择三个不同步长
mu_values = [0.001, 0.005, 0.02]  # 小、中、大步长，根据实际调整

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, mu in enumerate(mu_values):
    w_lms, y_lms, e_lms, w_hist = lms_filter(
        noisy_signal, clean_signal, M_lms, mu
    )
    
    # TODO: 计算 MSE 学习曲线 (对 e^2 做滑动平均或直接用 e^2)
    mse_curve = None  # e_lms**2 或滑动平均
    
    # TODO: 在 axes[0,0] 绘制不同 mu 的学习曲线
    # TODO: 在 axes[1,0] 绘制最终滤波器输出 vs 干净信号 (选一个mu)
    # TODO: 打印最终 SNR

plt.tight_layout()
plt.savefig('step4_lms_filter.png', dpi=150)
plt.show()
```

**思考题**：
- 步长 $\mu$ 太大会怎样？太小又会怎样？画出收敛速度 vs 稳态误差的权衡图。
- LMS 的最终滤波器系数与 Wiener 解是否一致？为什么？
- LMS 相比 Wiener 滤波的优势和劣势各是什么？

---

### Step 5：一维 Kalman 滤波器（20 分）

**目标**：建立状态空间模型，实现 Kalman 滤波完成目标跟踪。

**背景**：

状态空间模型：
- 状态转移：$\mathbf{x}[k] = \mathbf{F} \mathbf{x}[k-1] + \mathbf{w}[k]$
- 观测方程：$z[k] = \mathbf{H} \mathbf{x}[k] + v[k]$

一维匀速运动模型：
- 状态：$\mathbf{x} = [position, velocity]^T$
- $\mathbf{F} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$, $\mathbf{H} = [1, 0]$

Kalman 滤波五个核心公式（预测 + 更新）：

**预测**：
$$\hat{\mathbf{x}}^-_k = \mathbf{F} \hat{\mathbf{x}}_{k-1}$$
$$\mathbf{P}^-_k = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$$

**更新**：
$$\mathbf{K}_k = \mathbf{P}^-_k \mathbf{H}^T (\mathbf{H} \mathbf{P}^-_k \mathbf{H}^T + R)^{-1}$$
$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}^-_k + \mathbf{K}_k (z_k - \mathbf{H} \hat{\mathbf{x}}^-_k)$$
$$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}^-_k$$

**任务**：

1. 建立上述状态空间模型（设定 $\mathbf{Q}$, $R$）；
2. 实现 Kalman 滤波的预测-更新循环；
3. 在跟踪数据上运行，绘制真实轨迹、观测、滤波结果；
4. 分析滤波效果。

**代码模板**：

```python
def kalman_filter_1d(observations, dt, Q, R, x0, P0):
    """
    一维 Kalman 滤波器（匀速运动模型）
    
    Parameters:
        observations: 观测序列 (位置)
        dt: 时间步长
        Q: 过程噪声协方差矩阵 (2×2)
        R: 观测噪声方差 (标量)
        x0: 初始状态估计 [position, velocity]
        P0: 初始协方差矩阵 (2×2)
    Returns:
        x_est: 状态估计历史 (N, 2)
        P_est: 协方差历史 (N, 2, 2)
        K_hist: 增益历史 (N, 2)
    """
    N = len(observations)
    
    # 状态转移矩阵
    F = np.array([[1, dt],
                  [0, 1]])
    
    # 观测矩阵
    H = np.array([[1, 0]])
    
    # 初始化
    x = x0.copy()
    P = P0.copy()
    
    x_est = np.zeros((N, 2))
    P_est = np.zeros((N, 2, 2))
    K_hist = np.zeros((N, 2))
    
    for k in range(N):
        # ===== 预测 =====
        x_pred = # TODO: F @ x
        P_pred = # TODO: F @ P @ F.T + Q
        
        # ===== 计算 Kalman 增益 =====
        S = # TODO: H @ P_pred @ H.T + R  (新息协方差, 标量)
        K = # TODO: P_pred @ H.T / S  (2×1 向量, 注意 S 是标量)
        
        # ===== 更新 =====
        z = observations[k]
        innovation = # TODO: z - H @ x_pred  (新息/残差)
        x = # TODO: x_pred + K.flatten() * innovation
        P = # TODO: (I - K @ H) @ P_pred
        
        # 保存
        x_est[k] = x
        P_est[k] = P
        K_hist[k] = K.flatten()
    
    return x_est, P_est, K_hist

# ---- 设置参数 ----
dt = project_data['track_dt']
Q = np.array([[0.01, 0],
              [0, 0.01]])  # 过程噪声协方差，可调
R = project_data['track_meas_std']**2  # 观测噪声方差
x0 = np.array([observations[0], 0.0])  # 初始状态
P0 = np.array([[10, 0],
               [0, 10]])  # 初始不确定性

# ---- 运行 Kalman 滤波 ----
x_est, P_est, K_hist = kalman_filter_1d(observations, dt, Q, R, x0, P0)

# ---- 绘图 ----
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# TODO: axes[0] —— 位置：真实轨迹、观测值、Kalman 估计
# 提示：观测用散点 plot(..., 'o', markersize=2, alpha=0.3)

# TODO: axes[1] —— 速度：真实速度、Kalman 估计速度

# TODO: axes[2] —— Kalman 增益 K[:,0] (位置增益) 随时间变化

plt.tight_layout()
plt.savefig('step5_kalman_filter.png', dpi=150)
plt.show()

# ---- 定量评估 ----
pos_rmse_obs = np.sqrt(np.mean((observations - true_position)**2))
pos_rmse_kal = np.sqrt(np.mean((x_est[:, 0] - true_position)**2))
print(f"观测 RMSE: {pos_rmse_obs:.4f}")
print(f"Kalman RMSE: {pos_rmse_kal:.4f}")
print(f"RMSE 改善: {(1 - pos_rmse_kal/pos_rmse_obs)*100:.1f}%")
```

**思考题**：
- 改变过程噪声 $\mathbf{Q}$ 的大小，滤波器的跟踪灵敏度如何变化？
- Kalman 增益 $\mathbf{K}$ 为什么会在初始阶段较大，随后逐渐减小？这意味着什么？
- 如果目标突然变向（非匀速运动），如何调整模型？

---

### Step 6：方法对比与综合分析（10 分）

**目标**：系统性对比 Wiener、LMS、Kalman 三种方法。

**任务**：

制作一个综合对比表格与图表，至少包含以下维度：

| 对比维度 | Wiener 滤波 | LMS 自适应 | Kalman 滤波 |
|----------|-------------|-----------|-------------|
| 是否需要先验统计信息 | | | |
| 计算复杂度（每步） | | | |
| 能否处理非平稳信号 | | | |
| 去噪 SNR 改善 (dB) | | | |
| 适用场景举例 | | | |

**代码模板**：

```python
# ---- 在同一去噪任务上对比三种方法 ----

# 方法1: Wiener 滤波 (Step 3 的结果)
# 方法2: LMS 滤波 (Step 4 的结果)
# 方法3: 将 Kalman 滤波应用于去噪（需要重新建模）

# Kalman 用于信号去噪的状态空间模型
# 状态: x = [s[n], s[n-1], ..., s[n-L+1]]^T (信号的最近 L 个样本)
# 简化版: 只跟踪信号值和其导数
def kalman_for_denoising(noisy_sig, Q_val, R_val):
    """
    用 Kalman 滤波做一维信号去噪
    
    简化模型：状态 = [信号值]，即标量 Kalman 滤波
    """
    N = len(noisy_sig)
    x_est = np.zeros(N)
    
    # 初始化
    x = noisy_sig[0]
    P = 1.0
    
    for k in range(N):
        # 预测
        x_pred = x  # 假设信号缓慢变化
        P_pred = P + Q_val
        
        # 更新
        K = P_pred / (P_pred + R_val)
        x = x_pred + K * (noisy_sig[k] - x_pred)
        P = (1 - K) * P_pred
        
        x_est[k] = x
    
    return x_est

# TODO: 运行三种方法，计算各自的 SNR
# TODO: 绘制对比图：三种方法的去噪结果叠加在一起
# TODO: 制作对比表格（填写上表）

fig, ax = plt.subplots(figsize=(14, 6))
# TODO: 绘制干净信号 + 三种滤波结果
ax.legend()
plt.savefig('step6_comparison.png', dpi=150)
plt.show()
```

**思考题**（写入报告，至少 200 字）：
- 三种方法各自的优势和局限性是什么？
- 在什么实际应用场景中你会选择哪种方法？给出至少 3 个具体场景。
- 如果信号是非平稳的（如语音），哪种方法最合适？为什么？

---

## 5. 评分细则（满分 100 分）

| 评分项 | 分值 | 评分标准 |
|--------|------|----------|
| **Step 1：统计分析** | **15** | |
| 　均值方差计算正确 | 3 | 数值正确即得分 |
| 　自相关函数实现正确 | 5 | 归一化、lag 范围正确 |
| 　绘图规范（标题/标签/图例） | 3 | 缺一项扣 1 分 |
| 　思考题回答 | 4 | 回答有理有据 |
| **Step 2：PSD 估计** | **15** | |
| 　周期图法实现正确 | 5 | FFT + 归一化正确 |
| 　Welch 法使用正确 | 3 | 参数合理 |
| 　对比分析到位 | 4 | 图文结合 |
| 　思考题回答 | 3 | |
| **Step 3：Wiener 滤波** | **20** | |
| 　自相关矩阵构造正确 | 5 | Toeplitz 结构正确 |
| 　Wiener-Hopf 方程求解 | 5 | 使用 np.linalg.solve |
| 　去噪效果明显 | 4 | SNR 改善 > 5 dB |
| 　滤波器长度实验 | 3 | 多个 M 值对比 |
| 　思考题回答 | 3 | |
| **Step 4：LMS 滤波** | **20** | |
| 　LMS 算法实现正确 | 6 | 更新公式正确 |
| 　步长实验（≥3 个值） | 5 | 含收敛/发散情况 |
| 　学习曲线绘制 | 3 | MSE vs 迭代次数 |
| 　与 Wiener 对比分析 | 3 | 系数对比、SNR 对比 |
| 　思考题回答 | 3 | |
| **Step 5：Kalman 滤波** | **20** | |
| 　状态空间模型正确 | 5 | F, H, Q, R 设置合理 |
| 　预测-更新循环正确 | 6 | 五个公式全部正确 |
| 　跟踪结果可视化 | 4 | 三维图（位置/速度/增益） |
| 　定量评估 | 2 | RMSE 计算 |
| 　思考题回答 | 3 | |
| **Step 6：综合对比** | **10** | |
| 　对比表格完整 | 3 | 所有维度填写 |
| 　场景分析合理 | 4 | 至少 3 个场景 |
| 　报告质量（排版/逻辑） | 3 | |
| **合计** | **100** | |

**加分项**（最多 +5 分）：
- 实现归一化 LMS（NLMS）算法并对比（+2）
- 实现二维 Kalman 滤波（带加速度状态）（+2）
- 报告中包含数学推导过程（Wiener-Hopf 推导 / Kalman 增益推导）（+1）

**扣分项**：
- 代码不能运行：对应步骤 0 分
- 使用禁止的库函数（如 `scipy.signal.wiener`）：全项目扣 20 分
- 抄袭：0 分并上报

---

## 6. 提交要求

### 6.1 提交物

| 文件 | 格式 | 说明 |
|------|------|------|
| 报告 | `.pdf` | 不超过 15 页（含图表），需包含所有步骤的结果与思考题 |
| 代码 | `.py` 或 `.ipynb` | 一个完整可运行的脚本/Notebook |
| 图片 | `.png` | 所有步骤的输出图（也应嵌入报告中） |

### 6.2 命名规范

```
学号_姓名_项目E/
├── report.pdf
├── code/
│   └── adaptive_filter_project.py (或 .ipynb)
├── figures/
│   ├── step1_statistical_analysis.png
│   ├── step2_psd_estimation.png
│   ├── step3_wiener_filter.png
│   ├── step4_lms_filter.png
│   ├── step5_kalman_filter.png
│   └── step6_comparison.png
└── README.md  (简要说明运行方式)
```

### 6.3 时间节点

| 节点 | 时间 | 内容 |
|------|------|------|
| 启动 | 第 2 周 | 阅读说明书，理解任务 |
| 期中检查 | 第 6 周 | 提交 Step 1–3 的代码与初步报告 |
| 终期提交 | 第 12 周周三 23:59 | 提交全部内容 |
| 答辩 | 第 12 周周五 | 5 分钟展示 + 3 分钟提问 |

---

## 7. FAQ

**Q1：我可以使用 `scipy.signal.welch` 吗？**  
A：可以。Welch 法属于 PSD 估计工具，允许使用。但 Wiener 滤波器、LMS、Kalman 滤波器的核心算法必须自行实现。

**Q2：LMS 发散了怎么办？**  
A：说明步长 $\mu$ 过大。减小 $\mu$ 直到收敛，并在报告中记录发散的阈值与理论值对比。

**Q3：Wiener 滤波需要知道干净信号，实际中怎么办？**  
A：本项目为教学目的，假设已知干净信号来设计滤波器。在报告中讨论实际场景下的替代方案（如用噪声的统计特性估计）。

**Q4：Kalman 滤波的 Q 和 R 如何设定？**  
A：Q 反映你对运动模型的信任程度（越小越信任模型），R 反映观测噪声大小。可以尝试不同值并分析影响。

**Q5：报告需要用 LaTeX 吗？**  
A：不强制，但推荐。Word 也可，但公式需清晰可读。

**Q6：Step 5 的 Kalman 滤波能用矩阵形式写吗？**  
A：必须用矩阵形式。一维跟踪的状态是 2 维的（位置+速度），这是理解 Kalman 滤波的关键。

---

## 8. 参考资源

### 教材
- [1] S. Haykin, *Adaptive Filter Theory*, 5th Ed., Pearson. （第 2、5、6 章）
- [2] 殷勤业等, 《随机信号处理》, 西安交通大学出版社.
- [3] R. G. Brown, P. Y. C. Hwang, *Introduction to Random Signals and Applied Kalman Filtering*, 4th Ed., Wiley.

### 在线资源
- [4] [Kalman Filter Tutorial](https://www.kalmanfilter.net/) — 交互式可视化讲解
- [5] [scipy.signal 文档](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [6] [numpy.fft 文档](https://numpy.org/doc/stable/reference/routines.fft.html)
- [7] Steve Brunton 的 YouTube 系列：[Kalman Filter](https://www.youtube.com/watch?v=CaCcOwJ-2LY)

### 工具
- Python 3.8+
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.4
- Jupyter Notebook（可选但推荐）

---

*本说明书由智能信号处理课程组编制，如有疑问请在课程群中提问或联系助教。*

*最后更新：2026 年春季学期*

---

> **课程关联**：本项目对应[[欢迎|智能信号处理课程]]W5-W7知识模块。涉及知识点：[[知识点/统计信号处理/随机信号基础|随机信号基础]]、[[知识点/统计信号处理/功率谱估计|功率谱估计]]、[[知识点/统计信号处理/维纳滤波|维纳滤波]]、[[知识点/统计信号处理/自适应滤波|自适应滤波]]（LMS）、[[知识点/统计信号处理/卡尔曼滤波|卡尔曼滤波]]。关联实验：[[实验4_统计信号处理|实验4]]。
