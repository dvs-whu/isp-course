# 迷你项目C：图像频域滤波与时频增强

## 1. 项目概述

本项目探索二维傅里叶变换在图像处理中的应用，包括频域滤波（低通、高通）用于图像去噪与边缘增强，以及短时傅里叶变换（STFT）在图像局部频率分析中的应用。学生将对比空间域卷积与频域滤波的效果与效率，并使用客观指标评估处理质量。

**适用课程：** 智能信号处理（大二）
**项目性质：** 个人项目
**技术栈：** Python、NumPy、SciPy、Matplotlib、scikit-image
**预计耗时：** 8–12 小时

---

## 2. 学习目标

完成本项目后，你应该能够：

1. **掌握 2D FFT**：理解二维离散傅里叶变换的数学原理，能够对图像执行 FFT 并正确可视化频谱（幅度谱、对数幅度谱）。
2. **频域滤波器设计**：设计理想、巴特沃斯、高斯三种低通/高通滤波器，理解截止频率与滤波器阶数对结果的影响。
3. **图像去噪**：利用频域低通滤波去除图像中的高频噪声，并对比不同滤波器的去噪效果。
4. **边缘增强与锐化**：利用频域高通滤波提取图像边缘信息，实现图像锐化。
5. **时频分析**：理解 2D STFT 的原理，分析图像的局部频率特征。
6. **性能评估**：使用 PSNR 和 SSIM 客观评价图像质量，对比空间域与频域方法的效果和速度。

---

## 3. 数据准备

### 3.1 使用 scikit-image 内置图像

```python
from skimage import data, color, img_as_float

# 灰度图像
camera = img_as_float(data.camera())      # 512×512 摄影师
coins = img_as_float(data.coins())        # 303×384 硬币

# 彩色图像（需转灰度）
astro = img_as_float(color.rgb2gray(data.astronaut()))  # 512×512 宇航员
coffee = img_as_float(color.rgb2gray(data.coffee()))    # 400×600 咖啡杯
```

> **要求：** 至少选择 **2 张不同图像** 进行实验。

### 3.2 添加噪声

```python
import numpy as np

def add_gaussian_noise(image, sigma=0.1):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image + noise, 0, 1)
    return noisy

def add_salt_pepper_noise(image, amount=0.05):
    """添加椒盐噪声"""
    noisy = image.copy()
    # 椒噪声
    num_salt = int(amount * image.size / 2)
    coords = tuple(np.random.randint(0, d, num_salt) for d in image.shape)
    noisy[coords] = 0
    # 盐噪声
    coords = tuple(np.random.randint(0, d, num_salt) for d in image.shape)
    noisy[coords] = 1
    return noisy
```

---

## 4. 任务步骤

### 步骤 1：图像 2D FFT 与频谱可视化

**目标：** 对图像执行二维傅里叶变换，理解频域表示。

**知识点：** 2D DFT、频谱中心化、对数幅度谱

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

# --- 加载图像 ---
image = img_as_float(data.camera())

# --- 2D FFT ---
# TODO: 对图像执行二维快速傅里叶变换
F = np.fft.____________(image)          # 填入正确的 fft 函数

# TODO: 将零频分量移到频谱中心
F_shifted = np.fft.____________(F)      # 填入正确的 fftshift 函数

# --- 计算幅度谱 ---
magnitude = np.abs(F_shifted)

# TODO: 计算对数幅度谱（加 1 避免 log(0)）
magnitude_log = np.____________(____________)  # 填入 log 函数和参数

# --- 可视化 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('原始图像')

# TODO: 显示幅度谱（使用 gray colormap）
axes[1].imshow(____________, cmap='gray')       # 填入变量名
axes[1].set_title('幅度谱（对数尺度）')

# TODO: 显示幅度谱（使用 hot colormap，注意设置插值方式）
axes[2].imshow(____________, cmap='hot', interpolation='____________')  # 填入变量和插值方式
axes[2].set_title('幅度谱（热力图）')

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig('step1_spectrum.png', dpi=150)
plt.show()
```

**实验任务：**
- 对比不同图像的频谱特征
- 观察图像旋转 45° 后频谱的变化（提示：`np.rot90` 或 `scipy.ndimage.rotate`）
- 将频谱的幅度设为 1、保留相位，做逆 FFT 看看结果（相位的重要性）

---

### 步骤 2：频域低通滤波器设计与去噪

**目标：** 设计三种低通滤波器，对含噪图像进行去噪对比。

**知识点：** 理想 LPF、巴特沃斯 LPF、高斯 LPF、截止频率

```python
def ideal_lowpass(shape, cutoff):
    """理想低通滤波器"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols).reshape(1, -1)
    # TODO: 计算到中心的距离矩阵 D
    D = np.sqrt(____________)            # 填入距离公式
    # TODO: 构造理想低通滤波器（D <= cutoff 为 1，否则为 0）
    H = np.____________.astype(np.float64)  # 填入条件
    return H

def butterworth_lowpass(shape, cutoff, order=2):
    """巴特沃斯低通滤波器"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols).reshape(1, -1)
    D = np.sqrt((u - crow)**2 + (v - ccol)**2)
    # TODO: 巴特沃斯滤波器公式 H = 1 / (1 + (D/D0)^(2n))
    H = ____________                      # 填入公式
    return H

def gaussian_lowpass(shape, cutoff):
    """高斯低通滤波器"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols).reshape(1, -1)
    D_sq = (u - crow)**2 + (v - ccol)**2
    # TODO: 高斯滤波器公式 H = exp(-D^2 / (2*D0^2))
    H = ____________                      # 填入公式
    return H

def frequency_filter(image, H):
    """频域滤波：FFT -> 乘以滤波器 -> IFFT"""
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    # TODO: 频域滤波（逐元素相乘）
    G_shifted = ____________              # 填入滤波操作
    G = np.fft.ifftshift(G_shifted)
    # TODO: 执行逆 FFT 并取实部
    result = np.____________(G)           # 填入逆 FFT
    return np.abs(result)

# --- 添加噪声 ---
image = img_as_float(data.camera())
noisy = add_gaussian_noise(image, sigma=0.1)

# --- 应用三种滤波器 ---
cutoff = 50
H_ideal = ideal_lowpass(image.shape, cutoff)
H_butter = butterworth_lowpass(image.shape, cutoff, order=2)
H_gauss = gaussian_lowpass(image.shape, cutoff)

denoised_ideal = frequency_filter(noisy, H_ideal)
denoised_butter = frequency_filter(noisy, H_butter)
denoised_gauss = frequency_filter(noisy, H_gauss)

# --- 可视化（自行补充） ---
# 要求：显示原图、含噪图、三种滤波器形状、三种去噪结果
```

**实验任务：**
- 改变截止频率 `cutoff = [20, 50, 100, 200]`，观察去噪效果与模糊程度的权衡
- 对比高斯噪声和椒盐噪声的去噪效果差异
- 可视化三种滤波器的 3D 曲面图（提示：`plot_surface`）

---

### 步骤 3：频域高通滤波与边缘增强

**目标：** 利用高通滤波提取边缘，实现图像锐化。

**知识点：** 高通滤波器、Unsharp Masking、频域锐化

```python
def ideal_highpass(shape, cutoff):
    """理想高通滤波器"""
    # TODO: 利用低通滤波器构造高通滤波器
    H_lp = ideal_lowpass(shape, cutoff)
    H_hp = ____________                   # 填入高通 = 1 - 低通
    return H_hp

def gaussian_highpass(shape, cutoff):
    """高斯高通滤波器"""
    H_lp = gaussian_lowpass(shape, cutoff)
    return ____________                   # 填入

# --- 边缘检测 ---
image = img_as_float(data.camera())
H_hp = gaussian_highpass(image.shape, cutoff=30)
edges = frequency_filter(image, H_hp)

# --- 图像锐化：Unsharp Masking ---
# 锐化图像 = 原图 + k × 高频细节
k = 1.5
sharpened = ____________                  # 填入锐化公式
sharpened = np.clip(sharpened, 0, 1)

# --- 频域高提升滤波（High-Boost Filtering）---
# H_hbf = a - H_lp，其中 a >= 1
a = 1.5
H_hbf = ____________ - gaussian_lowpass(image.shape, cutoff=30)
highboost = frequency_filter(image, H_hbf)
```

**实验任务：**
- 对比理想高通（振铃效应）与高斯高通的边缘提取效果
- 调节锐化参数 `k`，观察欠锐化与过锐化
- 将频域高通结果与 Sobel 算子结果对比

---

### 步骤 4：二维 STFT 局部频率分析

**目标：** 对图像进行分块 STFT，分析不同区域的频率特征。

**知识点：** 2D STFT、窗口函数、局部频谱

```python
from scipy.signal import stft
from scipy.ndimage import zoom

def image_stft_2d(image, window_size=64, overlap=32):
    """
    对图像进行 2D STFT 分析
    思路：先对每行做 1D STFT，再对结果的每列做 1D STFT
    """
    # 方法一：逐行-逐列级联
    rows, cols = image.shape
    win = np.hanning(window_size)

    # TODO: 对图像的每一行做 1D STFT
    # 提示：可以使用 scipy.signal.stft 沿 axis=1
    f_row, t_row, Zxx_row = stft(image, fs=1.0, window=win,
                                  nperseg=____________,    # 填入窗口大小
                                  noverlap=____________,   # 填入重叠大小
                                  axis=____________)       # 填入处理轴

    # TODO: 对结果的频率轴再做 1D STFT（沿 axis=0）
    # 注意：需要对 Zxx_row 的实部和虚部分别处理
    f_col, t_col, Zxx_2d = stft(np.real(Zxx_row), fs=1.0, window=win,
                                 nperseg=____________,
                                 noverlap=____________,
                                 axis=____________)

    return f_row, t_row, f_col, t_col, Zxx_2d

# 方法二：分块分析（更直观）
def block_frequency_analysis(image, block_size=32):
    """将图像分块，对每块计算频谱能量分布"""
    rows, cols = image.shape
    energy_map = np.zeros((rows // block_size, cols // block_size))

    for i in range(energy_map.shape[0]):
        for j in range(energy_map.shape[1]):
            block = image[i*block_size:(i+1)*block_size,
                         j*block_size:(j+1)*block_size]
            # TODO: 对每个块做 2D FFT 并计算总能量
            F_block = np.fft.fft2(____________)      # 填入变量
            energy_map[i, j] = np.sum(np.abs(____________)**2)  # 填入变量

    return energy_map

# --- 实验 ---
image = img_as_float(data.camera())

# 分块频率能量分析
energy = block_frequency_analysis(image, block_size=32)

# TODO: 可视化原图与频率能量分布图的对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('原始图像')
axes[1].imshow(energy, cmap='____________')          # 填入合适的 colormap
axes[1].set_title('局部频率能量分布')
plt.savefig('step4_stft.png', dpi=150)
plt.show()
```

**实验任务：**
- 用 `astronaut` 图像对比面部区域与背景区域的频率特征
- 尝试不同窗口大小对分析结果的影响
- 将高频能量占比较高的区域标记出来

---

### 步骤 5：空间域卷积 vs 频域滤波对比

**目标：** 验证卷积定理，对比两种方法的结果一致性和运算速度。

**知识点：** 卷积定理、计算复杂度

```python
import time
from scipy.ndimage import gaussian_filter

image = img_as_float(data.camera())

# ====== 方法一：空间域高斯滤波 ======
t0 = time.time()
sigma = 5
result_spatial = gaussian_filter(image, sigma=sigma)
t_spatial = time.time() - t0

# ====== 方法二：频域高斯滤波 ======
t0 = time.time()
# TODO: 构造频域高斯滤波器
H = gaussian_lowpass(image.shape, cutoff=____________)  # cutoff ≈ M/(2π*sigma)
# TODO: 频域滤波
result_freq = frequency_filter(image, H)
t_freq = time.time() - t0

# ====== 方法三：scipy.signal.fftconvolve ======
t0 = time.time()
# TODO: 先生成空间域高斯核，再用 fftconvolve
kernel_size = 6 * sigma + 1 if (6*sigma) % 2 == 0 else int(6*sigma) + 1
x = np.arange(kernel_size) - kernel_size // 2
kernel = np.exp(-x**2 / (2 * sigma**2))
kernel_2d = np.outer(kernel, kernel)
kernel_2d /= kernel_2d.sum()
result_fftconv = ____________(image, kernel_2d, mode='____________')  # 填入函数和 mode
t_fftconv = time.time() - t0

# --- 结果对比 ---
print(f"空间域滤波耗时: {t_spatial:.4f}s")
print(f"频域滤波耗时:   {t_freq:.4f}s")
print(f"FFT卷积耗时:    {t_fftconv:.4f}s")

# TODO: 计算两种方法结果之间的最大像素差
diff = np.max(np.abs(result_spatial - result_freq))
print(f"结果最大差异:   {diff:.6e}")

# TODO: 可视化三种方法的结果和差异图
```

**实验任务：**
- 测试不同图像尺寸（可用 `skimage.transform.resize` 缩放）下三种方法的耗时，绘制耗时-尺寸曲线
- 改变滤波核大小，观察空间域方法耗时增长速度
- 验证卷积定理：`FFT(A*B) = FFT(A) · FFT(B)`

---

### 步骤 6：客观质量评估与综合对比

**目标：** 使用 PSNR 和 SSIM 评估去噪效果，生成对比表格。

**知识点：** PSNR、SSIM、图像质量评价

```python
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 数据准备 ---
image = img_as_float(data.camera())
noisy_gauss = add_gaussian_noise(image, sigma=0.1)
noisy_sp = add_salt_pepper_noise(image, amount=0.05)

# --- 各种去噪方法 ---
methods = {}

# 理想低通
H = ideal_lowpass(image.shape, cutoff=50)
methods['理想LPF(c=50)'] = frequency_filter(noisy_gauss, H)

# 巴特沃斯低通
H = butterworth_lowpass(image.shape, cutoff=50, order=2)
methods['巴特沃斯LPF'] = frequency_filter(noisy_gauss, H)

# 高斯低通
H = gaussian_lowpass(image.shape, cutoff=50)
methods['高斯LPF'] = frequency_filter(noisy_gauss, H)

# 不同截止频率的高斯低通
for c in [20, 100, 200]:
    H = gaussian_lowpass(image.shape, cutoff=c)
    methods[f'高斯LPF(c={c})'] = frequency_filter(noisy_gauss, H)

# --- 评估函数 ---
def evaluate(original, processed, noisy_label=""):
    """计算 PSNR 和 SSIM"""
    p = psnr(original, processed)
    s = ssim(original, processed, data_range=1.0)
    return p, s

# --- 生成对比表格 ---
print(f"{'方法':<20} {'PSNR(dB)':<12} {'SSIM':<10}")
print("-" * 42)

# TODO: 先计算含噪图像的指标作为基线
p_noisy, s_noisy = evaluate(image, noisy_gauss)
print(f"{'含噪图像':<20} {p_noisy:<12.2f} {s_noisy:<10.4f}")

for name, result in methods.items():
    # TODO: 计算每种方法的 PSNR 和 SSIM
    p, s = evaluate(image, ____________)          # 填入变量
    print(f"{name:<20} {p:<12.2f} {s:<10.4f}")

# --- 可视化对比 ---
n_methods = len(methods) + 2  # 原图 + 含噪 + 各方法
cols = 4
rows = (n_methods + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()

axes[0].imshow(image, cmap='gray')
axes[0].set_title('原图')
axes[1].imshow(noisy_gauss, cmap='gray')
axes[1].set_title(f'含噪\nPSNR={p_noisy:.1f}dB')

for idx, (name, result) in enumerate(methods.items()):
    p, s = evaluate(image, result)
    axes[idx+2].imshow(result, cmap='gray')
    axes[idx+2].set_title(f'{name}\nPSNR={p:.1f}dB SSIM={s:.3f}')

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig('step6_comparison.png', dpi=150)
plt.show()
```

**实验任务：**
- 对椒盐噪声重复上述实验，分析频域滤波对椒盐噪声的适用性
- 绘制 PSNR 随截止频率变化的曲线
- 尝试对彩色图像的 RGB 三通道分别进行频域滤波
- 综合分析：何时选择频域滤波优于空间域滤波？

---

## 5. 评分细则（满分 100 分）

| 评分项 | 分值 | 说明 |
|--------|------|------|
| **步骤 1：FFT 与频谱可视化** | 15 分 | 正确执行 FFT 和频谱中心化（5 分）；对数幅度谱计算正确（5 分）；可视化清晰、有标题和 colorbar（5 分） |
| **步骤 2：低通滤波去噪** | 20 分 | 三种滤波器实现正确（10 分）；截止频率对比实验完整（5 分）；滤波器 3D 可视化（5 分） |
| **步骤 3：高通滤波锐化** | 15 分 | 高通滤波器实现正确（5 分）；Unsharp Masking / High-Boost 实现正确（5 分）；边缘对比分析（5 分） |
| **步骤 4：2D STFT 分析** | 15 分 | STFT / 分块频谱分析实现正确（8 分）；不同区域频率特征分析（4 分）；可视化效果好（3 分） |
| **步骤 5：空间域 vs 频域对比** | 15 分 | 三种方法实现正确（6 分）；耗时对比与曲线绘制（5 分）；卷积定理验证（4 分） |
| **步骤 6：质量评估** | 10 分 | PSNR/SSIM 计算正确（4 分）；对比表格完整（3 分）；综合分析有深度（3 分） |
| **报告质量** | 10 分 | 结构清晰（3 分）；图表美观（3 分）；分析论述有深度（2 分）；排版规范（2 分） |

---

## 6. 提交要求

### 6.1 提交内容

| 文件 | 说明 |
|------|------|
| `report.pdf` | 实验报告（PDF 格式，建议 8–15 页） |
| `code/` | 所有 Python 源代码（含注释） |
| `results/` | 关键结果图片（PNG，≥150 dpi） |
| `README.md` | 项目说明与运行指引 |

### 6.2 报告结构

1. **封面**：项目名称、姓名、学号、日期
2. **实验目的**
3. **原理简述**（2D FFT、各类滤波器、卷积定理）
4. **实验过程与结果**（每个步骤独立一节，包含代码片段、结果图、分析）
5. **综合对比与讨论**
6. **总结与心得**
7. **参考文献**

### 6.3 代码规范

- 使用函数封装各步骤，避免重复代码
- 关键步骤添加中文注释
- 确保代码可直接运行（无未定义变量）

---

## 7. FAQ

**Q1：频域滤波后图像边缘出现亮边/暗边怎么办？**
A：这是频谱泄漏导致的。建议在 FFT 前对图像边缘做窗函数处理，或使用 `np.pad` 进行镜像填充。

**Q2：理想低通滤波器为什么会有振铃效应？**
A：理想低通在频域的矩形形状对应空间域的 sinc 函数，其振荡特性导致振铃。巴特沃斯和高斯滤波器通过平滑过渡缓解此问题。

**Q3：巴特沃斯滤波器阶数如何选择？**
A：阶数越高越接近理想滤波器（过渡带越窄），但振铃也越明显。一般 n=2~4 是常用范围。

**Q4：截止频率 `cutoff` 和空间域高斯 `sigma` 有什么对应关系？**
A：近似关系为 `cutoff ≈ M / (2π × sigma)`，其中 M 为图像尺寸。但实际对比时建议直接可视化两种方法的结果差异。

**Q5：STFT 部分一定要用 scipy.signal.stft 吗？**
A：不强制。可以自己实现分块 FFT（方法二），原理更清晰。使用 scipy.signal.stft 需要理解其参数含义。

**Q6：彩色图像如何做频域滤波？**
A：有两种方式：(1) 转灰度后处理；(2) 对 R/G/B 三通道分别做频域滤波再合并。推荐先做方案(1)，有余力再做(2)。

**Q7：运行速度很慢怎么办？**
A：确保使用 NumPy 向量化操作，避免 Python for 循环遍历像素。对于大图像可先缩放到 256×256 进行调试。

---

## 8. 参考资源

### 教材与文献
1. Gonzalez & Woods,《数字图像处理》(第4版)，第4章（频域滤波）
2. Oppenheim & Schafer,《离散时间信号处理》

### 在线教程
3. [NumPy FFT 文档](https://numpy.org/doc/stable/reference/routines.fft.html)
4. [scikit-image 教程](https://scikit-image.org/docs/stable/user_guide.html)
5. [SciPy 信号处理模块](https://docs.scipy.org/doc/scipy/reference/signal.html)

### 代码参考
6. scikit-image 频域滤波示例：`skimage.filters`
7. `scipy.signal.stft` / `scipy.signal.istft` 文档

### 工具
8. Matplotlib 色彩映射参考：[Choosing Colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
9. `skimage.metrics` 模块：PSNR 和 SSIM 计算

---

> 💡 **提示：** 本项目的核心是理解"频域视角看图像"。建议先在小图像（64×64）上手动验证 FFT、滤波、IFFT 的每一步，确认理解后再处理大图像。

---

> **课程关联**：本项目对应[[欢迎|智能信号处理课程]]W6知识模块。涉及知识点：[[知识点/经典信号处理/傅里叶分析|傅里叶分析]]（二维FFT）、[[知识点/经典信号处理/数字滤波器设计|数字滤波器设计]]（频域滤波器）、[[知识点/图像信号处理/图像去噪|图像去噪]]、[[知识点/图像信号处理/图像重建|图像重建]]。关联实验：[[实验3_图像去噪与增强|实验3]]。
