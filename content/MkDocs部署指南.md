# MkDocs 静态网站部署指南

> **目标：** 将智能信号处理课程发布为静态网站，学生可通过浏览器访问  
> **技术栈：** MkDocs Material + GitHub Pages  
> **访问地址：** https://dvs-whu.github.io/isp-course/

---

## 🚀 快速开始

### 1. 本地预览

```bash
# 启动本地服务器
mkdocs serve

# 访问 http://127.0.0.1:8000
```

### 2. 构建站点

```bash
# 构建静态文件
mkdocs build --clean

# 生成的文件在 site/ 目录
```

### 3. 部署到GitHub Pages

```bash
# 提交更改
git add docs/ site/
git commit -m "docs: 更新课程内容"
git push

# GitHub Actions 会自动部署到 gh-pages 分支
```

---

## 📁 目录结构

```
智能信号处理/
├── mkdocs.yml              # MkDocs配置文件
├── docs/                   # 文档源文件
│   ├── index.md            # 首页
│   ├── stylesheets/
│   │   └── extra.css       # 自定义样式
│   ├── knowledge/          # 知识点
│   │   ├── classic/        # 经典信号处理
│   │   ├── time-freq/      # 时频分析
│   │   ├── statistical/    # 统计信号处理
│   │   ├── deep-learning/  # 深度学习
│   │   ├── image/          # 图像处理
│   │   ├── speech/         # 语音处理
│   │   └── event/          # 事件驱动
│   ├── projects/           # 项目
│   │   ├── project1/       # 项目一
│   │   └── project2/       # 项目二
│   ├── labs/               # 实验
│   └── resources/          # 资源
├── site/                   # 生成的静态文件（不要手动编辑）
└── scripts/
    └── sync-to-mkdocs.sh   # 从Obsidian同步内容
```

---

## 📝 更新内容流程

### 方式1：从Obsidian同步（推荐）

```bash
# 1. 在Obsidian中编辑内容
# 2. 运行同步脚本
./scripts/sync-to-mkdocs.sh

# 3. 本地预览
mkdocs serve

# 4. 提交并推送
git add docs/
git commit -m "docs: 更新XXX内容"
git push
```

### 方式2：直接编辑docs目录

```bash
# 1. 编辑 docs/ 目录下的文件
# 2. 本地预览
mkdocs serve

# 3. 提交并推送
git add docs/
git commit -m "docs: 更新XXX内容"
git push
```

---

## 🎨 自定义配置

### 修改主题颜色

编辑 `mkdocs.yml`：

```yaml
theme:
  palette:
    - scheme: default
      primary: indigo      # 主题色
      accent: indigo       # 强调色
```

可选颜色：red, pink, purple, deep purple, indigo, blue, light blue, cyan, teal, green, light green, lime, yellow, amber, orange, deep orange, brown, grey, blue grey

### 添加新页面

1. 在 `docs/` 目录创建新的 `.md` 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加页面引用

```yaml
nav:
  - 新页面: new-page.md
```

### 添加Mermaid图表

在Markdown文件中使用：

````markdown
```mermaid
graph LR
    A[信号] --> B[采样]
    B --> C[量化]
    C --> D[编码]
```
````

---

## 🔧 高级配置

### 添加数学公式支持

`mkdocs.yml` 已配置 `pymdownx.arithmatex`，在Markdown中使用：

```markdown
行内公式：$x^2 + y^2 = z^2$

块级公式：
$$
\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n
$$
```

### 添加代码高亮

````markdown
```python
import numpy as np
import matplotlib.pyplot as plt

# 生成信号
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 50 * t)

# 绘制频谱
plt.magnitude_spectrum(signal, Fs=1000)
plt.show()
```
````

### 添加Admonition提示框

```markdown
!!! note "注意"
    这是一个注意提示框

!!! warning "警告"
    这是一个警告提示框

!!! tip "提示"
    这是一个提示框
```

---

## 🌐 GitHub Pages 部署

### 1. 启用GitHub Pages

1. 进入仓库设置 (Settings)
2. 找到 "Pages" 选项
3. Source 选择 "Deploy from a branch"
4. Branch 选择 `gh-pages`，目录选择 `/ (root)`
5. 点击 "Save"

### 2. 配置自定义域名（可选）

1. 在仓库设置中添加自定义域名
2. 创建 `docs/CNAME` 文件，内容为你的域名
3. 配置DNS记录

### 3. 自动部署

`.github/workflows/deploy.yml` 已配置自动部署：
- 每次推送到 `main` 分支时自动构建
- 自动部署到 `gh-pages` 分支
- 约2-3分钟完成

---

## 📊 站点统计

当前站点：
- HTML文件：41个
- 总大小：6.7MB
- 知识点：30个
- 项目：2个
- 实验：6个

---

## 🛠️ 故障排除

### 问题：构建失败

```bash
# 清理并重新构建
mkdocs build --clean --verbose
```

### 问题：页面404

检查 `mkdocs.yml` 中的 `nav` 配置是否与实际文件路径一致。

### 问题：样式不生效

1. 清除浏览器缓存
2. 检查 `docs/stylesheets/extra.css` 是否存在
3. 确认 `mkdocs.yml` 中配置了 `extra_css`

### 问题：中文路径问题

确保文件名和路径使用UTF-8编码。

---

## 📚 相关资源

- [MkDocs文档](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown语法](https://www.markdownguide.org/basic-syntax/)
- [GitHub Pages文档](https://docs.github.com/en/pages)

---

*MkDocs部署指南 v1.0 · 2026-06-24*
