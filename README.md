# Icon Extractor

[English](#english) | [中文](#中文)

---

<a name="中文"></a>
## 中文

从合成图片中自动提取图标，转换为透明背景 PNG。**支持深色和浅色背景**。

### 特性

- **自动检测** - 智能识别深色/浅色背景
- **连通组件检测** - 自动识别图标区域
- **网格模式** - 支持规则排列的图标切割
- **边缘清理** - 处理抗锯齿边缘
- **一键安装** - 依赖自动安装

### 安装

```bash
git clone https://github.com/yueqianyun-cloud/icon-extractor-skill.git ~/.cursor/skills/icon-extractor
~/.cursor/skills/icon-extractor/install.sh
```

### 使用

```bash
# 自动检测背景
~/.cursor/skills/icon-extractor/extract 图片.png ./输出/

# 网格模式（规则排列）
~/.cursor/skills/icon-extractor/extract 图片.png ./输出/ -g 2x5

# 指定浅色背景
~/.cursor/skills/icon-extractor/extract 图片.png ./输出/ -b light
```

### 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `-b` | 背景类型: auto/dark/light | auto |
| `-g` | 网格模式: 行x列 | 无 |
| `-c` | 边缘清理 (0-3) | 2 |
| `-t` | 检测阈值 | 30 |
| `-m` | 最小像素数 | 500 |
| `-d` | 合并距离 | 20 |
| `-n` | 名称文件 | 无 |

---

<a name="english"></a>
## English

Extract icons from composite images with transparent background. **Supports both dark and light backgrounds**.

### Features

- **Auto Detection** - Smart dark/light background detection
- **Connected Components** - Auto icon region detection
- **Grid Mode** - Support regular icon grid cutting
- **Edge Cleaning** - Handle anti-aliased edges
- **One-click Install** - Auto dependency installation

### Install

```bash
git clone https://github.com/yueqianyun-cloud/icon-extractor-skill.git ~/.cursor/skills/icon-extractor
~/.cursor/skills/icon-extractor/install.sh
```

### Usage

```bash
# Auto detect background
~/.cursor/skills/icon-extractor/extract image.png ./output/

# Grid mode
~/.cursor/skills/icon-extractor/extract image.png ./output/ -g 2x5

# Light background
~/.cursor/skills/icon-extractor/extract image.png ./output/ -b light
```

### Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `-b` | Background: auto/dark/light | auto |
| `-g` | Grid mode: rowsxcols | None |
| `-c` | Edge clean (0-3) | 2 |
| `-t` | Threshold | 30 |
| `-m` | Min pixels | 500 |
| `-d` | Merge distance | 20 |
| `-n` | Names file | None |

---

## License

MIT
