# Icon Extractor

[English](#english) | [中文](#中文)

---

<a name="中文"></a>
## 中文

从合成图片中自动提取图标，转换为透明背景 PNG。

### 特性

- 自动检测图标区域（连通组件算法）
- 智能合并相邻区域（处理带标签的图标）
- 黑色背景自动转透明
- 一键安装，简单调用

### 安装

```bash
# 克隆仓库
git clone https://github.com/yueqianyun-cloud/icon-extractor-skill.git ~/.cursor/skills/icon-extractor

# 一键安装依赖
~/.cursor/skills/icon-extractor/install.sh
```

### 使用

```bash
# 提取图标
~/.cursor/skills/icon-extractor/extract 输入图片.png ./输出目录/
```

### 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `-t` | 黑色检测阈值 (0-255) | 30 |
| `-m` | 最小图标像素数 | 500 |
| `-d` | 区域合并距离 | 20 |
| `-p` | 裁剪边距 | 5 |
| `-n` | 名称文件（每行一个） | 无 |

### 示例

```bash
# 基础用法
./extract icons.png ./output/

# 处理小图标
./extract icons.png ./output/ -m 200

# 指定图标名称
./extract icons.png ./output/ -n names.txt
```

---

<a name="english"></a>
## English

Auto-extract icons from composite images with transparent background.

### Features

- Auto-detect icon regions (connected component algorithm)
- Smart merge adjacent regions (handles icons with labels)
- Black background to transparent
- One-click install, simple usage

### Install

```bash
# Clone
git clone https://github.com/yueqianyun-cloud/icon-extractor-skill.git ~/.cursor/skills/icon-extractor

# Install dependencies
~/.cursor/skills/icon-extractor/install.sh
```

### Usage

```bash
~/.cursor/skills/icon-extractor/extract input.png ./output/
```

### Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `-t` | Black threshold (0-255) | 30 |
| `-m` | Min icon pixels | 500 |
| `-d` | Merge distance | 20 |
| `-p` | Crop padding | 5 |
| `-n` | Names file | None |

---

## License

MIT
