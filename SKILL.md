---
name: icon-extractor
description: 从图片中批量提取图标并转换为透明背景PNG。自动检测深色/浅色背景。当用户需要裁剪图标、提取图标、切图、去除背景时使用。
---

# Icon Extractor - 图标提取工具

从合成图片中自动检测并提取独立图标，转换为透明背景 PNG。**支持深色和浅色背景**。

## 一键安装

```bash
.cursor/skills/icon-extractor/install.sh
```

## 使用方法

```bash
# 自动检测背景，智能提取
.cursor/skills/icon-extractor/extract 图片.png ./输出/

# 网格模式（规则排列的图标）
.cursor/skills/icon-extractor/extract 图片.png ./输出/ -g 2x5

# 指定背景类型
.cursor/skills/icon-extractor/extract 图片.png ./输出/ -b light
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-b` | 背景类型: auto/dark/light | auto |
| `-g` | 网格模式: 行x列 (如 2x5) | 无 |
| `-c` | 边缘清理强度 (0-3) | 2 |
| `-t` | 黑色检测阈值 | 30 |
| `-m` | 最小图标像素数 | 500 |
| `-d` | 区域合并距离 | 20 |
| `-n` | 图标名称文件 | 无 |

## 示例

```bash
# 黑色背景（自动检测）
./extract dark_bg.png ./output/

# 浅色背景 + 网格切割
./extract light_bg.png ./output/ -g 2x5

# 指定名称
./extract icons.png ./output/ -n names.txt
```
