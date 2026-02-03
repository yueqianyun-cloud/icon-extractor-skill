---
name: icon-extractor
description: 从图片中批量提取图标并转换为透明背景PNG。当用户需要裁剪图标、提取图标、切图、去除黑色背景时使用此工具。
---

# Icon Extractor - 图标提取工具

从合成图片中自动检测并提取独立图标，转换为透明背景 PNG。

## 一键安装

```bash
# 进入 skill 目录运行安装脚本
.cursor/skills/icon-extractor/install.sh
```

## 使用方法

```bash
# 简单调用
.cursor/skills/icon-extractor/extract <输入图片> <输出目录>

# 示例
.cursor/skills/icon-extractor/extract ./icons.png ./output/
```

## 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t` | 黑色检测阈值 | 30 |
| `-m` | 最小图标像素数 | 500 |
| `-d` | 区域合并距离 | 20 |
| `-p` | 裁剪边距 | 5 |
| `-n` | 图标名称文件 | 无 |

## 示例

```bash
# 基础用法
./extract icons.png ./output/

# 自定义参数
./extract icons.png ./output/ -t 50 -m 1000

# 指定名称
./extract icons.png ./output/ -n names.txt
```
