#!/bin/bash
# Icon Extractor - 一键安装脚本

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SKILL_DIR/.venv"

echo "=== Icon Extractor 安装 ==="

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

# 激活虚拟环境并安装依赖
echo "安装依赖..."
source "$VENV_DIR/bin/activate"
pip install -q Pillow

# 设置执行权限
chmod +x "$SKILL_DIR/extract"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "使用方法："
echo "  $SKILL_DIR/extract <输入图片> <输出目录>"
echo ""
echo "示例："
echo "  $SKILL_DIR/extract ./icons.png ./output/"
echo ""
