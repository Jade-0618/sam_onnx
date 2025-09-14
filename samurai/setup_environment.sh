#!/bin/bash

echo "🚀 SAMURAI ONNX 环境设置脚本"
echo "================================"

# 检查conda是否安装
echo "📋 检查conda是否安装..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装或未添加到PATH"
    echo "请先安装Anaconda或Miniconda"
    exit 1
fi

echo "✅ Conda已安装"

# 创建环境
echo "📋 创建samurai环境..."
conda create -n samurai python=3.10 -y
if [ $? -ne 0 ]; then
    echo "❌ 环境创建失败"
    exit 1
fi

echo "✅ 环境创建成功"

# 激活环境并安装依赖
echo "📋 激活环境并安装依赖..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate samurai
if [ $? -ne 0 ]; then
    echo "❌ 环境激活失败"
    exit 1
fi

echo "📦 安装Python包..."
pip install onnxruntime opencv-python numpy matplotlib pandas scipy loguru
if [ $? -ne 0 ]; then
    echo "❌ 包安装失败"
    exit 1
fi

echo "✅ 所有依赖安装完成"

# 测试环境
echo "📋 测试环境..."
python -c "import onnxruntime; import cv2; import numpy; print('✅ 环境测试通过!')"
if [ $? -ne 0 ]; then
    echo "❌ 环境测试失败"
    exit 1
fi

echo ""
echo "🎉 SAMURAI ONNX 环境设置完成!"
echo ""
echo "使用方法:"
echo "  conda activate samurai"
echo "  python video_tracking_demo.py --input video.mp4 --bbox '100,100,200,150'"
echo ""
