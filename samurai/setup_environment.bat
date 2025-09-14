@echo off
echo 🚀 SAMURAI ONNX 环境设置脚本
echo ================================

echo 📋 检查conda是否安装...
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda未安装或未添加到PATH
    echo 请先安装Anaconda或Miniconda
    pause
    exit /b 1
)

echo ✅ Conda已安装

echo 📋 创建samurai环境...
conda create -n samurai python=3.10 -y
if %errorlevel% neq 0 (
    echo ❌ 环境创建失败
    pause
    exit /b 1
)

echo ✅ 环境创建成功

echo 📋 激活环境并安装依赖...
call conda activate samurai
if %errorlevel% neq 0 (
    echo ❌ 环境激活失败
    pause
    exit /b 1
)

echo 📦 安装Python包...
pip install onnxruntime opencv-python numpy matplotlib pandas scipy loguru
if %errorlevel% neq 0 (
    echo ❌ 包安装失败
    pause
    exit /b 1
)

echo ✅ 所有依赖安装完成

echo 📋 测试环境...
python -c "import onnxruntime; import cv2; import numpy; print('✅ 环境测试通过!')"
if %errorlevel% neq 0 (
    echo ❌ 环境测试失败
    pause
    exit /b 1
)

echo.
echo 🎉 SAMURAI ONNX 环境设置完成!
echo.
echo 使用方法:
echo   conda activate samurai
echo   python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150"
echo.
pause
