# SAMURAI ONNX 快速开始指南

## 🚀 环境设置

### 方法1: 使用自动设置脚本

**Windows用户:**
```bash
# 双击运行或在命令行执行
setup_environment.bat
```

**Linux/Mac用户:**
```bash
# 给脚本执行权限并运行
chmod +x setup_environment.sh
./setup_environment.sh
```

### 方法2: 手动设置

```bash
# 1. 创建conda环境
conda create -n samurai python=3.10 -y

# 2. 激活环境
conda activate samurai

# 3. 安装依赖
pip install onnxruntime opencv-python numpy matplotlib pandas scipy loguru
```

### 方法3: 使用environment.yml

```bash
# 使用conda环境文件创建环境
conda env create -f environment.yml
conda activate samurai
```

## 🎯 快速测试

### 1. 测试环境
```bash
conda activate samurai
python test_demo.py
```

### 2. 运行简单演示
```bash
conda activate samurai
python simple_tracking_example.py --demo
```

### 3. 追踪自定义视频
```bash
conda activate samurai
python video_tracking_demo.py --input your_video.mp4 --bbox "100,100,200,150"
```

## 📁 文件说明

### 核心文件
- `video_tracking_demo.py` - 完整的视频追踪演示
- `simple_tracking_example.py` - 简化的追踪示例
- `SAMURAI_ONNX_FINAL_DELIVERY.py` - 核心追踪类

### 环境文件
- `environment.yml` - Conda环境配置文件
- `setup_environment.bat` - Windows环境设置脚本
- `setup_environment.sh` - Linux/Mac环境设置脚本

### 文档文件
- `DEMO_USAGE_GUIDE.md` - 详细使用指南
- `QUICK_START.md` - 快速开始指南（本文件）

## 🔧 常见问题

### Q: 环境激活失败
**A:** 确保已安装Anaconda或Miniconda，并且conda命令在PATH中。

### Q: 包安装失败
**A:** 尝试使用国内镜像源：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime opencv-python numpy
```

### Q: 模型文件缺失
**A:** 确保 `onnx_models/` 目录包含必要的模型文件：
- `image_encoder_base_plus.onnx` (必需)
- `samurai_mock_end_to_end.onnx` (推荐)

### Q: 编码错误
**A:** 如果遇到Unicode编码错误，可以设置环境变量：
```bash
# Windows
set PYTHONIOENCODING=utf-8

# Linux/Mac
export PYTHONIOENCODING=utf-8
```

## 📊 性能参考

- **CPU推理**: ~0.5-1.0 FPS
- **GPU推理**: ~2-5 FPS (需要CUDA支持)
- **内存使用**: ~2-4GB
- **模型大小**: ~264MB (图像编码器)

## 🎉 开始使用

环境设置完成后，您可以：

1. **运行演示**: `python simple_tracking_example.py --demo`
2. **追踪视频**: `python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150"`
3. **查看帮助**: `python video_tracking_demo.py --help`

---

**注意**: 确保您的系统满足最低要求（Python 3.10+, 4GB+ RAM, 1GB+ 存储空间）。
