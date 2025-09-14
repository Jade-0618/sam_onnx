# SAMURAI ONNX 移植指南

本指南详细介绍如何将SAMURAI项目移植到ONNX，实现高效的跨平台推理。

## 概述

SAMURAI是基于SAM2的零样本视觉跟踪系统，主要挑战在于：
- 复杂的状态管理（内存银行、Kalman滤波器）
- 动态形状处理
- 跨帧依赖关系

## 移植策略

### 1. 模块化分解

将SAMURAI分解为独立的ONNX模型：

```
SAMURAI System
├── Image Encoder (✅ 已实现)
├── Prompt Encoder (🔄 部分实现)  
├── Mask Decoder (🔄 部分实现)
├── Memory Encoder (❌ 复杂)
└── Kalman Filter (✅ 纯Python实现)
```

### 2. 核心组件

#### 图像编码器 (Image Encoder)
- **状态**: ✅ 已完成
- **输入**: `[B, 3, H, W]` 图像张量
- **输出**: 多尺度特征图
- **ONNX文件**: `image_encoder_{model_size}.onnx`

#### 提示编码器 (Prompt Encoder)
- **状态**: 🔄 基础实现
- **输入**: 点坐标、标签、边界框、掩码
- **输出**: 稀疏和密集嵌入
- **ONNX文件**: `prompt_encoder_{model_size}.onnx`

#### 掩码解码器 (Mask Decoder)
- **状态**: 🔄 需要优化
- **输入**: 图像特征 + 提示嵌入
- **输出**: 掩码预测 + IoU分数
- **ONNX文件**: `mask_decoder_{model_size}.onnx`

#### Kalman滤波器
- **状态**: ✅ 纯Python实现
- **功能**: 目标状态预测和更新
- **实现**: `KalmanFilterONNX` 类

## 使用方法

### 1. 导出ONNX模型

```bash
# 导出所有组件
python scripts/export_onnx.py --components all --model_name base_plus

# 导出特定组件
python scripts/export_onnx.py --components image_encoder prompt_encoder --model_name base_plus

# 启用优化
python scripts/export_onnx.py --components all --optimize --dynamic_batch
```

### 2. ONNX推理

```bash
# 基本推理
python scripts/onnx_inference.py \
    --video_path demo.mp4 \
    --bbox "100,100,200,150" \
    --model_dir onnx_models

# GPU推理
python scripts/onnx_inference.py \
    --video_path demo.mp4 \
    --bbox "100,100,200,150" \
    --device cuda \
    --output_video output.mp4
```

### 3. Python API

```python
from scripts.onnx_inference import SAMURAIONNXPredictor

# 初始化预测器
predictor = SAMURAIONNXPredictor("onnx_models", device="cpu")

# 跟踪视频
results = predictor.track_video("video.mp4", (x, y, w, h))
```

## 性能优化

### 1. 模型优化

```python
# 启用ONNX优化
python scripts/export_onnx.py --optimize

# 使用TensorRT (NVIDIA GPU)
import onnxruntime as ort
providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
```

### 2. 推理优化

- **批处理**: 支持多帧并行处理
- **内存管理**: 固定大小的内存银行
- **精度**: FP16推理（GPU）

### 3. 部署优化

```python
# 量化模型
from onnxruntime.quantization import quantize_dynamic
quantize_dynamic(input_model, output_model)

# 图优化
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

## 限制和解决方案

### 1. 内存银行机制
**问题**: 动态内存管理难以导出
**解决方案**: 
- 固定大小的循环缓冲区
- 简化的内存更新策略

### 2. Flash Attention
**问题**: 可能不完全兼容ONNX
**解决方案**:
- 回退到标准注意力机制
- 使用ONNX兼容的注意力实现

### 3. 动态形状
**问题**: 不同视频分辨率
**解决方案**:
- 固定输入尺寸 + 预处理
- 动态轴配置

## 测试和验证

### 1. 精度验证

```bash
# 比较PyTorch vs ONNX结果
python scripts/validate_onnx.py \
    --pytorch_model configs/samurai/sam2.1_hiera_b+.yaml \
    --onnx_models onnx_models \
    --test_video test.mp4
```

### 2. 性能基准

```bash
# 性能测试
python scripts/benchmark_onnx.py \
    --model_dir onnx_models \
    --device cuda \
    --batch_sizes 1,4,8
```

## 依赖安装

```bash
# 基础依赖
pip install onnxruntime opencv-python numpy

# GPU支持
pip install onnxruntime-gpu

# 优化工具
pip install onnx onnxoptimizer

# TensorRT (可选)
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## 文件结构

```
samurai/
├── scripts/
│   ├── export_onnx.py          # ONNX导出脚本
│   ├── onnx_inference.py       # ONNX推理引擎
│   ├── validate_onnx.py        # 精度验证 (待实现)
│   └── benchmark_onnx.py       # 性能测试 (待实现)
├── onnx_models/                # 导出的ONNX模型
│   ├── image_encoder_base_plus.onnx
│   ├── prompt_encoder_base_plus.onnx
│   └── mask_decoder_base_plus.onnx
└── ONNX_MIGRATION_GUIDE.md     # 本指南
```

## 下一步计划

1. **完善掩码解码器导出** - 解决复杂输入输出结构
2. **实现内存编码器** - 设计状态无关的版本
3. **端到端优化** - 整合所有组件
4. **移动端部署** - ONNX.js 或 NCNN 支持
5. **量化和压缩** - 减少模型大小

## 常见问题

### Q: 为什么不导出完整模型？
A: SAMURAI包含复杂的状态管理和跨帧依赖，分模块导出更灵活且易于优化。

### Q: 性能提升如何？
A: 预期CPU推理提升2-3倍，GPU推理提升1.5-2倍（取决于硬件）。

### Q: 精度损失多少？
A: 理论上无损失，实际可能有微小差异（<1% mAP）。

### Q: 支持哪些平台？
A: Windows、Linux、macOS，以及移动端（通过ONNX.js或专用运行时）。

## 贡献

欢迎提交Issue和PR来改进ONNX移植：
- 性能优化建议
- 新平台支持
- Bug修复
- 文档改进

## 许可证

遵循原SAMURAI项目的许可证条款。
