# SAMURAI ONNX 视频追踪演示使用指南

本指南介绍如何使用SAMURAI ONNX进行视频目标追踪的演示脚本。

## 📁 文件说明

### 主要演示文件

1. **`video_tracking_demo.py`** - 完整的视频追踪演示脚本
   - 支持完整的命令行参数
   - 详细的进度显示和统计信息
   - 自动保存结果文件
   - 适合生产环境使用

2. **`simple_tracking_example.py`** - 简化的追踪示例
   - 简单的API接口
   - 自动创建测试视频
   - 适合快速测试和学习

3. **`SAMURAI_ONNX_FINAL_DELIVERY.py`** - 核心追踪类
   - 包含完整的SAMURAITracker类
   - 可以直接导入使用
   - 支持单帧和视频追踪

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
pip install onnxruntime opencv-python numpy
```

### 2. 模型文件检查

确保 `onnx_models/` 目录包含必要的ONNX模型文件：
- `image_encoder_base_plus.onnx` (必需)
- `samurai_mock_end_to_end.onnx` (推荐)
- `samurai_lightweight.onnx` (可选)

### 3. 运行演示

#### 使用完整演示脚本
```bash
# 基本用法
python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150"

# 保存输出视频
python video_tracking_demo.py --input video.mp4 --output result.mp4 --bbox "100,100,200,150"

# 使用GPU加速
python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150" --device cuda

# 指定模型目录
python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150" --model_dir custom_models
```

#### 使用简化示例
```bash
# 运行演示模式（自动创建测试视频）
python simple_tracking_example.py --demo

# 追踪自定义视频
python simple_tracking_example.py --video your_video.mp4 --bbox "100,100,200,150"
```

## 📋 参数说明

### 边界框格式
边界框使用 `"x,y,w,h"` 格式：
- `x`: 左上角x坐标
- `y`: 左上角y坐标  
- `w`: 宽度
- `h`: 高度

示例：`"100,100,200,150"` 表示从(100,100)开始，宽200像素，高150像素的矩形区域。

### 命令行参数

#### video_tracking_demo.py
- `--input, -i`: 输入视频文件路径（必需）
- `--bbox, -b`: 初始边界框（必需）
- `--output, -o`: 输出视频文件路径（可选）
- `--model_dir, -m`: ONNX模型目录（默认：onnx_models）
- `--device, -d`: 推理设备 cpu/cuda（默认：cpu）
- `--no_save`: 不保存结果文件

#### simple_tracking_example.py
- `--video`: 输入视频文件路径
- `--bbox`: 初始边界框
- `--demo`: 运行演示模式

## 📊 输出文件

### 自动生成的文件

1. **边界框结果文件** (`*_tracking_results.txt`)
   ```
   100,100,200,150
   102,98,198,152
   105,95,195,155
   ...
   ```

2. **详细结果文件** (`*_tracking_details.json`)
   ```json
   {
     "tracking_results": [...],
     "confidence_scores": [...],
     "stats": {
       "total_frames": 90,
       "processing_time": 45.2,
       "avg_fps": 1.99,
       "bbox_changes": 12
     },
     "video_info": {
       "width": 640,
       "height": 480,
       "fps": 30.0,
       "total_frames": 90
     }
   }
   ```

3. **输出视频** (如果指定了 `--output`)
   - 包含追踪边界框的可视化视频
   - 显示帧数、置信度、处理时间等信息

## 🔧 编程接口

### 直接使用SAMURAITracker类

```python
from SAMURAI_ONNX_FINAL_DELIVERY import SAMURAITracker

# 初始化追踪器
tracker = SAMURAITracker("onnx_models", device="cpu")

# 单帧预测
mask, confidence = tracker.predict_single_frame(image, bbox)

# 视频追踪
results = tracker.track_video("video.mp4", initial_bbox, "output.mp4")
```

### 使用SAMURAIONNXPredictor类

```python
from scripts.onnx_inference import SAMURAIONNXPredictor

# 初始化预测器
predictor = SAMURAIONNXPredictor("onnx_models", device="cpu")

# 预测掩码
mask, confidence, memory_features = predictor.predict_mask(image, bbox)

# 视频追踪
results = predictor.track_video("video.mp4", initial_bbox, "output.mp4")
```

## 🎯 使用技巧

### 1. 边界框选择
- 选择包含完整目标的边界框
- 避免边界框过大或过小
- 确保边界框在视频范围内

### 2. 性能优化
- 使用GPU加速：`--device cuda`
- 对于长视频，考虑分段处理
- 监控内存使用情况

### 3. 结果分析
- 查看置信度分数判断追踪质量
- 分析边界框变化频率
- 检查输出视频的可视化效果

## 🐛 常见问题

### Q1: 模型文件缺失
**A**: 确保 `onnx_models/` 目录包含必要的模型文件，特别是 `image_encoder_base_plus.onnx`。

### Q2: 边界框超出范围
**A**: 检查边界框坐标是否在视频尺寸范围内，格式是否正确。

### Q3: 追踪效果不佳
**A**: 
- 尝试调整初始边界框
- 检查目标是否清晰可见
- 考虑使用更高质量的模型

### Q4: 处理速度慢
**A**:
- 使用GPU加速：`--device cuda`
- 检查模型文件是否正确加载
- 考虑使用轻量级模型

## 📈 性能参考

### 典型性能指标
- **CPU推理**: ~0.5-1.0 FPS
- **GPU推理**: ~2-5 FPS
- **内存使用**: ~2-4GB
- **模型大小**: ~264MB (图像编码器)

### 系统要求
- **Python**: >= 3.8
- **内存**: >= 4GB RAM
- **存储**: >= 1GB 可用空间
- **GPU**: 可选，支持CUDA 11.0+

## 🔄 更新日志

- **v1.0**: 初始版本，支持基本视频追踪
- **v1.1**: 添加GPU支持和性能优化
- **v1.2**: 改进错误处理和进度显示
- **v1.3**: 添加详细统计信息和结果保存

## 📞 技术支持

如果遇到问题，请检查：
1. 依赖是否正确安装
2. 模型文件是否完整
3. 输入参数是否正确
4. 系统资源是否充足

---

**注意**: 本演示脚本基于SAMURAI ONNX移植版本，确保您已正确完成模型导出步骤。
