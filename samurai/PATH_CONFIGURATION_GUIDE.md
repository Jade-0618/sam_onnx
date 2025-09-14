# SAMURAI ONNX 路径配置指南

## 🎯 配置输入输出视频路径的方法

### 方法1：命令行参数（最简单）

```bash
# 激活环境
conda activate samurai

# 直接指定路径
python video_tracking_demo.py --input "C:\Users\YourName\Videos\my_video.mp4" --output "C:\Users\YourName\Videos\result.mp4" --bbox "100,100,200,150"
```

### 方法2：使用配置文件（推荐）

#### 步骤1：运行路径设置脚本
```bash
python setup_paths.py
```

#### 步骤2：按提示输入路径
```
SAMURAI ONNX 路径设置
==============================

请输入您的视频路径:
示例: C:\Users\YourName\Videos\my_video.mp4
或者: D:\Videos\input.mp4
输入视频路径: C:\Users\YourName\Videos\my_video.mp4

自动生成的输出路径: C:\Users\YourName\Videos\my_video_tracked.mp4
是否修改输出路径? (y/n): n

请输入初始边界框 (x, y, w, h):
示例: 100,100,200,150
边界框: 100,100,200,150

✅ 配置已保存到: config.py
```

#### 步骤3：直接运行追踪
```bash
python track_with_config.py
```

### 方法3：手动编辑配置文件

编辑 `config.py` 文件：

```python
DEFAULT_CONFIG = {
    # 修改这里的路径
    "input_video": r"C:\Users\YourName\Videos\my_video.mp4",
    "output_video": r"C:\Users\YourName\Videos\result.mp4",
    
    # 修改边界框 (x, y, w, h)
    "default_bbox": (100, 100, 200, 150),
    
    # 其他设置
    "device": "cpu",  # 或 "cuda"
}
```

## 📁 路径示例

### Windows路径示例
```bash
# 桌面上的视频
--input "C:\Users\YourName\Desktop\video.mp4"

# D盘Videos文件夹
--input "D:\Videos\input.mp4"

# 项目文件夹
--input "C:\Users\YourName\Desktop\py_for_ma\samurai\my_video.mp4"
```

### 相对路径示例
```bash
# 当前目录下的视频
--input "my_video.mp4"

# 上级目录的视频
--input "../videos/input.mp4"

# 子目录的视频
--input "videos/input.mp4"
```

## 🎯 边界框格式

边界框使用 `"x,y,w,h"` 格式：
- `x`: 左上角x坐标
- `y`: 左上角y坐标
- `w`: 宽度
- `h`: 高度

### 示例
```bash
# 从(100,100)开始，宽200像素，高150像素
--bbox "100,100,200,150"

# 从(50,50)开始，宽100像素，高100像素
--bbox "50,50,100,100"
```

## 🚀 快速开始

### 1. 最简单的使用方式
```bash
# 激活环境
conda activate samurai

# 设置路径
python setup_paths.py

# 运行追踪
python track_with_config.py
```

### 2. 命令行方式
```bash
# 激活环境
conda activate samurai

# 直接指定路径
python video_tracking_demo.py --input "your_video.mp4" --bbox "100,100,200,150"
```

### 3. 查看当前配置
```bash
python track_with_config.py --config
```

## 🔧 常见问题

### Q: 路径包含空格怎么办？
**A:** 使用引号包围路径：
```bash
python video_tracking_demo.py --input "C:\Users\My Name\Videos\my video.mp4"
```

### Q: 如何批量处理多个视频？
**A:** 可以编写批处理脚本：
```python
import os
from track_with_config import track_video_with_config

video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video in video_files:
    track_video_with_config(input_path=video)
```

### Q: 输出路径可以自动生成吗？
**A:** 可以，如果不指定输出路径，会自动在输入文件同目录生成：
```
输入: C:\Videos\input.mp4
输出: C:\Videos\input_tracked.mp4
```

### Q: 如何修改默认配置？
**A:** 编辑 `config.py` 文件中的 `DEFAULT_CONFIG` 字典。

## 📊 配置选项说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `input_video` | 输入视频路径 | `"C:\Videos\input.mp4"` |
| `output_video` | 输出视频路径 | `"C:\Videos\output.mp4"` |
| `default_bbox` | 默认边界框 | `(100, 100, 200, 150)` |
| `device` | 推理设备 | `"cpu"` 或 `"cuda"` |
| `model_dir` | 模型目录 | `"onnx_models"` |

## 🎉 总结

现在您有三种方式配置路径：

1. **命令行参数** - 最灵活，适合临时使用
2. **配置文件** - 最方便，适合重复使用
3. **路径设置脚本** - 最友好，适合初学者

选择最适合您的方式开始使用吧！
