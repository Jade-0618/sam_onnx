#!/usr/bin/env python3
"""
SAMURAI ONNX 配置文件
自动生成的配置
"""

import os
from pathlib import Path

# 默认路径配置
DEFAULT_CONFIG = {
    # 输入视频路径
    "input_video": r"C:\Users\l\Videos\test_video.mp4",
    
    # 输出视频路径
    "output_video": r"C:\Users\l\Videos\test_video_tracked.mp4",
    
    # 默认边界框 (x, y, w, h)
    "default_bbox": (100, 100, 200, 150),
    
    # 模型目录
    "model_dir": "onnx_models",
    
    # 推理设备
    "device": "cpu",  # 或 "cuda"
    
    # 视频处理参数
    "save_results": True,
    "show_progress": True,
}

# 常用路径预设
COMMON_PATHS = {
    "desktop": str(Path.home() / "Desktop"),
    "documents": str(Path.home() / "Documents"),
    "project": str(Path(__file__).parent),
    "videos": str(Path.home() / "Videos"),
    "downloads": str(Path.home() / "Downloads"),
}

def get_input_path(filename=None):
    """获取输入视频路径"""
    if filename:
        return filename
    return DEFAULT_CONFIG["input_video"]

def get_output_path(input_path=None):
    """获取输出视频路径"""
    if input_path:
        input_file = Path(input_path)
        return str(input_file.parent / f"{input_file.stem}_tracked{input_file.suffix}")
    return DEFAULT_CONFIG["output_video"]

def get_bbox():
    """获取默认边界框"""
    return DEFAULT_CONFIG["default_bbox"]

def print_config():
    """打印当前配置"""
    print("当前配置:")
    print(f"  输入视频: {DEFAULT_CONFIG['input_video']}")
    print(f"  输出视频: {DEFAULT_CONFIG['output_video']}")
    print(f"  默认边界框: {DEFAULT_CONFIG['default_bbox']}")
    print(f"  模型目录: {DEFAULT_CONFIG['model_dir']}")
    print(f"  推理设备: {DEFAULT_CONFIG['device']}")

if __name__ == "__main__":
    print_config()
