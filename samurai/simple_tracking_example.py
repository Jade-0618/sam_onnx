#!/usr/bin/env python3
"""
SAMURAI ONNX 简单追踪示例

这是一个简化的示例脚本，展示如何使用SAMURAI ONNX进行视频追踪。
适合快速测试和集成到其他项目中。

使用方法:
    python simple_tracking_example.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from scripts.onnx_inference import SAMURAIONNXPredictor
except ImportError:
    print("❌ 无法导入 SAMURAIONNXPredictor")
    print("请确保 scripts/onnx_inference.py 文件存在")
    sys.exit(1)

def create_test_video(output_path: str, duration: int = 3):
    """创建一个简单的测试视频"""
    print(f"创建测试视频: {output_path}")
    
    width, height = 640, 480
    fps = 30
    total_frames = duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # 创建背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # 深灰色背景
        
        # 添加一些噪声
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # 移动的目标（绿色圆圈）
        t = frame_idx / total_frames
        center_x = int(100 + (width - 200) * t)  # 从左到右移动
        center_y = int(height // 2 + 50 * np.sin(2 * np.pi * t * 2))  # 上下摆动
        radius = 25
        
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), -1)
        
        # 添加一些干扰物
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
        cv2.rectangle(frame, (width-100, height-100), (width-50, height-50), (255, 0, 0), -1)
        
        writer.write(frame)
    
    writer.release()
    
    # 返回初始边界框（绿色圆圈的位置）
    initial_bbox = (100 - radius, height // 2 - radius, 2 * radius, 2 * radius)
    return initial_bbox

def simple_tracking_demo():
    """简单的追踪演示"""
    print("SAMURAI ONNX 简单追踪演示")
    print("=" * 40)
    
    # 创建测试视频
    test_video = "test_video.mp4"
    output_video = "tracking_result.mp4"
    
    if not os.path.exists(test_video):
        initial_bbox = create_test_video(test_video)
    else:
        print(f"使用现有测试视频: {test_video}")
        initial_bbox = (75, 205, 50, 50)  # 默认边界框
    
    print(f"初始边界框: {initial_bbox}")
    
    # 检查模型文件
    model_dir = "onnx_models"
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        print("请先运行模型导出脚本")
        return
    
    # 初始化追踪器
    print("初始化追踪器...")
    try:
        predictor = SAMURAIONNXPredictor(model_dir, device="cpu")
        print("追踪器初始化成功")
    except Exception as e:
        print(f"追踪器初始化失败: {e}")
        return
    
    # 执行追踪
    print("开始追踪...")
    try:
        results = predictor.track_video(test_video, initial_bbox, output_video)
        print(f"追踪完成! 处理了 {len(results)} 帧")
        
        # 保存结果
        with open("tracking_results.txt", 'w') as f:
            for bbox in results:
                f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
        print("结果已保存到 tracking_results.txt")
        
        if os.path.exists(output_video):
            print(f"输出视频已保存到: {output_video}")
        
    except Exception as e:
        print(f"追踪失败: {e}")
        import traceback
        traceback.print_exc()

def track_custom_video(video_path: str, bbox_str: str):
    """追踪自定义视频"""
    print(f"追踪自定义视频: {video_path}")
    
    # 解析边界框
    try:
        bbox_parts = bbox_str.split(',')
        if len(bbox_parts) != 4:
            raise ValueError("边界框格式错误")
        initial_bbox = tuple(map(int, bbox_parts))
    except Exception as e:
        print(f"边界框解析失败: {e}")
        return
    
    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return
    
    # 初始化追踪器
    model_dir = "onnx_models"
    try:
        predictor = SAMURAIONNXPredictor(model_dir, device="cpu")
    except Exception as e:
        print(f"追踪器初始化失败: {e}")
        return
    
    # 执行追踪
    output_video = f"{Path(video_path).stem}_tracked.mp4"
    try:
        results = predictor.track_video(video_path, initial_bbox, output_video)
        print(f"追踪完成! 处理了 {len(results)} 帧")
        print(f"输出视频: {output_video}")
        
        # 保存结果
        results_file = f"{Path(video_path).stem}_results.txt"
        with open(results_file, 'w') as f:
            for bbox in results:
                f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
        print(f"结果文件: {results_file}")
        
    except Exception as e:
        print(f"追踪失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAMURAI ONNX 简单追踪示例")
    parser.add_argument("--video", help="输入视频文件路径")
    parser.add_argument("--bbox", help="初始边界框 'x,y,w,h'")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    
    args = parser.parse_args()
    
    if args.video and args.bbox:
        # 追踪自定义视频
        track_custom_video(args.video, args.bbox)
    else:
        # 运行演示
        simple_tracking_demo()

if __name__ == "__main__":
    main()
