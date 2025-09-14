#!/usr/bin/env python3
"""
SAMURAI ONNX 视频追踪演示脚本

这个脚本提供了一个完整的视频目标追踪演示，支持：
- 指定输入视频路径
- 指定输出视频路径
- 指定初始边界框
- 实时进度显示
- 详细的性能统计

使用方法:
    python video_tracking_demo.py --input video.mp4 --output result.mp4 --bbox "100,100,200,150"
    python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150"  # 只保存结果文件
    python video_tracking_demo.py --help  # 查看所有选项

依赖:
    - onnxruntime
    - opencv-python
    - numpy
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from scripts.onnx_inference import SAMURAIONNXPredictor
except ImportError:
    print("❌ 无法导入 SAMURAIONNXPredictor")
    print("请确保 scripts/onnx_inference.py 文件存在")
    sys.exit(1)

class VideoTrackingDemo:
    """
    SAMURAI ONNX 视频追踪演示类
    
    提供完整的视频目标追踪功能，包括：
    - 视频读取和处理
    - 目标追踪
    - 结果可视化
    - 性能统计
    """
    
    def __init__(self, model_dir: str = "onnx_models", device: str = "cpu"):
        """
        初始化视频追踪演示
        
        Args:
            model_dir: ONNX模型文件目录
            device: 推理设备 ("cpu" 或 "cuda")
        """
        self.model_dir = model_dir
        self.device = device
        
        # 初始化追踪器
        print("🚀 初始化 SAMURAI ONNX 追踪器...")
        try:
            self.predictor = SAMURAIONNXPredictor(model_dir, device)
            print("✅ 追踪器初始化成功")
        except Exception as e:
            print(f"❌ 追踪器初始化失败: {e}")
            raise
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'processing_time': 0,
            'avg_fps': 0,
            'bbox_changes': 0,
            'confidence_scores': []
        }
    
    def validate_inputs(self, input_path: str, bbox: Tuple[int, int, int, int]) -> bool:
        """
        验证输入参数
        
        Args:
            input_path: 输入视频路径
            bbox: 初始边界框 (x, y, w, h)
            
        Returns:
            验证是否通过
        """
        # 检查视频文件
        if not os.path.exists(input_path):
            print(f"❌ 输入视频文件不存在: {input_path}")
            return False
        
        # 检查视频格式
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {input_path}")
            return False
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 视频信息:")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.1f} fps")
        print(f"   总帧数: {frame_count}")
        
        # 检查边界框
        x, y, w, h = bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            print(f"❌ 无效的边界框: {bbox}")
            return False
        
        if x + w > width or y + h > height:
            print(f"❌ 边界框超出视频范围: {bbox} (视频尺寸: {width}x{height})")
            return False
        
        print(f"🎯 初始边界框: ({x}, {y}, {w}, {h})")
        
        cap.release()
        return True
    
    def track_video(self, input_path: str, initial_bbox: Tuple[int, int, int, int],
                   output_path: Optional[str] = None, save_results: bool = True) -> Dict[str, Any]:
        """
        执行视频追踪
        
        Args:
            input_path: 输入视频路径
            initial_bbox: 初始边界框 (x, y, w, h)
            output_path: 输出视频路径（可选）
            save_results: 是否保存结果到文件
            
        Returns:
            追踪结果字典
        """
        # 验证输入
        if not self.validate_inputs(input_path, initial_bbox):
            raise ValueError("输入验证失败")
        
        # 打开视频
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化输出视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"📝 将保存输出视频到: {output_path}")
        
        # 追踪结果
        tracking_results = []
        confidence_scores = []
        
        # 开始追踪
        print(f"\n🎬 开始追踪视频...")
        start_time = time.time()
        
        frame_idx = 0
        prev_bbox = initial_bbox
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 执行追踪
                frame_start = time.time()
                mask, confidence, memory_features = self.predictor.predict_mask(frame, initial_bbox)
                frame_time = time.time() - frame_start
                
                # 从掩码更新边界框
                if mask.any():
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1, x2 = x_indices.min(), x_indices.max()
                        y1, y2 = y_indices.min(), y_indices.max()
                        
                        # 检查掩码范围是否过大
                        mask_area = np.sum(mask)
                        image_area = frame.shape[0] * frame.shape[1]
                        
                        if mask_area > image_area * 0.5:  # 如果掩码超过50%的图像
                            print(f"帧 {frame_idx+1}: 掩码过大 ({mask_area}/{image_area}), 保持原边界框")
                            current_bbox = prev_bbox
                        else:
                            # 添加边距
                            padding = 5
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(width - 1, x2 + padding)
                            y2 = min(height - 1, y2 + padding)
                            
                            current_bbox = (x1, y1, x2 - x1, y2 - y1)
                            
                            # 检查边界框变化
                            if current_bbox != prev_bbox:
                                self.stats['bbox_changes'] += 1
                                prev_bbox = current_bbox
                    else:
                        current_bbox = prev_bbox
                else:
                    current_bbox = prev_bbox
                
                # 保存结果
                tracking_results.append(current_bbox)
                confidence_scores.append(confidence)
                
                # 绘制结果
                if writer:
                    self._draw_tracking_info(frame, current_bbox, confidence, frame_idx, frame_time)
                    writer.write(frame)
                
                frame_idx += 1
                
                # 进度显示
                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    progress = frame_idx / total_frames * 100
                    elapsed = time.time() - start_time
                    current_fps = frame_idx / elapsed
                    print(f"   进度: {frame_idx}/{total_frames} ({progress:.1f}%) - {current_fps:.2f} fps")
                
            except Exception as e:
                print(f"⚠️  帧 {frame_idx} 处理失败: {e}")
                # 使用上一帧的结果
                if tracking_results:
                    tracking_results.append(tracking_results[-1])
                    confidence_scores.append(confidence_scores[-1])
                else:
                    tracking_results.append(initial_bbox)
                    confidence_scores.append(0.0)
                frame_idx += 1
        
        # 清理资源
        cap.release()
        if writer:
            writer.release()
        
        # 计算统计信息
        total_time = time.time() - start_time
        self.stats.update({
            'total_frames': len(tracking_results),
            'processing_time': total_time,
            'avg_fps': len(tracking_results) / total_time,
            'confidence_scores': confidence_scores
        })
        
        # 保存结果
        results = {
            'tracking_results': tracking_results,
            'confidence_scores': confidence_scores,
            'stats': self.stats,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames
            }
        }
        
        if save_results:
            self._save_results(input_path, results)
        
        return results
    
    def _draw_tracking_info(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                           confidence: float, frame_idx: int, frame_time: float):
        """
        在帧上绘制追踪信息
        
        Args:
            frame: 输入帧
            bbox: 边界框 (x, y, w, h)
            confidence: 置信度
            frame_idx: 帧索引
            frame_time: 处理时间
        """
        x, y, w, h = bbox
        
        # 绘制边界框
        color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # 绘制信息文本
        info_text = [
            f"Frame: {frame_idx}",
            f"Conf: {confidence:.3f}",
            f"Time: {frame_time*1000:.1f}ms",
            f"BBox: ({x},{y},{w},{h})"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _save_results(self, input_path: str, results: Dict[str, Any]):
        """
        保存追踪结果到文件
        
        Args:
            input_path: 输入视频路径
            results: 追踪结果
        """
        base_name = Path(input_path).stem
        
        # 保存边界框结果
        bbox_file = f"{base_name}_tracking_results.txt"
        with open(bbox_file, 'w') as f:
            for bbox in results['tracking_results']:
                f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
        print(f"📄 边界框结果保存到: {bbox_file}")
        
        # 保存详细结果
        json_file = f"{base_name}_tracking_details.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📊 详细结果保存到: {json_file}")
    
    def print_statistics(self, results: Dict[str, Any]):
        """
        打印追踪统计信息
        
        Args:
            results: 追踪结果
        """
        stats = results['stats']
        video_info = results['video_info']
        
        print(f"\n📊 追踪统计信息:")
        print(f"=" * 40)
        print(f"总帧数: {stats['total_frames']}")
        print(f"处理时间: {stats['processing_time']:.2f}秒")
        print(f"平均速度: {stats['avg_fps']:.2f} fps")
        print(f"边界框变化次数: {stats['bbox_changes']}")
        
        if stats['confidence_scores']:
            avg_conf = np.mean(stats['confidence_scores'])
            min_conf = np.min(stats['confidence_scores'])
            max_conf = np.max(stats['confidence_scores'])
            print(f"置信度统计:")
            print(f"  平均: {avg_conf:.3f}")
            print(f"  最小: {min_conf:.3f}")
            print(f"  最大: {max_conf:.3f}")
        
        print(f"\n视频信息:")
        print(f"  分辨率: {video_info['width']}x{video_info['height']}")
        print(f"  帧率: {video_info['fps']:.1f} fps")
        print(f"  总帧数: {video_info['total_frames']}")

def parse_bbox(bbox_str: str) -> Tuple[int, int, int, int]:
    """
    解析边界框字符串
    
    Args:
        bbox_str: 边界框字符串 "x,y,w,h"
        
    Returns:
        边界框元组 (x, y, w, h)
    """
    try:
        parts = bbox_str.split(',')
        if len(parts) != 4:
            raise ValueError("边界框格式错误")
        
        bbox = tuple(map(int, parts))
        if any(x < 0 for x in bbox):
            raise ValueError("边界框值不能为负数")
        
        return bbox
    except Exception as e:
        raise ValueError(f"边界框解析失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="SAMURAI ONNX 视频追踪演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本追踪
  python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150"
  
  # 保存输出视频
  python video_tracking_demo.py --input video.mp4 --output result.mp4 --bbox "100,100,200,150"
  
  # 使用GPU加速
  python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150" --device cuda
  
  # 指定模型目录
  python video_tracking_demo.py --input video.mp4 --bbox "100,100,200,150" --model_dir custom_models
        """
    )
    
    parser.add_argument("--input", "-i", required=True, 
                       help="输入视频文件路径")
    parser.add_argument("--bbox", "-b", required=True,
                       help="初始边界框，格式: 'x,y,w,h'")
    parser.add_argument("--output", "-o",
                       help="输出视频文件路径（可选）")
    parser.add_argument("--model_dir", "-m", default="onnx_models",
                       help="ONNX模型文件目录 (默认: onnx_models)")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"],
                       help="推理设备 (默认: cpu)")
    parser.add_argument("--no_save", action="store_true",
                       help="不保存结果文件")
    
    args = parser.parse_args()
    
    try:
        # 解析边界框
        initial_bbox = parse_bbox(args.bbox)
        
        # 检查输入文件
        if not os.path.exists(args.input):
            print(f"❌ 输入文件不存在: {args.input}")
            return 1
        
        # 检查模型目录
        if not os.path.exists(args.model_dir):
            print(f"❌ 模型目录不存在: {args.model_dir}")
            print("请确保ONNX模型文件已正确导出")
            return 1
        
        # 创建演示实例
        demo = VideoTrackingDemo(args.model_dir, args.device)
        
        # 执行追踪
        print(f"🎯 开始追踪视频: {args.input}")
        results = demo.track_video(
            args.input, 
            initial_bbox, 
            args.output, 
            save_results=not args.no_save
        )
        
        # 打印统计信息
        demo.print_statistics(results)
        
        print(f"\n✅ 追踪完成!")
        if args.output:
            print(f"📹 输出视频: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 追踪失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
