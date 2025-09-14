#!/usr/bin/env python3
"""
修复版本的SAMURAI追踪器
添加了调试信息和更好的边界框处理
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from SAMURAI_ONNX_FINAL_DELIVERY import SAMURAITracker

class FixedSAMURAITracker(SAMURAITracker):
    """修复版本的SAMURAI追踪器"""
    
    def __init__(self, model_dir: str = "onnx_models", device: str = "cpu", debug: bool = False):
        super().__init__(model_dir, device)
        self.debug = debug
        self.bbox_history = []
    
    def predict_single_frame(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """预测单帧，添加调试信息"""
        
        if self.debug:
            print(f"预测帧: 图像尺寸={image.shape}, 边界框={bbox}")
        
        # 调用父类方法
        mask, confidence = super().predict_single_frame(image, bbox)
        
        if self.debug:
            print(f"掩码统计: 形状={mask.shape}, 非零像素={np.sum(mask)}, 置信度={confidence:.3f}")
            
            # 分析掩码
            if mask.any():
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    print(f"掩码范围: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        
        return mask, confidence
    
    def track_video(self, video_path: str, initial_bbox: Tuple[int, int, int, int], 
                   output_path: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
        """修复版本的视频追踪"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 转换初始边界框格式
        x, y, w, h = initial_bbox
        current_bbox = (x, y, x + w, y + h)  # 转换为 (x1, y1, x2, y2)
        
        results = []
        frame_idx = 0
        start_time = time.time()
        
        print(f"开始跟踪，初始边界框: {initial_bbox}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 预测掩码
                mask, confidence = self.predict_single_frame(frame, current_bbox)
                
                # 从掩码更新边界框
                if mask.any():
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1, x2 = x_indices.min(), x_indices.max()
                        y1, y2 = y_indices.min(), y_indices.max()
                        
                        # 添加边距
                        padding = 5
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(width - 1, x2 + padding)
                        y2 = min(height - 1, y2 + padding)
                        
                        new_bbox = (x1, y1, x2, y2)
                        
                        if self.debug:
                            print(f"帧 {frame_idx+1}: 边界框变化 {current_bbox} -> {new_bbox}")
                        
                        current_bbox = new_bbox
                    else:
                        if self.debug:
                            print(f"帧 {frame_idx+1}: 掩码为空，保持边界框")
                else:
                    if self.debug:
                        print(f"帧 {frame_idx+1}: 掩码为空，保持边界框")
                
                # 转换回 (x, y, w, h) 格式
                x1, y1, x2, y2 = current_bbox
                result_bbox = (x1, y1, x2 - x1, y2 - y1)
                results.append(result_bbox)
                self.bbox_history.append(result_bbox)
                
                # 绘制结果
                if writer:
                    # 绘制边界框
                    color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制信息
                    cv2.putText(frame, f"Frame {frame_idx+1}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"BBox: ({x1},{y1},{x2-x1},{y2-y1})", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    writer.write(frame)
                
                frame_idx += 1
                
                # 进度显示
                if frame_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_idx / elapsed
                    progress = frame_idx / total_frames * 100
                    print(f"   进度: {frame_idx}/{total_frames} ({progress:.1f}%) - {fps_current:.2f} fps")
                    
            except Exception as e:
                print(f"帧 {frame_idx} 处理失败: {e}")
                # 使用上一帧的边界框
                if results:
                    results.append(results[-1])
                else:
                    results.append(initial_bbox)
                frame_idx += 1
        
        cap.release()
        if writer:
            writer.release()
        
        # 统计结果
        elapsed_total = time.time() - start_time
        avg_fps = len(results) / elapsed_total
        
        print(f"跟踪完成!")
        print(f"   总帧数: {len(results)}")
        print(f"   总时间: {elapsed_total:.1f}s")
        print(f"   平均速度: {avg_fps:.2f} fps")
        if output_path:
            print(f"   输出视频: {output_path}")
        
        # 分析边界框变化
        self.analyze_bbox_changes()
        
        return results
    
    def analyze_bbox_changes(self):
        """分析边界框变化"""
        if len(self.bbox_history) < 2:
            return
        
        changes = 0
        for i in range(1, len(self.bbox_history)):
            if self.bbox_history[i] != self.bbox_history[i-1]:
                changes += 1
        
        print(f"\n边界框变化分析:")
        print(f"  总帧数: {len(self.bbox_history)}")
        print(f"  变化次数: {changes}")
        print(f"  变化率: {changes/len(self.bbox_history)*100:.1f}%")
        
        if changes == 0:
            print(f"❌ 边界框完全没有变化！")
            print(f"可能原因:")
            print(f"  1. 掩码预测失败")
            print(f"  2. 边界框更新逻辑有问题")
            print(f"  3. 模型输出异常")
        else:
            print(f"✅ 边界框有变化，追踪正常")
        
        # 显示边界框范围
        if self.bbox_history:
            x_coords = [bbox[0] for bbox in self.bbox_history]
            y_coords = [bbox[1] for bbox in self.bbox_history]
            w_coords = [bbox[2] for bbox in self.bbox_history]
            h_coords = [bbox[3] for bbox in self.bbox_history]
            
            print(f"边界框统计:")
            print(f"  X范围: {min(x_coords)} - {max(x_coords)}")
            print(f"  Y范围: {min(y_coords)} - {max(y_coords)}")
            print(f"  宽度范围: {min(w_coords)} - {max(w_coords)}")
            print(f"  高度范围: {min(h_coords)} - {max(h_coords)}")

def test_fixed_tracker():
    """测试修复版本的追踪器"""
    print("测试修复版本的追踪器...")
    
    # 创建测试视频
    test_video = "test_tracking.mp4"
    if not os.path.exists(test_video):
        print("创建测试视频...")
        width, height = 640, 480
        fps = 10
        total_frames = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(test_video, fourcc, fps, (width, height))
        
        for frame_idx in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)
            
            noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            t = frame_idx / total_frames
            center_x = int(100 + (width - 200) * t)
            center_y = int(height // 2 + 50 * np.sin(2 * np.pi * t * 2))
            
            cv2.rectangle(frame, (center_x-25, center_y-25), (center_x+25, center_y+25), (0, 0, 255), -1)
            
            writer.write(frame)
        
        writer.release()
    
    # 测试追踪
    initial_bbox = (75, 205, 50, 50)
    
    print(f"使用修复版本的追踪器...")
    tracker = FixedSAMURAITracker("onnx_models", device="cpu", debug=True)
    
    results = tracker.track_video(test_video, initial_bbox, "fixed_result.mp4")
    
    print(f"追踪完成! 处理了 {len(results)} 帧")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="修复版本的SAMURAI追踪器")
    parser.add_argument("--video", help="输入视频路径")
    parser.add_argument("--bbox", help="边界框 'x,y,w,h'")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--test", action="store_true", help="运行测试")
    
    args = parser.parse_args()
    
    if args.test:
        test_fixed_tracker()
        return
    
    if not args.video or not args.bbox:
        print("请提供视频路径和边界框")
        return
    
    # 解析边界框
    try:
        bbox_parts = args.bbox.split(',')
        if len(bbox_parts) != 4:
            raise ValueError("边界框格式错误")
        bbox = tuple(map(int, bbox_parts))
    except Exception as e:
        print(f"边界框解析失败: {e}")
        return
    
    # 执行追踪
    tracker = FixedSAMURAITracker("onnx_models", device="cpu", debug=args.debug)
    results = tracker.track_video(args.video, bbox, args.output)
    
    print(f"追踪完成! 处理了 {len(results)} 帧")

if __name__ == "__main__":
    main()
