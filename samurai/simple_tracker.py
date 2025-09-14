#!/usr/bin/env python3
"""
简化版本的SAMURAI追踪器
使用更稳定的追踪策略
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

class SimpleTracker:
    """简化版本的追踪器，使用更稳定的策略"""
    
    def __init__(self, model_dir: str = "onnx_models", device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        
        # 初始化ONNX Runtime
        try:
            import onnxruntime as ort
            self.ort = ort
            if device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # 加载模型
            self.sessions = {}
            self._load_models(providers)
            
        except ImportError:
            print("ONNXRuntime not found. Install with: pip install onnxruntime")
            sys.exit(1)
        
        # 追踪参数
        self.template = None
        self.template_bbox = None
        self.search_size = 200  # 搜索区域大小
        
        print(f"简化追踪器初始化完成")
        print(f"   设备: {device}")
        print(f"   可用模型: {list(self.sessions.keys())}")
    
    def _load_models(self, providers):
        """加载ONNX模型"""
        model_files = {
            'image_encoder': ['image_encoder_base_plus.onnx'],
            'end_to_end': ['samurai_mock_end_to_end.onnx', 'samurai_lightweight.onnx']
        }
        
        for model_name, filenames in model_files.items():
            for filename in filenames:
                model_path = self.model_dir / filename
                if model_path.exists():
                    try:
                        session = self.ort.InferenceSession(str(model_path), providers=providers)
                        self.sessions[model_name] = session
                        print(f"✅ 加载 {model_name}: {filename}")
                        break
                    except Exception as e:
                        print(f"⚠️  加载 {filename} 失败: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # 调整尺寸到1024x1024
        image_resized = cv2.resize(image_rgb, (1024, 1024))
        
        # 归一化到[0,1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 转换为CHW格式并添加batch维度
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_tensor, axis=0)
        
        return image_batch
    
    def predict_mask_simple(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """简化的掩码预测"""
        
        # 如果只有图像编码器，使用模板匹配
        if 'image_encoder' in self.sessions and 'end_to_end' not in self.sessions:
            return self._template_matching(image, bbox)
        
        # 如果有端到端模型，尝试使用但限制输出
        if 'end_to_end' in self.sessions:
            return self._predict_with_limits(image, bbox)
        
        # 默认使用模板匹配
        return self._template_matching(image, bbox)
    
    def _template_matching(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """使用模板匹配进行追踪"""
        
        x, y, w, h = bbox
        
        # 确保边界框在图像范围内
        x = max(0, min(x, image.shape[1] - w))
        y = max(0, min(y, image.shape[0] - h))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        # 如果是第一帧，保存模板
        if self.template is None:
            self.template = image[y:y+h, x:x+w].copy()
            self.template_bbox = (x, y, w, h)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask[y:y+h, x:x+w] = 1
            return mask, 0.8
        
        # 模板匹配
        result = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 如果匹配度太低，保持原位置
        if max_val < 0.3:
            x, y, w, h = self.template_bbox
        else:
            x, y = max_loc
            w, h = self.template.shape[1], self.template.shape[0]
        
        # 创建掩码
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 1
        
        # 更新模板
        self.template = image[y:y+h, x:x+w].copy()
        self.template_bbox = (x, y, w, h)
        
        return mask, max_val
    
    def _predict_with_limits(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """使用端到端模型但限制输出范围"""
        
        try:
            # 预处理图像
            input_tensor = self.preprocess_image(image)
            
            # 准备提示
            x1, y1, x2, y2 = bbox
            point_labels = np.array([[1]], dtype=np.int64)
            box_coords = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            
            # 运行推理
            session = self.sessions['end_to_end']
            input_names = [inp.name for inp in session.get_inputs()]
            
            inputs = {"image": input_tensor}
            if "point_labels" in input_names:
                inputs["point_labels"] = point_labels
            if "box_coords" in input_names:
                inputs["box_coords"] = box_coords
            
            outputs = session.run(None, inputs)
            
            # 处理输出
            masks = outputs[0]
            iou_predictions = outputs[1] if len(outputs) > 1 else np.array([[0.8]])
            
            # 选择最佳掩码
            if len(masks.shape) == 4 and masks.shape[1] > 1:
                best_idx = np.argmax(iou_predictions[0])
                best_mask = masks[0, best_idx]
                confidence = float(iou_predictions[0, best_idx])
            else:
                best_mask = masks[0, 0] if len(masks.shape) == 4 else masks[0]
                confidence = float(iou_predictions[0, 0]) if len(iou_predictions.shape) == 2 else float(iou_predictions[0])
            
            # 调整掩码尺寸
            mask_resized = cv2.resize(best_mask, (image.shape[1], image.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # 限制掩码范围 - 如果掩码太大，使用原始边界框
            mask_area = np.sum(mask_binary)
            image_area = image.shape[0] * image.shape[1]
            
            if mask_area > image_area * 0.5:  # 如果掩码超过50%的图像
                print(f"掩码过大 ({mask_area}/{image_area}), 使用原始边界框")
                mask_binary = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                x1, y1, x2, y2 = bbox
                mask_binary[y1:y2, x1:x2] = 1
                confidence = 0.5
            
            return mask_binary, confidence
            
        except Exception as e:
            print(f"端到端预测失败: {e}")
            return self._template_matching(image, bbox)
    
    def track_video(self, video_path: str, initial_bbox: Tuple[int, int, int, int], 
                   output_path: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
        """视频追踪"""
        
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
                mask, confidence = self.predict_mask_simple(frame, current_bbox)
                
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
                        
                        current_bbox = (x1, y1, x2, y2)
                
                # 转换回 (x, y, w, h) 格式
                x1, y1, x2, y2 = current_bbox
                result_bbox = (x1, y1, x2 - x1, y2 - y1)
                results.append(result_bbox)
                
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
        self._analyze_results(results)
        
        return results
    
    def _analyze_results(self, results):
        """分析追踪结果"""
        if len(results) < 2:
            return
        
        changes = 0
        for i in range(1, len(results)):
            if results[i] != results[i-1]:
                changes += 1
        
        print(f"\n边界框变化分析:")
        print(f"  总帧数: {len(results)}")
        print(f"  变化次数: {changes}")
        print(f"  变化率: {changes/len(results)*100:.1f}%")
        
        if changes == 0:
            print(f"❌ 边界框完全没有变化！")
        else:
            print(f"✅ 边界框有变化，追踪正常")
        
        # 显示边界框范围
        if results:
            x_coords = [bbox[0] for bbox in results]
            y_coords = [bbox[1] for bbox in results]
            w_coords = [bbox[2] for bbox in results]
            h_coords = [bbox[3] for bbox in results]
            
            print(f"边界框统计:")
            print(f"  X范围: {min(x_coords)} - {max(x_coords)}")
            print(f"  Y范围: {min(y_coords)} - {max(y_coords)}")
            print(f"  宽度范围: {min(w_coords)} - {max(w_coords)}")
            print(f"  高度范围: {min(h_coords)} - {max(h_coords)}")

def test_simple_tracker():
    """测试简化追踪器"""
    print("测试简化追踪器...")
    
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
    
    print(f"使用简化追踪器...")
    tracker = SimpleTracker("onnx_models", device="cpu")
    
    results = tracker.track_video(test_video, initial_bbox, "simple_result.mp4")
    
    print(f"追踪完成! 处理了 {len(results)} 帧")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简化版本的SAMURAI追踪器")
    parser.add_argument("--video", help="输入视频路径")
    parser.add_argument("--bbox", help="边界框 'x,y,w,h'")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--test", action="store_true", help="运行测试")
    
    args = parser.parse_args()
    
    if args.test:
        test_simple_tracker()
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
    tracker = SimpleTracker("onnx_models", device="cpu")
    results = tracker.track_video(args.video, bbox, args.output)
    
    print(f"追踪完成! 处理了 {len(results)} 帧")

if __name__ == "__main__":
    main()
