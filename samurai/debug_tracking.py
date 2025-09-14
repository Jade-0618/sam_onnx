#!/usr/bin/env python3
"""
SAMURAI ONNX 追踪调试脚本
帮助诊断追踪问题
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from SAMURAI_ONNX_FINAL_DELIVERY import SAMURAITracker

def analyze_video(video_path):
    """分析视频内容"""
    print(f"分析视频: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.1f} fps")
    print(f"  总帧数: {total_frames}")
    
    # 分析前几帧
    print(f"\n分析前5帧...")
    for i in range(min(5, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 计算帧的统计信息
        mean_color = np.mean(frame, axis=(0, 1))
        std_color = np.std(frame, axis=(0, 1))
        
        print(f"  帧 {i+1}: 平均颜色 {mean_color}, 标准差 {std_color}")
    
    cap.release()

def test_bbox_validity(video_path, bbox):
    """测试边界框的有效性"""
    print(f"\n测试边界框: {bbox}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    x, y, w, h = bbox
    
    print(f"视频尺寸: {width}x{height}")
    print(f"边界框: x={x}, y={y}, w={w}, h={h}")
    
    # 检查边界框是否在视频范围内
    if x < 0 or y < 0 or x + w > width or y + h > height:
        print(f"❌ 边界框超出视频范围!")
        print(f"   边界框范围: ({x}, {y}) 到 ({x+w}, {y+h})")
        print(f"   视频范围: (0, 0) 到 ({width}, {height})")
        return False
    
    # 检查边界框大小
    if w <= 0 or h <= 0:
        print(f"❌ 边界框大小无效: w={w}, h={h}")
        return False
    
    print(f"✅ 边界框有效")
    
    # 读取第一帧并显示边界框区域
    ret, frame = cap.read()
    if ret:
        roi = frame[y:y+h, x:x+w]
        roi_mean = np.mean(roi)
        roi_std = np.std(roi)
        print(f"边界框区域统计: 平均值={roi_mean:.1f}, 标准差={roi_std:.1f}")
        
        # 检查区域是否有足够的变化
        if roi_std < 10:
            print(f"⚠️  边界框区域变化较小 (std={roi_std:.1f})，可能影响追踪")
    
    cap.release()
    return True

def debug_tracking(video_path, bbox, max_frames=10):
    """调试追踪过程"""
    print(f"\n调试追踪过程...")
    
    # 初始化追踪器
    tracker = SAMURAITracker("onnx_models", device="cpu")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 转换边界框格式
    x, y, w, h = bbox
    current_bbox = (x, y, x + w, y + h)  # 转换为 (x1, y1, x2, y2)
    
    print(f"开始追踪，初始边界框: {current_bbox}")
    
    frame_idx = 0
    bbox_history = []
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # 执行预测
            mask, confidence, memory_features = tracker.predict_mask(frame, current_bbox)
            
            print(f"帧 {frame_idx+1}:")
            print(f"  置信度: {confidence:.3f}")
            print(f"  掩码形状: {mask.shape}")
            print(f"  掩码像素数: {np.sum(mask)}")
            
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
                    
                    # 检查边界框变化
                    if new_bbox != current_bbox:
                        print(f"  边界框变化: {current_bbox} -> {new_bbox}")
                        current_bbox = new_bbox
                    else:
                        print(f"  边界框未变化: {current_bbox}")
                else:
                    print(f"  掩码为空，保持原边界框")
            else:
                print(f"  掩码为空，保持原边界框")
            
            # 转换回 (x, y, w, h) 格式
            x1, y1, x2, y2 = current_bbox
            result_bbox = (x1, y1, x2 - x1, y2 - y1)
            bbox_history.append(result_bbox)
            
            print(f"  结果边界框: {result_bbox}")
            
        except Exception as e:
            print(f"  帧 {frame_idx+1} 处理失败: {e}")
        
        frame_idx += 1
    
    cap.release()
    
    # 分析边界框变化
    print(f"\n边界框变化分析:")
    if len(bbox_history) > 1:
        changes = 0
        for i in range(1, len(bbox_history)):
            if bbox_history[i] != bbox_history[i-1]:
                changes += 1
        
        print(f"  总帧数: {len(bbox_history)}")
        print(f"  变化次数: {changes}")
        print(f"  变化率: {changes/len(bbox_history)*100:.1f}%")
        
        if changes == 0:
            print(f"❌ 边界框完全没有变化！")
            print(f"可能原因:")
            print(f"  1. 边界框区域没有明显的目标")
            print(f"  2. 模型预测失败")
            print(f"  3. 边界框设置不当")
        else:
            print(f"✅ 边界框有变化，追踪正常")
    else:
        print(f"❌ 只处理了1帧，无法分析变化")

def suggest_solutions():
    """建议解决方案"""
    print(f"\n🔧 解决方案建议:")
    print(f"1. 检查边界框设置:")
    print(f"   - 确保边界框包含明显的目标")
    print(f"   - 边界框不要太大或太小")
    print(f"   - 避免边界框在视频边缘")
    
    print(f"\n2. 检查视频内容:")
    print(f"   - 确保目标在视频中清晰可见")
    print(f"   - 避免目标太小或太模糊")
    print(f"   - 确保目标有足够的对比度")
    
    print(f"\n3. 调整参数:")
    print(f"   - 尝试不同的边界框位置")
    print(f"   - 使用更小的边界框")
    print(f"   - 检查视频质量")
    
    print(f"\n4. 测试建议:")
    print(f"   - 先用简单的测试视频")
    print(f"   - 选择有明显移动的目标")
    print(f"   - 确保目标与背景有对比")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAMURAI ONNX 追踪调试")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--bbox", required=True, help="边界框 'x,y,w,h'")
    parser.add_argument("--frames", type=int, default=10, help="调试帧数")
    
    args = parser.parse_args()
    
    # 解析边界框
    try:
        bbox_parts = args.bbox.split(',')
        if len(bbox_parts) != 4:
            raise ValueError("边界框格式错误")
        bbox = tuple(map(int, bbox_parts))
    except Exception as e:
        print(f"边界框解析失败: {e}")
        return
    
    print("SAMURAI ONNX 追踪调试工具")
    print("=" * 40)
    
    # 分析视频
    analyze_video(args.video)
    
    # 测试边界框
    if not test_bbox_validity(args.video, bbox):
        print(f"\n❌ 边界框无效，请检查设置")
        suggest_solutions()
        return
    
    # 调试追踪
    debug_tracking(args.video, bbox, args.frames)
    
    # 建议解决方案
    suggest_solutions()

if __name__ == "__main__":
    main()
