#!/usr/bin/env python3
"""
测试追踪问题的简单脚本
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_test_video():
    """创建一个简单的测试视频"""
    print("创建测试视频...")
    
    width, height = 640, 480
    fps = 10
    duration = 3  # 3秒
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("test_tracking.mp4", fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # 创建背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # 深灰色背景
        
        # 添加一些噪声
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # 移动的目标（红色矩形）
        t = frame_idx / total_frames
        center_x = int(100 + (width - 200) * t)  # 从左到右移动
        center_y = int(height // 2 + 50 * np.sin(2 * np.pi * t * 2))  # 上下摆动
        
        # 绘制目标
        cv2.rectangle(frame, (center_x-25, center_y-25), (center_x+25, center_y+25), (0, 0, 255), -1)
        
        # 添加一些干扰物
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), -1)
        cv2.rectangle(frame, (width-100, height-100), (width-50, height-50), (255, 0, 0), -1)
        
        writer.write(frame)
    
    writer.release()
    
    # 返回初始边界框（红色矩形的初始位置）
    initial_bbox = (75, 205, 50, 50)  # x, y, w, h
    return initial_bbox

def test_simple_tracking():
    """测试简单追踪"""
    print("测试简单追踪...")
    
    # 创建测试视频
    test_video = "test_tracking.mp4"
    if not os.path.exists(test_video):
        initial_bbox = create_test_video()
    else:
        initial_bbox = (75, 205, 50, 50)
    
    print(f"测试视频: {test_video}")
    print(f"初始边界框: {initial_bbox}")
    
    # 初始化追踪器
    try:
        from SAMURAI_ONNX_FINAL_DELIVERY import SAMURAITracker
        tracker = SAMURAITracker("onnx_models", device="cpu")
        print("追踪器初始化成功")
    except Exception as e:
        print(f"追踪器初始化失败: {e}")
        return
    
    # 执行追踪
    try:
        results = tracker.track_video(test_video, initial_bbox, "test_result.mp4")
        print(f"追踪完成! 处理了 {len(results)} 帧")
        
        # 分析结果
        print(f"\n边界框变化分析:")
        changes = 0
        for i in range(1, len(results)):
            if results[i] != results[i-1]:
                changes += 1
        
        print(f"总帧数: {len(results)}")
        print(f"变化次数: {changes}")
        print(f"变化率: {changes/len(results)*100:.1f}%")
        
        if changes == 0:
            print(f"❌ 边界框完全没有变化！")
            print(f"可能的问题:")
            print(f"  1. 模型预测失败")
            print(f"  2. 掩码生成有问题")
            print(f"  3. 边界框更新逻辑有问题")
        else:
            print(f"✅ 边界框有变化，追踪正常")
            
        # 显示前几个边界框
        print(f"\n前5个边界框:")
        for i in range(min(5, len(results))):
            print(f"  帧 {i+1}: {results[i]}")
            
    except Exception as e:
        print(f"追踪失败: {e}")
        import traceback
        traceback.print_exc()

def analyze_bbox_issue():
    """分析边界框问题"""
    print(f"\n🔍 边界框问题分析:")
    print(f"常见原因:")
    print(f"1. 边界框设置不当:")
    print(f"   - 边界框太大，包含了整个画面")
    print(f"   - 边界框太小，目标不清晰")
    print(f"   - 边界框位置不对，没有包含目标")
    
    print(f"\n2. 视频内容问题:")
    print(f"   - 目标不够明显")
    print(f"   - 背景太复杂")
    print(f"   - 目标移动太快")
    
    print(f"\n3. 模型问题:")
    print(f"   - 模型预测失败")
    print(f"   - 掩码生成有问题")
    print(f"   - 置信度过低")
    
    print(f"\n4. 代码问题:")
    print(f"   - 边界框更新逻辑错误")
    print(f"   - 掩码处理有问题")
    print(f"   - 坐标转换错误")

def main():
    """主函数"""
    print("SAMURAI ONNX 追踪问题测试")
    print("=" * 40)
    
    # 测试简单追踪
    test_simple_tracking()
    
    # 分析问题
    analyze_bbox_issue()
    
    print(f"\n💡 建议:")
    print(f"1. 运行调试脚本:")
    print(f"   python debug_tracking.py --video test_tracking.mp4 --bbox '75,205,50,50'")
    
    print(f"\n2. 检查您的视频:")
    print(f"   - 确保目标清晰可见")
    print(f"   - 边界框包含目标")
    print(f"   - 目标有足够的对比度")
    
    print(f"\n3. 尝试不同的边界框:")
    print(f"   - 使用更小的边界框")
    print(f"   - 调整边界框位置")
    print(f"   - 确保边界框在视频范围内")

if __name__ == "__main__":
    main()
