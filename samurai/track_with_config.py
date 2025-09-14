#!/usr/bin/env python3
"""
使用配置文件的视频追踪脚本
可以预先在config.py中设置路径，然后直接运行
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import DEFAULT_CONFIG, get_input_path, get_output_path, get_bbox, print_config
from SAMURAI_ONNX_FINAL_DELIVERY import SAMURAITracker

def track_video_with_config(input_path=None, output_path=None, bbox=None, device=None):
    """
    使用配置文件进行视频追踪
    
    Args:
        input_path: 输入视频路径（可选，使用配置文件中的默认值）
        output_path: 输出视频路径（可选，自动生成）
        bbox: 边界框 (x,y,w,h)（可选，使用配置文件中的默认值）
        device: 推理设备（可选，使用配置文件中的默认值）
    """
    
    # 使用配置文件中的默认值
    if input_path is None:
        input_path = get_input_path()
    if output_path is None:
        output_path = get_output_path(input_path)
    if bbox is None:
        bbox = get_bbox()
    if device is None:
        device = DEFAULT_CONFIG["device"]
    
    # 检查输入文件
    if not os.path.exists(input_path):
        print(f"错误: 输入视频文件不存在: {input_path}")
        print("请检查config.py中的input_video路径，或使用命令行参数指定")
        return False
    
    # 确保输出目录存在
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"开始追踪视频...")
    print(f"输入视频: {input_path}")
    print(f"输出视频: {output_path}")
    print(f"边界框: {bbox}")
    print(f"设备: {device}")
    print("-" * 50)
    
    try:
        # 初始化追踪器
        tracker = SAMURAITracker(DEFAULT_CONFIG["model_dir"], device)
        
        # 执行追踪
        results = tracker.track_video(input_path, bbox, output_path)
        
        print(f"\n追踪完成!")
        print(f"处理了 {len(results)} 帧")
        print(f"输出视频已保存到: {output_path}")
        
        # 保存结果文件
        results_file = f"{Path(input_path).stem}_results.txt"
        with open(results_file, 'w') as f:
            for bbox_result in results:
                f.write(f"{bbox_result[0]},{bbox_result[1]},{bbox_result[2]},{bbox_result[3]}\n")
        print(f"边界框结果已保存到: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"追踪失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用配置文件的视频追踪")
    parser.add_argument("--input", "-i", help="输入视频路径（覆盖配置文件）")
    parser.add_argument("--output", "-o", help="输出视频路径（覆盖配置文件）")
    parser.add_argument("--bbox", "-b", help="边界框 'x,y,w,h'（覆盖配置文件）")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda"], help="推理设备（覆盖配置文件）")
    parser.add_argument("--config", action="store_true", help="显示当前配置")
    
    args = parser.parse_args()
    
    # 显示配置
    if args.config:
        print_config()
        return
    
    # 解析边界框
    bbox = None
    if args.bbox:
        try:
            bbox_parts = args.bbox.split(',')
            if len(bbox_parts) != 4:
                raise ValueError("边界框格式错误")
            bbox = tuple(map(int, bbox_parts))
        except Exception as e:
            print(f"边界框解析失败: {e}")
            return
    
    # 执行追踪
    success = track_video_with_config(
        input_path=args.input,
        output_path=args.output,
        bbox=bbox,
        device=args.device
    )
    
    if success:
        print("\n✅ 追踪成功完成!")
    else:
        print("\n❌ 追踪失败!")

if __name__ == "__main__":
    main()
