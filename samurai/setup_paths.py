#!/usr/bin/env python3
"""
路径设置脚本
帮助您快速设置输入输出视频路径
"""

import os
from pathlib import Path

def setup_paths():
    """交互式设置路径"""
    print("SAMURAI ONNX 路径设置")
    print("=" * 30)
    
    # 获取用户输入
    print("\n请输入您的视频路径:")
    print("示例: C:\\Users\\YourName\\Videos\\my_video.mp4")
    print("或者: D:\\Videos\\input.mp4")
    
    input_path = input("输入视频路径: ").strip()
    
    if not input_path:
        print("未输入路径，使用默认值")
        input_path = "input_video.mp4"
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"⚠️  警告: 文件不存在: {input_path}")
        print("请确保路径正确，或者稍后手动修改config.py")
    
    # 生成输出路径
    input_file = Path(input_path)
    output_path = str(input_file.parent / f"{input_file.stem}_tracked{input_file.suffix}")
    
    print(f"\n自动生成的输出路径: {output_path}")
    
    # 询问是否修改输出路径
    modify_output = input("是否修改输出路径? (y/n): ").strip().lower()
    if modify_output == 'y':
        output_path = input("输出视频路径: ").strip()
        if not output_path:
            output_path = str(input_file.parent / f"{input_file.stem}_tracked{input_file.suffix}")
    
    # 设置边界框
    print(f"\n请输入初始边界框 (x, y, w, h):")
    print("示例: 100,100,200,150")
    print("这表示从(100,100)开始，宽200像素，高150像素的矩形区域")
    
    bbox_input = input("边界框: ").strip()
    if bbox_input:
        try:
            bbox_parts = bbox_input.split(',')
            if len(bbox_parts) == 4:
                bbox = tuple(map(int, bbox_parts))
            else:
                raise ValueError("格式错误")
        except:
            print("边界框格式错误，使用默认值")
            bbox = (100, 100, 200, 150)
    else:
        bbox = (100, 100, 200, 150)
    
    # 生成配置文件内容
    config_content = f'''#!/usr/bin/env python3
"""
SAMURAI ONNX 配置文件
自动生成的配置
"""

import os
from pathlib import Path

# 默认路径配置
DEFAULT_CONFIG = {{
    # 输入视频路径
    "input_video": r"{input_path}",
    
    # 输出视频路径
    "output_video": r"{output_path}",
    
    # 默认边界框 (x, y, w, h)
    "default_bbox": {bbox},
    
    # 模型目录
    "model_dir": "onnx_models",
    
    # 推理设备
    "device": "cpu",  # 或 "cuda"
    
    # 视频处理参数
    "save_results": True,
    "show_progress": True,
}}

# 常用路径预设
COMMON_PATHS = {{
    "desktop": str(Path.home() / "Desktop"),
    "documents": str(Path.home() / "Documents"),
    "project": str(Path(__file__).parent),
    "videos": str(Path.home() / "Videos"),
    "downloads": str(Path.home() / "Downloads"),
}}

def get_input_path(filename=None):
    """获取输入视频路径"""
    if filename:
        return filename
    return DEFAULT_CONFIG["input_video"]

def get_output_path(input_path=None):
    """获取输出视频路径"""
    if input_path:
        input_file = Path(input_path)
        return str(input_file.parent / f"{{input_file.stem}}_tracked{{input_file.suffix}}")
    return DEFAULT_CONFIG["output_video"]

def get_bbox():
    """获取默认边界框"""
    return DEFAULT_CONFIG["default_bbox"]

def print_config():
    """打印当前配置"""
    print("当前配置:")
    print(f"  输入视频: {{DEFAULT_CONFIG['input_video']}}")
    print(f"  输出视频: {{DEFAULT_CONFIG['output_video']}}")
    print(f"  默认边界框: {{DEFAULT_CONFIG['default_bbox']}}")
    print(f"  模型目录: {{DEFAULT_CONFIG['model_dir']}}")
    print(f"  推理设备: {{DEFAULT_CONFIG['device']}}")

if __name__ == "__main__":
    print_config()
'''
    
    # 保存配置文件
    config_file = "config.py"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\n✅ 配置已保存到: {config_file}")
    print("\n现在您可以使用以下命令进行追踪:")
    print(f"python track_with_config.py")
    print("\n或者查看当前配置:")
    print(f"python track_with_config.py --config")

def main():
    """主函数"""
    try:
        setup_paths()
    except KeyboardInterrupt:
        print("\n\n设置已取消")
    except Exception as e:
        print(f"\n设置失败: {e}")

if __name__ == "__main__":
    main()
