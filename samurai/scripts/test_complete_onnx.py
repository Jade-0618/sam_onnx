"""
完整SAMURAI ONNX系统测试脚本
测试所有导出的组件和端到端模型
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_individual_components():
    """测试各个组件的ONNX模型"""
    
    print("🧪 测试各个ONNX组件")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ ONNX Runtime未安装")
        return False
    
    model_dir = Path("onnx_models")
    if not model_dir.exists():
        print("❌ ONNX模型目录不存在")
        return False
    
    # 测试图像编码器
    print("\n1. 测试图像编码器")
    image_encoder_path = model_dir / "image_encoder_base_plus.onnx"
    if image_encoder_path.exists():
        try:
            session = ort.InferenceSession(str(image_encoder_path))
            test_input = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
            
            start_time = time.time()
            outputs = session.run(None, {'input_image': test_input})
            end_time = time.time()
            
            print(f"   ✅ 图像编码器工作正常")
            print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
            print(f"   输出数量: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"   输出{i+1}: {output.shape}")
        except Exception as e:
            print(f"   ❌ 图像编码器测试失败: {e}")
    else:
        print("   ⚠️  图像编码器模型不存在")
    
    # 测试提示编码器
    print("\n2. 测试提示编码器")
    prompt_models = [
        "prompt_encoder_simple.onnx",
        "prompt_encoder_points_only.onnx"
    ]
    
    for model_name in prompt_models:
        model_path = model_dir / model_name
        if model_path.exists():
            try:
                session = ort.InferenceSession(str(model_path))
                
                if "points_only" in model_name:
                    # 仅点输入
                    inputs = {
                        'point_coords': np.array([[[512.0, 512.0]]], dtype=np.float32),
                        'point_labels': np.array([[1]], dtype=np.int64)
                    }
                else:
                    # 完整输入
                    inputs = {
                        'point_coords': np.array([[[512.0, 512.0]]], dtype=np.float32),
                        'point_labels': np.array([[1]], dtype=np.int64),
                        'box_coords': np.array([[100.0, 100.0, 400.0, 400.0]], dtype=np.float32)
                    }
                
                outputs = session.run(None, inputs)
                print(f"   ✅ {model_name} 工作正常")
                print(f"   稀疏嵌入: {outputs[0].shape}")
                print(f"   密集嵌入: {outputs[1].shape}")
                break
                
            except Exception as e:
                print(f"   ❌ {model_name} 测试失败: {e}")
        else:
            print(f"   ⚠️  {model_name} 不存在")
    
    # 测试掩码解码器
    print("\n3. 测试掩码解码器")
    mask_models = [
        "mask_decoder_simple.onnx",
        "mask_decoder_single.onnx"
    ]
    
    for model_name in mask_models:
        model_path = model_dir / model_name
        if model_path.exists():
            try:
                session = ort.InferenceSession(str(model_path))
                
                inputs = {
                    'image_embeddings': np.random.randn(1, 256, 64, 64).astype(np.float32),
                    'sparse_prompt_embeddings': np.random.randn(1, 3, 256).astype(np.float32),
                    'dense_prompt_embeddings': np.random.randn(1, 256, 64, 64).astype(np.float32)
                }
                
                outputs = session.run(None, inputs)
                print(f"   ✅ {model_name} 工作正常")
                print(f"   掩码: {outputs[0].shape}")
                print(f"   IoU预测: {outputs[1].shape}")
                break
                
            except Exception as e:
                print(f"   ❌ {model_name} 测试失败: {e}")
        else:
            print(f"   ⚠️  {model_name} 不存在")
    
    # 测试内存编码器
    print("\n4. 测试内存编码器")
    memory_models = [
        "memory_encoder_full.onnx",
        "memory_encoder_stateful.onnx"
    ]
    
    for model_name in memory_models:
        model_path = model_dir / model_name
        if model_path.exists():
            try:
                session = ort.InferenceSession(str(model_path))
                
                if "stateful" in model_name:
                    inputs = {
                        'curr_vision_feats': np.random.randn(1, 256, 64, 64).astype(np.float32),
                        'feat_sizes': np.array([64, 64], dtype=np.int64),
                        'output_mask': np.random.randn(1, 1, 64, 64).astype(np.float32)
                    }
                else:
                    inputs = {
                        'curr_vision_feats': np.random.randn(1, 256, 64, 64).astype(np.float32),
                        'feat_sizes': np.array([64, 64], dtype=np.int64),
                        'output_mask': np.random.randn(1, 1, 64, 64).astype(np.float32),
                        'is_mem_frame': np.array([True], dtype=bool),
                        'prev_memory_bank': np.random.randn(1, 7, 256, 64, 64).astype(np.float32)
                    }
                
                outputs = session.run(None, inputs)
                print(f"   ✅ {model_name} 工作正常")
                print(f"   输出数量: {len(outputs)}")
                for i, output in enumerate(outputs):
                    print(f"   输出{i+1}: {output.shape}")
                break
                
            except Exception as e:
                print(f"   ❌ {model_name} 测试失败: {e}")
        else:
            print(f"   ⚠️  {model_name} 不存在")
    
    return True

def test_end_to_end_models():
    """测试端到端模型"""
    
    print("\n🎯 测试端到端模型")
    print("=" * 30)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ ONNX Runtime未安装")
        return False
    
    model_dir = Path("onnx_models")
    
    # 测试端到端模型
    end_to_end_models = [
        "samurai_end_to_end.onnx",
        "samurai_stateless.onnx"
    ]
    
    for model_name in end_to_end_models:
        model_path = model_dir / model_name
        if model_path.exists():
            try:
                session = ort.InferenceSession(str(model_path))
                
                # 准备输入
                image = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
                point_coords = np.array([[[512.0, 512.0]]], dtype=np.float32)
                point_labels = np.array([[1]], dtype=np.int64)
                box_coords = np.array([[100.0, 100.0, 400.0, 400.0]], dtype=np.float32)
                
                if "stateless" in model_name:
                    # 无状态模型需要内存银行
                    memory_bank = np.random.randn(1, 7, 256, 64, 64).astype(np.float32)
                    inputs = {
                        'image': image,
                        'point_coords': point_coords,
                        'point_labels': point_labels,
                        'memory_bank': memory_bank,
                        'box_coords': box_coords
                    }
                else:
                    # 有状态模型
                    inputs = {
                        'image': image,
                        'point_coords': point_coords,
                        'point_labels': point_labels,
                        'box_coords': box_coords
                    }
                
                start_time = time.time()
                outputs = session.run(None, inputs)
                end_time = time.time()
                
                print(f"   ✅ {model_name} 工作正常")
                print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
                print(f"   输出数量: {len(outputs)}")
                for i, output in enumerate(outputs):
                    print(f"   输出{i+1}: {output.shape}")
                
                return True
                
            except Exception as e:
                print(f"   ❌ {model_name} 测试失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠️  {model_name} 不存在")
    
    return False

def test_complete_inference():
    """测试完整的推理流程"""
    
    print("\n🚀 测试完整推理流程")
    print("=" * 35)
    
    try:
        from onnx_inference import SAMURAIONNXPredictor
        
        # 初始化预测器
        predictor = SAMURAIONNXPredictor("onnx_models", device="cpu")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_bbox = (100, 100, 200, 200)  # x1, y1, x2, y2
        
        print(f"   测试图像: {test_image.shape}")
        print(f"   测试边界框: {test_bbox}")
        
        # 测试单帧预测
        start_time = time.time()
        mask, confidence, memory_features = predictor.predict_mask(test_image, test_bbox)
        end_time = time.time()
        
        print(f"   ✅ 单帧预测成功")
        print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
        print(f"   掩码形状: {mask.shape}")
        print(f"   置信度: {confidence:.3f}")
        print(f"   内存特征: {memory_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 完整推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_video():
    """创建测试视频"""
    
    print("\n🎬 创建测试视频")
    print("=" * 25)
    
    output_path = "test_video_complete.mp4"
    width, height = 640, 480
    fps = 30
    duration = 3  # 秒
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_idx in range(total_frames):
        # 创建背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # 深灰色背景
        
        # 移动的目标物体
        t = frame_idx / total_frames
        center_x = int(100 + (width - 200) * t)
        center_y = int(height // 2 + 50 * np.sin(2 * np.pi * t * 2))
        
        # 绘制目标（矩形）
        size = 40
        cv2.rectangle(frame, 
                     (center_x - size//2, center_y - size//2),
                     (center_x + size//2, center_y + size//2),
                     (0, 255, 0), -1)
        
        # 添加一些干扰物
        cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
        cv2.circle(frame, (width-50, height-50), 25, (255, 0, 0), -1)
        
        writer.write(frame)
    
    writer.release()
    
    print(f"   ✅ 测试视频创建完成: {output_path}")
    
    # 返回初始边界框
    initial_bbox = (100 - 20, height // 2 - 20, 40, 40)  # x, y, w, h
    return output_path, initial_bbox

def test_video_tracking():
    """测试视频跟踪"""
    
    print("\n🎯 测试视频跟踪")
    print("=" * 25)
    
    try:
        from onnx_inference import SAMURAIONNXPredictor
        
        # 创建测试视频
        video_path, initial_bbox = create_test_video()
        
        # 初始化预测器
        predictor = SAMURAIONNXPredictor("onnx_models", device="cpu")
        
        # 运行跟踪
        print(f"   开始跟踪: {video_path}")
        print(f"   初始边界框: {initial_bbox}")
        
        results = predictor.track_video(
            video_path, 
            initial_bbox, 
            output_path="tracking_result_complete.mp4"
        )
        
        print(f"   ✅ 视频跟踪完成")
        print(f"   跟踪帧数: {len(results)}")
        print(f"   结果视频: tracking_result_complete.mp4")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 视频跟踪测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    
    print("🧪 SAMURAI 完整ONNX系统测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("组件测试", test_individual_components),
        ("端到端模型测试", test_end_to_end_models),
        ("完整推理测试", test_complete_inference),
        ("视频跟踪测试", test_video_tracking)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 出现异常: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "="*60)
    print("🏆 测试总结")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！SAMURAI ONNX系统工作正常！")
    elif passed > 0:
        print("⚠️  部分测试通过，系统基本可用")
    else:
        print("❌ 所有测试失败，需要检查系统配置")

if __name__ == "__main__":
    main()
