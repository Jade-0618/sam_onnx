"""
简化的端到端模型导出
避免复杂依赖，直接使用现有的图像编码器创建端到端模型
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

def create_mock_end_to_end_model():
    """创建一个模拟的端到端模型用于测试"""
    
    print("🔧 创建模拟端到端模型")
    print("=" * 30)
    
    class MockSAMURAIEndToEnd(nn.Module):
        """模拟的SAMURAI端到端模型"""
        
        def __init__(self):
            super().__init__()
            
            # 模拟图像编码器 - 简化版本
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((64, 64))
            )
            
            # 模拟提示编码器
            self.point_embed = nn.Embedding(2, 256)  # 正负点嵌入
            self.box_embed = nn.Linear(4, 256)
            
            # 模拟掩码解码器
            self.mask_decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 3个掩码
                nn.Sigmoid()
            )
            
            # IoU预测头
            self.iou_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # 3个掩码的IoU
            )
            
            # 内存编码器 - 简化版本
            self.memory_encoder = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1)
            )
            
        def forward(self, image, point_coords, point_labels, box_coords):
            """
            前向传播
            
            Args:
                image: [B, 3, H, W] 输入图像
                point_coords: [B, N, 2] 点坐标
                point_labels: [B, N] 点标签
                box_coords: [B, 4] 边界框坐标
                
            Returns:
                masks: [B, 3, H, W] 预测掩码
                iou_predictions: [B, 3] IoU预测
                memory_features: [B, 256, 64, 64] 内存特征
            """
            batch_size = image.shape[0]
            
            # 1. 图像编码
            image_features = self.image_encoder(image)  # [B, 256, 64, 64]
            
            # 2. 提示编码 (简化)
            # 点提示
            point_embeds = self.point_embed(point_labels.long())  # [B, N, 256]
            point_embeds = point_embeds.mean(dim=1, keepdim=True)  # [B, 1, 256]
            
            # 框提示
            box_embeds = self.box_embed(box_coords)  # [B, 256]
            box_embeds = box_embeds.unsqueeze(1)  # [B, 1, 256]
            
            # 合并提示
            prompt_embeds = (point_embeds + box_embeds).squeeze(1)  # [B, 256]
            
            # 3. 特征融合
            prompt_embeds = prompt_embeds.unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
            prompt_embeds = prompt_embeds.expand(-1, -1, 64, 64)  # [B, 256, 64, 64]
            
            fused_features = image_features + prompt_embeds
            
            # 4. 掩码解码
            masks = self.mask_decoder(fused_features)  # [B, 3, 1024, 1024]
            
            # 5. IoU预测
            iou_predictions = self.iou_head(fused_features)  # [B, 3]
            iou_predictions = torch.sigmoid(iou_predictions)
            
            # 6. 内存编码
            memory_features = self.memory_encoder(fused_features)  # [B, 256, 64, 64]
            
            return masks, iou_predictions, memory_features
    
    return MockSAMURAIEndToEnd()

def export_mock_end_to_end():
    """导出模拟的端到端模型"""
    
    print("🚀 导出模拟端到端SAMURAI模型")
    print("=" * 50)
    
    # 创建模型
    model = create_mock_end_to_end_model()
    model.eval()
    
    # 创建测试输入
    batch_size = 1
    image_size = 1024
    
    image = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    point_coords = torch.tensor([[[512.0, 512.0]]], dtype=torch.float32)
    point_labels = torch.tensor([[1]], dtype=torch.int64)
    box_coords = torch.tensor([[100.0, 100.0, 400.0, 400.0]], dtype=torch.float32)
    
    print(f"测试输入:")
    print(f"  图像: {image.shape}")
    print(f"  点坐标: {point_coords.shape}")
    print(f"  点标签: {point_labels.shape}")
    print(f"  边界框: {box_coords.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        try:
            masks, iou_predictions, memory_features = model(
                image, point_coords, point_labels, box_coords
            )
            print(f"\n输出:")
            print(f"  掩码: {masks.shape}")
            print(f"  IoU预测: {iou_predictions.shape}")
            print(f"  内存特征: {memory_features.shape}")
            
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 导出ONNX
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "samurai_mock_end_to_end.onnx")
    
    print(f"\n导出到: {output_path}")
    
    try:
        torch.onnx.export(
            model,
            (image, point_coords, point_labels, box_coords),
            output_path,
            input_names=["image", "point_coords", "point_labels", "box_coords"],
            output_names=["masks", "iou_predictions", "memory_features"],
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            verbose=True
        )
        
        print("✅ 模拟端到端模型导出成功!")
        
        # 验证导出的模型
        try:
            session = ort.InferenceSession(output_path)
            
            # 测试ONNX模型
            onnx_inputs = {
                "image": image.numpy(),
                "point_coords": point_coords.numpy(),
                "point_labels": point_labels.numpy(),
                "box_coords": box_coords.numpy()
            }
            
            onnx_outputs = session.run(None, onnx_inputs)
            
            print(f"\nONNX验证:")
            print(f"  ONNX掩码: {onnx_outputs[0].shape}")
            print(f"  ONNX IoU: {onnx_outputs[1].shape}")
            print(f"  ONNX内存特征: {onnx_outputs[2].shape}")
            
            # 比较输出差异
            mask_diff = np.mean(np.abs(masks.numpy() - onnx_outputs[0]))
            iou_diff = np.mean(np.abs(iou_predictions.numpy() - onnx_outputs[1]))
            memory_diff = np.mean(np.abs(memory_features.numpy() - onnx_outputs[2]))
            
            print(f"  掩码差异: {mask_diff:.6f}")
            print(f"  IoU差异: {iou_diff:.6f}")
            print(f"  内存特征差异: {memory_diff:.6f}")
            
            if mask_diff < 1e-5 and iou_diff < 1e-5 and memory_diff < 1e-5:
                print("✅ ONNX模型验证通过!")
            else:
                print("⚠️  ONNX模型存在精度差异")
                
        except Exception as e:
            print(f"ONNX验证失败: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_lightweight_end_to_end():
    """创建轻量级端到端模型"""
    
    print("\n🪶 创建轻量级端到端模型")
    print("=" * 40)
    
    class LightweightSAMURAI(nn.Module):
        """轻量级SAMURAI模型"""
        
        def __init__(self):
            super().__init__()
            
            # 轻量级图像编码器
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4),  # 1024 -> 256
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4),  # 256 -> 64
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU()
            )
            
            # 简单的掩码解码器
            self.mask_decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=4, padding=0),  # 64 -> 256
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=4, padding=0),   # 256 -> 1024
                nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1),  # 单掩码输出
                nn.Sigmoid()
            )
            
            # IoU预测
            self.iou_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
        def forward(self, image, point_coords, point_labels, box_coords):
            """轻量级前向传播"""
            
            # 图像编码
            features = self.image_encoder(image)  # [B, 128, 64, 64]
            
            # 掩码解码
            mask = self.mask_decoder(features)  # [B, 1, 1024, 1024]
            
            # IoU预测
            iou = self.iou_head(features)  # [B, 1]
            
            # 内存特征就是编码特征
            memory_features = features  # [B, 128, 64, 64]
            
            return mask, iou, memory_features
    
    return LightweightSAMURAI()

def export_lightweight_model():
    """导出轻量级模型"""
    
    model = create_lightweight_end_to_end()
    model.eval()
    
    # 测试输入
    image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    point_coords = torch.tensor([[[512.0, 512.0]]], dtype=torch.float32)
    point_labels = torch.tensor([[1]], dtype=torch.int64)
    box_coords = torch.tensor([[100.0, 100.0, 400.0, 400.0]], dtype=torch.float32)
    
    print(f"轻量级模型输入:")
    print(f"  图像: {image.shape}")
    
    # 测试
    with torch.no_grad():
        mask, iou, memory_features = model(image, point_coords, point_labels, box_coords)
        print(f"  掩码: {mask.shape}")
        print(f"  IoU: {iou.shape}")
        print(f"  内存特征: {memory_features.shape}")
    
    # 导出
    output_path = os.path.join("onnx_models", "samurai_lightweight.onnx")
    
    try:
        torch.onnx.export(
            model,
            (image, point_coords, point_labels, box_coords),
            output_path,
            input_names=["image", "point_coords", "point_labels", "box_coords"],
            output_names=["mask", "iou", "memory_features"],
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        print(f"✅ 轻量级模型导出成功: {output_path}")
        
        # 检查模型大小
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   模型大小: {model_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 轻量级模型导出失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 简化端到端SAMURAI模型导出")
    print("=" * 50)
    
    success1 = export_mock_end_to_end()
    success2 = export_lightweight_model()
    
    print("\n" + "=" * 50)
    print("🏆 导出总结")
    
    if success1:
        print("✅ 模拟端到端模型: onnx_models/samurai_mock_end_to_end.onnx")
    else:
        print("❌ 模拟端到端模型导出失败")
    
    if success2:
        print("✅ 轻量级模型: onnx_models/samurai_lightweight.onnx")
    else:
        print("❌ 轻量级模型导出失败")
    
    if success1 or success2:
        print("\n🎉 至少一个端到端模型导出成功!")
        print("💡 这些模型可以用于测试端到端推理流程")
        print("💡 虽然是模拟模型，但具有完整的输入输出接口")
    else:
        print("\n❌ 所有导出都失败了")

if __name__ == "__main__":
    main()
