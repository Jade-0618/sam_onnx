"""
简化的提示编码器ONNX导出
专门处理提示编码器的复杂输入结构
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sam2_path = os.path.join(project_root, "sam2")
sys.path.insert(0, sam2_path)

from sam2.build_sam import build_sam2_video_predictor

class SimplePromptEncoder(nn.Module):
    """
    简化的提示编码器，专门用于ONNX导出
    将复杂的可选输入简化为固定输入
    """
    
    def __init__(self, original_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        
    def forward(self, point_coords, point_labels, box_coords):
        """
        简化的前向传播
        
        Args:
            point_coords: [B, N, 2] 点坐标
            point_labels: [B, N] 点标签 (0=负点, 1=正点)
            box_coords: [B, 4] 边界框坐标 [x1, y1, x2, y2]
        
        Returns:
            sparse_embeddings: [B, N+2, C] 稀疏嵌入 (点+框角点)
            dense_embeddings: [B, C, H, W] 密集嵌入
        """
        # 处理点提示
        points = (point_coords, point_labels)
        
        # 处理框提示
        boxes = box_coords
        
        # 不使用掩码提示
        masks = None
        
        # 调用原始编码器
        sparse_embeddings, dense_embeddings = self.original_encoder(
            points=points,
            boxes=boxes, 
            masks=masks
        )
        
        return sparse_embeddings, dense_embeddings

def export_simple_prompt_encoder():
    """导出简化的提示编码器"""
    
    print("🔧 导出简化提示编码器")
    print("=" * 40)
    
    # 加载模型
    model_path = os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    config_path = os.path.join(project_root, "sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml")
    
    print(f"加载模型: {model_path}")
    predictor = build_sam2_video_predictor(config_path, model_path, device="cpu")
    predictor.eval()
    
    # 创建简化编码器
    simple_encoder = SimplePromptEncoder(predictor.sam_prompt_encoder)
    simple_encoder.eval()
    
    # 创建测试输入
    batch_size = 1
    num_points = 3  # 减少点数量以简化
    
    # 点坐标 (归一化到0-1024)
    point_coords = torch.tensor([[[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]]], dtype=torch.float32)
    
    # 点标签 (1=正点, 0=负点)
    point_labels = torch.tensor([[1, 1, 0]], dtype=torch.int64)
    
    # 边界框 [x1, y1, x2, y2]
    box_coords = torch.tensor([[50.0, 50.0, 200.0, 200.0]], dtype=torch.float32)
    
    print(f"测试输入:")
    print(f"  点坐标: {point_coords.shape} = {point_coords}")
    print(f"  点标签: {point_labels.shape} = {point_labels}")
    print(f"  边界框: {box_coords.shape} = {box_coords}")
    
    # 测试前向传播
    with torch.no_grad():
        sparse_emb, dense_emb = simple_encoder(point_coords, point_labels, box_coords)
        print(f"\n输出:")
        print(f"  稀疏嵌入: {sparse_emb.shape}")
        print(f"  密集嵌入: {dense_emb.shape}")
    
    # 导出ONNX
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prompt_encoder_simple.onnx")
    
    print(f"\n导出到: {output_path}")
    
    try:
        torch.onnx.export(
            simple_encoder,
            (point_coords, point_labels, box_coords),
            output_path,
            input_names=["point_coords", "point_labels", "box_coords"],
            output_names=["sparse_embeddings", "dense_embeddings"],
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            verbose=True
        )
        
        print("✅ 提示编码器导出成功!")
        
        # 验证导出的模型
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        
        # 测试ONNX模型
        onnx_inputs = {
            "point_coords": point_coords.numpy(),
            "point_labels": point_labels.numpy(),
            "box_coords": box_coords.numpy()
        }
        
        onnx_outputs = session.run(None, onnx_inputs)
        
        print(f"\nONNX验证:")
        print(f"  ONNX稀疏嵌入: {onnx_outputs[0].shape}")
        print(f"  ONNX密集嵌入: {onnx_outputs[1].shape}")
        
        # 比较输出差异
        sparse_diff = np.mean(np.abs(sparse_emb.numpy() - onnx_outputs[0]))
        dense_diff = np.mean(np.abs(dense_emb.numpy() - onnx_outputs[1]))
        
        print(f"  稀疏嵌入差异: {sparse_diff:.6f}")
        print(f"  密集嵌入差异: {dense_diff:.6f}")
        
        if sparse_diff < 1e-5 and dense_diff < 1e-5:
            print("✅ ONNX模型验证通过!")
        else:
            print("⚠️  ONNX模型存在精度差异")
            
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

class PointOnlyPromptEncoder(nn.Module):
    """
    仅支持点提示的编码器 - 最简化版本
    """
    
    def __init__(self, original_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        
    def forward(self, point_coords, point_labels):
        """
        仅处理点提示
        
        Args:
            point_coords: [B, N, 2] 点坐标
            point_labels: [B, N] 点标签
        """
        points = (point_coords, point_labels)
        
        sparse_embeddings, dense_embeddings = self.original_encoder(
            points=points,
            boxes=None,
            masks=None
        )
        
        return sparse_embeddings, dense_embeddings

def export_point_only_encoder():
    """导出仅支持点的编码器"""
    
    print("\n🎯 导出点提示编码器")
    print("=" * 30)
    
    # 加载模型
    model_path = os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    config_path = os.path.join(project_root, "sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml")
    
    predictor = build_sam2_video_predictor(config_path, model_path, device="cpu")
    predictor.eval()
    
    # 创建点编码器
    point_encoder = PointOnlyPromptEncoder(predictor.sam_prompt_encoder)
    point_encoder.eval()
    
    # 创建测试输入
    point_coords = torch.tensor([[[512.0, 512.0], [256.0, 256.0]]], dtype=torch.float32)
    point_labels = torch.tensor([[1, 0]], dtype=torch.int64)
    
    print(f"点编码器输入:")
    print(f"  点坐标: {point_coords.shape}")
    print(f"  点标签: {point_labels.shape}")
    
    # 测试
    with torch.no_grad():
        sparse_emb, dense_emb = point_encoder(point_coords, point_labels)
        print(f"  稀疏嵌入: {sparse_emb.shape}")
        print(f"  密集嵌入: {dense_emb.shape}")
    
    # 导出
    output_path = os.path.join("onnx_models", "prompt_encoder_points_only.onnx")
    
    try:
        torch.onnx.export(
            point_encoder,
            (point_coords, point_labels),
            output_path,
            input_names=["point_coords", "point_labels"],
            output_names=["sparse_embeddings", "dense_embeddings"],
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        print(f"✅ 点编码器导出成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 点编码器导出失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 SAMURAI 提示编码器 ONNX 导出")
    print("=" * 50)
    
    success1 = export_simple_prompt_encoder()
    success2 = export_point_only_encoder()
    
    print("\n" + "=" * 50)
    if success1 or success2:
        print("🎉 至少一个提示编码器导出成功!")
        
        if success1:
            print("✅ 完整提示编码器: onnx_models/prompt_encoder_simple.onnx")
        if success2:
            print("✅ 点提示编码器: onnx_models/prompt_encoder_points_only.onnx")
    else:
        print("❌ 所有导出都失败了")

if __name__ == "__main__":
    main()
