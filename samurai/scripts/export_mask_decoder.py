"""
掩码解码器ONNX导出
处理SAM2掩码解码器的复杂输入输出结构
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

class SimpleMaskDecoder(nn.Module):
    """
    简化的掩码解码器，用于ONNX导出
    """
    
    def __init__(self, original_decoder, prompt_encoder):
        super().__init__()
        self.original_decoder = original_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings):
        """
        简化的掩码解码器前向传播
        
        Args:
            image_embeddings: [B, C, H, W] 图像特征
            sparse_prompt_embeddings: [B, N, C] 稀疏提示嵌入
            dense_prompt_embeddings: [B, C, H, W] 密集提示嵌入
            
        Returns:
            masks: [B, num_masks, H, W] 预测掩码
            iou_predictions: [B, num_masks] IoU预测
        """
        # 获取位置编码
        image_pe = self.prompt_encoder.get_dense_pe()
        
        # 调用原始解码器
        masks, iou_predictions, _, _ = self.original_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=None
        )
        
        return masks, iou_predictions

def export_mask_decoder():
    """导出掩码解码器"""
    
    print("🎭 导出掩码解码器")
    print("=" * 30)
    
    # 加载模型
    model_path = os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    config_path = os.path.join(project_root, "sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml")
    
    print(f"加载模型: {model_path}")
    predictor = build_sam2_video_predictor(config_path, model_path, device="cpu")
    predictor.eval()
    
    # 创建简化解码器
    simple_decoder = SimpleMaskDecoder(predictor.sam_mask_decoder, predictor.sam_prompt_encoder)
    simple_decoder.eval()
    
    # 创建测试输入
    batch_size = 1
    embed_dim = 256
    
    # 图像嵌入 (来自图像编码器的输出)
    image_embeddings = torch.randn(batch_size, embed_dim, 64, 64, dtype=torch.float32)
    
    # 稀疏提示嵌入 (来自提示编码器)
    num_sparse_prompts = 5  # 3个点 + 2个框角点
    sparse_prompt_embeddings = torch.randn(batch_size, num_sparse_prompts, embed_dim, dtype=torch.float32)
    
    # 密集提示嵌入 (来自提示编码器)
    dense_prompt_embeddings = torch.randn(batch_size, embed_dim, 64, 64, dtype=torch.float32)
    
    print(f"测试输入:")
    print(f"  图像嵌入: {image_embeddings.shape}")
    print(f"  稀疏提示嵌入: {sparse_prompt_embeddings.shape}")
    print(f"  密集提示嵌入: {dense_prompt_embeddings.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        try:
            masks, iou_predictions = simple_decoder(
                image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings
            )
            print(f"\n输出:")
            print(f"  掩码: {masks.shape}")
            print(f"  IoU预测: {iou_predictions.shape}")
            
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 导出ONNX
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mask_decoder_simple.onnx")
    
    print(f"\n导出到: {output_path}")
    
    try:
        torch.onnx.export(
            simple_decoder,
            (image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings),
            output_path,
            input_names=["image_embeddings", "sparse_prompt_embeddings", "dense_prompt_embeddings"],
            output_names=["masks", "iou_predictions"],
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            verbose=True
        )
        
        print("✅ 掩码解码器导出成功!")
        
        # 验证导出的模型
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        
        # 测试ONNX模型
        onnx_inputs = {
            "image_embeddings": image_embeddings.numpy(),
            "sparse_prompt_embeddings": sparse_prompt_embeddings.numpy(),
            "dense_prompt_embeddings": dense_prompt_embeddings.numpy()
        }
        
        onnx_outputs = session.run(None, onnx_inputs)
        
        print(f"\nONNX验证:")
        print(f"  ONNX掩码: {onnx_outputs[0].shape}")
        print(f"  ONNX IoU: {onnx_outputs[1].shape}")
        
        # 比较输出差异
        mask_diff = np.mean(np.abs(masks.numpy() - onnx_outputs[0]))
        iou_diff = np.mean(np.abs(iou_predictions.numpy() - onnx_outputs[1]))
        
        print(f"  掩码差异: {mask_diff:.6f}")
        print(f"  IoU差异: {iou_diff:.6f}")
        
        if mask_diff < 1e-4 and iou_diff < 1e-4:
            print("✅ ONNX模型验证通过!")
        else:
            print("⚠️  ONNX模型存在精度差异")
            
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

class SingleMaskDecoder(nn.Module):
    """
    单掩码解码器 - 简化版本
    """
    
    def __init__(self, original_decoder, prompt_encoder):
        super().__init__()
        self.original_decoder = original_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings):
        """
        单掩码输出版本
        """
        image_pe = self.prompt_encoder.get_dense_pe()
        
        masks, iou_predictions, _, _ = self.original_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,  # 单掩码输出
            repeat_image=False,
            high_res_features=None
        )
        
        return masks, iou_predictions

def export_single_mask_decoder():
    """导出单掩码解码器"""
    
    print("\n🎯 导出单掩码解码器")
    print("=" * 35)
    
    # 加载模型
    model_path = os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    config_path = os.path.join(project_root, "sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml")
    
    predictor = build_sam2_video_predictor(config_path, model_path, device="cpu")
    predictor.eval()
    
    # 创建单掩码解码器
    single_decoder = SingleMaskDecoder(predictor.sam_mask_decoder, predictor.sam_prompt_encoder)
    single_decoder.eval()
    
    # 创建测试输入
    batch_size = 1
    embed_dim = 256
    
    image_embeddings = torch.randn(batch_size, embed_dim, 64, 64, dtype=torch.float32)
    sparse_prompt_embeddings = torch.randn(batch_size, 3, embed_dim, dtype=torch.float32)  # 减少提示数量
    dense_prompt_embeddings = torch.randn(batch_size, embed_dim, 64, 64, dtype=torch.float32)
    
    print(f"单掩码解码器输入:")
    print(f"  图像嵌入: {image_embeddings.shape}")
    print(f"  稀疏提示嵌入: {sparse_prompt_embeddings.shape}")
    print(f"  密集提示嵌入: {dense_prompt_embeddings.shape}")
    
    # 测试
    with torch.no_grad():
        try:
            masks, iou_predictions = single_decoder(
                image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings
            )
            print(f"  掩码: {masks.shape}")
            print(f"  IoU预测: {iou_predictions.shape}")
            
        except Exception as e:
            print(f"单掩码解码器测试失败: {e}")
            return False
    
    # 导出
    output_path = os.path.join("onnx_models", "mask_decoder_single.onnx")
    
    try:
        torch.onnx.export(
            single_decoder,
            (image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings),
            output_path,
            input_names=["image_embeddings", "sparse_prompt_embeddings", "dense_prompt_embeddings"],
            output_names=["mask", "iou_prediction"],
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        print(f"✅ 单掩码解码器导出成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 单掩码解码器导出失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 SAMURAI 掩码解码器 ONNX 导出")
    print("=" * 50)
    
    success1 = export_mask_decoder()
    success2 = export_single_mask_decoder()
    
    print("\n" + "=" * 50)
    if success1 or success2:
        print("🎉 至少一个掩码解码器导出成功!")
        
        if success1:
            print("✅ 多掩码解码器: onnx_models/mask_decoder_simple.onnx")
        if success2:
            print("✅ 单掩码解码器: onnx_models/mask_decoder_single.onnx")
    else:
        print("❌ 所有导出都失败了")

if __name__ == "__main__":
    main()
