"""
ONNX Model Validation Script

Compare PyTorch and ONNX model outputs to ensure accuracy.
"""

import os
import sys
import torch
import numpy as np
import argparse
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import onnxruntime as ort
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have installed all dependencies")
    sys.exit(1)

class ModelValidator:
    """Validate ONNX models against PyTorch reference."""
    
    def __init__(self, pytorch_config: str, pytorch_checkpoint: str, 
                 onnx_model_dir: str, device: str = "cpu"):
        """
        Initialize validator.
        
        Args:
            pytorch_config: Path to PyTorch model config
            pytorch_checkpoint: Path to PyTorch checkpoint
            onnx_model_dir: Directory containing ONNX models
            device: Device for PyTorch inference
        """
        self.device = device
        self.onnx_model_dir = Path(onnx_model_dir)
        
        # Load PyTorch model
        print("Loading PyTorch model...")
        self.pytorch_model = build_sam2_video_predictor(
            pytorch_config, pytorch_checkpoint, device=device
        )
        self.pytorch_model.eval()
        
        # Load ONNX models
        print("Loading ONNX models...")
        self.onnx_sessions = self._load_onnx_models()
        
    def _load_onnx_models(self) -> Dict[str, ort.InferenceSession]:
        """Load ONNX model sessions."""
        sessions = {}
        
        # Determine providers
        if self.device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Model files to load
        model_files = {
            'image_encoder': 'image_encoder_base_plus.onnx',
            'prompt_encoder': 'prompt_encoder_base_plus.onnx',
            'mask_decoder': 'mask_decoder_base_plus.onnx',
        }
        
        for model_name, filename in model_files.items():
            model_path = self.onnx_model_dir / filename
            if model_path.exists():
                print(f"  Loading {model_name}")
                sessions[model_name] = ort.InferenceSession(
                    str(model_path), providers=providers
                )
            else:
                print(f"  Warning: {filename} not found")
        
        return sessions
    
    def validate_image_encoder(self, test_images: List[np.ndarray]) -> Dict[str, float]:
        """Validate image encoder accuracy."""
        print("\n=== Validating Image Encoder ===")
        
        if 'image_encoder' not in self.onnx_sessions:
            print("ONNX image encoder not available")
            return {}
        
        results = {
            'max_diff': 0.0,
            'mean_diff': 0.0,
            'relative_error': 0.0,
            'num_tests': 0
        }
        
        total_max_diff = 0.0
        total_mean_diff = 0.0
        total_relative_error = 0.0
        
        for i, image in enumerate(test_images):
            print(f"  Testing image {i+1}/{len(test_images)}")
            
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_input = torch.from_numpy(input_tensor).to(self.device)
                pytorch_output = self.pytorch_model.image_encoder(pytorch_input)
            
            # ONNX inference
            onnx_output = self.onnx_sessions['image_encoder'].run(
                None, {'input_image': input_tensor}
            )
            
            # Compare outputs
            if isinstance(pytorch_output, dict):
                # Handle dictionary output
                for key in pytorch_output.keys():
                    if isinstance(pytorch_output[key], torch.Tensor):
                        pt_tensor = pytorch_output[key].cpu().numpy()
                        # Find corresponding ONNX output (simplified)
                        onnx_tensor = onnx_output[0]  # Assume first output
                        
                        diff_stats = self._compute_diff_stats(pt_tensor, onnx_tensor)
                        total_max_diff = max(total_max_diff, diff_stats['max_diff'])
                        total_mean_diff += diff_stats['mean_diff']
                        total_relative_error += diff_stats['relative_error']
                        break
            else:
                # Handle tensor output
                pt_tensor = pytorch_output.cpu().numpy()
                onnx_tensor = onnx_output[0]
                
                diff_stats = self._compute_diff_stats(pt_tensor, onnx_tensor)
                total_max_diff = max(total_max_diff, diff_stats['max_diff'])
                total_mean_diff += diff_stats['mean_diff']
                total_relative_error += diff_stats['relative_error']
        
        num_tests = len(test_images)
        results.update({
            'max_diff': total_max_diff,
            'mean_diff': total_mean_diff / num_tests,
            'relative_error': total_relative_error / num_tests,
            'num_tests': num_tests
        })
        
        print(f"  Max difference: {results['max_diff']:.6f}")
        print(f"  Mean difference: {results['mean_diff']:.6f}")
        print(f"  Relative error: {results['relative_error']:.6f}")
        
        return results
    
    def validate_prompt_encoder(self) -> Dict[str, float]:
        """Validate prompt encoder accuracy."""
        print("\n=== Validating Prompt Encoder ===")
        
        if 'prompt_encoder' not in self.onnx_sessions:
            print("ONNX prompt encoder not available")
            return {}
        
        # Create test inputs
        point_coords = np.random.rand(1, 5, 2).astype(np.float32) * 512
        point_labels = np.random.randint(0, 2, (1, 5)).astype(np.int32)
        boxes = np.random.rand(1, 4).astype(np.float32) * 512
        mask_input = np.random.rand(1, 1, 256, 256).astype(np.float32)
        
        try:
            # PyTorch inference
            with torch.no_grad():
                pt_point_coords = torch.from_numpy(point_coords).to(self.device)
                pt_point_labels = torch.from_numpy(point_labels).to(self.device)
                pt_boxes = torch.from_numpy(boxes).to(self.device)
                pt_mask_input = torch.from_numpy(mask_input).to(self.device)
                
                pt_sparse, pt_dense = self.pytorch_model.sam_prompt_encoder(
                    points=(pt_point_coords, pt_point_labels),
                    boxes=pt_boxes,
                    masks=pt_mask_input,
                )
            
            # ONNX inference
            onnx_outputs = self.onnx_sessions['prompt_encoder'].run(
                None, {
                    'point_coords': point_coords,
                    'point_labels': point_labels,
                    'boxes': boxes,
                    'mask_input': mask_input
                }
            )
            
            # Compare outputs
            sparse_diff = self._compute_diff_stats(
                pt_sparse.cpu().numpy(), onnx_outputs[0]
            )
            dense_diff = self._compute_diff_stats(
                pt_dense.cpu().numpy(), onnx_outputs[1]
            )
            
            results = {
                'sparse_max_diff': sparse_diff['max_diff'],
                'sparse_mean_diff': sparse_diff['mean_diff'],
                'dense_max_diff': dense_diff['max_diff'],
                'dense_mean_diff': dense_diff['mean_diff'],
            }
            
            print(f"  Sparse embeddings max diff: {results['sparse_max_diff']:.6f}")
            print(f"  Dense embeddings max diff: {results['dense_max_diff']:.6f}")
            
            return results
            
        except Exception as e:
            print(f"  Validation failed: {e}")
            return {}
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Resize to model input size
        target_size = 1024
        h, w = image.shape[:2]
        
        # Maintain aspect ratio
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize and pad
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalize
        normalized = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert to CHW format with batch dimension
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return input_tensor.astype(np.float32)
    
    def _compute_diff_stats(self, tensor1: np.ndarray, tensor2: np.ndarray) -> Dict[str, float]:
        """Compute difference statistics between two tensors."""
        # Ensure same shape
        if tensor1.shape != tensor2.shape:
            print(f"Warning: Shape mismatch {tensor1.shape} vs {tensor2.shape}")
            # Try to reshape or return large error
            return {
                'max_diff': float('inf'),
                'mean_diff': float('inf'),
                'relative_error': float('inf')
            }
        
        diff = np.abs(tensor1 - tensor2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Relative error
        tensor1_norm = np.linalg.norm(tensor1)
        if tensor1_norm > 0:
            relative_error = np.linalg.norm(diff) / tensor1_norm
        else:
            relative_error = 0.0
        
        return {
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'relative_error': float(relative_error)
        }
    
    def generate_test_images(self, num_images: int = 5) -> List[np.ndarray]:
        """Generate test images for validation."""
        images = []
        
        for i in range(num_images):
            # Create random test image
            h, w = np.random.randint(480, 1080), np.random.randint(640, 1920)
            image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            images.append(image)
        
        return images
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("Starting ONNX model validation...")
        
        results = {}
        
        # Generate test data
        test_images = self.generate_test_images(5)
        
        # Validate components
        results['image_encoder'] = self.validate_image_encoder(test_images)
        results['prompt_encoder'] = self.validate_prompt_encoder()
        
        # Summary
        print("\n=== Validation Summary ===")
        for component, metrics in results.items():
            if metrics:
                print(f"{component}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.6f}")
                    else:
                        print(f"  {metric}: {value}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Validate ONNX models")
    parser.add_argument("--pytorch_config", required=True, help="PyTorch model config")
    parser.add_argument("--pytorch_checkpoint", required=True, help="PyTorch checkpoint")
    parser.add_argument("--onnx_models", required=True, help="ONNX models directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pytorch_config):
        print(f"Config file not found: {args.pytorch_config}")
        return
    
    if not os.path.exists(args.pytorch_checkpoint):
        print(f"Checkpoint not found: {args.pytorch_checkpoint}")
        return
    
    if not os.path.exists(args.onnx_models):
        print(f"ONNX models directory not found: {args.onnx_models}")
        return
    
    # Run validation
    validator = ModelValidator(
        args.pytorch_config,
        args.pytorch_checkpoint,
        args.onnx_models,
        args.device
    )
    
    results = validator.run_full_validation()
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
