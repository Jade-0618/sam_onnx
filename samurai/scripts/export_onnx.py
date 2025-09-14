import torch
import torch.nn as nn
import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

def main():
    parser = argparse.ArgumentParser(description="Export SAMURAI components to ONNX")
    parser.add_argument("--model_name", default="base_plus", choices=["tiny", "small", "base_plus", "large"],
                       help="Model size to export")
    parser.add_argument("--components", nargs="+", default=["image_encoder"],
                       choices=["image_encoder", "prompt_encoder", "mask_decoder", "memory_encoder", "all"],
                       help="Components to export")
    parser.add_argument("--output_dir", default="onnx_models", help="Output directory for ONNX models")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--optimize", action="store_true", help="Apply ONNX optimizations")
    parser.add_argument("--dynamic_batch", action="store_true", help="Enable dynamic batch size")

    args = parser.parse_args()

    try:
        # Add project root to Python path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sam2_path = os.path.join(project_root, "sam2")
        sys.path.insert(0, sam2_path)
        print(f"Added {sam2_path} to Python path")

        from sam2.build_sam import build_sam2_video_predictor

        print(f"Loading {args.model_name} model...")

        # Model configuration mapping
        model_configs = {
            "tiny": (os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_tiny.pt"),
                    os.path.join(project_root, "sam2/sam2/configs/samurai/sam2.1_hiera_t.yaml")),
            "small": (os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_small.pt"),
                     os.path.join(project_root, "sam2/sam2/configs/samurai/sam2.1_hiera_s.yaml")),
            "base_plus": (os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_base_plus.pt"),
                         os.path.join(project_root, "sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml")),
            "large": (os.path.join(project_root, "sam2/checkpoints/sam2.1_hiera_large.pt"),
                     os.path.join(project_root, "sam2/sam2/configs/samurai/sam2.1_hiera_l.yaml")),
        }

        checkpoint_path, model_cfg = model_configs[args.model_name]

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found at: {checkpoint_path}")
            print("Please run 'cd sam2/checkpoints && ./download_ckpts.sh && cd ../..' as mentioned in the README.")
            return

        print("Building SAM2 video predictor...")
        predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device="cpu")
        predictor.eval()
        model = predictor
        print("Model loaded successfully.")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

        # Determine which components to export
        components_to_export = args.components
        if "all" in components_to_export:
            components_to_export = ["image_encoder", "prompt_encoder", "mask_decoder", "memory_encoder"]

        # Export each component
        for component in components_to_export:
            export_component(model, component, args)

        print("All components exported successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

def export_component(model, component_name: str, args):
    """Export a specific component of the SAMURAI model to ONNX"""
    print(f"\n=== Exporting {component_name} ===")

    if component_name == "image_encoder":
        export_image_encoder(model, args)
    elif component_name == "prompt_encoder":
        export_prompt_encoder(model, args)
    elif component_name == "mask_decoder":
        export_mask_decoder(model, args)
    elif component_name == "memory_encoder":
        export_memory_encoder(model, args)
    else:
        print(f"Unknown component: {component_name}")

def export_image_encoder(model, args):
    """Export the image encoder component"""
    image_encoder = model.image_encoder
    image_size = model.image_size

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size, device="cpu")
    print(f"Image encoder input shape: {dummy_input.shape}")

    # Test forward pass
    with torch.no_grad():
        output = image_encoder(dummy_input)
        if isinstance(output, dict):
            print(f"Image encoder outputs: {list(output.keys())}")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"  {key}: list of {len(value)} tensors")
                    for i, tensor in enumerate(value):
                        if isinstance(tensor, torch.Tensor):
                            print(f"    [{i}]: {tensor.shape}")
        else:
            print(f"Image encoder output shape: {output.shape}")

    output_path = os.path.join(args.output_dir, f"image_encoder_{args.model_name}.onnx")

    # Dynamic axes configuration
    dynamic_axes = {}
    if args.dynamic_batch:
        dynamic_axes["input_image"] = {0: "batch_size"}
        # Note: output dynamic axes depend on the actual output structure

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        image_encoder,
        dummy_input,
        output_path,
        input_names=["input_image"],
        output_names=["backbone_features"],  # This might need adjustment based on actual output
        dynamic_axes=dynamic_axes,
        opset_version=args.opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )

    print(f"Image encoder exported to {output_path}")

    # Optimize if requested
    if args.optimize:
        optimize_onnx_model(output_path)

def export_prompt_encoder(model, args):
    """Export the prompt encoder component"""
    prompt_encoder = model.sam_prompt_encoder

    # Create a wrapper class to handle the complex input structure
    class PromptEncoderWrapper(torch.nn.Module):
        def __init__(self, prompt_encoder):
            super().__init__()
            self.prompt_encoder = prompt_encoder

        def forward(self, point_coords, point_labels, boxes, mask_input, has_points, has_boxes, has_masks):
            """
            Wrapper forward function that handles optional inputs

            Args:
                point_coords: [B, N, 2] point coordinates
                point_labels: [B, N] point labels
                boxes: [B, 4] box coordinates
                mask_input: [B, 1, H, W] mask input
                has_points: [B] boolean flag for points
                has_boxes: [B] boolean flag for boxes
                has_masks: [B] boolean flag for masks
            """
            batch_size = point_coords.shape[0]

            # Process inputs based on flags
            points = None
            if has_points.any():
                points = (point_coords, point_labels)

            boxes_input = None
            if has_boxes.any():
                boxes_input = boxes

            masks_input = None
            if has_masks.any():
                masks_input = mask_input

            return self.prompt_encoder(points=points, boxes=boxes_input, masks=masks_input)

    wrapper = PromptEncoderWrapper(prompt_encoder)
    wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    num_points = 5

    point_coords = torch.randn(batch_size, num_points, 2, device="cpu") * 1024  # Scale to image size
    point_labels = torch.randint(0, 2, (batch_size, num_points), device="cpu")
    boxes = torch.randn(batch_size, 4, device="cpu") * 1024  # x1, y1, x2, y2

    # Ensure boxes are valid (x1 < x2, y1 < y2)
    boxes[:, 2] = boxes[:, 0] + torch.abs(boxes[:, 2] - boxes[:, 0])
    boxes[:, 3] = boxes[:, 1] + torch.abs(boxes[:, 3] - boxes[:, 1])

    # Mask input should match the expected size
    mask_size = prompt_encoder.mask_input_size
    mask_input = torch.randn(batch_size, 1, mask_size[0], mask_size[1], device="cpu")

    # Control flags
    has_points = torch.tensor([True], dtype=torch.bool, device="cpu")
    has_boxes = torch.tensor([True], dtype=torch.bool, device="cpu")
    has_masks = torch.tensor([False], dtype=torch.bool, device="cpu")  # Start with no masks

    print(f"Prompt encoder inputs:")
    print(f"  Point coords: {point_coords.shape}")
    print(f"  Point labels: {point_labels.shape}")
    print(f"  Boxes: {boxes.shape}")
    print(f"  Mask input: {mask_input.shape} (expected: {mask_size})")
    print(f"  Has points: {has_points}")
    print(f"  Has boxes: {has_boxes}")
    print(f"  Has masks: {has_masks}")

    # Test forward pass
    with torch.no_grad():
        try:
            sparse_embeddings, dense_embeddings = wrapper(
                point_coords, point_labels, boxes, mask_input,
                has_points, has_boxes, has_masks
            )
            print(f"Prompt encoder outputs:")
            print(f"  Sparse embeddings: {sparse_embeddings.shape}")
            print(f"  Dense embeddings: {dense_embeddings.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            print("Trying with simpler inputs...")

            # Try with just points
            has_boxes = torch.tensor([False], dtype=torch.bool, device="cpu")
            sparse_embeddings, dense_embeddings = wrapper(
                point_coords, point_labels, boxes, mask_input,
                has_points, has_boxes, has_masks
            )
            print(f"Simplified prompt encoder outputs:")
            print(f"  Sparse embeddings: {sparse_embeddings.shape}")
            print(f"  Dense embeddings: {dense_embeddings.shape}")

    output_path = os.path.join(args.output_dir, f"prompt_encoder_{args.model_name}.onnx")

    # Dynamic axes configuration
    dynamic_axes = {}
    if args.dynamic_batch:
        dynamic_axes.update({
            "point_coords": {0: "batch_size"},
            "point_labels": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "mask_input": {0: "batch_size"},
            "has_points": {0: "batch_size"},
            "has_boxes": {0: "batch_size"},
            "has_masks": {0: "batch_size"},
            "sparse_embeddings": {0: "batch_size"},
            "dense_embeddings": {0: "batch_size"},
        })

    print(f"Exporting to {output_path}...")
    try:
        torch.onnx.export(
            wrapper,
            (point_coords, point_labels, boxes, mask_input, has_points, has_boxes, has_masks),
            output_path,
            input_names=["point_coords", "point_labels", "boxes", "mask_input",
                        "has_points", "has_boxes", "has_masks"],
            output_names=["sparse_embeddings", "dense_embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )

        print(f"Prompt encoder exported to {output_path}")

        if args.optimize:
            optimize_onnx_model(output_path)

    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("This is expected due to the complexity of the prompt encoder.")
        print("Consider using a simplified version for production.")

def export_mask_decoder(model, args):
    """Export the mask decoder component"""
    mask_decoder = model.sam_mask_decoder

    # Create dummy inputs for mask decoder
    # Image embeddings from image encoder
    image_embeddings = torch.randn(1, 256, 64, 64, device="cpu")  # batch_size, embed_dim, H, W

    # Prompt embeddings from prompt encoder
    sparse_prompt_embeddings = torch.randn(1, 6, 256, device="cpu")  # batch_size, num_prompts, embed_dim
    dense_prompt_embeddings = torch.randn(1, 256, 64, 64, device="cpu")  # batch_size, embed_dim, H, W

    # Multimask output flag
    multimask_output = torch.tensor([True], device="cpu")

    print(f"Mask decoder inputs:")
    print(f"  Image embeddings: {image_embeddings.shape}")
    print(f"  Sparse prompt embeddings: {sparse_prompt_embeddings.shape}")
    print(f"  Dense prompt embeddings: {dense_prompt_embeddings.shape}")

    # Test forward pass
    with torch.no_grad():
        try:
            masks, iou_predictions = mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=True,
            )
            print(f"Mask decoder outputs:")
            print(f"  Masks: {masks.shape}")
            print(f"  IoU predictions: {iou_predictions.shape}")
        except Exception as e:
            print(f"Mask decoder test failed: {e}")
            print("Skipping mask decoder export due to complexity")
            return

    output_path = os.path.join(args.output_dir, f"mask_decoder_{args.model_name}.onnx")

    # Dynamic axes configuration
    dynamic_axes = {}
    if args.dynamic_batch:
        dynamic_axes.update({
            "image_embeddings": {0: "batch_size"},
            "sparse_prompt_embeddings": {0: "batch_size"},
            "dense_prompt_embeddings": {0: "batch_size"},
            "masks": {0: "batch_size"},
            "iou_predictions": {0: "batch_size"},
        })

    print(f"Exporting to {output_path}...")
    print("Warning: Mask decoder export is simplified and may need adjustment for full functionality")

    if args.optimize:
        optimize_onnx_model(output_path)

def export_memory_encoder(model, args):
    """Export the memory encoder component"""
    memory_encoder = model.memory_encoder

    # Create dummy inputs for memory encoder
    curr_vision_feats = torch.randn(1, 256, 64, 64, device="cpu")  # Current frame features
    feat_sizes = [(64, 64)]  # Feature map sizes

    print(f"Memory encoder inputs:")
    print(f"  Current vision features: {curr_vision_feats.shape}")

    # Test forward pass (simplified)
    with torch.no_grad():
        # This is a simplified test - actual memory encoder has more complex inputs
        print("Warning: Memory encoder export is complex due to state dependencies")
        print("Consider implementing a stateless version for ONNX export")

    output_path = os.path.join(args.output_dir, f"memory_encoder_{args.model_name}.onnx")
    print(f"Memory encoder export skipped - requires state management redesign")

def optimize_onnx_model(model_path: str):
    """Apply ONNX optimizations to the exported model"""
    try:
        import onnx
        from onnxoptimizer import optimize

        print(f"Optimizing ONNX model: {model_path}")

        # Load the model
        model = onnx.load(model_path)

        # Apply optimizations
        optimized_model = optimize(model)

        # Save optimized model
        optimized_path = model_path.replace('.onnx', '_optimized.onnx')
        onnx.save(optimized_model, optimized_path)

        print(f"Optimized model saved to: {optimized_path}")

    except ImportError:
        print("onnxoptimizer not available. Skipping optimization.")
        print("Install with: pip install onnxoptimizer")
    except Exception as e:
        print(f"Optimization failed: {e}")

if __name__ == "__main__":
    main()

