"""
SAMURAI ONNX Demo Script

Demonstrate ONNX-based inference on a sample video.
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.onnx_inference import SAMURAIONNXPredictor

def create_sample_video(output_path: str, duration: int = 5, fps: int = 30):
    """Create a sample video with a moving object for testing."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_idx in range(total_frames):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add some noise
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Moving object (circle)
        t = frame_idx / total_frames
        center_x = int(100 + (width - 200) * t)
        center_y = int(height // 2 + 50 * np.sin(2 * np.pi * t * 3))
        radius = 30
        
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), -1)
        
        # Add some distractor objects
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 0, 255), -1)
        cv2.rectangle(frame, (width-100, height-100), (width-50, height-50), (255, 0, 0), -1)
        
        writer.write(frame)
    
    writer.release()
    print(f"Sample video created: {output_path}")
    
    # Return initial bounding box for the green circle
    initial_bbox = (100 - radius, height // 2 - radius, 2 * radius, 2 * radius)
    return initial_bbox

def visualize_tracking_results(video_path: str, results: list, output_path: str):
    """Visualize tracking results on the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(results):
            x, y, w, h = results[frame_idx]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add frame number and bbox info
            text = f"Frame {frame_idx}: ({x},{y},{w},{h})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(frame)
        frame_idx += 1
    
    cap.release()
    writer.release()
    print(f"Tracking visualization saved: {output_path}")

def run_demo(model_dir: str, device: str = "cpu", create_sample: bool = True):
    """Run complete ONNX demo."""
    print("=== SAMURAI ONNX Demo ===")
    
    # Paths
    demo_dir = Path("demo_output")
    demo_dir.mkdir(exist_ok=True)
    
    sample_video = demo_dir / "sample_video.mp4"
    output_video = demo_dir / "tracking_result.mp4"
    results_txt = demo_dir / "tracking_results.txt"
    
    # Create or use existing sample video
    if create_sample or not sample_video.exists():
        print("Creating sample video...")
        initial_bbox = create_sample_video(str(sample_video))
    else:
        print(f"Using existing video: {sample_video}")
        # Default bbox for existing video
        initial_bbox = (100, 200, 60, 60)
    
    print(f"Initial bounding box: {initial_bbox}")
    
    # Check if ONNX models exist
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Model directory not found: {model_dir}")
        print("Please run the export script first:")
        print("  python scripts/export_onnx.py --components all")
        return
    
    # Initialize ONNX predictor
    print("Initializing ONNX predictor...")
    try:
        predictor = SAMURAIONNXPredictor(model_dir, device=device)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        print("Make sure ONNX models are exported and onnxruntime is installed")
        return
    
    # Run tracking
    print("Running ONNX tracking...")
    try:
        results = predictor.track_video(
            str(sample_video), 
            initial_bbox, 
            output_path=None  # We'll create visualization separately
        )
        
        print(f"Tracking completed! Processed {len(results)} frames")
        
        # Save results to text file
        with open(results_txt, 'w') as f:
            for i, bbox in enumerate(results):
                f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
        
        print(f"Results saved to: {results_txt}")
        
        # Create visualization
        print("Creating tracking visualization...")
        visualize_tracking_results(str(sample_video), results, str(output_video))
        
        # Print some statistics
        print("\n=== Tracking Statistics ===")
        print(f"Total frames: {len(results)}")
        
        if results:
            # Calculate average bbox size
            avg_width = np.mean([bbox[2] for bbox in results])
            avg_height = np.mean([bbox[3] for bbox in results])
            print(f"Average bbox size: {avg_width:.1f} x {avg_height:.1f}")
            
            # Calculate movement
            if len(results) > 1:
                movements = []
                for i in range(1, len(results)):
                    prev_center = (results[i-1][0] + results[i-1][2]/2, results[i-1][1] + results[i-1][3]/2)
                    curr_center = (results[i][0] + results[i][2]/2, results[i][1] + results[i][3]/2)
                    movement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                    movements.append(movement)
                
                avg_movement = np.mean(movements)
                print(f"Average movement per frame: {avg_movement:.2f} pixels")
        
        print(f"\nDemo completed successfully!")
        print(f"Check the following files:")
        print(f"  - Input video: {sample_video}")
        print(f"  - Output video: {output_video}")
        print(f"  - Results text: {results_txt}")
        
    except Exception as e:
        print(f"Tracking failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="SAMURAI ONNX Demo")
    parser.add_argument("--model_dir", default="onnx_models", help="ONNX models directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--video_path", help="Path to existing video (optional)")
    parser.add_argument("--bbox", help="Initial bbox as 'x,y,w,h' (required if using existing video)")
    parser.add_argument("--no_sample", action="store_true", help="Don't create sample video")
    
    args = parser.parse_args()
    
    if args.video_path:
        # Use existing video
        if not args.bbox:
            print("Error: --bbox is required when using existing video")
            return
        
        bbox_parts = args.bbox.split(',')
        if len(bbox_parts) != 4:
            print("Error: bbox must be in format 'x,y,w,h'")
            return
        
        initial_bbox = tuple(map(int, bbox_parts))
        
        print("Using existing video...")
        try:
            predictor = SAMURAIONNXPredictor(args.model_dir, device=args.device)
            results = predictor.track_video(args.video_path, initial_bbox)
            
            # Save results
            output_txt = args.video_path.replace('.mp4', '_onnx_results.txt')
            with open(output_txt, 'w') as f:
                for bbox in results:
                    f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
            
            print(f"Results saved to: {output_txt}")
            
        except Exception as e:
            print(f"Tracking failed: {e}")
    else:
        # Run full demo
        run_demo(args.model_dir, args.device, not args.no_sample)

if __name__ == "__main__":
    main()
