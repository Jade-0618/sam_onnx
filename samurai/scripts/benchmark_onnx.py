"""
ONNX Performance Benchmark Script

Benchmark ONNX models performance across different configurations.
"""

import os
import sys
import time
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from statistics import mean, stdev

try:
    import onnxruntime as ort
    import psutil
    import GPUtil
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install onnxruntime psutil gputil")
    sys.exit(1)

class ONNXBenchmark:
    """Benchmark ONNX models performance."""
    
    def __init__(self, model_dir: str, device: str = "cpu"):
        """
        Initialize benchmark.
        
        Args:
            model_dir: Directory containing ONNX models
            device: Device for inference ("cpu" or "cuda")
        """
        self.model_dir = Path(model_dir)
        self.device = device
        
        # Setup providers
        if device == "cuda":
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
        
        # Load models
        self.sessions = self._load_models()
        
        # System info
        self.system_info = self._get_system_info()
        
    def _load_models(self) -> Dict[str, ort.InferenceSession]:
        """Load ONNX model sessions."""
        sessions = {}
        
        model_files = {
            'image_encoder': 'image_encoder_base_plus.onnx',
            'prompt_encoder': 'prompt_encoder_base_plus.onnx',
            'mask_decoder': 'mask_decoder_base_plus.onnx',
        }
        
        for model_name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                print(f"Loading {model_name}")
                
                # Configure session options for performance
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                
                # Enable parallel execution for CPU
                if self.device == "cpu":
                    session_options.intra_op_num_threads = psutil.cpu_count(logical=False)
                    session_options.inter_op_num_threads = 1
                
                sessions[model_name] = ort.InferenceSession(
                    str(model_path), 
                    sess_options=session_options,
                    providers=self.providers
                )
            else:
                print(f"Warning: {filename} not found")
        
        return sessions
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform,
            'onnxruntime_version': ort.__version__,
        }
        
        # GPU info
        if self.device == "cuda":
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info.update({
                        'gpu_name': gpu.name,
                        'gpu_memory_gb': gpu.memoryTotal / 1024,
                        'gpu_driver': gpu.driver,
                    })
            except:
                info['gpu_info'] = "Not available"
        
        return info
    
    def benchmark_image_encoder(self, batch_sizes: List[int] = [1, 2, 4], 
                               num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark image encoder performance."""
        print("\n=== Benchmarking Image Encoder ===")
        
        if 'image_encoder' not in self.sessions:
            return {'error': 'Image encoder not available'}
        
        session = self.sessions['image_encoder']
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size {batch_size}")
            
            # Create dummy input
            input_shape = (batch_size, 3, 1024, 1024)
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                session.run(None, {'input_image': dummy_input})
            
            # Benchmark
            times = []
            memory_usage = []
            
            for i in range(num_runs):
                # Memory before
                if self.device == "cuda":
                    try:
                        gpu_mem_before = GPUtil.getGPUs()[0].memoryUsed
                    except:
                        gpu_mem_before = 0
                else:
                    cpu_mem_before = psutil.Process().memory_info().rss / (1024**2)
                
                # Time inference
                start_time = time.perf_counter()
                outputs = session.run(None, {'input_image': dummy_input})
                end_time = time.perf_counter()
                
                inference_time = end_time - start_time
                times.append(inference_time)
                
                # Memory after
                if self.device == "cuda":
                    try:
                        gpu_mem_after = GPUtil.getGPUs()[0].memoryUsed
                        memory_usage.append(gpu_mem_after - gpu_mem_before)
                    except:
                        memory_usage.append(0)
                else:
                    cpu_mem_after = psutil.Process().memory_info().rss / (1024**2)
                    memory_usage.append(cpu_mem_after - cpu_mem_before)
                
                if (i + 1) % 20 == 0:
                    print(f"    Completed {i + 1}/{num_runs} runs")
            
            # Calculate statistics
            batch_results = {
                'batch_size': batch_size,
                'input_shape': input_shape,
                'num_runs': num_runs,
                'mean_time_ms': mean(times) * 1000,
                'std_time_ms': stdev(times) * 1000 if len(times) > 1 else 0,
                'min_time_ms': min(times) * 1000,
                'max_time_ms': max(times) * 1000,
                'throughput_fps': batch_size / mean(times),
                'mean_memory_mb': mean(memory_usage) if memory_usage else 0,
            }
            
            results[f'batch_{batch_size}'] = batch_results
            
            print(f"    Mean time: {batch_results['mean_time_ms']:.2f} ms")
            print(f"    Throughput: {batch_results['throughput_fps']:.2f} FPS")
        
        return results
    
    def benchmark_prompt_encoder(self, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark prompt encoder performance."""
        print("\n=== Benchmarking Prompt Encoder ===")
        
        if 'prompt_encoder' not in self.sessions:
            return {'error': 'Prompt encoder not available'}
        
        session = self.sessions['prompt_encoder']
        
        # Create dummy inputs
        point_coords = np.random.rand(1, 5, 2).astype(np.float32) * 512
        point_labels = np.random.randint(0, 2, (1, 5)).astype(np.int32)
        boxes = np.random.rand(1, 4).astype(np.float32) * 512
        mask_input = np.random.rand(1, 1, 256, 256).astype(np.float32)
        
        inputs = {
            'point_coords': point_coords,
            'point_labels': point_labels,
            'boxes': boxes,
            'mask_input': mask_input
        }
        
        # Warmup
        for _ in range(10):
            try:
                session.run(None, inputs)
            except Exception as e:
                return {'error': f'Prompt encoder inference failed: {e}'}
        
        # Benchmark
        times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            outputs = session.run(None, inputs)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")
        
        results = {
            'num_runs': num_runs,
            'mean_time_ms': mean(times) * 1000,
            'std_time_ms': stdev(times) * 1000 if len(times) > 1 else 0,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
        }
        
        print(f"  Mean time: {results['mean_time_ms']:.2f} ms")
        
        return results
    
    def benchmark_end_to_end(self, num_frames: int = 100) -> Dict[str, Any]:
        """Benchmark end-to-end video processing."""
        print(f"\n=== Benchmarking End-to-End ({num_frames} frames) ===")
        
        if 'image_encoder' not in self.sessions:
            return {'error': 'Image encoder not available'}
        
        # Simulate video processing
        frame_times = []
        total_start = time.perf_counter()
        
        for frame_idx in range(num_frames):
            frame_start = time.perf_counter()
            
            # Generate dummy frame
            dummy_frame = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
            
            # Image encoding
            image_features = self.sessions['image_encoder'].run(
                None, {'input_image': dummy_frame}
            )
            
            # Simulate other processing (Kalman filter, etc.)
            time.sleep(0.001)  # 1ms simulation
            
            frame_end = time.perf_counter()
            frame_times.append(frame_end - frame_start)
            
            if (frame_idx + 1) % 20 == 0:
                print(f"  Processed {frame_idx + 1}/{num_frames} frames")
        
        total_end = time.perf_counter()
        total_time = total_end - total_start
        
        results = {
            'num_frames': num_frames,
            'total_time_s': total_time,
            'mean_frame_time_ms': mean(frame_times) * 1000,
            'std_frame_time_ms': stdev(frame_times) * 1000 if len(frame_times) > 1 else 0,
            'fps': num_frames / total_time,
            'real_time_factor': (num_frames / 30) / total_time,  # Assuming 30 FPS video
        }
        
        print(f"  Total time: {results['total_time_s']:.2f} s")
        print(f"  Average FPS: {results['fps']:.2f}")
        print(f"  Real-time factor: {results['real_time_factor']:.2f}x")
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("Starting ONNX performance benchmark...")
        print(f"Device: {self.device}")
        print(f"Providers: {self.providers}")
        
        results = {
            'system_info': self.system_info,
            'device': self.device,
            'providers': self.providers,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Run benchmarks
        results['image_encoder'] = self.benchmark_image_encoder([1])  # Only test batch size 1
        results['prompt_encoder'] = self.benchmark_prompt_encoder()
        results['end_to_end'] = self.benchmark_end_to_end(50)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models")
    parser.add_argument("--model_dir", required=True, help="ONNX models directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4], 
                       help="Batch sizes to test")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of runs per test")
    parser.add_argument("--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Model directory not found: {args.model_dir}")
        return
    
    # Run benchmark
    benchmark = ONNXBenchmark(args.model_dir, args.device)
    results = benchmark.run_full_benchmark()
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    if 'image_encoder' in results and 'batch_1' in results['image_encoder']:
        ie_results = results['image_encoder']['batch_1']
        print(f"Image Encoder (batch=1): {ie_results['mean_time_ms']:.2f} ms, {ie_results['throughput_fps']:.2f} FPS")
    
    if 'prompt_encoder' in results and 'mean_time_ms' in results['prompt_encoder']:
        pe_results = results['prompt_encoder']
        print(f"Prompt Encoder: {pe_results['mean_time_ms']:.2f} ms")
    
    if 'end_to_end' in results and 'fps' in results['end_to_end']:
        e2e_results = results['end_to_end']
        print(f"End-to-End: {e2e_results['fps']:.2f} FPS ({e2e_results['real_time_factor']:.2f}x real-time)")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
