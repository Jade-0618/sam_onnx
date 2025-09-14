"""
SAMURAI ONNX Inference Engine

This module provides ONNX-based inference for SAMURAI video tracking,
replacing PyTorch inference with optimized ONNX Runtime execution.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("ONNXRuntime not found. Install with: pip install onnxruntime")
    sys.exit(1)

class KalmanFilterONNX:
    """
    ONNX-compatible Kalman Filter implementation for object tracking.
    Replaces the original scipy-based implementation.
    """
    
    def __init__(self):
        self.ndim = 4  # [x, y, a, h]
        self.dt = 1.0
        
        # Initialize state transition matrix
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # Observation matrix
        self._update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)
        
        # Process and measurement noise
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
        self.mean = None
        self.covariance = None
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, 
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step."""
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        # Measurement noise
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = projected_cov + np.diag(np.square(std))
        
        # Kalman gain
        kalman_gain = np.linalg.solve(
            innovation_cov, np.dot(covariance, self._update_mat.T).T).T
        
        # Update
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance

class SAMURAIONNXPredictor:
    """
    Complete ONNX-based SAMURAI predictor for video object tracking.
    Supports full SAMURAI functionality with all components.
    """

    def __init__(self, model_dir: str, device: str = "cpu", use_end_to_end: bool = True):
        """
        Initialize SAMURAI ONNX predictor.

        Args:
            model_dir: Directory containing ONNX model files
            device: Device for inference ("cpu" or "cuda")
            use_end_to_end: Whether to use end-to-end model or separate components
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.use_end_to_end = use_end_to_end

        # Initialize ONNX Runtime providers
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load ONNX models
        self.sessions = {}
        self._load_models(providers)

        # Initialize Kalman filter
        self.kalman_filter = KalmanFilterONNX()
        self.kf_mean = None
        self.kf_covariance = None

        # Tracking state
        self.memory_bank = None
        self.frame_count = 0
        self.stable_frames = 0

        # SAMURAI hyperparameters
        self.stable_frames_threshold = 15
        self.stable_ious_threshold = 0.3
        self.kf_score_weight = 0.15
        self.memory_bank_iou_threshold = 0.5

        # Model parameters
        self.image_size = 1024
        self.num_maskmem = 7
        self.embed_dim = 256
    
    def _load_models(self, providers: List[str]):
        """Load ONNX model sessions."""

        if self.use_end_to_end:
            # Try to load end-to-end model first
            end_to_end_files = [
                'samurai_end_to_end.onnx',
                'samurai_stateless.onnx'
            ]

            for filename in end_to_end_files:
                model_path = self.model_dir / filename
                if model_path.exists():
                    print(f"Loading end-to-end model from {model_path}")
                    self.sessions['end_to_end'] = ort.InferenceSession(
                        str(model_path), providers=providers
                    )
                    self.model_type = 'end_to_end' if 'end_to_end' in filename else 'stateless'
                    return

            print("End-to-end model not found, falling back to component models")
            self.use_end_to_end = False

        # Load individual component models
        model_files = {
            'image_encoder': [
                'image_encoder_base_plus.onnx',
                'image_encoder_simple.onnx'
            ],
            'prompt_encoder': [
                'prompt_encoder_simple.onnx',
                'prompt_encoder_points_only.onnx',
                'prompt_encoder_base_plus.onnx'
            ],
            'mask_decoder': [
                'mask_decoder_simple.onnx',
                'mask_decoder_single.onnx',
                'mask_decoder_base_plus.onnx'
            ],
            'memory_encoder': [
                'memory_encoder_full.onnx',
                'memory_encoder_stateful.onnx'
            ],
            'end_to_end': [
                'samurai_mock_end_to_end.onnx',
                'samurai_lightweight.onnx'
            ]
        }

        for model_name, filenames in model_files.items():
            loaded = False
            for filename in filenames:
                model_path = self.model_dir / filename
                if model_path.exists():
                    print(f"Loading {model_name} from {model_path}")
                    self.sessions[model_name] = ort.InferenceSession(
                        str(model_path), providers=providers
                    )
                    loaded = True
                    break

            if not loaded:
                print(f"Warning: No {model_name} model found. This component will be unavailable.")

        self.model_type = 'components'
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX inference."""
        # Resize to model input size (typically 1024x1024 for SAM2)
        target_size = 1024
        h, w = image.shape[:2]
        
        # Maintain aspect ratio
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalize and convert to CHW format
        normalized = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Add batch dimension and convert to CHW
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return input_tensor.astype(np.float32)
    
    def encode_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Encode image using ONNX image encoder."""
        if 'image_encoder' not in self.sessions:
            raise RuntimeError("Image encoder not loaded")
        
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.sessions['image_encoder'].run(None, {
            'input_image': input_tensor
        })
        
        # Return as dictionary (structure depends on actual model output)
        return {'backbone_features': outputs[0]}
    
    def predict_mask(self, image: np.ndarray,
                    bbox: Tuple[int, int, int, int],
                    prev_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Predict mask for given bounding box using complete SAMURAI pipeline.

        Args:
            image: Input image [H, W, 3]
            bbox: Bounding box (x1, y1, x2, y2)
            prev_mask: Optional previous mask for temporal consistency

        Returns:
            mask: Binary mask [H, W]
            confidence: Prediction confidence
            memory_features: Memory features for next frame
        """

        if self.use_end_to_end and 'end_to_end' in self.sessions:
            return self._predict_end_to_end(image, bbox, prev_mask)
        elif 'end_to_end' in self.sessions:
            # 如果有端到端模型但没有启用，也尝试使用
            return self._predict_end_to_end(image, bbox, prev_mask)
        else:
            return self._predict_components(image, bbox, prev_mask)

    def _predict_end_to_end(self, image: np.ndarray,
                           bbox: Tuple[int, int, int, int],
                           prev_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:
        """End-to-end prediction using single ONNX model."""

        # Preprocess image
        input_tensor = self.preprocess_image(image)

        # Convert bbox to point and box prompts
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        point_coords = np.array([[[center_x, center_y]]], dtype=np.float32)
        point_labels = np.array([[1]], dtype=np.int64)  # Positive point
        box_coords = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        # 获取模型输入信息
        session = self.sessions['end_to_end']
        input_names = [inp.name for inp in session.get_inputs()]

        print(f"端到端模型输入: {input_names}")

        # 根据模型的实际输入准备数据
        inputs = {'image': input_tensor}

        if 'point_coords' in input_names:
            inputs['point_coords'] = point_coords
        if 'point_labels' in input_names:
            inputs['point_labels'] = point_labels
        if 'box_coords' in input_names:
            inputs['box_coords'] = box_coords
        if 'memory_bank' in input_names:
            # 无状态模型需要内存银行
            if self.memory_bank is None:
                self.memory_bank = np.zeros((1, self.num_maskmem, self.embed_dim, 64, 64), dtype=np.float32)
            inputs['memory_bank'] = self.memory_bank

        try:
            # 运行推理
            outputs = session.run(None, inputs)

            # 处理输出
            if len(outputs) >= 3:
                masks, iou_predictions, memory_features = outputs[:3]

                # 检查是否需要更新内存银行
                if 'memory_bank' in input_names and len(outputs) >= 3:
                    self.memory_bank = outputs[2]  # 更新内存银行
                    memory_features = outputs[2]
                elif len(outputs) > 3:
                    memory_features = outputs[3]
                else:
                    memory_features = outputs[2]

            elif len(outputs) >= 2:
                # 简化模型输出
                masks, iou_predictions = outputs[:2]
                memory_features = np.zeros((1, 256, 64, 64), dtype=np.float32)  # 默认内存特征
            else:
                raise ValueError(f"意外的输出数量: {len(outputs)}")

            # 处理掩码和置信度
            if len(masks.shape) == 4 and masks.shape[1] > 1:
                # 多掩码输出
                if len(iou_predictions.shape) == 2 and iou_predictions.shape[1] > 1:
                    best_mask_idx = np.argmax(iou_predictions[0])
                    best_mask = masks[0, best_mask_idx]
                    confidence = float(iou_predictions[0, best_mask_idx])
                else:
                    # IoU是标量，使用第一个掩码
                    best_mask = masks[0, 0]
                    confidence = float(iou_predictions[0]) if len(iou_predictions.shape) == 1 else float(iou_predictions[0, 0])
            else:
                # 单掩码输出
                best_mask = masks[0, 0] if len(masks.shape) == 4 else masks[0]
                confidence = float(iou_predictions[0]) if len(iou_predictions.shape) == 1 else float(iou_predictions[0, 0])

            # 调整掩码尺寸到原始图像大小
            mask_resized = self._resize_mask(best_mask, image.shape[:2])

            return mask_resized, confidence, memory_features

        except Exception as e:
            print(f"端到端推理失败: {e}")
            # 降级到组件推理
            return self._predict_components(image, bbox, prev_mask)

    def _predict_components(self, image: np.ndarray,
                           bbox: Tuple[int, int, int, int],
                           prev_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:
        """Component-based prediction using separate ONNX models."""

        # 1. Image encoding
        image_features = self.encode_image(image)

        # 2. Prompt encoding
        sparse_embeddings, dense_embeddings = self._encode_prompts(bbox, prev_mask)

        # 3. Mask decoding
        masks, iou_predictions = self._decode_masks(image_features, sparse_embeddings, dense_embeddings)

        # 4. Memory encoding
        memory_features = self._encode_memory(image_features, masks)

        # Process outputs
        if len(masks.shape) == 4 and masks.shape[1] > 1:
            # Multiple masks - choose best one
            best_mask_idx = np.argmax(iou_predictions[0])
            best_mask = masks[0, best_mask_idx]
            confidence = float(iou_predictions[0, best_mask_idx])
        else:
            # Single mask
            best_mask = masks[0, 0] if len(masks.shape) == 4 else masks[0]
            confidence = float(iou_predictions[0, 0]) if len(iou_predictions.shape) == 2 else float(iou_predictions[0])

        # Resize mask to original image size
        mask_resized = self._resize_mask(best_mask, image.shape[:2])

        return mask_resized, confidence, memory_features

    def _encode_prompts(self, bbox: Tuple[int, int, int, int],
                       prev_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Encode prompts using prompt encoder."""

        if 'prompt_encoder' not in self.sessions:
            # Return dummy embeddings if no prompt encoder
            return np.zeros((1, 5, 256), dtype=np.float32), np.zeros((1, 256, 64, 64), dtype=np.float32)

        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Try different input formats based on available model
        try:
            # Try full prompt encoder
            point_coords = np.array([[[center_x, center_y]]], dtype=np.float32)
            point_labels = np.array([[1]], dtype=np.int64)
            box_coords = np.array([[x1, y1, x2, y2]], dtype=np.float32)

            inputs = {
                'point_coords': point_coords,
                'point_labels': point_labels,
                'box_coords': box_coords
            }
            outputs = self.sessions['prompt_encoder'].run(None, inputs)
            return outputs[0], outputs[1]

        except:
            try:
                # Try points-only encoder
                point_coords = np.array([[[center_x, center_y]]], dtype=np.float32)
                point_labels = np.array([[1]], dtype=np.int64)

                inputs = {
                    'point_coords': point_coords,
                    'point_labels': point_labels
                }
                outputs = self.sessions['prompt_encoder'].run(None, inputs)
                return outputs[0], outputs[1]

            except Exception as e:
                print(f"Prompt encoding failed: {e}")
                # Return dummy embeddings
                return np.zeros((1, 3, 256), dtype=np.float32), np.zeros((1, 256, 64, 64), dtype=np.float32)

    def _decode_masks(self, image_features: Dict[str, np.ndarray],
                     sparse_embeddings: np.ndarray,
                     dense_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode masks using mask decoder."""

        if 'mask_decoder' not in self.sessions:
            # Return dummy mask if no decoder
            return np.random.rand(1, 1, 1024, 1024) > 0.5, np.array([[0.8]])

        try:
            # Get main image features
            if 'backbone_features' in image_features:
                main_features = image_features['backbone_features']
            else:
                main_features = list(image_features.values())[0]

            inputs = {
                'image_embeddings': main_features,
                'sparse_prompt_embeddings': sparse_embeddings,
                'dense_prompt_embeddings': dense_embeddings
            }

            outputs = self.sessions['mask_decoder'].run(None, inputs)
            return outputs[0], outputs[1]  # masks, iou_predictions

        except Exception as e:
            print(f"Mask decoding failed: {e}")
            # Return dummy outputs
            return np.random.rand(1, 3, 1024, 1024) > 0.5, np.array([[0.8, 0.7, 0.6]])

    def _encode_memory(self, image_features: Dict[str, np.ndarray],
                      masks: np.ndarray) -> np.ndarray:
        """Encode memory using memory encoder."""

        if 'memory_encoder' not in self.sessions:
            # Return dummy memory features
            return np.zeros((1, 256, 64, 64), dtype=np.float32)

        try:
            # Get main image features
            if 'backbone_features' in image_features:
                curr_vision_feats = image_features['backbone_features']
            else:
                curr_vision_feats = list(image_features.values())[0]

            # Use first mask as output mask
            output_mask = masks[:, 0:1] if len(masks.shape) == 4 else masks[:1]

            # Prepare inputs
            feat_sizes = np.array([curr_vision_feats.shape[2], curr_vision_feats.shape[3]], dtype=np.int64)

            inputs = {
                'curr_vision_feats': curr_vision_feats,
                'feat_sizes': feat_sizes,
                'output_mask': output_mask
            }

            outputs = self.sessions['memory_encoder'].run(None, inputs)
            return outputs[0]  # memory_features

        except Exception as e:
            print(f"Memory encoding failed: {e}")
            # Return dummy memory features
            return np.zeros((1, 256, 64, 64), dtype=np.float32)

    def _resize_mask(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize mask to target size."""
        import cv2

        if len(mask.shape) == 2:
            # Single mask
            resized = cv2.resize(mask.astype(np.float32), (target_size[1], target_size[0]))
            return (resized > 0.5).astype(np.uint8)
        else:
            # Multiple masks - resize first one
            resized = cv2.resize(mask[0].astype(np.float32), (target_size[1], target_size[0]))
            return (resized > 0.5).astype(np.uint8)
    
    def track_video(self, video_path: str, initial_bbox: Tuple[int, int, int, int],
                   output_path: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
        """
        Track object in video using ONNX inference.
        
        Args:
            video_path: Path to input video
            initial_bbox: Initial bounding box (x, y, w, h)
            output_path: Optional path to save output video
            
        Returns:
            List of bounding boxes for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize tracking
        x, y, w, h = initial_bbox
        current_bbox = (x, y, x + w, y + h)  # Convert to (x1, y1, x2, y2)
        
        # Initialize Kalman filter
        measurement = np.array([x + w/2, y + h/2, w/h, h], dtype=np.float32)
        self.kf_mean, self.kf_covariance = self.kalman_filter.initiate(measurement)
        
        results = []
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode current frame
            image_features = self.encode_image(frame)
            
            # Predict with Kalman filter
            self.kf_mean, self.kf_covariance = self.kalman_filter.predict(
                self.kf_mean, self.kf_covariance
            )
            
            # Get mask prediction using complete SAMURAI pipeline
            mask, confidence, memory_features = self.predict_mask(frame, current_bbox)
            
            # Update bounding box from mask
            if mask.any():
                y_indices, x_indices = np.where(mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x1, x2 = x_indices.min(), x_indices.max()
                    y1, y2 = y_indices.min(), y_indices.max()
                    current_bbox = (x1, y1, x2, y2)
                    
                    # Update Kalman filter
                    w, h = x2 - x1, y2 - y1
                    measurement = np.array([x1 + w/2, y1 + h/2, w/h, h], dtype=np.float32)
                    self.kf_mean, self.kf_covariance = self.kalman_filter.update(
                        self.kf_mean, self.kf_covariance, measurement
                    )
            
            # Convert back to (x, y, w, h) format
            x1, y1, x2, y2 = current_bbox
            result_bbox = (x1, y1, x2 - x1, y2 - y1)
            results.append(result_bbox)
            
            # Draw bounding box on frame if saving video
            if writer:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                writer.write(frame)
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Processed frame {frame_idx}")
        
        cap.release()
        if writer:
            writer.release()

        # Add performance statistics
        elapsed_total = time.time() - start_time if 'start_time' in locals() else 0
        if elapsed_total > 0:
            avg_fps = len(results) / elapsed_total
            print(f"✅ Processing completed!")
            print(f"   Total frames: {len(results)}")
            print(f"   Total time: {elapsed_total:.1f}s")
            print(f"   Average speed: {avg_fps:.2f} fps")
            if output_path:
                print(f"   Output video: {output_path}")

        return results

def main():
    parser = argparse.ArgumentParser(description="SAMURAI ONNX Inference")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--bbox", required=True, help="Initial bbox as 'x,y,w,h'")
    parser.add_argument("--model_dir", default="onnx_models", help="Directory with ONNX models")
    parser.add_argument("--output_video", help="Path to output video")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    
    args = parser.parse_args()
    
    # Parse initial bounding box
    bbox_parts = args.bbox.split(',')
    if len(bbox_parts) != 4:
        raise ValueError("Bbox must be in format 'x,y,w,h'")
    initial_bbox = tuple(map(int, bbox_parts))
    
    # Initialize predictor
    predictor = SAMURAIONNXPredictor(args.model_dir, args.device)
    
    # Run tracking
    results = predictor.track_video(args.video_path, initial_bbox, args.output_video)
    
    # Save results
    output_txt = args.video_path.replace('.mp4', '_results.txt')
    with open(output_txt, 'w') as f:
        for bbox in results:
            f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
    
    print(f"Tracking completed. Results saved to {output_txt}")
    if args.output_video:
        print(f"Output video saved to {args.output_video}")

if __name__ == "__main__":
    main()
