"""
YOLO Model Inference Engine for AI Campus Security System

This module provides YOLO model loading, inference, and detection processing
with GPU acceleration support for edge devices.
"""

import torch
import numpy as np
import cv2
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO
import threading
from queue import Queue, Empty


@dataclass
class Detection:
    """Represents a single object detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized coordinates)
    timestamp: float


@dataclass
class InferenceResult:
    """Results from YOLO inference"""
    camera_id: str
    frame_id: str
    detections: List[Detection]
    inference_time: float
    timestamp: float
    frame_shape: Tuple[int, int, int]  # height, width, channels


@dataclass
class ModelConfig:
    """Configuration for YOLO model"""
    model_path: str
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    input_size: Tuple[int, int] = (640, 640)
    use_gpu: bool = True
    gpu_device: int = 0
    batch_size: int = 1
    half_precision: bool = False


class YOLODetector:
    """
    YOLO model wrapper for object detection with GPU acceleration
    
    Features:
    - Model loading and initialization
    - Batch inference processing
    - Confidence and IoU filtering
    - GPU acceleration support
    - Thread-safe inference
    """
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.logger = logging.getLogger(f"yolo_detector")
        
        # Model state
        self.model: Optional[YOLO] = None
        self.device = None
        self.class_names: Dict[int, str] = {}
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # Thread safety
        self._inference_lock = threading.Lock()
        
        # Initialize model
        self._load_model()
        
    def _load_model(self) -> bool:
        """Load YOLO model and configure device"""
        try:
            self.logger.info(f"Loading YOLO model from {self.config.model_path}")
            
            # Check if model file exists
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            
            # Load YOLO model
            self.model = YOLO(self.config.model_path)
            
            # Configure device
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = f"cuda:{self.config.gpu_device}"
                self.logger.info(f"Using GPU device: {self.device}")
                
                # Move model to GPU
                self.model.to(self.device)
                
                # Enable half precision if requested and supported
                if self.config.half_precision:
                    self.model.half()
                    self.logger.info("Enabled half precision inference")
            else:
                self.device = "cpu"
                self.logger.info("Using CPU for inference")
            
            # Get class names
            if hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            else:
                # Default COCO class names for security-relevant objects
                self.class_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                    5: 'bus', 7: 'truck', 15: 'cat', 16: 'dog',
                    24: 'handbag', 26: 'suitcase', 27: 'frisbee',
                    28: 'skis', 29: 'snowboard', 32: 'sports ball'
                }
            
            self.logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_objects(self, frame: np.ndarray, camera_id: str, frame_id: str = None) -> InferenceResult:
        """
        Perform object detection on a single frame
        
        Args:
            frame: Input frame (RGB format, normalized 0-1)
            camera_id: Identifier for the camera
            frame_id: Optional frame identifier
            
        Returns:
            InferenceResult with detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        frame_id = frame_id or f"{camera_id}_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            with self._inference_lock:
                # Prepare frame for inference
                input_frame = self._preprocess_frame(frame)
                
                # Run inference
                results = self.model(
                    input_frame,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_detections,
                    device=self.device,
                    verbose=False
                )
                
                # Process results
                detections = self._process_results(results[0], frame.shape)
                
                inference_time = time.time() - start_time
                
                # Update performance metrics
                self.inference_count += 1
                self.total_inference_time += inference_time
                
                return InferenceResult(
                    camera_id=camera_id,
                    frame_id=frame_id,
                    detections=detections,
                    inference_time=inference_time,
                    timestamp=time.time(),
                    frame_shape=frame.shape
                )
                
        except Exception as e:
            self.logger.error(f"Inference failed for {camera_id}: {e}")
            return InferenceResult(
                camera_id=camera_id,
                frame_id=frame_id,
                detections=[],
                inference_time=time.time() - start_time,
                timestamp=time.time(),
                frame_shape=frame.shape
            )
    
    def detect_batch(self, frames: List[Tuple[np.ndarray, str]], frame_ids: List[str] = None) -> List[InferenceResult]:
        """
        Perform batch inference on multiple frames
        
        Args:
            frames: List of (frame, camera_id) tuples
            frame_ids: Optional list of frame identifiers
            
        Returns:
            List of InferenceResult objects
        """
        if not frames:
            return []
        
        if frame_ids is None:
            frame_ids = [f"{cam_id}_{int(time.time() * 1000)}_{i}" 
                        for i, (_, cam_id) in enumerate(frames)]
        
        results = []
        
        try:
            # Process frames in batches
            batch_size = min(self.config.batch_size, len(frames))
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_ids = frame_ids[i:i + batch_size]
                
                # Process batch
                batch_results = self._process_batch(batch_frames, batch_ids)
                results.extend(batch_results)
                
        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            # Return empty results for failed batch
            for (frame, camera_id), frame_id in zip(frames, frame_ids):
                results.append(InferenceResult(
                    camera_id=camera_id,
                    frame_id=frame_id,
                    detections=[],
                    inference_time=0.0,
                    timestamp=time.time(),
                    frame_shape=frame.shape
                ))
        
        return results
    
    def _process_batch(self, batch_frames: List[Tuple[np.ndarray, str]], batch_ids: List[str]) -> List[InferenceResult]:
        """Process a batch of frames"""
        start_time = time.time()
        
        with self._inference_lock:
            # Prepare batch input
            input_frames = []
            original_shapes = []
            
            for frame, camera_id in batch_frames:
                input_frame = self._preprocess_frame(frame)
                input_frames.append(input_frame)
                original_shapes.append(frame.shape)
            
            # Run batch inference
            results = self.model(
                input_frames,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                device=self.device,
                verbose=False
            )
            
            # Process results
            batch_results = []
            inference_time = time.time() - start_time
            
            for i, (result, (frame, camera_id), frame_id, original_shape) in enumerate(
                zip(results, batch_frames, batch_ids, original_shapes)
            ):
                detections = self._process_results(result, original_shape)
                
                batch_results.append(InferenceResult(
                    camera_id=camera_id,
                    frame_id=frame_id,
                    detections=detections,
                    inference_time=inference_time / len(batch_frames),  # Approximate per-frame time
                    timestamp=time.time(),
                    frame_shape=original_shape
                ))
            
            # Update performance metrics
            self.inference_count += len(batch_frames)
            self.total_inference_time += inference_time
            
            return batch_results
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLO inference
        
        Args:
            frame: Input frame (RGB, normalized 0-1)
            
        Returns:
            Preprocessed frame ready for inference
        """
        # Convert from normalized float to uint8 if needed
        if frame.dtype == np.float32 and frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        
        # Resize to model input size
        target_height, target_width = self.config.input_size
        if frame.shape[:2] != (target_height, target_width):
            frame = cv2.resize(frame, (target_width, target_height))
        
        return frame
    
    def _process_results(self, result, original_shape: Tuple[int, int, int]) -> List[Detection]:
        """
        Process YOLO inference results into Detection objects
        
        Args:
            result: YOLO result object
            original_shape: Original frame shape (height, width, channels)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        try:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Normalize coordinates to original frame size
                orig_height, orig_width = original_shape[:2]
                model_height, model_width = self.config.input_size
                
                for i in range(len(boxes)):
                    # Convert to normalized coordinates (0-1)
                    x1, y1, x2, y2 = boxes[i]
                    
                    # Scale back to original frame coordinates
                    x1 = (x1 / model_width) * orig_width / orig_width  # Normalize to 0-1
                    y1 = (y1 / model_height) * orig_height / orig_height
                    x2 = (x2 / model_width) * orig_width / orig_width
                    y2 = (y2 / model_height) * orig_height / orig_height
                    
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        timestamp=time.time()
                    )
                    
                    detections.append(detection)
                    
        except Exception as e:
            self.logger.error(f"Error processing detection results: {e}")
        
        return detections
    
    def filter_detections(self, detections: List[Detection], 
                         min_confidence: float = None,
                         allowed_classes: List[str] = None) -> List[Detection]:
        """
        Filter detections based on confidence and class criteria
        
        Args:
            detections: List of Detection objects
            min_confidence: Minimum confidence threshold
            allowed_classes: List of allowed class names
            
        Returns:
            Filtered list of detections
        """
        filtered = detections
        
        # Filter by confidence
        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
        
        # Filter by allowed classes
        if allowed_classes is not None:
            allowed_set = set(allowed_classes)
            filtered = [d for d in filtered if d.class_name in allowed_set]
        
        return filtered
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_inference_time = (
            self.total_inference_time / self.inference_count 
            if self.inference_count > 0 else 0.0
        )
        
        return {
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": avg_inference_time,
            "fps_capability": 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0,
            "device": self.device,
            "model_path": self.config.model_path,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_count = 0
        self.total_inference_time = 0.0
    
    def warmup(self, num_iterations: int = 5):
        """
        Warm up the model with dummy inference calls
        
        Args:
            num_iterations: Number of warmup iterations
        """
        self.logger.info(f"Warming up model with {num_iterations} iterations")
        
        # Create dummy frame
        dummy_frame = np.random.rand(*self.config.input_size, 3).astype(np.float32)
        
        for i in range(num_iterations):
            try:
                self.detect_objects(dummy_frame, "warmup", f"warmup_{i}")
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i} failed: {e}")
        
        # Reset stats after warmup
        self.reset_stats()
        self.logger.info("Model warmup completed")