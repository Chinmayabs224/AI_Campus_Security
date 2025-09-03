"""
Batch Processing Manager for YOLO Inference

This module manages batch processing of frames from multiple camera streams
to optimize GPU utilization and throughput.
"""

import asyncio
import threading
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from queue import Queue, Empty, Full
from dataclasses import dataclass
from collections import defaultdict, deque

from .yolo_detector import YOLODetector, ModelConfig, InferenceResult, Detection


@dataclass
class FrameRequest:
    """Represents a frame processing request"""
    camera_id: str
    frame: np.ndarray
    frame_id: str
    timestamp: float
    callback: Optional[Callable[[InferenceResult], None]] = None


@dataclass
class BatchProcessorConfig:
    """Configuration for batch processor"""
    max_batch_size: int = 8
    batch_timeout_ms: int = 50  # Maximum time to wait for batch to fill
    max_queue_size: int = 100
    processing_threads: int = 2
    enable_frame_dropping: bool = True
    max_frame_age_ms: int = 1000  # Drop frames older than this


class BatchProcessor:
    """
    Manages batch processing of frames from multiple camera streams
    
    Features:
    - Automatic batching for optimal GPU utilization
    - Frame dropping for real-time processing
    - Per-camera queue management
    - Asynchronous result callbacks
    """
    
    def __init__(self, yolo_detector: YOLODetector, config: BatchProcessorConfig):
        self.detector = yolo_detector
        self.config = config
        self.logger = logging.getLogger("batch_processor")
        
        # Processing queues
        self.frame_queue = Queue(maxsize=config.max_queue_size)
        self.result_callbacks: Dict[str, Callable] = {}
        
        # Per-camera tracking
        self.camera_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'frames_processed': 0,
            'frames_dropped': 0,
            'last_frame_time': 0,
            'avg_processing_time': 0.0,
            'queue_size': 0
        })
        
        # Processing state
        self._running = False
        self._processing_threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        
        # Performance tracking
        self.total_batches_processed = 0
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        
    def start(self) -> bool:
        """Start the batch processor"""
        if self._running:
            self.logger.warning("Batch processor already running")
            return True
        
        try:
            self.logger.info("Starting batch processor")
            self._running = True
            self._stop_event.clear()
            
            # Start processing threads
            for i in range(self.config.processing_threads):
                thread = threading.Thread(
                    target=self._processing_loop,
                    name=f"BatchProcessor-{i}",
                    daemon=True
                )
                thread.start()
                self._processing_threads.append(thread)
            
            self.logger.info(f"Started {len(self._processing_threads)} processing threads")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start batch processor: {e}")
            return False
    
    def stop(self):
        """Stop the batch processor"""
        if not self._running:
            return
        
        self.logger.info("Stopping batch processor")
        self._running = False
        self._stop_event.set()
        
        # Wait for threads to finish
        for thread in self._processing_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self._processing_threads.clear()
        self.logger.info("Batch processor stopped")
    
    def submit_frame(self, camera_id: str, frame: np.ndarray, 
                    frame_id: str = None, callback: Callable = None) -> bool:
        """
        Submit a frame for processing
        
        Args:
            camera_id: Camera identifier
            frame: Frame data (RGB, normalized 0-1)
            frame_id: Optional frame identifier
            callback: Optional callback for results
            
        Returns:
            True if frame was queued successfully
        """
        if not self._running:
            return False
        
        frame_id = frame_id or f"{camera_id}_{int(time.time() * 1000)}"
        current_time = time.time()
        
        # Check if we should drop old frames
        if self.config.enable_frame_dropping:
            if self._should_drop_frame(camera_id, current_time):
                self.camera_stats[camera_id]['frames_dropped'] += 1
                self.logger.debug(f"Dropped frame from {camera_id} due to queue pressure")
                return False
        
        # Create frame request
        request = FrameRequest(
            camera_id=camera_id,
            frame=frame,
            frame_id=frame_id,
            timestamp=current_time,
            callback=callback
        )
        
        try:
            # Try to add to queue (non-blocking)
            self.frame_queue.put_nowait(request)
            self.camera_stats[camera_id]['queue_size'] += 1
            return True
            
        except Full:
            # Queue is full, drop frame
            if self.config.enable_frame_dropping:
                self.camera_stats[camera_id]['frames_dropped'] += 1
                self.logger.debug(f"Dropped frame from {camera_id} - queue full")
                return False
            else:
                # Block until space available
                try:
                    self.frame_queue.put(request, timeout=0.1)
                    return True
                except Full:
                    return False
    
    def set_camera_callback(self, camera_id: str, callback: Callable[[InferenceResult], None]):
        """Set default callback for a camera"""
        self.result_callbacks[camera_id] = callback
    
    def _processing_loop(self):
        """Main processing loop for batch inference"""
        while self._running and not self._stop_event.is_set():
            try:
                # Collect batch of frames
                batch = self._collect_batch()
                
                if not batch:
                    time.sleep(0.001)  # Short sleep if no frames available
                    continue
                
                # Process batch
                self._process_batch(batch)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _collect_batch(self) -> List[FrameRequest]:
        """Collect a batch of frames for processing"""
        batch = []
        batch_start_time = time.time()
        timeout_seconds = self.config.batch_timeout_ms / 1000.0
        
        while (len(batch) < self.config.max_batch_size and 
               time.time() - batch_start_time < timeout_seconds):
            
            try:
                # Try to get frame with short timeout
                request = self.frame_queue.get(timeout=0.01)
                
                # Check if frame is too old
                if self.config.enable_frame_dropping:
                    frame_age_ms = (time.time() - request.timestamp) * 1000
                    if frame_age_ms > self.config.max_frame_age_ms:
                        self.camera_stats[request.camera_id]['frames_dropped'] += 1
                        self.frame_queue.task_done()
                        continue
                
                batch.append(request)
                self.camera_stats[request.camera_id]['queue_size'] -= 1
                
            except Empty:
                # No frames available, continue collecting or timeout
                if batch:  # If we have some frames, process them
                    break
                continue
        
        return batch
    
    def _process_batch(self, batch: List[FrameRequest]):
        """Process a batch of frame requests"""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Prepare batch data
            frames = [(req.frame, req.camera_id) for req in batch]
            frame_ids = [req.frame_id for req in batch]
            
            # Run batch inference
            results = self.detector.detect_batch(frames, frame_ids)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.total_batches_processed += 1
            self.total_frames_processed += len(batch)
            self.total_processing_time += processing_time
            
            # Process results and call callbacks
            for request, result in zip(batch, results):
                # Update camera stats
                camera_stats = self.camera_stats[request.camera_id]
                camera_stats['frames_processed'] += 1
                camera_stats['last_frame_time'] = time.time()
                
                # Update average processing time
                if camera_stats['avg_processing_time'] == 0:
                    camera_stats['avg_processing_time'] = processing_time
                else:
                    # Exponential moving average
                    alpha = 0.1
                    camera_stats['avg_processing_time'] = (
                        alpha * processing_time + 
                        (1 - alpha) * camera_stats['avg_processing_time']
                    )
                
                # Call callbacks
                try:
                    if request.callback:
                        request.callback(result)
                    elif request.camera_id in self.result_callbacks:
                        self.result_callbacks[request.camera_id](result)
                except Exception as e:
                    self.logger.error(f"Error in result callback for {request.camera_id}: {e}")
                
                # Mark task as done
                self.frame_queue.task_done()
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            # Mark all tasks as done even on error
            for _ in batch:
                self.frame_queue.task_done()
    
    def _should_drop_frame(self, camera_id: str, current_time: float) -> bool:
        """Determine if a frame should be dropped based on queue pressure"""
        camera_stats = self.camera_stats[camera_id]
        
        # Drop if queue is getting too full
        if camera_stats['queue_size'] > self.config.max_batch_size * 2:
            return True
        
        # Drop if we're processing frames too slowly
        if camera_stats['avg_processing_time'] > 0:
            expected_fps = 1.0 / camera_stats['avg_processing_time']
            if expected_fps < 10:  # If we can't maintain at least 10 FPS
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_batch_processing_time = (
            self.total_processing_time / self.total_batches_processed
            if self.total_batches_processed > 0 else 0.0
        )
        
        avg_frame_processing_time = (
            self.total_processing_time / self.total_frames_processed
            if self.total_frames_processed > 0 else 0.0
        )
        
        return {
            "total_batches_processed": self.total_batches_processed,
            "total_frames_processed": self.total_frames_processed,
            "total_processing_time": self.total_processing_time,
            "avg_batch_processing_time": avg_batch_processing_time,
            "avg_frame_processing_time": avg_frame_processing_time,
            "current_queue_size": self.frame_queue.qsize(),
            "processing_threads": len(self._processing_threads),
            "camera_stats": dict(self.camera_stats),
            "detector_stats": self.detector.get_performance_stats()
        }
    
    def get_camera_statistics(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific camera"""
        if camera_id not in self.camera_stats:
            return None
        
        stats = self.camera_stats[camera_id].copy()
        
        # Calculate derived metrics
        if stats['frames_processed'] > 0:
            total_frames = stats['frames_processed'] + stats['frames_dropped']
            stats['drop_rate'] = stats['frames_dropped'] / total_frames
            stats['processing_fps'] = 1.0 / stats['avg_processing_time'] if stats['avg_processing_time'] > 0 else 0
        else:
            stats['drop_rate'] = 0.0
            stats['processing_fps'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.total_batches_processed = 0
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        self.camera_stats.clear()
        self.detector.reset_stats()


class InferenceManager:
    """
    High-level manager for YOLO inference with batch processing
    
    Combines YOLO detector with batch processor for optimal performance
    """
    
    def __init__(self, model_config: ModelConfig, batch_config: BatchProcessorConfig):
        self.logger = logging.getLogger("inference_manager")
        
        # Initialize components
        self.detector = YOLODetector(model_config)
        self.batch_processor = BatchProcessor(self.detector, batch_config)
        
        # Result handlers
        self.result_handlers: Dict[str, Callable] = {}
        
    def start(self) -> bool:
        """Start the inference manager"""
        try:
            self.logger.info("Starting inference manager")
            
            # Warm up the model
            self.detector.warmup()
            
            # Start batch processor
            if not self.batch_processor.start():
                return False
            
            self.logger.info("Inference manager started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start inference manager: {e}")
            return False
    
    def stop(self):
        """Stop the inference manager"""
        self.logger.info("Stopping inference manager")
        self.batch_processor.stop()
    
    def register_camera(self, camera_id: str, result_handler: Callable[[InferenceResult], None]):
        """Register a camera with result handler"""
        self.result_handlers[camera_id] = result_handler
        self.batch_processor.set_camera_callback(camera_id, result_handler)
        self.logger.info(f"Registered camera: {camera_id}")
    
    def process_frame(self, camera_id: str, frame: np.ndarray, frame_id: str = None) -> bool:
        """
        Submit frame for processing
        
        Args:
            camera_id: Camera identifier
            frame: Frame data (RGB, normalized 0-1)
            frame_id: Optional frame identifier
            
        Returns:
            True if frame was accepted for processing
        """
        return self.batch_processor.submit_frame(camera_id, frame, frame_id)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "inference_manager": {
                "registered_cameras": len(self.result_handlers),
                "model_config": {
                    "model_path": self.detector.config.model_path,
                    "confidence_threshold": self.detector.config.confidence_threshold,
                    "device": self.detector.device
                }
            },
            "batch_processor": self.batch_processor.get_statistics()
        }