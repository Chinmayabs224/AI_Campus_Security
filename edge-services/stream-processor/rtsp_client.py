"""
RTSP Stream Processing Service for AI Campus Security System

This module handles RTSP stream ingestion from IP cameras with health monitoring
and automatic reconnection capabilities.
"""

import asyncio
import cv2
import numpy as np
import time
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue, Empty


class StreamStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class CameraConfig:
    """Configuration for a single camera stream"""
    camera_id: str
    rtsp_url: str
    fps: int = 30
    width: int = 640
    height: int = 480
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
    health_check_interval: int = 30


@dataclass
class StreamHealth:
    """Health status information for a camera stream"""
    camera_id: str
    status: StreamStatus
    last_frame_time: Optional[float]
    frames_received: int
    reconnect_attempts: int
    error_message: Optional[str]


class RTSPStreamProcessor:
    """
    Handles RTSP stream processing with health monitoring and reconnection logic.
    
    Features:
    - Asynchronous frame processing
    - Automatic reconnection on stream failures
    - Health monitoring and status reporting
    - Frame preprocessing pipeline
    """
    
    def __init__(self, camera_config: CameraConfig, frame_callback: Callable[[str, np.ndarray], None]):
        self.config = camera_config
        self.frame_callback = frame_callback
        self.logger = logging.getLogger(f"rtsp_client.{camera_config.camera_id}")
        
        # Stream state
        self.cap: Optional[cv2.VideoCapture] = None
        self.health = StreamHealth(
            camera_id=camera_config.camera_id,
            status=StreamStatus.DISCONNECTED,
            last_frame_time=None,
            frames_received=0,
            reconnect_attempts=0,
            error_message=None
        )
        
        # Control flags
        self._running = False
        self._stop_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        
        # Frame processing
        self.frame_queue = Queue(maxsize=10)  # Buffer for frame processing
        
    def start(self) -> bool:
        """Start the RTSP stream processing"""
        if self._running:
            self.logger.warning("Stream processor already running")
            return True
            
        self.logger.info(f"Starting RTSP stream for camera {self.config.camera_id}")
        self._running = True
        self._stop_event.clear()
        
        # Start stream processing thread
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        
        # Start health monitoring thread
        self._health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self._health_thread.start()
        
        return True
        
    def stop(self):
        """Stop the RTSP stream processing"""
        if not self._running:
            return
            
        self.logger.info(f"Stopping RTSP stream for camera {self.config.camera_id}")
        self._running = False
        self._stop_event.set()
        
        # Close video capture
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Wait for threads to finish
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5)
            
        self.health.status = StreamStatus.DISCONNECTED
        
    def get_health_status(self) -> StreamHealth:
        """Get current health status of the stream"""
        return self.health
        
    def _connect_stream(self) -> bool:
        """Establish connection to RTSP stream"""
        try:
            self.health.status = StreamStatus.CONNECTING
            self.logger.info(f"Connecting to RTSP stream: {self.config.rtsp_url}")
            
            # Create VideoCapture with optimized settings
            self.cap = cv2.VideoCapture(self.config.rtsp_url)
            
            # Configure capture properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            
            # Test connection
            if not self.cap.isOpened():
                raise Exception("Failed to open RTSP stream")
                
            # Try to read a frame to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise Exception("Failed to read initial frame from stream")
                
            self.health.status = StreamStatus.CONNECTED
            self.health.reconnect_attempts = 0
            self.health.error_message = None
            self.logger.info(f"Successfully connected to camera {self.config.camera_id}")
            return True
            
        except Exception as e:
            self.health.status = StreamStatus.ERROR
            self.health.error_message = str(e)
            self.logger.error(f"Failed to connect to camera {self.config.camera_id}: {e}")
            
            if self.cap:
                self.cap.release()
                self.cap = None
            return False 
   
    def _stream_loop(self):
        """Main stream processing loop"""
        while self._running and not self._stop_event.is_set():
            try:
                # Connect if not connected
                if self.health.status != StreamStatus.CONNECTED:
                    if not self._connect_stream():
                        self._handle_reconnection()
                        continue
                
                # Read frame from stream
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.logger.warning(f"Failed to read frame from camera {self.config.camera_id}")
                    self.health.status = StreamStatus.ERROR
                    self.health.error_message = "Failed to read frame"
                    self._handle_reconnection()
                    continue
                
                # Update health metrics
                self.health.last_frame_time = time.time()
                self.health.frames_received += 1
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                
                # Add to processing queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((self.config.camera_id, processed_frame))
                except:
                    # Queue is full, skip this frame
                    pass
                
                # Process frames from queue
                self._process_frame_queue()
                
                # Small delay to prevent CPU overload
                time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                self.logger.error(f"Error in stream loop for camera {self.config.camera_id}: {e}")
                self.health.status = StreamStatus.ERROR
                self.health.error_message = str(e)
                self._handle_reconnection()
                
    def _process_frame_queue(self):
        """Process frames from the queue"""
        try:
            while not self.frame_queue.empty():
                camera_id, frame = self.frame_queue.get_nowait()
                # Call the frame callback for processing
                if self.frame_callback:
                    self.frame_callback(camera_id, frame)
                self.frame_queue.task_done()
        except Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing frame queue: {e}")
            
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for AI inference
        
        Args:
            frame: Raw frame from camera
            
        Returns:
            Preprocessed frame ready for inference
        """
        try:
            # Resize frame to target dimensions
            if frame.shape[:2] != (self.config.height, self.config.width):
                frame = cv2.resize(frame, (self.config.width, self.config.height))
            
            # Convert BGR to RGB (YOLO expects RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error preprocessing frame: {e}")
            return frame
            
    def _handle_reconnection(self):
        """Handle stream reconnection logic"""
        if self.health.reconnect_attempts >= self.config.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts reached for camera {self.config.camera_id}")
            self.health.status = StreamStatus.ERROR
            return
            
        self.health.status = StreamStatus.RECONNECTING
        self.health.reconnect_attempts += 1
        
        self.logger.info(f"Attempting reconnection {self.health.reconnect_attempts}/{self.config.max_reconnect_attempts} for camera {self.config.camera_id}")
        
        # Close existing connection
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Wait before reconnecting
        time.sleep(self.config.reconnect_interval)
        
    def _health_monitor_loop(self):
        """Monitor stream health and detect stale connections"""
        while self._running and not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if we haven't received frames recently
                if (self.health.last_frame_time and 
                    current_time - self.health.last_frame_time > self.config.health_check_interval):
                    
                    self.logger.warning(f"No frames received for {current_time - self.health.last_frame_time:.1f}s from camera {self.config.camera_id}")
                    self.health.status = StreamStatus.ERROR
                    self.health.error_message = "Stream appears stale"
                
                # Sleep for health check interval
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                time.sleep(5)


class RTSPStreamManager:
    """
    Manages multiple RTSP streams with centralized health monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger("rtsp_manager")
        self.streams: Dict[str, RTSPStreamProcessor] = {}
        self.frame_callbacks: Dict[str, Callable] = {}
        
    def add_camera(self, camera_config: CameraConfig, frame_callback: Callable[[str, np.ndarray], None]) -> bool:
        """
        Add a new camera stream
        
        Args:
            camera_config: Camera configuration
            frame_callback: Function to call when frames are received
            
        Returns:
            True if camera was added successfully
        """
        if camera_config.camera_id in self.streams:
            self.logger.warning(f"Camera {camera_config.camera_id} already exists")
            return False
            
        try:
            stream_processor = RTSPStreamProcessor(camera_config, frame_callback)
            self.streams[camera_config.camera_id] = stream_processor
            self.frame_callbacks[camera_config.camera_id] = frame_callback
            
            self.logger.info(f"Added camera {camera_config.camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add camera {camera_config.camera_id}: {e}")
            return False
            
    def start_camera(self, camera_id: str) -> bool:
        """Start processing for a specific camera"""
        if camera_id not in self.streams:
            self.logger.error(f"Camera {camera_id} not found")
            return False
            
        return self.streams[camera_id].start()
        
    def stop_camera(self, camera_id: str) -> bool:
        """Stop processing for a specific camera"""
        if camera_id not in self.streams:
            self.logger.error(f"Camera {camera_id} not found")
            return False
            
        self.streams[camera_id].stop()
        return True
        
    def start_all_cameras(self) -> Dict[str, bool]:
        """Start all camera streams"""
        results = {}
        for camera_id in self.streams:
            results[camera_id] = self.start_camera(camera_id)
        return results
        
    def stop_all_cameras(self):
        """Stop all camera streams"""
        for camera_id in self.streams:
            self.stop_camera(camera_id)
            
    def get_health_status(self, camera_id: Optional[str] = None) -> Dict[str, StreamHealth]:
        """Get health status for cameras"""
        if camera_id:
            if camera_id in self.streams:
                return {camera_id: self.streams[camera_id].get_health_status()}
            return {}
        
        return {cid: stream.get_health_status() for cid, stream in self.streams.items()}
        
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera from management"""
        if camera_id not in self.streams:
            return False
            
        # Stop the stream first
        self.stop_camera(camera_id)
        
        # Remove from tracking
        del self.streams[camera_id]
        del self.frame_callbacks[camera_id]
        
        self.logger.info(f"Removed camera {camera_id}")
        return True