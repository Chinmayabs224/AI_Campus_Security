"""
Incident Clip Extraction Service

This module handles the extraction of video clips from camera streams
for security incidents with configurable duration and buffering.
"""

import os
import time
import cv2
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque
from pathlib import Path
import logging
from queue import Queue, Empty
import json
from datetime import datetime

from .models import SecurityEvent, IncidentClip, EventBuffer


class ClipExtractionConfig:
    """Configuration for clip extraction"""
    
    def __init__(self):
        # Clip timing
        self.pre_event_duration = 10.0  # seconds before event
        self.post_event_duration = 15.0  # seconds after event
        self.max_clip_duration = 60.0  # maximum clip length
        
        # Video settings
        self.output_fps = 15.0
        self.output_resolution = (640, 480)
        self.video_codec = 'mp4v'
        self.video_quality = 80  # 0-100
        
        # Storage settings
        self.clips_directory = "clips"
        self.max_clips_per_camera = 100
        self.max_storage_gb = 10.0
        
        # Buffer settings
        self.frame_buffer_size = 900  # frames (30 seconds at 30fps)
        self.frame_buffer_memory_limit = 500 * 1024 * 1024  # 500MB


class FrameBuffer:
    """
    Circular buffer for storing video frames with timestamps
    """
    
    def __init__(self, camera_id: str, config: ClipExtractionConfig):
        self.camera_id = camera_id
        self.config = config
        self.logger = logging.getLogger(f"frame_buffer.{camera_id}")
        
        # Frame storage
        self.frames: deque = deque(maxlen=config.frame_buffer_size)
        self.timestamps: deque = deque(maxlen=config.frame_buffer_size)
        
        # Memory management
        self.current_memory_usage = 0
        self.lock = threading.Lock()
        
        # Statistics
        self.frames_added = 0
        self.frames_dropped = 0
    
    def add_frame(self, frame: np.ndarray, timestamp: float) -> bool:
        """
        Add a frame to the buffer
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
            
        Returns:
            True if frame was added successfully
        """
        with self.lock:
            try:
                # Check memory usage
                frame_size = frame.nbytes
                
                # Remove old frames if memory limit exceeded
                while (self.current_memory_usage + frame_size > self.config.frame_buffer_memory_limit 
                       and self.frames):
                    old_frame = self.frames.popleft()
                    self.timestamps.popleft()
                    self.current_memory_usage -= old_frame.nbytes
                
                # Add new frame
                self.frames.append(frame.copy())
                self.timestamps.append(timestamp)
                self.current_memory_usage += frame_size
                self.frames_added += 1
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error adding frame to buffer: {e}")
                self.frames_dropped += 1
                return False
    
    def get_frames_in_range(self, start_time: float, end_time: float) -> List[Tuple[np.ndarray, float]]:
        """
        Get frames within a time range
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of (frame, timestamp) tuples
        """
        with self.lock:
            result = []
            
            for frame, timestamp in zip(self.frames, self.timestamps):
                if start_time <= timestamp <= end_time:
                    result.append((frame.copy(), timestamp))
            
            return result
    
    def get_buffer_info(self) -> Dict:
        """Get buffer status information"""
        with self.lock:
            return {
                'camera_id': self.camera_id,
                'frame_count': len(self.frames),
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'memory_limit_mb': self.config.frame_buffer_memory_limit / (1024 * 1024),
                'frames_added': self.frames_added,
                'frames_dropped': self.frames_dropped,
                'oldest_timestamp': self.timestamps[0] if self.timestamps else None,
                'newest_timestamp': self.timestamps[-1] if self.timestamps else None
            }


class ClipExtractor:
    """
    Extracts video clips from frame buffers for security incidents
    """
    
    def __init__(self, config: ClipExtractionConfig):
        self.config = config
        self.logger = logging.getLogger("clip_extractor")
        
        # Frame buffers per camera
        self.frame_buffers: Dict[str, FrameBuffer] = {}
        
        # Clip processing
        self.extraction_queue = Queue(maxsize=100)
        self.processing = False
        self.process_thread: Optional[threading.Thread] = None
        
        # Storage management
        self.clips_directory = Path(config.clips_directory)
        self.clips_directory.mkdir(exist_ok=True)
        
        # Callbacks
        self.clip_callbacks: List[Callable[[IncidentClip], None]] = []
        
        # Statistics
        self.stats = {
            'clips_extracted': 0,
            'clips_failed': 0,
            'total_clip_duration': 0.0,
            'storage_used_mb': 0.0
        }
    
    def start(self) -> bool:
        """Start the clip extraction service"""
        if self.processing:
            return True
        
        self.logger.info("Starting Clip Extraction Service")
        self.processing = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_extraction_queue, daemon=True)
        self.process_thread.start()
        
        return True
    
    def stop(self):
        """Stop the clip extraction service"""
        if not self.processing:
            return
        
        self.logger.info("Stopping Clip Extraction Service")
        self.processing = False
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=10)
    
    def add_camera(self, camera_id: str) -> bool:
        """Add a camera for frame buffering"""
        if camera_id in self.frame_buffers:
            return True
        
        try:
            self.frame_buffers[camera_id] = FrameBuffer(camera_id, self.config)
            self.logger.info(f"Added frame buffer for camera {camera_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add camera {camera_id}: {e}")
            return False
    
    def add_frame(self, camera_id: str, frame: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Add a frame to the camera's buffer
        
        Args:
            camera_id: Camera identifier
            frame: Video frame
            timestamp: Frame timestamp (current time if None)
            
        Returns:
            True if frame was added successfully
        """
        if camera_id not in self.frame_buffers:
            self.add_camera(camera_id)
        
        timestamp = timestamp or time.time()
        return self.frame_buffers[camera_id].add_frame(frame, timestamp)
    
    def extract_clip_for_event(self, event: SecurityEvent) -> Optional[IncidentClip]:
        """
        Request clip extraction for a security event
        
        Args:
            event: Security event requiring clip extraction
            
        Returns:
            IncidentClip object if extraction was queued successfully
        """
        try:
            # Calculate clip timing
            start_time = event.timestamp - self.config.pre_event_duration
            end_time = event.timestamp + self.config.post_event_duration
            
            # Ensure clip doesn't exceed maximum duration
            if end_time - start_time > self.config.max_clip_duration:
                end_time = start_time + self.config.max_clip_duration
            
            # Create clip object
            clip = IncidentClip(
                event_id=event.event_id,
                camera_id=event.camera_id,
                start_timestamp=start_time,
                end_timestamp=end_time,
                metadata={
                    'event_type': event.event_type.value,
                    'severity': event.severity.value,
                    'confidence': event.confidence_score
                }
            )
            
            # Queue for extraction
            try:
                self.extraction_queue.put_nowait(clip)
                self.logger.info(f"Queued clip extraction for event {event.event_id}")
                return clip
            except:
                self.logger.warning("Extraction queue full, dropping clip request")
                return None
                
        except Exception as e:
            self.logger.error(f"Error queuing clip extraction: {e}")
            return None
    
    def _process_extraction_queue(self):
        """Process clip extraction requests"""
        while self.processing:
            try:
                # Get extraction request
                try:
                    clip = self.extraction_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Extract the clip
                success = self._extract_clip(clip)
                
                if success:
                    self.stats['clips_extracted'] += 1
                    self.stats['total_clip_duration'] += clip.duration
                    
                    # Call callbacks
                    for callback in self.clip_callbacks:
                        try:
                            callback(clip)
                        except Exception as e:
                            self.logger.error(f"Error in clip callback: {e}")
                else:
                    self.stats['clips_failed'] += 1
                
                self.extraction_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in extraction processing: {e}")
    
    def _extract_clip(self, clip: IncidentClip) -> bool:
        """
        Extract video clip from frame buffer
        
        Args:
            clip: IncidentClip object with extraction parameters
            
        Returns:
            True if extraction was successful
        """
        try:
            camera_id = clip.camera_id
            
            # Check if camera buffer exists
            if camera_id not in self.frame_buffers:
                self.logger.error(f"No frame buffer for camera {camera_id}")
                return False
            
            # Get frames from buffer
            frame_buffer = self.frame_buffers[camera_id]
            frames = frame_buffer.get_frames_in_range(clip.start_timestamp, clip.end_timestamp)
            
            if not frames:
                self.logger.warning(f"No frames available for clip {clip.clip_id}")
                return False
            
            # Create output file path
            clip_filename = self._generate_clip_filename(clip)
            clip_path = self.clips_directory / clip_filename
            
            # Extract video clip
            success = self._write_video_clip(frames, clip_path, clip)
            
            if success:
                clip.file_path = str(clip_path)
                clip.file_size = clip_path.stat().st_size
                clip.extracted = True
                
                # Update storage statistics
                self.stats['storage_used_mb'] += clip.file_size / (1024 * 1024)
                
                self.logger.info(f"Successfully extracted clip {clip.clip_id} to {clip_path}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error extracting clip {clip.clip_id}: {e}")
            return False
    
    def _write_video_clip(self, frames: List[Tuple[np.ndarray, float]], 
                         output_path: Path, clip: IncidentClip) -> bool:
        """
        Write frames to video file
        
        Args:
            frames: List of (frame, timestamp) tuples
            output_path: Output video file path
            clip: IncidentClip object to update
            
        Returns:
            True if video was written successfully
        """
        try:
            if not frames:
                return False
            
            # Sort frames by timestamp
            frames.sort(key=lambda x: x[1])
            
            # Get video properties from first frame
            first_frame = frames[0][0]
            height, width = first_frame.shape[:2]
            
            # Resize frames if needed
            target_width, target_height = self.config.output_resolution
            if (width, height) != (target_width, target_height):
                resize_frames = True
            else:
                resize_frames = False
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.config.output_fps,
                (target_width, target_height)
            )
            
            if not writer.isOpened():
                self.logger.error(f"Failed to open video writer for {output_path}")
                return False
            
            # Write frames
            frames_written = 0
            for frame, timestamp in frames:
                try:
                    # Convert RGB to BGR for OpenCV
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Assume RGB, convert to BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    # Resize if needed
                    if resize_frames:
                        frame_bgr = cv2.resize(frame_bgr, (target_width, target_height))
                    
                    # Ensure frame is uint8
                    if frame_bgr.dtype != np.uint8:
                        if frame_bgr.max() <= 1.0:
                            frame_bgr = (frame_bgr * 255).astype(np.uint8)
                        else:
                            frame_bgr = frame_bgr.astype(np.uint8)
                    
                    writer.write(frame_bgr)
                    frames_written += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error writing frame: {e}")
                    continue
            
            writer.release()
            
            # Update clip metadata
            clip.resolution = (target_width, target_height)
            clip.fps = self.config.output_fps
            clip.metadata['frames_written'] = frames_written
            clip.metadata['original_frame_count'] = len(frames)
            
            self.logger.info(f"Wrote {frames_written} frames to {output_path}")
            return frames_written > 0
            
        except Exception as e:
            self.logger.error(f"Error writing video clip: {e}")
            return False
    
    def _generate_clip_filename(self, clip: IncidentClip) -> str:
        """Generate filename for clip"""
        timestamp_str = datetime.fromtimestamp(clip.start_timestamp).strftime("%Y%m%d_%H%M%S")
        return f"{clip.camera_id}_{timestamp_str}_{clip.clip_id[:8]}.{clip.format}"
    
    def add_clip_callback(self, callback: Callable[[IncidentClip], None]):
        """Add callback for when clips are extracted"""
        self.clip_callbacks.append(callback)
    
    def get_buffer_status(self) -> Dict[str, Dict]:
        """Get status of all frame buffers"""
        return {
            camera_id: buffer.get_buffer_info()
            for camera_id, buffer in self.frame_buffers.items()
        }
    
    def get_statistics(self) -> Dict:
        """Get extraction statistics"""
        return {
            **self.stats,
            'active_cameras': len(self.frame_buffers),
            'extraction_queue_size': self.extraction_queue.qsize(),
            'processing': self.processing
        }
    
    def cleanup_old_clips(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up old clip files
        
        Args:
            max_age_hours: Maximum age of clips to keep
            
        Returns:
            Number of clips removed
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            removed_count = 0
            
            for clip_file in self.clips_directory.glob("*.mp4"):
                try:
                    file_age = current_time - clip_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_size = clip_file.stat().st_size
                        clip_file.unlink()
                        removed_count += 1
                        self.stats['storage_used_mb'] -= file_size / (1024 * 1024)
                        
                except Exception as e:
                    self.logger.warning(f"Error removing clip file {clip_file}: {e}")
            
            # Ensure storage stats don't go negative
            self.stats['storage_used_mb'] = max(0, self.stats['storage_used_mb'])
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old clip files")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error during clip cleanup: {e}")
            return 0