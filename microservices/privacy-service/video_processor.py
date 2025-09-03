"""
Video processing for face detection and redaction.
"""
import cv2
import numpy as np
import logging
import time
import base64
import tempfile
import os
from typing import List, Optional, Dict, Any

from models import PrivacyZone, VideoProcessingResult
from face_detector import FaceDetector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Video processing for face detection and redaction."""
    
    def __init__(self, face_detector: FaceDetector):
        """Initialize video processor."""
        self.face_detector = face_detector
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        privacy_zones: List[PrivacyZone],
        blur_strength: int = 50,
        frame_skip: int = 1
    ) -> VideoProcessingResult:
        """
        Process video file with face detection and redaction.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            privacy_zones: Privacy zones to apply
            blur_strength: Blur strength for faces
            frame_skip: Process every Nth frame
            
        Returns:
            Video processing result
        """
        start_time = time.time()
        frames_processed = 0
        total_faces = 0
        
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process every Nth frame for face detection
                if frame_count % frame_skip == 0:
                    # Detect faces
                    faces = self.face_detector.detect_faces(frame)
                    total_faces += len(faces)
                    
                    # Apply face blurring
                    if faces:
                        frame = self.face_detector.blur_faces(frame, faces, blur_strength)
                    
                    # Apply privacy zones
                    if privacy_zones:
                        frame = self.face_detector.apply_privacy_zones(frame, privacy_zones)
                    
                    frames_processed += 1
                
                # Write frame to output
                out.write(frame)
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}%")
            
            # Release everything
            cap.release()
            out.release()
            
            processing_time = time.time() - start_time
            
            # Get output file size
            output_file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            logger.info(f"Video processing completed in {processing_time:.2f}s")
            
            return VideoProcessingResult(
                frames_processed=frames_processed,
                total_faces=total_faces,
                processing_time=processing_time,
                output_file_size=output_file_size,
                frame_rate=fps,
                resolution=(width, height)
            )
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            return VideoProcessingResult(
                frames_processed=frames_processed,
                total_faces=total_faces,
                processing_time=time.time() - start_time,
                output_file_size=0,
                frame_rate=0.0,
                resolution=(0, 0),
                error_message=str(e)
            )
    
    def process_base64_video(
        self,
        base64_data: str,
        privacy_zones: List[PrivacyZone],
        blur_strength: int = 50,
        frame_skip: int = 1
    ) -> Dict[str, Any]:
        """
        Process base64 encoded video.
        
        Args:
            base64_data: Base64 encoded video
            privacy_zones: Privacy zones to apply
            blur_strength: Blur strength for faces
            frame_skip: Process every Nth frame
            
        Returns:
            Processing result dictionary
        """
        try:
            # Decode base64 video
            video_data = base64.b64decode(base64_data)
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
                temp_input.write(video_data)
                input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
                output_path = temp_output.name
            
            try:
                # Process video
                result = self.process_video(
                    input_path=input_path,
                    output_path=output_path,
                    privacy_zones=privacy_zones,
                    blur_strength=blur_strength,
                    frame_skip=frame_skip
                )
                
                if result.error_message:
                    return {'error': result.error_message}
                
                # Read processed video
                with open(output_path, 'rb') as f:
                    processed_video = f.read()
                
                # Encode result
                result_b64 = base64.b64encode(processed_video).decode('utf-8')
                
                return {
                    'success': True,
                    'frames_processed': result.frames_processed,
                    'faces_detected': result.total_faces,
                    'privacy_zones_applied': len([z for z in privacy_zones if z.active]),
                    'processing_time': result.processing_time,
                    'redacted_video': result_b64
                }
                
            finally:
                # Cleanup temporary files
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except OSError:
                    pass
            
        except Exception as e:
            logger.error(f"Base64 video processing failed: {str(e)}")
            return {'error': str(e)}
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 30
    ) -> List[str]:
        """
        Extract frames from video for analysis.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_interval: Extract every Nth frame
            
        Returns:
            List of extracted frame paths
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frame_paths = []
            frame_count = 0
            saved_count = 0
            
            os.makedirs(output_dir, exist_ok=True)
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{saved_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    saved_count += 1
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frame_paths)} frames from video")
            return frame_paths
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {str(e)}")
            return []
    
    def create_video_from_frames(
        self,
        frame_paths: List[str],
        output_path: str,
        fps: float = 30.0
    ) -> bool:
        """
        Create video from processed frames.
        
        Args:
            frame_paths: List of frame file paths
            output_path: Output video path
            fps: Frames per second
            
        Returns:
            True if successful
        """
        try:
            if not frame_paths:
                return False
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(frame_paths[0])
            if first_frame is None:
                return False
            
            height, width, _ = first_frame.shape
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
            
            out.release()
            
            logger.info(f"Created video from {len(frame_paths)} frames")
            return True
            
        except Exception as e:
            logger.error(f"Video creation failed: {str(e)}")
            return False
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video file information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {'error': 'Could not open video file'}
            
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
            }
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            return {'error': str(e)}