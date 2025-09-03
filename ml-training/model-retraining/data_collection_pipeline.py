"""
Data Collection Pipeline for New Incident Footage
Collects and processes new incident data for model retraining
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import hashlib
import shutil
import cv2
import numpy as np
from dataclasses import dataclass, asdict
import yaml

@dataclass
class IncidentData:
    """Structure for incident data collection"""
    incident_id: str
    timestamp: datetime
    camera_id: str
    event_type: str
    confidence_score: float
    video_path: str
    metadata: Dict
    ground_truth_label: Optional[str] = None
    validation_status: str = "pending"  # pending, validated, rejected
    
class DataCollectionPipeline:
    """Pipeline for collecting and processing new incident footage for retraining"""
    
    def __init__(self, config_path: str = "config/retraining_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.collection_dir = Path(self.config['data_collection']['output_dir'])
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load retraining configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'data_collection': {
                    'output_dir': '../../data/retraining',
                    'min_confidence_threshold': 0.3,
                    'max_confidence_threshold': 0.9,
                    'collection_window_days': 7,
                    'max_samples_per_class': 1000,
                    'frame_extraction_interval': 30
                },
                'quality_filters': {
                    'min_resolution': [320, 240],
                    'max_blur_threshold': 100,
                    'min_brightness': 20,
                    'max_brightness': 235
                },
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'campus_security',
                    'table': 'incidents'
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data collection"""
        logger = logging.getLogger('data_collection')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def collect_recent_incidents(self, days_back: int = None) -> List[IncidentData]:
        """Collect recent incident data for retraining"""
        if days_back is None:
            days_back = self.config['data_collection']['collection_window_days']
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Simulate database query - in real implementation, connect to PostgreSQL
        incidents = await self._query_incidents_since(cutoff_date)
        
        collected_data = []
        for incident in incidents:
            try:
                incident_data = await self._process_incident(incident)
                if incident_data and await self._validate_data_quality(incident_data):
                    collected_data.append(incident_data)
                    self.logger.info(f"Collected incident {incident_data.incident_id}")
            except Exception as e:
                self.logger.error(f"Error processing incident {incident.get('id', 'unknown')}: {e}")
        
        return collected_data
    
    async def _query_incidents_since(self, cutoff_date: datetime) -> List[Dict]:
        """Query incidents from database since cutoff date"""
        # Simulate database query - replace with actual database connection
        sample_incidents = [
            {
                'id': f'incident_{i}',
                'timestamp': datetime.now() - timedelta(hours=i),
                'camera_id': f'cam_{i % 10}',
                'event_type': ['suspicious', 'violence', 'theft', 'normal'][i % 4],
                'confidence_score': 0.4 + (i % 6) * 0.1,
                'video_path': f'/evidence/incident_{i}.mp4',
                'metadata': {'location': f'building_{i % 5}', 'duration': 30 + i}
            }
            for i in range(50)  # Simulate 50 recent incidents
        ]
        
        return [inc for inc in sample_incidents if inc['timestamp'] >= cutoff_date]
    
    async def _process_incident(self, incident: Dict) -> Optional[IncidentData]:
        """Process individual incident data"""
        try:
            # Check confidence score range for retraining candidates
            confidence = incident['confidence_score']
            min_conf = self.config['data_collection']['min_confidence_threshold']
            max_conf = self.config['data_collection']['max_confidence_threshold']
            
            if not (min_conf <= confidence <= max_conf):
                return None  # Skip incidents outside confidence range
            
            incident_data = IncidentData(
                incident_id=incident['id'],
                timestamp=incident['timestamp'],
                camera_id=incident['camera_id'],
                event_type=incident['event_type'],
                confidence_score=confidence,
                video_path=incident['video_path'],
                metadata=incident['metadata']
            )
            
            # Copy video file to collection directory
            await self._copy_incident_video(incident_data)
            
            return incident_data
            
        except Exception as e:
            self.logger.error(f"Error processing incident: {e}")
            return None
    
    async def _copy_incident_video(self, incident_data: IncidentData):
        """Copy incident video to collection directory"""
        source_path = Path(incident_data.video_path)
        
        # Create organized directory structure
        class_dir = self.collection_dir / incident_data.event_type
        class_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp_str = incident_data.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{incident_data.incident_id}_{timestamp_str}.mp4"
        dest_path = class_dir / filename
        
        # Simulate file copy - in real implementation, copy from evidence storage
        if not dest_path.exists():
            # Create dummy video file for demonstration
            self._create_dummy_video(dest_path)
        
        # Update path in incident data
        incident_data.video_path = str(dest_path)
    
    def _create_dummy_video(self, path: Path):
        """Create dummy video file for demonstration"""
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, 20.0, (640, 480))
        
        for i in range(100):  # 5 second video at 20fps
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
    
    async def _validate_data_quality(self, incident_data: IncidentData) -> bool:
        """Validate data quality for training suitability"""
        try:
            video_path = Path(incident_data.video_path)
            if not video_path.exists():
                return False
            
            # Check video properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Check minimum resolution
            min_res = self.config['quality_filters']['min_resolution']
            if width < min_res[0] or height < min_res[1]:
                cap.release()
                return False
            
            # Sample frames for quality check
            quality_passed = await self._check_frame_quality(cap)
            cap.release()
            
            return quality_passed and frame_count > 10
            
        except Exception as e:
            self.logger.error(f"Quality validation error: {e}")
            return False
    
    async def _check_frame_quality(self, cap: cv2.VideoCapture) -> bool:
        """Check frame quality metrics"""
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(10, frame_count // 10)  # Sample 10 frames or 10% of video
        
        quality_scores = []
        
        for i in range(sample_frames):
            frame_idx = i * (frame_count // sample_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Check brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Check blur (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            min_brightness = self.config['quality_filters']['min_brightness']
            max_brightness = self.config['quality_filters']['max_brightness']
            min_blur = self.config['quality_filters']['max_blur_threshold']
            
            if (min_brightness <= brightness <= max_brightness and 
                blur_score >= min_blur):
                quality_scores.append(1)
            else:
                quality_scores.append(0)
        
        # Require at least 70% of frames to pass quality check
        return sum(quality_scores) / len(quality_scores) >= 0.7 if quality_scores else False
    
    async def extract_training_frames(self, incident_data: IncidentData) -> List[str]:
        """Extract frames from incident video for training"""
        video_path = Path(incident_data.video_path)
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_paths = []
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = self.config['data_collection']['frame_extraction_interval']
            
            for frame_idx in range(0, frame_count, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_filename = f"frame_{frame_idx:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
            
        finally:
            cap.release()
        
        return frame_paths
    
    async def save_collection_metadata(self, collected_data: List[IncidentData]):
        """Save collection metadata for tracking"""
        metadata = {
            'collection_timestamp': datetime.now().isoformat(),
            'total_incidents': len(collected_data),
            'incidents_by_class': {},
            'incidents': [asdict(incident) for incident in collected_data]
        }
        
        # Count incidents by class
        for incident in collected_data:
            event_type = incident.event_type
            metadata['incidents_by_class'][event_type] = (
                metadata['incidents_by_class'].get(event_type, 0) + 1
            )
        
        metadata_path = self.collection_dir / f"collection_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved collection metadata to {metadata_path}")
        return metadata_path

async def main():
    """Example usage of data collection pipeline"""
    pipeline = DataCollectionPipeline()
    
    # Collect recent incidents
    incidents = await pipeline.collect_recent_incidents(days_back=7)
    print(f"Collected {len(incidents)} incidents for retraining")
    
    # Extract frames from collected incidents
    for incident in incidents[:3]:  # Process first 3 incidents
        frames = await pipeline.extract_training_frames(incident)
        print(f"Extracted {len(frames)} frames from incident {incident.incident_id}")
    
    # Save metadata
    await pipeline.save_collection_metadata(incidents)

if __name__ == "__main__":
    asyncio.run(main())