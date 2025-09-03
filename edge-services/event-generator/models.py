"""
Security Event Data Models and Validation

This module defines the core data structures for security events, incidents,
and evidence clips with comprehensive validation and serialization support.
"""

import uuid
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib
from pathlib import Path


class EventType(Enum):
    """Types of security events that can be detected"""
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    CROWDING = "crowding"
    ABANDONED_OBJECT = "abandoned_object"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    VIOLENCE = "violence"
    THEFT = "theft"
    VANDALISM = "vandalism"
    FIRE = "fire"
    MEDICAL_EMERGENCY = "medical_emergency"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Severity levels for security events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(Enum):
    """Status of security events"""
    DETECTED = "detected"
    VALIDATED = "validated"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


@dataclass
class BoundingBox:
    """Represents a bounding box for object detection"""
    x1: float  # Normalized coordinates (0-1)
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    
    def __post_init__(self):
        """Validate bounding box coordinates"""
        if not (0 <= self.x1 <= 1 and 0 <= self.y1 <= 1 and 
                0 <= self.x2 <= 1 and 0 <= self.y2 <= 1):
            raise ValueError("Bounding box coordinates must be normalized (0-1)")
        
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Invalid bounding box: x1 < x2 and y1 < y2 required")
        
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
    
    def area(self) -> float:
        """Calculate the area of the bounding box"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box"""
        # Calculate intersection
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class SecurityEvent:
    """
    Core security event data structure
    
    Represents a single security event detected by the AI system
    with all necessary metadata for processing and analysis.
    """
    # Core identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Event classification
    event_type: EventType = EventType.UNKNOWN
    severity: SeverityLevel = SeverityLevel.LOW
    status: EventStatus = EventStatus.DETECTED
    
    # Detection data
    confidence_score: float = 0.0
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    
    # Location and context
    location: Optional[str] = None
    zone_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_id: Optional[str] = None
    model_version: Optional[str] = None
    
    # Processing information
    created_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    
    def __post_init__(self):
        """Validate event data after initialization"""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate the security event data
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not self.camera_id:
            raise ValueError("Camera ID is required")
        
        if not (0 <= self.confidence_score <= 1):
            raise ValueError("Confidence score must be between 0 and 1")
        
        if self.timestamp <= 0:
            raise ValueError("Timestamp must be positive")
        
        # Validate bounding boxes
        for bbox in self.bounding_boxes:
            if not isinstance(bbox, BoundingBox):
                raise ValueError("All bounding boxes must be BoundingBox instances")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        data = asdict(self)
        
        # Convert enums to strings
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        
        # Convert timestamp to ISO format for readability
        data['timestamp_iso'] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        data['created_at_iso'] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityEvent':
        """Create SecurityEvent from dictionary"""
        # Convert string enums back to enum objects
        if 'event_type' in data and isinstance(data['event_type'], str):
            data['event_type'] = EventType(data['event_type'])
        
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = SeverityLevel(data['severity'])
        
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = EventStatus(data['status'])
        
        # Convert bounding box dictionaries back to BoundingBox objects
        if 'bounding_boxes' in data:
            bboxes = []
            for bbox_data in data['bounding_boxes']:
                if isinstance(bbox_data, dict):
                    bboxes.append(BoundingBox(**bbox_data))
                else:
                    bboxes.append(bbox_data)
            data['bounding_boxes'] = bboxes
        
        # Remove ISO timestamp fields if present
        data.pop('timestamp_iso', None)
        data.pop('created_at_iso', None)
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SecurityEvent':
        """Create SecurityEvent from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_hash(self) -> str:
        """Generate a hash for event deduplication"""
        # Create hash based on key identifying fields
        hash_data = {
            'camera_id': self.camera_id,
            'event_type': self.event_type.value,
            'timestamp_rounded': int(self.timestamp / 5) * 5,  # Round to 5-second intervals
            'location': self.location,
            'zone_id': self.zone_id
        }
        
        # Include bounding box centers for spatial deduplication
        if self.bounding_boxes:
            centers = [bbox.center() for bbox in self.bounding_boxes]
            hash_data['bbox_centers'] = sorted(centers)
        
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def is_similar_to(self, other: 'SecurityEvent', 
                     time_threshold: float = 10.0,
                     spatial_threshold: float = 0.1) -> bool:
        """
        Check if this event is similar to another event for deduplication
        
        Args:
            other: Another SecurityEvent to compare
            time_threshold: Maximum time difference in seconds
            spatial_threshold: Maximum spatial distance for bounding boxes
            
        Returns:
            True if events are similar enough to be considered duplicates
        """
        # Check basic criteria
        if (self.camera_id != other.camera_id or 
            self.event_type != other.event_type or
            abs(self.timestamp - other.timestamp) > time_threshold):
            return False
        
        # Check spatial similarity if bounding boxes exist
        if self.bounding_boxes and other.bounding_boxes:
            # Find best matching bounding boxes
            max_iou = 0.0
            for bbox1 in self.bounding_boxes:
                for bbox2 in other.bounding_boxes:
                    iou = bbox1.iou(bbox2)
                    max_iou = max(max_iou, iou)
            
            # Consider similar if IoU is above threshold
            return max_iou > spatial_threshold
        
        return True


@dataclass
class IncidentClip:
    """
    Represents a video clip extracted for a security incident
    """
    clip_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""
    camera_id: str = ""
    
    # Clip timing
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    duration: float = 0.0
    
    # File information
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    format: str = "mp4"
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    
    # Processing status
    extracted: bool = False
    uploaded: bool = False
    redacted: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Calculate duration if not provided"""
        if self.duration == 0.0 and self.end_timestamp > self.start_timestamp:
            self.duration = self.end_timestamp - self.start_timestamp
    
    def validate(self) -> bool:
        """Validate clip data"""
        if not self.event_id or not self.camera_id:
            raise ValueError("Event ID and Camera ID are required")
        
        if self.start_timestamp >= self.end_timestamp:
            raise ValueError("Start timestamp must be before end timestamp")
        
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert clip to dictionary"""
        data = asdict(self)
        
        # Add ISO timestamps for readability
        data['start_timestamp_iso'] = datetime.fromtimestamp(self.start_timestamp, tz=timezone.utc).isoformat()
        data['end_timestamp_iso'] = datetime.fromtimestamp(self.end_timestamp, tz=timezone.utc).isoformat()
        data['created_at_iso'] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        
        return data


@dataclass
class EventBuffer:
    """
    Represents a buffer of events for local storage during network outages
    """
    buffer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str = ""
    events: List[SecurityEvent] = field(default_factory=list)
    clips: List[IncidentClip] = field(default_factory=list)
    
    # Buffer metadata
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    max_size: int = 1000  # Maximum number of events
    max_age: float = 86400  # Maximum age in seconds (24 hours)
    
    # Status
    is_full: bool = False
    needs_sync: bool = False
    
    def add_event(self, event: SecurityEvent) -> bool:
        """
        Add an event to the buffer
        
        Returns:
            True if event was added, False if buffer is full
        """
        if len(self.events) >= self.max_size:
            self.is_full = True
            return False
        
        self.events.append(event)
        self.last_updated = time.time()
        self.needs_sync = True
        return True
    
    def add_clip(self, clip: IncidentClip) -> bool:
        """Add a clip to the buffer"""
        self.clips.append(clip)
        self.last_updated = time.time()
        self.needs_sync = True
        return True
    
    def get_events_to_sync(self) -> List[SecurityEvent]:
        """Get events that need to be synchronized"""
        return [event for event in self.events if event.processed_at is None]
    
    def mark_events_synced(self, event_ids: List[str]):
        """Mark events as synchronized"""
        current_time = time.time()
        for event in self.events:
            if event.event_id in event_ids:
                event.processed_at = current_time
        
        # Check if all events are synced
        unsynced = [e for e in self.events if e.processed_at is None]
        if not unsynced:
            self.needs_sync = False
    
    def cleanup_old_events(self) -> int:
        """
        Remove old events from buffer
        
        Returns:
            Number of events removed
        """
        current_time = time.time()
        initial_count = len(self.events)
        
        # Remove events older than max_age that have been processed
        self.events = [
            event for event in self.events
            if (current_time - event.created_at < self.max_age or 
                event.processed_at is None)
        ]
        
        # Remove old clips
        self.clips = [
            clip for clip in self.clips
            if current_time - clip.created_at < self.max_age
        ]
        
        removed_count = initial_count - len(self.events)
        
        # Update full status
        if len(self.events) < self.max_size:
            self.is_full = False
        
        return removed_count
    
    def get_size_info(self) -> Dict[str, Any]:
        """Get buffer size information"""
        return {
            'total_events': len(self.events),
            'unsynced_events': len(self.get_events_to_sync()),
            'total_clips': len(self.clips),
            'max_size': self.max_size,
            'is_full': self.is_full,
            'needs_sync': self.needs_sync,
            'oldest_event_age': (
                time.time() - min(e.created_at for e in self.events)
                if self.events else 0
            )
        }