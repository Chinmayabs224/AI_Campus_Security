"""
Data models for Privacy Service.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import uuid


class RedactionType(str, Enum):
    """Types of redaction."""
    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK_BOX = "black_box"
    MASK = "mask"


class DSARStatus(str, Enum):
    """DSAR request status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DSARRequestType(str, Enum):
    """Types of DSAR requests."""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    RESTRICTION = "restriction"


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def area(self) -> int:
        """Calculate bounding box area."""
        return self.width * self.height
    
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Face:
    """Detected face information."""
    bounding_box: BoundingBox
    landmarks: Optional[List[Tuple[int, int]]] = None
    confidence: float = 0.0
    face_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bounding_box': self.bounding_box.to_dict(),
            'landmarks': self.landmarks,
            'confidence': self.confidence,
            'face_id': self.face_id
        }


@dataclass
class PrivacyZone:
    """Privacy zone configuration."""
    zone_id: str
    name: str
    coordinates: List[Tuple[int, int]]  # Polygon coordinates
    redaction_type: RedactionType = RedactionType.BLUR
    blur_strength: int = 50
    active: bool = True
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'zone_id': self.zone_id,
            'name': self.name,
            'coordinates': self.coordinates,
            'redaction_type': self.redaction_type.value,
            'blur_strength': self.blur_strength,
            'active': self.active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PrivacyZone':
        """Create from dictionary."""
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        return cls(
            zone_id=data['zone_id'],
            name=data['name'],
            coordinates=data['coordinates'],
            redaction_type=RedactionType(data.get('redaction_type', RedactionType.BLUR)),
            blur_strength=data.get('blur_strength', 50),
            active=data.get('active', True),
            created_at=created_at
        )


@dataclass
class RedactionRequest:
    """Request for content redaction."""
    request_id: str
    file_type: str  # 'image' or 'video'
    privacy_zones: List[PrivacyZone]
    blur_strength: int = 50
    redaction_type: RedactionType = RedactionType.BLUR
    detect_faces: bool = True
    apply_privacy_zones: bool = True
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'file_type': self.file_type,
            'privacy_zones': [zone.to_dict() for zone in self.privacy_zones],
            'blur_strength': self.blur_strength,
            'redaction_type': self.redaction_type.value,
            'detect_faces': self.detect_faces,
            'apply_privacy_zones': self.apply_privacy_zones,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class RedactionResult:
    """Result of redaction operation."""
    request_id: str
    success: bool
    faces_detected: int = 0
    privacy_zones_applied: int = 0
    processing_time: float = 0.0
    file_size_original: int = 0
    file_size_redacted: int = 0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.completed_at is None:
            self.completed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'success': self.success,
            'faces_detected': self.faces_detected,
            'privacy_zones_applied': self.privacy_zones_applied,
            'processing_time': self.processing_time,
            'file_size_original': self.file_size_original,
            'file_size_redacted': self.file_size_redacted,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class DSARRequest:
    """Data Subject Access Request."""
    request_id: str
    request_type: DSARRequestType
    subject_email: str
    subject_name: Optional[str] = None
    subject_id: Optional[str] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    description: Optional[str] = None
    status: DSARStatus = DSARStatus.PENDING
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'request_type': self.request_type.value,
            'subject_email': self.subject_email,
            'subject_name': self.subject_name,
            'subject_id': self.subject_id,
            'date_range_start': self.date_range_start.isoformat() if self.date_range_start else None,
            'date_range_end': self.date_range_end.isoformat() if self.date_range_end else None,
            'description': self.description,
            'status': self.status.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DSARRequest':
        """Create from dictionary."""
        date_range_start = None
        date_range_end = None
        created_at = None
        updated_at = None
        completed_at = None
        
        if data.get('date_range_start'):
            date_range_start = datetime.fromisoformat(data['date_range_start'])
        if data.get('date_range_end'):
            date_range_end = datetime.fromisoformat(data['date_range_end'])
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])
        if data.get('completed_at'):
            completed_at = datetime.fromisoformat(data['completed_at'])
        
        return cls(
            request_id=data['request_id'],
            request_type=DSARRequestType(data['request_type']),
            subject_email=data['subject_email'],
            subject_name=data.get('subject_name'),
            subject_id=data.get('subject_id'),
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            description=data.get('description'),
            status=DSARStatus(data.get('status', DSARStatus.PENDING)),
            created_at=created_at,
            updated_at=updated_at,
            completed_at=completed_at
        )


@dataclass
class VideoProcessingResult:
    """Result of video processing operation."""
    frames_processed: int
    total_faces: int
    processing_time: float
    output_file_size: int
    frame_rate: float
    resolution: Tuple[int, int]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'frames_processed': self.frames_processed,
            'total_faces': self.total_faces,
            'processing_time': self.processing_time,
            'output_file_size': self.output_file_size,
            'frame_rate': self.frame_rate,
            'resolution': self.resolution,
            'error_message': self.error_message
        }