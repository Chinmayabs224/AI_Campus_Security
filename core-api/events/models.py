"""
Event data models and schemas.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime, Float, JSON, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from core.database import Base


class EventType(str, Enum):
    """Security event types."""
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    CROWDING = "crowding"
    ABANDONED_OBJECT = "abandoned_object"
    VIOLENCE = "violence"
    THEFT = "theft"
    VANDALISM = "vandalism"
    FIRE = "fire"
    MEDICAL_EMERGENCY = "medical_emergency"
    UNKNOWN = "unknown"


class ThreatLevel(str, Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BoundingBox(BaseModel):
    """Bounding box for detected objects."""
    x: float = Field(..., ge=0, le=1, description="X coordinate (normalized)")
    y: float = Field(..., ge=0, le=1, description="Y coordinate (normalized)")
    width: float = Field(..., ge=0, le=1, description="Width (normalized)")
    height: float = Field(..., ge=0, le=1, description="Height (normalized)")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    class_name: str = Field(..., description="Detected object class")


class EventMetadata(BaseModel):
    """Additional event metadata."""
    camera_location: Optional[str] = None
    weather_conditions: Optional[str] = None
    lighting_conditions: Optional[str] = None
    people_count: Optional[int] = None
    vehicle_count: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


class SecurityEvent(Base):
    """Security event database model."""
    __tablename__ = "events"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    threat_level = Column(String(20), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    bounding_boxes = Column(JSON, nullable=True)
    event_metadata = Column(JSON, nullable=True)
    processed = Column(String(20), default="pending", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic schemas for API
class SecurityEventCreate(BaseModel):
    """Schema for creating security events."""
    camera_id: str = Field(..., min_length=1, max_length=50, description="Camera identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: EventType = Field(..., description="Type of security event")
    threat_level: ThreatLevel = Field(..., description="Threat severity level")
    confidence_score: float = Field(..., ge=0, le=1, description="AI model confidence score")
    bounding_boxes: Optional[List[BoundingBox]] = Field(default=None, description="Detected object bounding boxes")
    metadata: Optional[EventMetadata] = Field(default=None, description="Additional event metadata")
    
    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Validate timestamp is not too far in the future."""
        if v > datetime.utcnow().replace(microsecond=0):
            # Allow small clock skew (5 minutes)
            max_future = datetime.utcnow().replace(microsecond=0)
            max_future = max_future.replace(minute=max_future.minute + 5)
            if v > max_future:
                raise ValueError("Timestamp cannot be more than 5 minutes in the future")
        return v
    
    @validator("confidence_score")
    def validate_confidence(cls, v):
        """Validate confidence score is reasonable."""
        if v < 0.1:
            raise ValueError("Confidence score too low for valid detection")
        return v


class SecurityEventResponse(BaseModel):
    """Schema for security event responses."""
    id: UUID
    camera_id: str
    timestamp: datetime
    event_type: EventType
    threat_level: ThreatLevel
    confidence_score: float
    bounding_boxes: Optional[List[BoundingBox]]
    metadata: Optional[EventMetadata]
    processed: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class EventFilter(BaseModel):
    """Schema for filtering events."""
    camera_id: Optional[str] = None
    event_type: Optional[EventType] = None
    threat_level: Optional[ThreatLevel] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    processed: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class EventStats(BaseModel):
    """Event statistics response."""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_threat_level: Dict[str, int]
    avg_confidence: float
    time_range: Dict[str, datetime]


class EdgeDeviceAuth(BaseModel):
    """Edge device authentication schema."""
    device_id: str = Field(..., min_length=1, max_length=100)
    device_secret: str = Field(..., min_length=32)
    location: Optional[str] = None
    capabilities: Optional[List[str]] = None