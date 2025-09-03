"""
Incident management data models and schemas.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from core.database import Base


class IncidentStatus(str, Enum):
    """Incident status values."""
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentPriority(str, Enum):
    """Incident priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Incident(Base):
    """Incident database model."""
    __tablename__ = "incidents"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default=IncidentStatus.OPEN.value, index=True)
    severity = Column(String(20), nullable=False, index=True)
    priority = Column(String(20), nullable=False, default=IncidentPriority.NORMAL.value, index=True)
    
    # Assignment and ownership
    assigned_to = Column(String(100), nullable=True, index=True)
    created_by = Column(String(100), nullable=True)
    
    # Location and context
    location = Column(String(200), nullable=True)
    camera_ids = Column(String(500), nullable=True)  # Comma-separated camera IDs
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    assigned_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    
    # Escalation
    escalated = Column(Boolean, default=False, index=True)
    escalated_at = Column(DateTime, nullable=True)
    escalation_reason = Column(Text, nullable=True)


class IncidentEvent(Base):
    """Association between incidents and events."""
    __tablename__ = "incident_events"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(PGUUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    event_id = Column(PGUUID(as_uuid=True), ForeignKey("events.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class IncidentNote(Base):
    """Notes and comments on incidents."""
    __tablename__ = "incident_notes"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(PGUUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    author = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    note_type = Column(String(50), default="comment")  # comment, status_change, assignment, etc.
    created_at = Column(DateTime, default=datetime.utcnow)


class IncidentStatusHistory(Base):
    """Track incident status changes."""
    __tablename__ = "incident_status_history"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id = Column(PGUUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    previous_status = Column(String(20), nullable=True)
    new_status = Column(String(20), nullable=False)
    changed_by = Column(String(100), nullable=False)
    change_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic schemas for API
class IncidentCreate(BaseModel):
    """Schema for creating incidents."""
    title: str = Field(..., min_length=1, max_length=200, description="Incident title")
    description: Optional[str] = Field(None, description="Detailed incident description")
    severity: IncidentSeverity = Field(..., description="Incident severity level")
    priority: IncidentPriority = Field(default=IncidentPriority.NORMAL, description="Incident priority")
    location: Optional[str] = Field(None, max_length=200, description="Incident location")
    camera_ids: Optional[List[str]] = Field(None, description="Related camera IDs")
    event_ids: Optional[List[UUID]] = Field(None, description="Related event IDs")
    assigned_to: Optional[str] = Field(None, max_length=100, description="Assigned user")
    
    @validator("title")
    def validate_title(cls, v):
        """Validate title is not empty."""
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()


class IncidentUpdate(BaseModel):
    """Schema for updating incidents."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[IncidentStatus] = None
    severity: Optional[IncidentSeverity] = None
    priority: Optional[IncidentPriority] = None
    assigned_to: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    escalation_reason: Optional[str] = None


class IncidentResponse(BaseModel):
    """Schema for incident responses."""
    id: UUID
    title: str
    description: Optional[str]
    status: IncidentStatus
    severity: IncidentSeverity
    priority: IncidentPriority
    assigned_to: Optional[str]
    created_by: Optional[str]
    location: Optional[str]
    camera_ids: Optional[str]
    created_at: datetime
    updated_at: datetime
    assigned_at: Optional[datetime]
    resolved_at: Optional[datetime]
    closed_at: Optional[datetime]
    escalated: bool
    escalated_at: Optional[datetime]
    escalation_reason: Optional[str]
    
    class Config:
        from_attributes = True


class IncidentFilter(BaseModel):
    """Schema for filtering incidents."""
    status: Optional[IncidentStatus] = None
    severity: Optional[IncidentSeverity] = None
    priority: Optional[IncidentPriority] = None
    assigned_to: Optional[str] = None
    created_by: Optional[str] = None
    location: Optional[str] = None
    escalated: Optional[bool] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    search: Optional[str] = Field(None, description="Search in title and description")
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class IncidentNoteCreate(BaseModel):
    """Schema for creating incident notes."""
    content: str = Field(..., min_length=1, max_length=5000)
    note_type: str = Field(default="comment", max_length=50)


class IncidentNoteResponse(BaseModel):
    """Schema for incident note responses."""
    id: UUID
    incident_id: UUID
    author: str
    content: str
    note_type: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class IncidentAssignment(BaseModel):
    """Schema for incident assignment."""
    assigned_to: str = Field(..., max_length=100)
    assignment_reason: Optional[str] = Field(None, max_length=500)


class IncidentEscalation(BaseModel):
    """Schema for incident escalation."""
    escalation_reason: str = Field(..., min_length=1, max_length=1000)
    new_priority: Optional[IncidentPriority] = None
    new_severity: Optional[IncidentSeverity] = None


class IncidentStats(BaseModel):
    """Incident statistics response."""
    total_incidents: int
    incidents_by_status: Dict[str, int]
    incidents_by_severity: Dict[str, int]
    incidents_by_priority: Dict[str, int]
    avg_resolution_time_hours: Optional[float]
    escalated_incidents: int
    unassigned_incidents: int


class IncidentSummary(BaseModel):
    """Brief incident summary for dashboards."""
    id: UUID
    title: str
    status: IncidentStatus
    severity: IncidentSeverity
    priority: IncidentPriority
    assigned_to: Optional[str]
    created_at: datetime
    location: Optional[str]