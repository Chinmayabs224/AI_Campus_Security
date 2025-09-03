"""
Extended audit logging models and schemas.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB

from core.database import Base


class AuditAction(str, Enum):
    """Standard audit actions."""
    # Authentication actions
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    
    # User management
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    USER_ACTIVATE = "user_activate"
    USER_DEACTIVATE = "user_deactivate"
    
    # Role and permission changes
    ROLE_ASSIGN = "role_assign"
    ROLE_REVOKE = "role_revoke"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    
    # Data access
    DATA_VIEW = "data_view"
    DATA_EXPORT = "data_export"
    DATA_DOWNLOAD = "data_download"
    
    # Evidence management
    EVIDENCE_ACCESS = "evidence_access"
    EVIDENCE_DOWNLOAD = "evidence_download"
    EVIDENCE_DELETE = "evidence_delete"
    EVIDENCE_REDACT = "evidence_redact"
    
    # Incident management
    INCIDENT_CREATE = "incident_create"
    INCIDENT_UPDATE = "incident_update"
    INCIDENT_ASSIGN = "incident_assign"
    INCIDENT_RESOLVE = "incident_resolve"
    INCIDENT_CLOSE = "incident_close"
    
    # System administration
    CONFIG_CHANGE = "config_change"
    SYSTEM_BACKUP = "system_backup"
    SYSTEM_RESTORE = "system_restore"
    
    # Privacy and compliance
    DSAR_REQUEST = "dsar_request"
    DATA_RETENTION_POLICY = "data_retention_policy"
    PRIVACY_ZONE_UPDATE = "privacy_zone_update"
    
    # API and integration
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    API_ACCESS = "api_access"
    
    # Generic actions
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"


class ResourceType(str, Enum):
    """Resource types for audit logging."""
    USER = "user"
    INCIDENT = "incident"
    EVENT = "event"
    EVIDENCE = "evidence"
    API_KEY = "api_key"
    SESSION = "session"
    AUDIT_LOG = "audit_log"
    SYSTEM_CONFIG = "system_config"
    PRIVACY_ZONE = "privacy_zone"
    CAMERA = "camera"
    NOTIFICATION = "notification"


class ComplianceTag(str, Enum):
    """Compliance framework tags."""
    GDPR = "gdpr"
    FERPA = "ferpa"
    COPPA = "coppa"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"


class AuditLogExtended(Base):
    """Extended audit log model with compliance features."""
    __tablename__ = "audit_logs_extended"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User and session information
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    username = Column(String(100), nullable=True, index=True)  # Denormalized for performance
    session_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    api_key_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Action details
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True, index=True)
    resource_id = Column(String(100), nullable=True, index=True)
    
    # Request details
    endpoint = Column(String(255), nullable=True, index=True)
    method = Column(String(10), nullable=True)
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    
    # Compliance and context
    compliance_tags = Column(JSONB, nullable=True)  # List of compliance frameworks
    risk_level = Column(String(20), nullable=False, default="low", index=True)
    business_justification = Column(Text, nullable=True)
    
    # Data sensitivity
    contains_pii = Column(Boolean, default=False, nullable=False, index=True)
    data_classification = Column(String(50), nullable=True, index=True)  # public, internal, confidential, restricted
    
    # Additional context
    event_metadata = Column(JSONB, nullable=True)
    before_state = Column(JSONB, nullable=True)  # State before change
    after_state = Column(JSONB, nullable=True)   # State after change
    
    # Result and error handling
    success = Column(Boolean, nullable=False, default=True, index=True)
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timing and performance
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    duration_ms = Column(Integer, nullable=True)  # Request duration in milliseconds
    
    # Retention and archival
    retention_date = Column(DateTime, nullable=True, index=True)
    archived = Column(Boolean, default=False, nullable=False, index=True)


class AuditLogFilter(BaseModel):
    """Filter for audit log queries."""
    user_id: Optional[uuid.UUID] = None
    username: Optional[str] = None
    action: Optional[AuditAction] = None
    resource_type: Optional[ResourceType] = None
    resource_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    ip_address: Optional[str] = None
    success: Optional[bool] = None
    compliance_tag: Optional[ComplianceTag] = None
    risk_level: Optional[str] = None
    contains_pii: Optional[bool] = None
    data_classification: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    
    @validator("end_time")
    def validate_time_range(cls, v, values):
        """Validate that end_time is after start_time."""
        if v and values.get("start_time") and v <= values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v


class AuditLogCreate(BaseModel):
    """Schema for creating audit log entries."""
    action: AuditAction
    resource_type: Optional[ResourceType] = None
    resource_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    compliance_tags: Optional[List[ComplianceTag]] = None
    risk_level: str = Field(default="low")
    business_justification: Optional[str] = None
    contains_pii: bool = Field(default=False)
    data_classification: Optional[str] = None
    event_metadata: Optional[Dict[str, Any]] = None
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    
    @validator("risk_level")
    def validate_risk_level(cls, v):
        """Validate risk level."""
        if v not in ["low", "medium", "high", "critical"]:
            raise ValueError("risk_level must be one of: low, medium, high, critical")
        return v


class AuditLogResponse(BaseModel):
    """Schema for audit log responses."""
    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    username: Optional[str]
    session_id: Optional[uuid.UUID]
    api_key_id: Optional[uuid.UUID]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    endpoint: Optional[str]
    method: Optional[str]
    ip_address: Optional[str]
    compliance_tags: Optional[List[str]]
    risk_level: str
    business_justification: Optional[str]
    contains_pii: bool
    data_classification: Optional[str]
    event_metadata: Optional[Dict[str, Any]]
    success: bool
    error_code: Optional[str]
    error_message: Optional[str]
    timestamp: datetime
    duration_ms: Optional[int]
    
    class Config:
        from_attributes = True


class ComplianceReport(BaseModel):
    """Compliance report schema."""
    report_id: uuid.UUID
    framework: ComplianceTag
    start_date: datetime
    end_date: datetime
    total_events: int
    high_risk_events: int
    pii_access_events: int
    failed_access_attempts: int
    user_activity_summary: Dict[str, int]
    resource_access_summary: Dict[str, int]
    generated_at: datetime
    generated_by: str


class AuditStats(BaseModel):
    """Audit statistics response."""
    total_logs: int
    logs_by_action: Dict[str, int]
    logs_by_resource_type: Dict[str, int]
    logs_by_risk_level: Dict[str, int]
    failed_actions: int
    pii_access_count: int
    unique_users: int
    time_range: Dict[str, datetime]


class DataRetentionPolicy(BaseModel):
    """Data retention policy configuration."""
    resource_type: ResourceType
    retention_days: int = Field(..., ge=1, le=3650)  # 1 day to 10 years
    compliance_framework: ComplianceTag
    auto_archive: bool = Field(default=True)
    auto_delete: bool = Field(default=False)
    
    @validator("retention_days")
    def validate_retention_days(cls, v, values):
        """Validate retention days based on compliance framework."""
        framework = values.get("compliance_framework")
        if framework == ComplianceTag.GDPR and v > 2555:  # 7 years max for GDPR
            raise ValueError("GDPR retention cannot exceed 7 years")
        elif framework == ComplianceTag.FERPA and v > 1825:  # 5 years max for FERPA
            raise ValueError("FERPA retention cannot exceed 5 years")
        return v


class DSARRequest(BaseModel):
    """Data Subject Access Request schema."""
    request_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    subject_identifier: str = Field(..., description="Email, username, or other identifier")
    request_type: str = Field(..., description="access, rectification, erasure, portability")
    requested_data_types: List[str] = Field(..., description="Types of data requested")
    business_justification: Optional[str] = None
    requested_by: str = Field(..., description="Who made the request")
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: datetime = Field(..., description="Legal deadline for response")
    status: str = Field(default="pending", description="pending, in_progress, completed, rejected")
    
    @validator("request_type")
    def validate_request_type(cls, v):
        """Validate DSAR request type."""
        valid_types = ["access", "rectification", "erasure", "portability", "restriction"]
        if v not in valid_types:
            raise ValueError(f"request_type must be one of: {', '.join(valid_types)}")
        return v