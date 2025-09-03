"""
Authentication and authorization models.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum
import uuid

from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from core.database import Base


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    SECURITY_SUPERVISOR = "security_supervisor"
    SECURITY_GUARD = "security_guard"
    ANALYST = "analyst"
    VIEWER = "viewer"
    EDGE_DEVICE = "edge_device"


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default=UserRole.VIEWER.value)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # SAML attributes
    saml_name_id = Column(String(255), unique=True, nullable=True, index=True)
    saml_attributes = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    """User session model for JWT token management."""
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Session metadata
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class APIKey(Base):
    """API key model for edge device authentication."""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    prefix = Column(String(20), nullable=False, index=True)  # First few chars for identification
    
    # Permissions and restrictions
    scopes = Column(JSONB, nullable=False, default=list)  # List of allowed scopes
    ip_whitelist = Column(JSONB, nullable=True)  # Optional IP restrictions
    
    # Status and expiration
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


class AuditLog(Base):
    """Audit log model for compliance tracking."""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User and session information
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    api_key_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Action details
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True, index=True)
    resource_id = Column(String(100), nullable=True, index=True)
    
    # Request details
    endpoint = Column(String(255), nullable=True)
    method = Column(String(10), nullable=True)
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    
    # Additional context
    event_metadata = Column(JSONB, nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamp (immutable)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


# Pydantic models for API

class UserBase(BaseModel):
    """Base user model."""
    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., max_length=255)
    full_name: str = Field(..., min_length=1, max_length=255)
    role: UserRole = UserRole.VIEWER


class UserCreate(UserBase):
    """User creation model."""
    saml_name_id: Optional[str] = None
    saml_attributes: Optional[dict] = None


class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[str] = Field(None, max_length=255)
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """User response model."""
    id: uuid.UUID
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyCreate(BaseModel):
    """API key creation model."""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default_factory=list)
    ip_whitelist: Optional[List[str]] = None
    expires_at: Optional[datetime] = None


class APIKeyResponse(BaseModel):
    """API key response model."""
    id: uuid.UUID
    name: str
    prefix: str
    scopes: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    created_at: datetime
    last_used: Optional[datetime]
    
    class Config:
        from_attributes = True


class APIKeyWithSecret(APIKeyResponse):
    """API key response with secret (only returned on creation)."""
    key: str


class AuditLogResponse(BaseModel):
    """Audit log response model."""
    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    endpoint: Optional[str]
    method: Optional[str]
    ip_address: Optional[str]
    success: bool
    timestamp: datetime
    
    class Config:
        from_attributes = True