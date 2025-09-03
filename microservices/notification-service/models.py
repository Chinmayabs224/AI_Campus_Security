"""
Data models for the notification service
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

class NotificationChannel(Enum):
    """Supported notification channels"""
    PUSH = "push"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class IncidentType(Enum):
    """Types of security incidents"""
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    CROWDING = "crowding"
    ABANDONED_OBJECT = "abandoned_object"
    VIOLENCE = "violence"
    FIRE = "fire"
    MEDICAL_EMERGENCY = "medical_emergency"
    SYSTEM_ALERT = "system_alert"

@dataclass
class NotificationRequest:
    """Request model for sending notifications"""
    notification_id: str
    user_id: str
    title: str
    message: str
    channels: List[NotificationChannel]
    priority: NotificationPriority
    incident_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    scheduled_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None

@dataclass
class NotificationResponse:
    """Response model for notification delivery"""
    notification_id: str
    success: bool
    results: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0

@dataclass
class UserPreferences:
    """User notification preferences"""
    user_id: str
    push_enabled: bool = True
    sms_enabled: bool = True
    whatsapp_enabled: bool = False
    email_enabled: bool = True
    push_min_priority: NotificationPriority = NotificationPriority.LOW
    sms_min_priority: NotificationPriority = NotificationPriority.MEDIUM
    whatsapp_min_priority: NotificationPriority = NotificationPriority.HIGH
    email_min_priority: NotificationPriority = NotificationPriority.LOW
    quiet_hours_start: Optional[str] = None  # Format: "22:00"
    quiet_hours_end: Optional[str] = None    # Format: "08:00"
    timezone: str = "UTC"
    incident_types: List[IncidentType] = None  # None means all types
    
    def __post_init__(self):
        if self.incident_types is None:
            self.incident_types = list(IncidentType)

@dataclass
class DeliveryResult:
    """Result of a single notification delivery attempt"""
    channel: NotificationChannel
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class BulkNotificationRequest:
    """Request model for bulk notifications"""
    notifications: List[NotificationRequest]
    batch_id: str
    priority: NotificationPriority = NotificationPriority.MEDIUM

@dataclass
class NotificationTemplate:
    """Template for notification messages"""
    template_id: str
    name: str
    title_template: str
    message_template: str
    supported_channels: List[NotificationChannel]
    default_priority: NotificationPriority = NotificationPriority.MEDIUM
    variables: List[str] = None  # List of template variables like {incident_type}, {location}
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []

@dataclass
class NotificationStats:
    """Statistics for notification delivery"""
    total_sent: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    delivery_rate: float = 0.0
    average_delivery_time: float = 0.0
    channel_stats: Dict[str, Dict[str, int]] = None
    
    def __post_init__(self):
        if self.channel_stats is None:
            self.channel_stats = {}
        
        if self.total_sent > 0:
            self.delivery_rate = self.successful_deliveries / self.total_sent * 100

@dataclass
class AlertEscalation:
    """Configuration for alert escalation rules"""
    escalation_id: str
    incident_types: List[IncidentType]
    initial_channels: List[NotificationChannel]
    escalation_delay_minutes: int
    escalation_channels: List[NotificationChannel]
    escalation_users: List[str]
    max_escalation_levels: int = 3
    enabled: bool = True