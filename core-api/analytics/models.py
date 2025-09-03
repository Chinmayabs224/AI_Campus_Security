"""
Analytics data models and schemas.
"""
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID


class TimeRange(str, Enum):
    """Time range options for analytics."""
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1m"
    QUARTER = "3m"
    YEAR = "1y"


class MetricType(str, Enum):
    """Types of metrics for analytics."""
    INCIDENT_COUNT = "incident_count"
    RESPONSE_TIME = "response_time"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    DETECTION_ACCURACY = "detection_accuracy"
    SYSTEM_UPTIME = "system_uptime"
    CAMERA_STATUS = "camera_status"


class IncidentPattern(BaseModel):
    """Incident pattern analysis result."""
    pattern_type: str
    frequency: int
    locations: List[str]
    time_periods: List[str]
    confidence: float
    description: str


class HeatMapPoint(BaseModel):
    """Heat map data point."""
    location_id: str
    location_name: str
    latitude: float
    longitude: float
    incident_count: int
    severity_score: float
    last_incident: Optional[datetime]


class TrendData(BaseModel):
    """Trend analysis data."""
    metric: MetricType
    time_range: TimeRange
    data_points: List[Dict[str, Any]]
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_percentage: float
    prediction: Optional[Dict[str, Any]]


class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime
    total_incidents: int
    resolved_incidents: int
    average_response_time: float
    false_positive_rate: float
    system_uptime: float
    active_cameras: int
    total_cameras: int
    detection_accuracy: float


class SecurityHotspot(BaseModel):
    """Security hotspot identification."""
    location_id: str
    location_name: str
    risk_score: float
    incident_frequency: int
    peak_hours: List[int]
    recommended_actions: List[str]


class PredictiveInsight(BaseModel):
    """Predictive analytics insight."""
    insight_type: str
    description: str
    confidence: float
    recommended_action: str
    impact_level: str  # "low", "medium", "high"
    time_horizon: str  # "short", "medium", "long"


class AnalyticsRequest(BaseModel):
    """Request model for analytics queries."""
    time_range: TimeRange
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    location_ids: Optional[List[str]] = None
    incident_types: Optional[List[str]] = None
    metrics: Optional[List[MetricType]] = None


class AnalyticsResponse(BaseModel):
    """Response model for analytics data."""
    request_id: str
    generated_at: datetime
    time_range: TimeRange
    patterns: List[IncidentPattern]
    heat_map: List[HeatMapPoint]
    trends: List[TrendData]
    performance_metrics: PerformanceMetrics
    hotspots: List[SecurityHotspot]
    predictions: List[PredictiveInsight]


class ComplianceReport(BaseModel):
    """Compliance report model."""
    report_id: UUID
    report_type: str
    generated_at: datetime
    period_start: date
    period_end: date
    total_incidents: int
    privacy_violations: int
    data_access_requests: int
    retention_compliance: float
    audit_findings: List[str]
    recommendations: List[str]


class ChainOfCustody(BaseModel):
    """Evidence chain of custody record."""
    evidence_id: UUID
    incident_id: UUID
    created_at: datetime
    custody_chain: List[Dict[str, Any]]
    integrity_verified: bool
    access_log: List[Dict[str, Any]]
    retention_expires: datetime