"""
Event Generator Module

This module provides comprehensive event generation and local buffering
capabilities for the AI Campus Security system.
"""

from .models import (
    SecurityEvent, IncidentClip, EventBuffer, EventType, 
    SeverityLevel, EventStatus, BoundingBox
)
from .event_service import SecurityEventService, EventGenerationConfig
from .clip_extractor import ClipExtractor, ClipExtractionConfig
from .local_buffer import LocalBufferService, BufferConfig
from .service import EdgeEventService, EdgeEventConfig

__all__ = [
    'SecurityEvent', 'IncidentClip', 'EventBuffer', 'EventType',
    'SeverityLevel', 'EventStatus', 'BoundingBox',
    'SecurityEventService', 'EventGenerationConfig',
    'ClipExtractor', 'ClipExtractionConfig', 
    'LocalBufferService', 'BufferConfig',
    'EdgeEventService', 'EdgeEventConfig'
]