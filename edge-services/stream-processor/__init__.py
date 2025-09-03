"""
RTSP Stream Processing Module for AI Campus Security System

This module provides RTSP stream ingestion, processing, and health monitoring
capabilities for IP camera integration.
"""

from .rtsp_client import (
    RTSPStreamProcessor,
    RTSPStreamManager,
    CameraConfig,
    StreamHealth,
    StreamStatus
)
from .config import ConfigManager, StreamProcessorConfig, setup_logging
from .service import StreamProcessorService

__all__ = [
    'RTSPStreamProcessor',
    'RTSPStreamManager', 
    'CameraConfig',
    'StreamHealth',
    'StreamStatus',
    'ConfigManager',
    'StreamProcessorConfig',
    'StreamProcessorService',
    'setup_logging'
]