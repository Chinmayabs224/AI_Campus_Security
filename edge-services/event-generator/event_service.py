"""
Security Event Generation Service

This module handles the generation, validation, and processing of security events
from AI detection results with deduplication and aggregation capabilities.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Callable, Set
from collections import defaultdict, deque
import threading
from queue import Queue, Empty
import json

from .models import (
    SecurityEvent, EventType, SeverityLevel, EventStatus,
    BoundingBox, IncidentClip, EventBuffer
)
from ..inference_engine.yolo_detector import InferenceResult, Detection


class EventGenerationConfig:
    """Configuration for event generation"""
    
    def __init__(self):
        # Event thresholds
        self.min_confidence = 0.5
        self.high_confidence = 0.8
        self.critical_confidence = 0.9
        
        # Deduplication settings
        self.dedup_time_window = 10.0  # seconds
        self.dedup_spatial_threshold = 0.3  # IoU threshold
        
        # Aggregation settings
        self.aggregation_window = 5.0  # seconds
        self.min_detections_for_event = 2
        
        # Event type mappings
        self.class_to_event_type = {
            'person': EventType.INTRUSION,
            'car': EventType.SUSPICIOUS_ACTIVITY,
            'truck': EventType.SUSPICIOUS_ACTIVITY,
            'motorcycle': EventType.SUSPICIOUS_ACTIVITY,
            'bicycle': EventType.SUSPICIOUS_ACTIVITY,
            'handbag': EventType.ABANDONED_OBJECT,
            'suitcase': EventType.ABANDONED_OBJECT,
            'backpack': EventType.ABANDONED_OBJECT,
        }
        
        # Severity mappings
        self.confidence_to_severity = [
            (0.9, SeverityLevel.CRITICAL),
            (0.8, SeverityLevel.HIGH),
            (0.6, SeverityLevel.MEDIUM),
            (0.0, SeverityLevel.LOW)
        ]
        
        # Zone-based rules
        self.zone_rules = {}  # zone_id -> rules dict


class EventAggregator:
    """
    Aggregates multiple detections into coherent security events
    """
    
    def __init__(self, config: EventGenerationConfig):
        self.config = config
        self.logger = logging.getLogger("event_aggregator")
        
        # Detection buffers per camera
        self.detection_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
    
    def add_detection_result(self, result: InferenceResult) -> List[SecurityEvent]:
        """
        Add detection result and generate events if conditions are met
        
        Args:
            result: YOLO inference result
            
        Returns:
            List of generated security events
        """
        with self.lock:
            camera_id = result.camera_id
            
            # Add to buffer
            self.detection_buffers[camera_id].append(result)
            
            # Clean old detections
            self._cleanup_old_detections(camera_id)
            
            # Generate events from current buffer
            return self._generate_events_from_buffer(camera_id)
    
    def _cleanup_old_detections(self, camera_id: str):
        """Remove old detections from buffer"""
        current_time = time.time()
        buffer = self.detection_buffers[camera_id]
        
        while buffer and current_time - buffer[0].timestamp > self.config.aggregation_window:
            buffer.popleft()
    
    def _generate_events_from_buffer(self, camera_id: str) -> List[SecurityEvent]:
        """Generate events from detection buffer"""
        buffer = self.detection_buffers[camera_id]
        if len(buffer) < self.config.min_detections_for_event:
            return []
        
        events = []
        current_time = time.time()
        
        # Group detections by type and location
        detection_groups = self._group_detections(buffer)
        
        for group_key, detections in detection_groups.items():
            if len(detections) >= self.config.min_detections_for_event:
                event = self._create_event_from_detections(camera_id, detections)
                if event:
                    events.append(event)
        
        return events
    
    def _group_detections(self, buffer: deque) -> Dict[str, List[Detection]]:
        """Group detections by type and spatial proximity"""
        groups = defaultdict(list)
        
        for result in buffer:
            for detection in result.detections:
                # Filter by confidence
                if detection.confidence < self.config.min_confidence:
                    continue
                
                # Create group key based on class and spatial location
                center_x, center_y = self._get_detection_center(detection)
                spatial_key = f"{int(center_x * 10)}_{int(center_y * 10)}"  # 10x10 grid
                group_key = f"{detection.class_name}_{spatial_key}"
                
                groups[group_key].append(detection)
        
        return groups
    
    def _get_detection_center(self, detection: Detection) -> tuple:
        """Get center point of detection bounding box"""
        x1, y1, x2, y2 = detection.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _create_event_from_detections(self, camera_id: str, detections: List[Detection]) -> Optional[SecurityEvent]:
        """Create a security event from grouped detections"""
        if not detections:
            return None
        
        # Use the most recent detection as primary
        primary_detection = max(detections, key=lambda d: d.timestamp)
        
        # Calculate average confidence
        avg_confidence = sum(d.confidence for d in detections) / len(detections)
        
        # Determine event type
        event_type = self.config.class_to_event_type.get(
            primary_detection.class_name, 
            EventType.SUSPICIOUS_ACTIVITY
        )
        
        # Determine severity
        severity = self._calculate_severity(avg_confidence, len(detections))
        
        # Create bounding boxes
        bounding_boxes = []
        for detection in detections[-5:]:  # Keep last 5 detections
            bbox = BoundingBox(
                x1=detection.bbox[0],
                y1=detection.bbox[1],
                x2=detection.bbox[2],
                y2=detection.bbox[3],
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name
            )
            bounding_boxes.append(bbox)
        
        # Create event
        event = SecurityEvent(
            camera_id=camera_id,
            timestamp=primary_detection.timestamp,
            event_type=event_type,
            severity=severity,
            confidence_score=avg_confidence,
            bounding_boxes=bounding_boxes,
            metadata={
                'detection_count': len(detections),
                'detection_duration': detections[-1].timestamp - detections[0].timestamp,
                'primary_class': primary_detection.class_name,
                'all_classes': list(set(d.class_name for d in detections))
            }
        )
        
        return event
    
    def _calculate_severity(self, confidence: float, detection_count: int) -> SeverityLevel:
        """Calculate event severity based on confidence and detection count"""
        # Boost confidence based on detection count
        boosted_confidence = min(1.0, confidence + (detection_count - 1) * 0.1)
        
        for threshold, severity in self.config.confidence_to_severity:
            if boosted_confidence >= threshold:
                return severity
        
        return SeverityLevel.LOW


class EventDeduplicator:
    """
    Handles deduplication of similar security events
    """
    
    def __init__(self, config: EventGenerationConfig):
        self.config = config
        self.logger = logging.getLogger("event_deduplicator")
        
        # Recent events for deduplication
        self.recent_events: Dict[str, List[SecurityEvent]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def process_event(self, event: SecurityEvent) -> Optional[SecurityEvent]:
        """
        Process event for deduplication
        
        Args:
            event: Security event to process
            
        Returns:
            Event if it's unique, None if it's a duplicate
        """
        with self.lock:
            camera_id = event.camera_id
            
            # Clean old events
            self._cleanup_old_events(camera_id)
            
            # Check for duplicates
            for existing_event in self.recent_events[camera_id]:
                if event.is_similar_to(
                    existing_event,
                    time_threshold=self.config.dedup_time_window,
                    spatial_threshold=self.config.dedup_spatial_threshold
                ):
                    self.logger.debug(f"Duplicate event detected for camera {camera_id}")
                    return None
            
            # Add to recent events
            self.recent_events[camera_id].append(event)
            return event
    
    def _cleanup_old_events(self, camera_id: str):
        """Remove old events from deduplication buffer"""
        current_time = time.time()
        self.recent_events[camera_id] = [
            event for event in self.recent_events[camera_id]
            if current_time - event.timestamp <= self.config.dedup_time_window * 2
        ]


class SecurityEventService:
    """
    Main service for generating and processing security events
    """
    
    def __init__(self, config: Optional[EventGenerationConfig] = None):
        self.config = config or EventGenerationConfig()
        self.logger = logging.getLogger("security_event_service")
        
        # Components
        self.aggregator = EventAggregator(self.config)
        self.deduplicator = EventDeduplicator(self.config)
        
        # Event processing
        self.event_queue = Queue(maxsize=1000)
        self.event_callbacks: List[Callable[[SecurityEvent], None]] = []
        
        # Processing thread
        self._processing = False
        self._process_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'events_generated': 0,
            'events_deduplicated': 0,
            'detections_processed': 0,
            'start_time': time.time()
        }
    
    def start(self) -> bool:
        """Start the event processing service"""
        if self._processing:
            return True
        
        self.logger.info("Starting Security Event Service")
        self._processing = True
        
        # Start processing thread
        self._process_thread = threading.Thread(target=self._process_events, daemon=True)
        self._process_thread.start()
        
        return True
    
    def stop(self):
        """Stop the event processing service"""
        if not self._processing:
            return
        
        self.logger.info("Stopping Security Event Service")
        self._processing = False
        
        # Wait for processing thread
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=5)
    
    def add_event_callback(self, callback: Callable[[SecurityEvent], None]):
        """Add callback for event processing"""
        self.event_callbacks.append(callback)
    
    def process_detection_result(self, result: InferenceResult):
        """
        Process YOLO detection result and generate events
        
        Args:
            result: YOLO inference result
        """
        try:
            self.stats['detections_processed'] += 1
            
            # Generate events from detections
            events = self.aggregator.add_detection_result(result)
            
            # Process each generated event
            for event in events:
                # Apply deduplication
                unique_event = self.deduplicator.process_event(event)
                
                if unique_event:
                    # Add to processing queue
                    try:
                        self.event_queue.put_nowait(unique_event)
                        self.stats['events_generated'] += 1
                    except:
                        self.logger.warning("Event queue full, dropping event")
                else:
                    self.stats['events_deduplicated'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error processing detection result: {e}")
    
    def _process_events(self):
        """Main event processing loop"""
        while self._processing:
            try:
                # Get event from queue
                try:
                    event = self.event_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Apply additional processing
                processed_event = self._enhance_event(event)
                
                # Call event callbacks
                for callback in self.event_callbacks:
                    try:
                        callback(processed_event)
                    except Exception as e:
                        self.logger.error(f"Error in event callback: {e}")
                
                self.event_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
    
    def _enhance_event(self, event: SecurityEvent) -> SecurityEvent:
        """Apply additional processing to enhance event"""
        # Apply zone-based rules
        if event.zone_id and event.zone_id in self.config.zone_rules:
            zone_rules = self.config.zone_rules[event.zone_id]
            
            # Adjust severity based on zone
            if 'severity_multiplier' in zone_rules:
                multiplier = zone_rules['severity_multiplier']
                if multiplier > 1.0 and event.severity != SeverityLevel.CRITICAL:
                    # Upgrade severity
                    severity_levels = [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
                    current_index = severity_levels.index(event.severity)
                    new_index = min(len(severity_levels) - 1, current_index + 1)
                    event.severity = severity_levels[new_index]
        
        # Add processing timestamp
        event.processed_at = time.time()
        
        return event
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        runtime = time.time() - self.stats['start_time']
        
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'events_per_minute': (self.stats['events_generated'] / runtime * 60) if runtime > 0 else 0,
            'deduplication_rate': (
                self.stats['events_deduplicated'] / 
                (self.stats['events_generated'] + self.stats['events_deduplicated'])
                if (self.stats['events_generated'] + self.stats['events_deduplicated']) > 0 else 0
            ),
            'queue_size': self.event_queue.qsize(),
            'processing': self._processing
        }
    
    def reset_statistics(self):
        """Reset service statistics"""
        self.stats = {
            'events_generated': 0,
            'events_deduplicated': 0,
            'detections_processed': 0,
            'start_time': time.time()
        }