"""
Main Event Generation Service

This module integrates all event generation components into a unified service
for processing AI detections into security events with local buffering.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import json

from .models import SecurityEvent, IncidentClip, EventType, SeverityLevel
from .event_service import SecurityEventService, EventGenerationConfig
from .clip_extractor import ClipExtractor, ClipExtractionConfig
from .local_buffer import LocalBufferService, BufferConfig
from ..inference_engine.yolo_detector import InferenceResult


class EdgeEventConfig:
    """Combined configuration for edge event processing"""
    
    def __init__(self):
        # Service settings
        self.service_name = "edge-event-generator"
        self.log_level = "INFO"
        
        # Component configs
        self.event_config = EventGenerationConfig()
        self.clip_config = ClipExtractionConfig()
        self.buffer_config = BufferConfig()
        
        # Integration settings
        self.enable_clip_extraction = True
        self.enable_local_buffering = True
        self.auto_cleanup_interval = 3600  # 1 hour
        
        # Network settings
        self.central_api_url = None
        self.api_timeout = 30
        self.api_retry_attempts = 3
    
    @classmethod
    def from_file(cls, config_path: str) -> 'EdgeEventConfig':
        """Load configuration from JSON file"""
        config = cls()
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Update configuration from file
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Update component configs
            if 'event_config' in data:
                for key, value in data['event_config'].items():
                    if hasattr(config.event_config, key):
                        setattr(config.event_config, key, value)
            
            if 'clip_config' in data:
                for key, value in data['clip_config'].items():
                    if hasattr(config.clip_config, key):
                        setattr(config.clip_config, key, value)
            
            if 'buffer_config' in data:
                for key, value in data['buffer_config'].items():
                    if hasattr(config.buffer_config, key):
                        setattr(config.buffer_config, key, value)
                        
        except Exception as e:
            logging.warning(f"Error loading config from {config_path}: {e}")
        
        return config


class EdgeEventService:
    """
    Main edge event processing service that integrates:
    - Event generation from AI detections
    - Video clip extraction
    - Local buffering for network outages
    - Synchronization with central services
    """
    
    def __init__(self, config: Optional[EdgeEventConfig] = None):
        self.config = config or EdgeEventConfig()
        self.logger = logging.getLogger("edge_event_service")
        
        # Initialize components
        self.event_service = SecurityEventService(self.config.event_config)
        
        if self.config.enable_clip_extraction:
            self.clip_extractor = ClipExtractor(self.config.clip_config)
        else:
            self.clip_extractor = None
        
        if self.config.enable_local_buffering:
            self.buffer_service = LocalBufferService(self.config.buffer_config)
        else:
            self.buffer_service = None
        
        # Service state
        self.running = False
        self.cleanup_thread: Optional[threading.Thread] = None
        
        # External callbacks
        self.event_callbacks: List[Callable[[SecurityEvent], None]] = []
        self.clip_callbacks: List[Callable[[IncidentClip], None]] = []
        self.sync_callback: Optional[Callable[[List[SecurityEvent], List[IncidentClip]], bool]] = None
        
        # Statistics
        self.stats = {
            'detections_processed': 0,
            'events_generated': 0,
            'clips_extracted': 0,
            'events_buffered': 0,
            'clips_buffered': 0,
            'start_time': time.time()
        }
        
        # Setup component integration
        self._setup_component_integration()
    
    def _setup_component_integration(self):
        """Setup integration between components"""
        # Event service callbacks
        self.event_service.add_event_callback(self._handle_generated_event)
        
        # Clip extractor callbacks
        if self.clip_extractor:
            self.clip_extractor.add_clip_callback(self._handle_extracted_clip)
        
        # Buffer service callbacks
        if self.buffer_service and self.sync_callback:
            self.buffer_service.add_sync_callback(self.sync_callback)
    
    def start(self) -> bool:
        """Start the edge event service"""
        if self.running:
            return True
        
        self.logger.info("Starting Edge Event Service")
        
        try:
            # Start event service
            if not self.event_service.start():
                self.logger.error("Failed to start event service")
                return False
            
            # Start clip extractor
            if self.clip_extractor and not self.clip_extractor.start():
                self.logger.error("Failed to start clip extractor")
                return False
            
            # Start buffer service
            if self.buffer_service and not self.buffer_service.start():
                self.logger.error("Failed to start buffer service")
                return False
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            
            self.running = True
            self.logger.info("Edge Event Service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting service: {e}")
            return False
    
    def stop(self):
        """Stop the edge event service"""
        if not self.running:
            return
        
        self.logger.info("Stopping Edge Event Service")
        self.running = False
        
        # Stop components
        self.event_service.stop()
        
        if self.clip_extractor:
            self.clip_extractor.stop()
        
        if self.buffer_service:
            self.buffer_service.stop()
        
        # Stop cleanup thread
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        self.logger.info("Edge Event Service stopped")
    
    def add_camera(self, camera_id: str) -> bool:
        """
        Add a camera for frame buffering
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            True if camera was added successfully
        """
        success = True
        
        if self.clip_extractor:
            success &= self.clip_extractor.add_camera(camera_id)
        
        return success
    
    def process_frame(self, camera_id: str, frame, timestamp: Optional[float] = None) -> bool:
        """
        Process a video frame for clip extraction
        
        Args:
            camera_id: Camera identifier
            frame: Video frame
            timestamp: Frame timestamp
            
        Returns:
            True if frame was processed successfully
        """
        if not self.clip_extractor:
            return True
        
        return self.clip_extractor.add_frame(camera_id, frame, timestamp)
    
    def process_detection_result(self, result: InferenceResult):
        """
        Process AI detection result to generate security events
        
        Args:
            result: YOLO inference result
        """
        try:
            self.stats['detections_processed'] += 1
            
            # Process through event service
            self.event_service.process_detection_result(result)
            
        except Exception as e:
            self.logger.error(f"Error processing detection result: {e}")
    
    def _handle_generated_event(self, event: SecurityEvent):
        """Handle newly generated security event"""
        try:
            self.stats['events_generated'] += 1
            
            # Extract clip if enabled and event severity is high enough
            if (self.clip_extractor and 
                event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]):
                
                clip = self.clip_extractor.extract_clip_for_event(event)
                if clip:
                    self.logger.info(f"Queued clip extraction for event {event.event_id}")
            
            # Buffer event if local buffering is enabled
            if self.buffer_service:
                if self.buffer_service.buffer_event(event):
                    self.stats['events_buffered'] += 1
            
            # Call external event callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling generated event: {e}")
    
    def _handle_extracted_clip(self, clip: IncidentClip):
        """Handle newly extracted video clip"""
        try:
            self.stats['clips_extracted'] += 1
            
            # Buffer clip if local buffering is enabled
            if self.buffer_service:
                if self.buffer_service.buffer_clip(clip):
                    self.stats['clips_buffered'] += 1
            
            # Call external clip callbacks
            for callback in self.clip_callbacks:
                try:
                    callback(clip)
                except Exception as e:
                    self.logger.error(f"Error in clip callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling extracted clip: {e}")
    
    def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.running:
            try:
                time.sleep(self.config.auto_cleanup_interval)
                
                if not self.running:
                    break
                
                self.logger.info("Running periodic cleanup")
                
                # Cleanup old clips
                if self.clip_extractor:
                    removed_clips = self.clip_extractor.cleanup_old_clips(24.0)
                    if removed_clips > 0:
                        self.logger.info(f"Cleaned up {removed_clips} old clip files")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    def add_event_callback(self, callback: Callable[[SecurityEvent], None]):
        """Add callback for generated events"""
        self.event_callbacks.append(callback)
    
    def add_clip_callback(self, callback: Callable[[IncidentClip], None]):
        """Add callback for extracted clips"""
        self.clip_callbacks.append(callback)
    
    def set_sync_callback(self, callback: Callable[[List[SecurityEvent], List[IncidentClip]], bool]):
        """Set callback for synchronizing data with central services"""
        self.sync_callback = callback
        if self.buffer_service:
            self.buffer_service.add_sync_callback(callback)
    
    def set_network_check_callback(self, callback: Callable[[], bool]):
        """Set callback for checking network connectivity"""
        if self.buffer_service:
            self.buffer_service.set_network_check_callback(callback)
    
    def force_sync(self) -> bool:
        """Force immediate synchronization with central services"""
        if self.buffer_service:
            return self.buffer_service.force_sync()
        return False
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all components"""
        status = {
            'service': {
                'running': self.running,
                'uptime_seconds': time.time() - self.stats['start_time'],
                **self.stats
            }
        }
        
        # Event service status
        if self.event_service:
            status['event_service'] = self.event_service.get_statistics()
        
        # Clip extractor status
        if self.clip_extractor:
            status['clip_extractor'] = {
                'statistics': self.clip_extractor.get_statistics(),
                'buffer_status': self.clip_extractor.get_buffer_status()
            }
        
        # Buffer service status
        if self.buffer_service:
            status['buffer_service'] = self.buffer_service.get_status()
        
        return status
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event generation statistics"""
        if self.event_service:
            return self.event_service.get_statistics()
        return {}
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get local buffer status"""
        if self.buffer_service:
            return self.buffer_service.get_status()
        return {}
    
    def create_test_event(self, camera_id: str, event_type: EventType = EventType.INTRUSION) -> SecurityEvent:
        """Create a test event for debugging purposes"""
        from .models import BoundingBox
        
        test_event = SecurityEvent(
            camera_id=camera_id,
            event_type=event_type,
            severity=SeverityLevel.MEDIUM,
            confidence_score=0.75,
            bounding_boxes=[
                BoundingBox(
                    x1=0.3, y1=0.3, x2=0.7, y2=0.7,
                    confidence=0.75, class_id=0, class_name="person"
                )
            ],
            metadata={'test': True, 'generated_by': 'create_test_event'}
        )
        
        return test_event


def main():
    """Main entry point for standalone service"""
    import argparse
    import signal
    import sys
    
    parser = argparse.ArgumentParser(description='Edge Event Generation Service')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    if args.config:
        config = EdgeEventConfig.from_file(args.config)
    else:
        config = EdgeEventConfig()
    
    # Create and start service
    service = EdgeEventService(config)
    
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if service.start():
            logging.info("Service started successfully. Press Ctrl+C to stop.")
            
            # Keep service running
            while service.running:
                time.sleep(1)
        else:
            logging.error("Failed to start service")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Received interrupt, shutting down...")
    except Exception as e:
        logging.error(f"Service error: {e}")
        sys.exit(1)
    finally:
        service.stop()


if __name__ == '__main__':
    main()