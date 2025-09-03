"""
Test script for event generation functionality

This script tests the core event generation, clip extraction, and local buffering
functionality to ensure everything works correctly.
"""

import time
import numpy as np
import logging
from typing import List
import tempfile
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from .models import SecurityEvent, EventType, SeverityLevel, BoundingBox
from .service import EdgeEventService, EdgeEventConfig
from ..inference-engine.yolo_detector import InferenceResult, Detection


def create_test_detection_result(camera_id: str = "test_camera_01") -> InferenceResult:
    """Create a test detection result for testing"""
    
    # Create test detections
    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(0.3, 0.2, 0.7, 0.8),
            timestamp=time.time()
        ),
        Detection(
            class_id=26,
            class_name="handbag",
            confidence=0.75,
            bbox=(0.1, 0.6, 0.3, 0.9),
            timestamp=time.time()
        )
    ]
    
    return InferenceResult(
        camera_id=camera_id,
        frame_id=f"{camera_id}_{int(time.time() * 1000)}",
        detections=detections,
        inference_time=0.05,
        timestamp=time.time(),
        frame_shape=(480, 640, 3)
    )


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test video frame"""
    # Create a simple test pattern
    frame = np.random.rand(height, width, 3).astype(np.float32)
    
    # Add some structure to make it more realistic
    frame[height//4:3*height//4, width//4:3*width//4] = 0.8
    
    return frame


def test_event_generation():
    """Test basic event generation functionality"""
    print("\n=== Testing Event Generation ===")
    
    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create configuration
        config = EdgeEventConfig()
        config.clip_config.clips_directory = str(temp_dir / "clips")
        config.buffer_config.buffer_directory = str(temp_dir / "buffer")
        
        # Create service
        service = EdgeEventService(config)
        
        # Track generated events
        generated_events = []
        extracted_clips = []
        
        def event_callback(event: SecurityEvent):
            generated_events.append(event)
            print(f"Generated event: {event.event_type.value} from {event.camera_id} (confidence: {event.confidence_score:.2f})")
        
        def clip_callback(clip):
            extracted_clips.append(clip)
            print(f"Extracted clip: {clip.clip_id} for event {clip.event_id}")
        
        service.add_event_callback(event_callback)
        service.add_clip_callback(clip_callback)
        
        # Start service
        assert service.start(), "Failed to start service"
        print("Service started successfully")
        
        # Add test camera
        camera_id = "test_camera_01"
        assert service.add_camera(camera_id), "Failed to add camera"
        print(f"Added camera: {camera_id}")
        
        # Process some test frames
        print("Processing test frames...")
        for i in range(10):
            frame = create_test_frame()
            service.process_frame(camera_id, frame, time.time())
            time.sleep(0.1)
        
        # Process some detection results
        print("Processing detection results...")
        for i in range(5):
            result = create_test_detection_result(camera_id)
            service.process_detection_result(result)
            time.sleep(1)  # Allow time for aggregation
        
        # Wait for processing
        time.sleep(3)
        
        # Check results
        print(f"Generated {len(generated_events)} events")
        print(f"Extracted {len(extracted_clips)} clips")
        
        # Get service status
        status = service.get_comprehensive_status()
        print(f"Service status: {status['service']}")
        
        # Stop service
        service.stop()
        print("Service stopped")
        
        return len(generated_events) > 0
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_local_buffering():
    """Test local buffering functionality"""
    print("\n=== Testing Local Buffering ===")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create configuration with buffering enabled
        config = EdgeEventConfig()
        config.enable_local_buffering = True
        config.buffer_config.buffer_directory = str(temp_dir / "buffer")
        
        # Create service
        service = EdgeEventService(config)
        
        # Mock sync callback that always fails (to test buffering)
        sync_attempts = []
        
        def mock_sync_callback(events: List[SecurityEvent], clips) -> bool:
            sync_attempts.append((len(events), len(clips)))
            print(f"Mock sync called with {len(events)} events, {len(clips)} clips")
            return False  # Always fail to test buffering
        
        service.set_sync_callback(mock_sync_callback)
        
        # Start service
        assert service.start(), "Failed to start service"
        
        # Generate some events
        camera_id = "test_camera_02"
        service.add_camera(camera_id)
        
        for i in range(3):
            result = create_test_detection_result(camera_id)
            service.process_detection_result(result)
            time.sleep(0.5)
        
        # Wait for processing
        time.sleep(2)
        
        # Check buffer status
        buffer_status = service.get_buffer_status()
        print(f"Buffer status: {buffer_status}")
        
        # Verify events were buffered
        assert buffer_status.get('pending_events', 0) > 0, "No events were buffered"
        print(f"Successfully buffered {buffer_status['pending_events']} events")
        
        service.stop()
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_event_deduplication():
    """Test event deduplication functionality"""
    print("\n=== Testing Event Deduplication ===")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = EdgeEventConfig()
        config.buffer_config.buffer_directory = str(temp_dir / "buffer")
        
        service = EdgeEventService(config)
        
        generated_events = []
        service.add_event_callback(lambda event: generated_events.append(event))
        
        assert service.start(), "Failed to start service"
        
        camera_id = "test_camera_03"
        service.add_camera(camera_id)
        
        # Generate multiple similar detection results (should be deduplicated)
        print("Generating similar detection results...")
        for i in range(10):
            result = create_test_detection_result(camera_id)
            service.process_detection_result(result)
            time.sleep(0.2)  # Short interval to trigger deduplication
        
        # Wait for processing
        time.sleep(3)
        
        # Check that fewer events were generated than detection results
        print(f"Generated {len(generated_events)} events from 10 similar detection results")
        
        # Get deduplication statistics
        event_stats = service.get_event_statistics()
        print(f"Event statistics: {event_stats}")
        
        dedup_rate = event_stats.get('deduplication_rate', 0)
        print(f"Deduplication rate: {dedup_rate:.2%}")
        
        service.stop()
        
        # Should have deduplicated some events
        return len(generated_events) < 10 and dedup_rate > 0
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """Run all tests"""
    print("Starting Event Generation Tests")
    print("=" * 50)
    
    tests = [
        ("Event Generation", test_event_generation),
        ("Local Buffering", test_local_buffering),
        ("Event Deduplication", test_event_deduplication)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name} test...")
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"{test_name}: FAILED - {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)