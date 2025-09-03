#!/usr/bin/env python3
"""
Test script for RTSP Stream Processing Service

This script tests the RTSP stream processing functionality with mock cameras
and validates the health monitoring and reconnection logic.
"""

import time
import logging
import numpy as np
from typing import Any

from rtsp_client import CameraConfig, RTSPStreamManager
from config import setup_logging


def mock_frame_callback(camera_id: str, frame: np.ndarray):
    """Mock callback for processing frames"""
    print(f"Received frame from {camera_id}: shape={frame.shape}, dtype={frame.dtype}")


def test_rtsp_stream_processing():
    """Test RTSP stream processing with mock configuration"""
    
    # Setup logging
    setup_logging("DEBUG")
    logger = logging.getLogger("test_rtsp")
    
    logger.info("Starting RTSP Stream Processing Test")
    
    # Create test camera configurations
    # Note: These are example RTSP URLs - replace with actual camera URLs for real testing
    test_cameras = [
        CameraConfig(
            camera_id="test_camera_1",
            rtsp_url="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",  # Public test stream
            fps=15,
            width=640,
            height=480,
            reconnect_interval=3,
            max_reconnect_attempts=5,
            health_check_interval=10
        ),
        CameraConfig(
            camera_id="test_camera_2", 
            rtsp_url="rtsp://invalid.url.test/stream",  # Invalid URL to test error handling
            fps=30,
            width=640,
            height=480,
            reconnect_interval=2,
            max_reconnect_attempts=3,
            health_check_interval=5
        )
    ]
    
    # Create stream manager
    stream_manager = RTSPStreamManager()
    
    try:
        # Add cameras
        for camera_config in test_cameras:
            success = stream_manager.add_camera(camera_config, mock_frame_callback)
            logger.info(f"Added camera {camera_config.camera_id}: {success}")
        
        # Start all cameras
        logger.info("Starting camera streams...")
        results = stream_manager.start_all_cameras()
        
        for camera_id, success in results.items():
            logger.info(f"Camera {camera_id} start result: {success}")
        
        # Monitor for 30 seconds
        logger.info("Monitoring streams for 30 seconds...")
        for i in range(30):
            time.sleep(1)
            
            # Print health status every 5 seconds
            if i % 5 == 0:
                health_status = stream_manager.get_health_status()
                logger.info(f"Health check at {i}s:")
                
                for camera_id, health in health_status.items():
                    logger.info(f"  {camera_id}: {health.status.value} "
                              f"(frames: {health.frames_received}, "
                              f"reconnects: {health.reconnect_attempts})")
                    
                    if health.error_message:
                        logger.warning(f"    Error: {health.error_message}")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Stopping all camera streams...")
        stream_manager.stop_all_cameras()
        logger.info("Test cleanup completed")


def test_camera_config():
    """Test camera configuration functionality"""
    
    logger = logging.getLogger("test_config")
    logger.info("Testing camera configuration...")
    
    # Test camera config creation
    config = CameraConfig(
        camera_id="test_cam",
        rtsp_url="rtsp://test.example.com/stream",
        fps=25,
        width=1280,
        height=720
    )
    
    logger.info(f"Created config: {config}")
    
    # Test validation
    assert config.camera_id == "test_cam"
    assert config.fps == 25
    assert config.width == 1280
    assert config.height == 720
    
    logger.info("Camera configuration test passed")


if __name__ == "__main__":
    print("RTSP Stream Processing Test")
    print("=" * 40)
    
    try:
        # Test configuration
        test_camera_config()
        
        # Test stream processing
        test_rtsp_stream_processing()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()