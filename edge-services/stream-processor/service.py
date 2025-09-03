"""
Main RTSP Stream Processing Service

This module provides the main service interface for RTSP stream processing
with health monitoring, metrics, and management APIs.
"""

import asyncio
import signal
import sys
import time
import logging
from typing import Dict, Any, Optional, Callable
from threading import Event
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from .rtsp_client import RTSPStreamManager, CameraConfig, StreamHealth
from .config import ConfigManager, setup_logging


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks and metrics"""
    
    def __init__(self, stream_manager: RTSPStreamManager, *args, **kwargs):
        self.stream_manager = stream_manager
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for health and metrics"""
        if self.path == '/health':
            self._handle_health()
        elif self.path == '/metrics':
            self._handle_metrics()
        elif self.path.startswith('/camera/'):
            self._handle_camera_status()
        else:
            self.send_error(404, "Not Found")
    
    def _handle_health(self):
        """Return overall service health"""
        try:
            health_data = self.stream_manager.get_health_status()
            
            # Determine overall health
            overall_status = "healthy"
            for camera_id, health in health_data.items():
                if health.status.value in ['error', 'disconnected']:
                    overall_status = "unhealthy"
                    break
            
            response = {
                "status": overall_status,
                "timestamp": time.time(),
                "cameras": len(health_data),
                "active_streams": sum(1 for h in health_data.values() if h.status.value == "connected")
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")
    
    def _handle_metrics(self):
        """Return detailed metrics for all cameras"""
        try:
            health_data = self.stream_manager.get_health_status()
            
            metrics = {
                "timestamp": time.time(),
                "cameras": {}
            }
            
            for camera_id, health in health_data.items():
                metrics["cameras"][camera_id] = {
                    "status": health.status.value,
                    "frames_received": health.frames_received,
                    "last_frame_time": health.last_frame_time,
                    "reconnect_attempts": health.reconnect_attempts,
                    "error_message": health.error_message
                }
            
            self._send_json_response(metrics)
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")
    
    def _handle_camera_status(self):
        """Return status for a specific camera"""
        try:
            # Extract camera ID from path
            camera_id = self.path.split('/')[-1]
            health_data = self.stream_manager.get_health_status(camera_id)
            
            if not health_data:
                self.send_error(404, f"Camera {camera_id} not found")
                return
            
            health = health_data[camera_id]
            response = {
                "camera_id": camera_id,
                "status": health.status.value,
                "frames_received": health.frames_received,
                "last_frame_time": health.last_frame_time,
                "reconnect_attempts": health.reconnect_attempts,
                "error_message": health.error_message,
                "timestamp": time.time()
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")
    
    def _send_json_response(self, data: Dict[str, Any]):
        """Send JSON response"""
        response_data = json.dumps(data, indent=2)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_data)))
        self.end_headers()
        self.wfile.write(response_data.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        pass  # Suppress default HTTP logging


class StreamProcessorService:
    """
    Main RTSP Stream Processing Service
    
    Manages multiple camera streams with health monitoring and HTTP API
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("stream_service")
        self.config_manager = ConfigManager(config_path)
        self.stream_manager = RTSPStreamManager()
        
        # Service state
        self._running = False
        self._stop_event = Event()
        self._http_server: Optional[HTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        
        # Frame processing callback
        self.frame_callback: Optional[Callable[[str, Any], None]] = None
        
    def set_frame_callback(self, callback: Callable[[str, Any], None]):
        """Set the callback function for processing frames"""
        self.frame_callback = callback
        
    def start(self) -> bool:
        """Start the stream processing service"""
        try:
            self.logger.info("Starting RTSP Stream Processing Service")
            
            # Load configuration
            if not self.config_manager.load_config():
                self.logger.error("Failed to load configuration")
                return False
            
            # Setup logging
            setup_logging(self.config_manager.service_config.log_level)
            
            # Add cameras to stream manager
            for camera_config in self.config_manager.get_all_camera_configs():
                if not self.stream_manager.add_camera(camera_config, self._default_frame_callback):
                    self.logger.error(f"Failed to add camera {camera_config.camera_id}")
                    
            # Start HTTP server for health checks
            self._start_http_server()
            
            # Start all camera streams
            results = self.stream_manager.start_all_cameras()
            
            # Log results
            successful = sum(1 for success in results.values() if success)
            self.logger.info(f"Started {successful}/{len(results)} camera streams")
            
            self._running = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            return False
    
    def stop(self):
        """Stop the stream processing service"""
        if not self._running:
            return
            
        self.logger.info("Stopping RTSP Stream Processing Service")
        self._running = False
        self._stop_event.set()
        
        # Stop all camera streams
        self.stream_manager.stop_all_cameras()
        
        # Stop HTTP server
        self._stop_http_server()
        
        self.logger.info("Service stopped")
    
    def run(self):
        """Run the service (blocking)"""
        if not self.start():
            return False
            
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info("Service running. Press Ctrl+C to stop.")
            
            # Main service loop
            while self._running and not self._stop_event.is_set():
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.stop()
            
        return True
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}")
        self.stop()
    
    def _start_http_server(self):
        """Start HTTP server for health checks and metrics"""
        try:
            port = self.config_manager.service_config.metrics_port
            
            # Create handler with stream manager reference
            def handler_factory(*args, **kwargs):
                return HealthHandler(self.stream_manager, *args, **kwargs)
            
            self._http_server = HTTPServer(('0.0.0.0', port), handler_factory)
            self._http_thread = threading.Thread(
                target=self._http_server.serve_forever,
                daemon=True
            )
            self._http_thread.start()
            
            self.logger.info(f"HTTP server started on port {port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start HTTP server: {e}")
    
    def _stop_http_server(self):
        """Stop HTTP server"""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server.server_close()
            
        if self._http_thread and self._http_thread.is_alive():
            self._http_thread.join(timeout=5)
    
    def _default_frame_callback(self, camera_id: str, frame: Any):
        """Default frame processing callback"""
        if self.frame_callback:
            self.frame_callback(camera_id, frame)
        else:
            # Default behavior - just log frame reception
            self.logger.debug(f"Received frame from camera {camera_id}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        health_data = self.stream_manager.get_health_status()
        
        return {
            "service_running": self._running,
            "total_cameras": len(health_data),
            "active_streams": sum(1 for h in health_data.values() if h.status.value == "connected"),
            "cameras": {cid: {
                "status": health.status.value,
                "frames_received": health.frames_received,
                "last_frame_time": health.last_frame_time,
                "error_message": health.error_message
            } for cid, health in health_data.items()},
            "timestamp": time.time()
        }


def main():
    """Main entry point for the service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RTSP Stream Processing Service')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Setup basic logging
    setup_logging(args.log_level)
    
    # Create and run service
    service = StreamProcessorService(args.config)
    
    try:
        success = service.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()