"""
Configuration management for RTSP Stream Processor
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .rtsp_client import CameraConfig


@dataclass
class StreamProcessorConfig:
    """Configuration for the stream processor service"""
    log_level: str = "INFO"
    max_concurrent_streams: int = 10
    frame_buffer_size: int = 10
    health_check_interval: int = 30
    metrics_port: int = 8080
    config_file_path: str = "/etc/edge-security/cameras.json"


class ConfigManager:
    """Manages configuration for camera streams and service settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("config_manager")
        self.config_path = config_path or os.getenv("CAMERA_CONFIG_PATH", "/etc/edge-security/cameras.json")
        self.service_config = StreamProcessorConfig()
        self.camera_configs: Dict[str, CameraConfig] = {}
        
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.warning(f"Config file not found: {self.config_path}")
                self._create_default_config()
                return True
                
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Load service configuration
            if 'service' in config_data:
                service_data = config_data['service']
                self.service_config = StreamProcessorConfig(**service_data)
                
            # Load camera configurations
            if 'cameras' in config_data:
                self.camera_configs = {}
                for camera_data in config_data['cameras']:
                    camera_config = CameraConfig(**camera_data)
                    self.camera_configs[camera_config.camera_id] = camera_config
                    
            self.logger.info(f"Loaded configuration for {len(self.camera_configs)} cameras")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
            
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_data = {
                'service': asdict(self.service_config),
                'cameras': [asdict(config) for config in self.camera_configs.values()]
            }
            
            # Ensure directory exists
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            self.logger.info(f"Saved configuration to {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
            
    def add_camera(self, camera_config: CameraConfig) -> bool:
        """Add a new camera configuration"""
        if camera_config.camera_id in self.camera_configs:
            self.logger.warning(f"Camera {camera_config.camera_id} already exists")
            return False
            
        self.camera_configs[camera_config.camera_id] = camera_config
        self.logger.info(f"Added camera configuration: {camera_config.camera_id}")
        return True
        
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera configuration"""
        if camera_id not in self.camera_configs:
            self.logger.warning(f"Camera {camera_id} not found")
            return False
            
        del self.camera_configs[camera_id]
        self.logger.info(f"Removed camera configuration: {camera_id}")
        return True
        
    def get_camera_config(self, camera_id: str) -> Optional[CameraConfig]:
        """Get configuration for a specific camera"""
        return self.camera_configs.get(camera_id)
        
    def get_all_camera_configs(self) -> List[CameraConfig]:
        """Get all camera configurations"""
        return list(self.camera_configs.values())
        
    def update_camera_config(self, camera_id: str, **kwargs) -> bool:
        """Update configuration for a specific camera"""
        if camera_id not in self.camera_configs:
            self.logger.warning(f"Camera {camera_id} not found")
            return False
            
        try:
            current_config = self.camera_configs[camera_id]
            updated_data = asdict(current_config)
            updated_data.update(kwargs)
            
            self.camera_configs[camera_id] = CameraConfig(**updated_data)
            self.logger.info(f"Updated camera configuration: {camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update camera {camera_id}: {e}")
            return False
            
    def _create_default_config(self):
        """Create a default configuration file"""
        try:
            default_config = {
                'service': asdict(self.service_config),
                'cameras': [
                    {
                        'camera_id': 'camera_001',
                        'rtsp_url': 'rtsp://admin:password@192.168.1.100:554/stream1',
                        'fps': 30,
                        'width': 640,
                        'height': 480,
                        'reconnect_interval': 5,
                        'max_reconnect_attempts': 10,
                        'health_check_interval': 30
                    }
                ]
            }
            
            # Ensure directory exists
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                
            self.logger.info(f"Created default configuration at {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create default configuration: {e}")


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/var/log/edge-security/stream-processor.log', mode='a')
        ]
    )