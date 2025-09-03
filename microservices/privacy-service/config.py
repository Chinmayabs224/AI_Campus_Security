"""
Configuration settings for Privacy Service.
"""
import os
from datetime import timedelta


class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 500 * 1024 * 1024))  # 500MB
    
    # Redis settings
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 1))
    
    # Privacy zone settings
    PRIVACY_ZONE_TTL = int(os.environ.get('PRIVACY_ZONE_TTL', 86400))  # 24 hours
    
    # Face detection settings
    FACE_DETECTION_CONFIDENCE = float(os.environ.get('FACE_DETECTION_CONFIDENCE', 0.7))
    FACE_DETECTION_DEVICE = os.environ.get('FACE_DETECTION_DEVICE', 'cpu')  # 'cpu' or 'cuda'
    
    # Video processing settings
    VIDEO_FRAME_SKIP = int(os.environ.get('VIDEO_FRAME_SKIP', 1))
    VIDEO_MAX_RESOLUTION = os.environ.get('VIDEO_MAX_RESOLUTION', '1920x1080')
    
    # DSAR settings
    DSAR_RETENTION_DAYS = int(os.environ.get('DSAR_RETENTION_DAYS', 30))
    DSAR_PROCESSING_TIMEOUT = int(os.environ.get('DSAR_PROCESSING_TIMEOUT', 3600))  # 1 hour
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # File upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/tmp/privacy-service')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
    
    # Model settings
    FACENET_MODEL_PATH = os.environ.get('FACENET_MODEL_PATH', 'models/facenet.pth')
    FACE_DETECTION_BATCH_SIZE = int(os.environ.get('FACE_DETECTION_BATCH_SIZE', 32))
    
    # Performance settings
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 4))
    PROCESSING_TIMEOUT = int(os.environ.get('PROCESSING_TIMEOUT', 300))  # 5 minutes


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    
    # Use in-memory Redis for testing
    REDIS_HOST = 'localhost'
    REDIS_DB = 15  # Use different DB for testing


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}