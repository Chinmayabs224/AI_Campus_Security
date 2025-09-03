"""
Configuration settings for the campus security API.
"""
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    ENVIRONMENT: str = Field(default="development", description="Environment (development, staging, production)")
    HOST: str = Field(default="0.0.0.0", description="Host to bind the server")
    PORT: int = Field(default=8000, description="Port to bind the server")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Security settings
    SECRET_KEY: str = Field(..., description="Secret key for JWT tokens")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration in minutes")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration in days")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    ALLOWED_HOSTS: Optional[List[str]] = Field(default=None, description="Allowed hosts")
    
    # Database settings
    DATABASE_URL: str = Field(..., description="PostgreSQL database URL")
    DATABASE_POOL_SIZE: int = Field(default=20, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, description="Database max overflow connections")
    DATABASE_POOL_TIMEOUT: int = Field(default=30, description="Database pool timeout in seconds")
    
    # Redis settings
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    REDIS_POOL_SIZE: int = Field(default=10, description="Redis connection pool size")
    REDIS_TIMEOUT: int = Field(default=5, description="Redis operation timeout in seconds")
    
    # Rate limiting settings
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per window")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limit window in seconds")
    
    # MinIO/S3 settings
    MINIO_ENDPOINT: str = Field(default="localhost:9000", description="MinIO endpoint")
    MINIO_ACCESS_KEY: str = Field(..., description="MinIO access key")
    MINIO_SECRET_KEY: str = Field(..., description="MinIO secret key")
    MINIO_SECURE: bool = Field(default=False, description="Use HTTPS for MinIO")
    EVIDENCE_BUCKET: str = Field(default="evidence", description="Evidence storage bucket")
    
    # SSO/SAML settings
    SAML_ENTITY_ID: Optional[str] = Field(default=None, description="SAML entity ID")
    SAML_SSO_URL: Optional[str] = Field(default=None, description="SAML SSO URL")
    SAML_X509_CERT: Optional[str] = Field(default=None, description="SAML X.509 certificate")
    
    # Notification settings
    NOTIFICATION_SERVICE_URL: str = Field(
        default="http://localhost:8001",
        description="Notification service URL"
    )
    
    # Edge device settings
    EDGE_DEVICE_TOKEN_EXPIRE_HOURS: int = Field(
        default=24,
        description="Edge device token expiration in hours"
    )
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting."""
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level setting."""
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()