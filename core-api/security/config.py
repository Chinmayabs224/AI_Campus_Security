"""
Security configuration for campus security system.
"""
from typing import List, Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class SecuritySettings(BaseSettings):
    """Security-specific settings."""
    
    # Vault settings
    VAULT_URL: str = Field(default="http://localhost:8200", description="HashiCorp Vault URL")
    VAULT_TOKEN: Optional[str] = Field(default=None, description="Vault authentication token")
    VAULT_ROLE_ID: Optional[str] = Field(default=None, description="Vault AppRole role ID")
    VAULT_SECRET_ID: Optional[str] = Field(default=None, description="Vault AppRole secret ID")
    VAULT_MOUNT_POINT: str = Field(default="secret", description="Vault secrets mount point")
    VAULT_NAMESPACE: Optional[str] = Field(default=None, description="Vault namespace")
    
    # Network security settings
    TRUSTED_NETWORKS: List[str] = Field(
        default=["10.0.0.0/16", "192.168.0.0/16", "172.16.0.0/12"],
        description="Trusted network ranges"
    )
    BLOCKED_NETWORKS: List[str] = Field(
        default=[],
        description="Blocked network ranges"
    )
    
    # Security monitoring settings
    ENABLE_VULNERABILITY_SCANNING: bool = Field(default=True, description="Enable container vulnerability scanning")
    VULNERABILITY_SCAN_SCHEDULE: str = Field(default="0 2 * * *", description="Vulnerability scan cron schedule")
    MAX_CRITICAL_VULNERABILITIES: int = Field(default=0, description="Maximum allowed critical vulnerabilities")
    MAX_HIGH_VULNERABILITIES: int = Field(default=5, description="Maximum allowed high vulnerabilities")
    
    # Incident response settings
    ENABLE_AUTO_INCIDENT_RESPONSE: bool = Field(default=True, description="Enable automatic incident response")
    INCIDENT_NOTIFICATION_WEBHOOK: Optional[str] = Field(default=None, description="Webhook URL for incident notifications")
    SECURITY_TEAM_EMAIL: str = Field(default="security@campus.edu", description="Security team email")
    
    # Rate limiting settings (enhanced)
    RATE_LIMIT_BURST_MULTIPLIER: int = Field(default=2, description="Rate limit burst multiplier")
    RATE_LIMIT_BLOCK_DURATION: int = Field(default=300, description="Rate limit block duration in seconds")
    
    # Security headers settings
    ENABLE_HSTS: bool = Field(default=True, description="Enable HTTP Strict Transport Security")
    HSTS_MAX_AGE: int = Field(default=31536000, description="HSTS max age in seconds")
    ENABLE_CSP: bool = Field(default=True, description="Enable Content Security Policy")
    CSP_POLICY: str = Field(
        default="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        description="Content Security Policy"
    )
    
    # Authentication settings (enhanced)
    PASSWORD_MIN_LENGTH: int = Field(default=12, description="Minimum password length")
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(default=True, description="Require uppercase in passwords")
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(default=True, description="Require lowercase in passwords")
    PASSWORD_REQUIRE_NUMBERS: bool = Field(default=True, description="Require numbers in passwords")
    PASSWORD_REQUIRE_SYMBOLS: bool = Field(default=True, description="Require symbols in passwords")
    
    # Session security settings
    SESSION_TIMEOUT_MINUTES: int = Field(default=30, description="Session timeout in minutes")
    MAX_CONCURRENT_SESSIONS: int = Field(default=3, description="Maximum concurrent sessions per user")
    
    # API security settings
    API_KEY_LENGTH: int = Field(default=32, description="API key length")
    API_KEY_EXPIRY_DAYS: int = Field(default=90, description="API key expiry in days")
    
    # Audit settings
    AUDIT_LOG_RETENTION_DAYS: int = Field(default=2555, description="Audit log retention in days (7 years)")
    ENABLE_AUDIT_LOG_ENCRYPTION: bool = Field(default=True, description="Enable audit log encryption")
    
    # Compliance settings
    ENABLE_GDPR_COMPLIANCE: bool = Field(default=True, description="Enable GDPR compliance features")
    ENABLE_FERPA_COMPLIANCE: bool = Field(default=True, description="Enable FERPA compliance features")
    DATA_RETENTION_DAYS: int = Field(default=2555, description="Default data retention in days")
    
    # Security scanning settings
    ENABLE_DEPENDENCY_SCANNING: bool = Field(default=True, description="Enable dependency vulnerability scanning")
    ENABLE_CONTAINER_SCANNING: bool = Field(default=True, description="Enable container image scanning")
    ENABLE_SECRET_SCANNING: bool = Field(default=True, description="Enable secret scanning in code")
    
    # Firewall settings
    ENABLE_WAF: bool = Field(default=True, description="Enable Web Application Firewall")
    WAF_BLOCK_SUSPICIOUS_REQUESTS: bool = Field(default=True, description="Block suspicious requests")
    WAF_LOG_ALL_REQUESTS: bool = Field(default=False, description="Log all requests for analysis")
    
    # Encryption settings
    ENCRYPTION_ALGORITHM: str = Field(default="AES-256-GCM", description="Default encryption algorithm")
    KEY_ROTATION_DAYS: int = Field(default=90, description="Key rotation interval in days")
    
    class Config:
        env_file = ".env"
        env_prefix = "SECURITY_"
        case_sensitive = True


# Global security settings instance
security_settings = SecuritySettings()