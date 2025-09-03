"""
Security hardening utilities for campus security system.
"""
import os
import subprocess
import json
from typing import List, Dict, Any
import structlog

from .config import security_settings
from .network_policies import network_policies

logger = structlog.get_logger()


class SecurityHardening:
    """Security hardening utilities and configurations."""
    
    def __init__(self):
        self.hardening_checks = []
        self.applied_configurations = []
    
    async def apply_system_hardening(self) -> Dict[str, Any]:
        """Apply comprehensive system hardening."""
        results = {
            "success": True,
            "applied_configurations": [],
            "failed_configurations": [],
            "warnings": []
        }
        
        hardening_tasks = [
            ("Network Security", self._harden_network_security),
            ("File Permissions", self._harden_file_permissions),
            ("Service Configuration", self._harden_service_configuration),
            ("Logging Configuration", self._harden_logging_configuration),
            ("Container Security", self._harden_container_security),
            ("Database Security", self._harden_database_security),
            ("API Security", self._harden_api_security)
        ]
        
        for task_name, task_func in hardening_tasks:
            try:
                logger.info("Applying security hardening", task=task_name)
                task_result = await task_func()
                
                if task_result.get("success", True):
                    results["applied_configurations"].append({
                        "task": task_name,
                        "details": task_result.get("details", [])
                    })
                    logger.info("Security hardening completed", task=task_name)
                else:
                    results["failed_configurations"].append({
                        "task": task_name,
                        "error": task_result.get("error", "Unknown error")
                    })
                    logger.error("Security hardening failed", task=task_name, 
                               error=task_result.get("error"))
                
                if task_result.get("warnings"):
                    results["warnings"].extend(task_result["warnings"])
                    
            except Exception as e:
                results["failed_configurations"].append({
                    "task": task_name,
                    "error": str(e)
                })
                logger.error("Security hardening task failed", task=task_name, error=str(e))
                results["success"] = False
        
        return results
    
    async def _harden_network_security(self) -> Dict[str, Any]:
        """Apply network security hardening."""
        details = []
        
        # Initialize default firewall rules
        network_policies.initialize_default_rules()
        details.append("Default firewall rules initialized")
        
        # Add trusted internal networks
        internal_networks = [
            "10.0.0.0/8",
            "172.16.0.0/12", 
            "192.168.0.0/16"
        ]
        
        for network in internal_networks:
            if network_policies.add_trusted_network(network):
                details.append(f"Added trusted network: {network}")
        
        # Block known malicious networks
        malicious_networks = [
            "0.0.0.0/8",      # Invalid source addresses
            "127.0.0.0/8",    # Loopback (should not come from external)
            "169.254.0.0/16", # Link-local addresses
            "224.0.0.0/4",    # Multicast addresses
            "240.0.0.0/4"     # Reserved addresses
        ]
        
        for network in malicious_networks:
            if network_policies.add_blocked_network(network):
                details.append(f"Blocked malicious network: {network}")
        
        # Configure rate limiting
        rate_limit_rules = [
            ("0.0.0.0/0", 100, 200),      # Global default
            ("10.0.0.0/8", 500, 1000),    # Internal networks
            ("192.168.0.0/16", 500, 1000), # Private networks
            ("172.16.0.0/12", 500, 1000)   # Private networks
        ]
        
        for network, rpm, burst in rate_limit_rules:
            network_policies.add_rate_limit_rule(network, rpm, burst)
            details.append(f"Rate limit configured for {network}: {rpm} req/min")
        
        return {"success": True, "details": details}
    
    async def _harden_file_permissions(self) -> Dict[str, Any]:
        """Apply file permission hardening."""
        details = []
        warnings = []
        
        # Critical files and their required permissions
        critical_files = {
            ".env": "600",
            "requirements.txt": "644",
            "main.py": "644",
            "security/": "755"
        }
        
        for file_path, required_perm in critical_files.items():
            try:
                if os.path.exists(file_path):
                    # Set secure permissions
                    os.chmod(file_path, int(required_perm, 8))
                    details.append(f"Set permissions {required_perm} on {file_path}")
                else:
                    warnings.append(f"File not found: {file_path}")
            except Exception as e:
                warnings.append(f"Failed to set permissions on {file_path}: {str(e)}")
        
        # Ensure log directory has proper permissions
        log_dirs = ["logs", "/var/log/campus-security"]
        for log_dir in log_dirs:
            try:
                if os.path.exists(log_dir):
                    os.chmod(log_dir, 0o750)
                    details.append(f"Set permissions 750 on {log_dir}")
            except Exception as e:
                warnings.append(f"Failed to set permissions on {log_dir}: {str(e)}")
        
        return {"success": True, "details": details, "warnings": warnings}
    
    async def _harden_service_configuration(self) -> Dict[str, Any]:
        """Apply service configuration hardening."""
        details = []
        warnings = []
        
        # Disable unnecessary services (if running on Linux)
        try:
            if os.name == 'posix':
                unnecessary_services = [
                    "telnet", "ftp", "rsh", "rlogin", "rexec",
                    "finger", "chargen", "daytime", "echo", "discard"
                ]
                
                for service in unnecessary_services:
                    try:
                        result = subprocess.run(
                            ["systemctl", "is-enabled", service],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0 and "enabled" in result.stdout:
                            subprocess.run(["systemctl", "disable", service], check=True)
                            details.append(f"Disabled unnecessary service: {service}")
                    except subprocess.CalledProcessError:
                        # Service doesn't exist or already disabled
                        pass
                    except Exception as e:
                        warnings.append(f"Failed to check/disable service {service}: {str(e)}")
        except Exception as e:
            warnings.append(f"Service hardening failed: {str(e)}")
        
        return {"success": True, "details": details, "warnings": warnings}
    
    async def _harden_logging_configuration(self) -> Dict[str, Any]:
        """Apply logging configuration hardening."""
        details = []
        
        # Ensure secure logging configuration
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "security": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
                }
            },
            "handlers": {
                "security_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "logs/security.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10,
                    "formatter": "security",
                    "level": "INFO"
                },
                "audit_file": {
                    "class": "logging.handlers.RotatingFileHandler", 
                    "filename": "logs/audit.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 50,  # Keep more audit logs
                    "formatter": "security",
                    "level": "INFO"
                }
            },
            "loggers": {
                "security": {
                    "handlers": ["security_file"],
                    "level": "INFO",
                    "propagate": False
                },
                "audit": {
                    "handlers": ["audit_file"],
                    "level": "INFO", 
                    "propagate": False
                }
            }
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", mode=0o750, exist_ok=True)
        details.append("Created secure logs directory")
        
        # Save logging configuration
        with open("logging_config.json", "w") as f:
            json.dump(logging_config, f, indent=2)
        details.append("Secure logging configuration saved")
        
        return {"success": True, "details": details}
    
    async def _harden_container_security(self) -> Dict[str, Any]:
        """Apply container security hardening."""
        details = []
        warnings = []
        
        # Create secure Dockerfile recommendations
        dockerfile_security_rules = [
            "# Security hardening for container",
            "USER 1000:1000  # Run as non-root user",
            "RUN apt-get update && apt-get upgrade -y && apt-get clean",
            "RUN rm -rf /var/lib/apt/lists/*",
            "COPY --chown=1000:1000 . /app",
            "RUN chmod -R 755 /app",
            "RUN find /app -name '*.py' -exec chmod 644 {} \\;",
            "HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\",
            "  CMD curl -f http://localhost:8000/health || exit 1"
        ]
        
        # Write security recommendations to file
        with open("Dockerfile.security", "w") as f:
            f.write("\n".join(dockerfile_security_rules))
        details.append("Container security recommendations saved to Dockerfile.security")
        
        # Create .dockerignore for security
        dockerignore_content = [
            ".env",
            ".env.*",
            "*.log",
            "logs/",
            ".git/",
            ".pytest_cache/",
            "__pycache__/",
            "*.pyc",
            ".coverage",
            "node_modules/",
            ".DS_Store",
            "Thumbs.db",
            "*.tmp",
            "*.swp",
            "*.swo"
        ]
        
        with open(".dockerignore", "w") as f:
            f.write("\n".join(dockerignore_content))
        details.append("Secure .dockerignore created")
        
        return {"success": True, "details": details, "warnings": warnings}
    
    async def _harden_database_security(self) -> Dict[str, Any]:
        """Apply database security hardening."""
        details = []
        warnings = []
        
        # Database security configuration
        db_security_config = {
            "postgresql": {
                "ssl_mode": "require",
                "log_connections": True,
                "log_disconnections": True,
                "log_statement": "all",
                "log_duration": True,
                "shared_preload_libraries": "pg_stat_statements",
                "max_connections": 100,
                "password_encryption": "scram-sha-256"
            },
            "connection_security": {
                "pool_size": 20,
                "max_overflow": 30,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "echo": False  # Don't log SQL in production
            }
        }
        
        # Save database security configuration
        with open("database_security_config.json", "w") as f:
            json.dump(db_security_config, f, indent=2)
        details.append("Database security configuration saved")
        
        # Create database backup script
        backup_script = """#!/bin/bash
# Secure database backup script
set -euo pipefail

BACKUP_DIR="/secure/backups/database"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="campus_security"

# Create backup directory with secure permissions
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

# Create encrypted backup
pg_dump "$DB_NAME" | gpg --cipher-algo AES256 --compress-algo 1 --symmetric --output "$BACKUP_DIR/backup_$DATE.sql.gpg"

# Set secure permissions on backup
chmod 600 "$BACKUP_DIR/backup_$DATE.sql.gpg"

# Remove backups older than 30 days
find "$BACKUP_DIR" -name "backup_*.sql.gpg" -mtime +30 -delete

echo "Secure backup completed: backup_$DATE.sql.gpg"
"""
        
        with open("secure_backup.sh", "w") as f:
            f.write(backup_script)
        os.chmod("secure_backup.sh", 0o750)
        details.append("Secure database backup script created")
        
        return {"success": True, "details": details, "warnings": warnings}
    
    async def _harden_api_security(self) -> Dict[str, Any]:
        """Apply API security hardening."""
        details = []
        
        # API security headers configuration
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=(), usb=()",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "X-Permitted-Cross-Domain-Policies": "none",
            "X-Download-Options": "noopen",
            "X-DNS-Prefetch-Control": "off"
        }
        
        # Save security headers configuration
        with open("api_security_headers.json", "w") as f:
            json.dump(security_headers, f, indent=2)
        details.append("API security headers configuration saved")
        
        # Create rate limiting configuration
        rate_limiting_config = {
            "global_rate_limit": {
                "requests_per_minute": 100,
                "burst_size": 200,
                "block_duration_seconds": 300
            },
            "endpoint_specific_limits": {
                "/api/v1/auth/login": {
                    "requests_per_minute": 5,
                    "burst_size": 10,
                    "block_duration_seconds": 900
                },
                "/api/v1/auth/register": {
                    "requests_per_minute": 3,
                    "burst_size": 5,
                    "block_duration_seconds": 1800
                },
                "/api/v1/incidents": {
                    "requests_per_minute": 50,
                    "burst_size": 100
                }
            },
            "ip_whitelist": [
                "127.0.0.1",
                "::1"
            ]
        }
        
        with open("rate_limiting_config.json", "w") as f:
            json.dump(rate_limiting_config, f, indent=2)
        details.append("Rate limiting configuration saved")
        
        return {"success": True, "details": details}
    
    def generate_security_checklist(self) -> List[Dict[str, Any]]:
        """Generate security hardening checklist."""
        checklist = [
            {
                "category": "Network Security",
                "items": [
                    {"task": "Configure firewall rules", "status": "automated"},
                    {"task": "Set up network segmentation", "status": "automated"},
                    {"task": "Enable DDoS protection", "status": "manual"},
                    {"task": "Configure VPN access", "status": "manual"}
                ]
            },
            {
                "category": "Application Security", 
                "items": [
                    {"task": "Enable security headers", "status": "automated"},
                    {"task": "Configure rate limiting", "status": "automated"},
                    {"task": "Set up input validation", "status": "implemented"},
                    {"task": "Enable HTTPS/TLS", "status": "manual"}
                ]
            },
            {
                "category": "Data Security",
                "items": [
                    {"task": "Encrypt data at rest", "status": "manual"},
                    {"task": "Encrypt data in transit", "status": "manual"},
                    {"task": "Set up secure backups", "status": "automated"},
                    {"task": "Configure data retention", "status": "manual"}
                ]
            },
            {
                "category": "Access Control",
                "items": [
                    {"task": "Implement multi-factor authentication", "status": "manual"},
                    {"task": "Set up role-based access control", "status": "implemented"},
                    {"task": "Configure session management", "status": "implemented"},
                    {"task": "Enable audit logging", "status": "implemented"}
                ]
            },
            {
                "category": "Monitoring & Incident Response",
                "items": [
                    {"task": "Set up security monitoring", "status": "implemented"},
                    {"task": "Configure incident response", "status": "implemented"},
                    {"task": "Enable vulnerability scanning", "status": "implemented"},
                    {"task": "Set up alerting", "status": "implemented"}
                ]
            }
        ]
        
        return checklist
    
    def export_hardening_report(self, output_file: str = "security_hardening_report.json"):
        """Export security hardening report."""
        report = {
            "generated_at": "2024-01-01T00:00:00Z",
            "security_checklist": self.generate_security_checklist(),
            "applied_configurations": self.applied_configurations,
            "recommendations": [
                "Enable HTTPS/TLS encryption for all endpoints",
                "Set up regular security audits and penetration testing",
                "Implement automated security scanning in CI/CD pipeline",
                "Configure centralized logging and SIEM integration",
                "Set up regular backup testing and disaster recovery procedures",
                "Enable multi-factor authentication for all admin accounts",
                "Implement network segmentation and micro-segmentation",
                "Set up regular vulnerability assessments",
                "Configure automated patch management",
                "Implement data loss prevention (DLP) controls"
            ],
            "compliance_frameworks": [
                "NIST Cybersecurity Framework",
                "ISO 27001",
                "GDPR (Data Protection)",
                "FERPA (Educational Records)",
                "SOC 2 Type II"
            ]
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("Security hardening report exported", output_file=output_file)


# Global security hardening instance
security_hardening = SecurityHardening()