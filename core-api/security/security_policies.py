"""
Comprehensive security policies and enforcement for campus security system.
"""
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class PolicyViolationSeverity(Enum):
    """Security policy violation severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PolicyViolation:
    """Security policy violation."""
    policy_name: str
    description: str
    severity: PolicyViolationSeverity
    details: Dict[str, Any]
    remediation: str


class SecurityPolicyEngine:
    """Security policy engine for enforcing security policies."""
    
    def __init__(self):
        self.policies = {}
        self.violations = []
        self.initialize_default_policies()
    
    def initialize_default_policies(self):
        """Initialize default security policies."""
        
        # Password policy
        self.policies["password_policy"] = {
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_symbols": True,
            "max_age_days": 90,
            "history_count": 12,
            "lockout_attempts": 5,
            "lockout_duration_minutes": 30
        }
        
        # Session policy
        self.policies["session_policy"] = {
            "max_idle_minutes": 30,
            "max_session_duration_hours": 8,
            "max_concurrent_sessions": 3,
            "require_reauth_for_sensitive": True,
            "secure_cookie_flags": True
        }
        
        # API security policy
        self.policies["api_security_policy"] = {
            "rate_limit_per_minute": 100,
            "max_request_size_mb": 10,
            "allowed_content_types": ["application/json", "multipart/form-data"],
            "require_authentication": True,
            "require_https": True,
            "enable_cors": False,
            "max_upload_size_mb": 50
        }
        
        # Data classification policy
        self.policies["data_classification_policy"] = {
            "public": {
                "encryption_required": False,
                "access_logging": False,
                "retention_days": 365
            },
            "internal": {
                "encryption_required": True,
                "access_logging": True,
                "retention_days": 2555  # 7 years
            },
            "confidential": {
                "encryption_required": True,
                "access_logging": True,
                "retention_days": 2555,
                "require_approval": True
            },
            "restricted": {
                "encryption_required": True,
                "access_logging": True,
                "retention_days": 2555,
                "require_approval": True,
                "require_mfa": True
            }
        }
        
        # Network access policy
        self.policies["network_access_policy"] = {
            "allowed_protocols": ["https", "ssh", "postgresql"],
            "blocked_ports": [23, 21, 135, 139, 445, 1433, 3389],
            "require_vpn_for_admin": True,
            "geo_blocking_enabled": True,
            "blocked_countries": ["CN", "RU", "KP", "IR"],
            "intrusion_detection": True
        }
        
        # Audit policy
        self.policies["audit_policy"] = {
            "log_all_access": True,
            "log_failed_attempts": True,
            "log_admin_actions": True,
            "log_data_access": True,
            "retention_years": 7,
            "real_time_monitoring": True,
            "alert_on_anomalies": True
        }
        
        # Compliance policy
        self.policies["compliance_policy"] = {
            "gdpr_compliance": True,
            "ferpa_compliance": True,
            "data_subject_rights": True,
            "privacy_by_design": True,
            "consent_management": True,
            "breach_notification_hours": 72
        }
        
        logger.info("Default security policies initialized")
    
    def validate_password(self, password: str, username: str = None) -> Tuple[bool, List[str]]:
        """Validate password against security policy."""
        policy = self.policies["password_policy"]
        violations = []
        
        # Check minimum length
        if len(password) < policy["min_length"]:
            violations.append(f"Password must be at least {policy['min_length']} characters long")
        
        # Check character requirements
        if policy["require_uppercase"] and not re.search(r'[A-Z]', password):
            violations.append("Password must contain at least one uppercase letter")
        
        if policy["require_lowercase"] and not re.search(r'[a-z]', password):
            violations.append("Password must contain at least one lowercase letter")
        
        if policy["require_numbers"] and not re.search(r'\d', password):
            violations.append("Password must contain at least one number")
        
        if policy["require_symbols"] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            violations.append("Password must contain at least one special character")
        
        # Check for common patterns
        if username and username.lower() in password.lower():
            violations.append("Password must not contain the username")
        
        # Check for common weak passwords
        weak_patterns = [
            r'password', r'123456', r'qwerty', r'admin', r'letmein',
            r'welcome', r'monkey', r'dragon', r'master', r'shadow'
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                violations.append("Password contains common weak patterns")
                break
        
        return len(violations) == 0, violations
    
    def validate_api_request(self, request_data: Dict[str, Any]) -> Tuple[bool, List[PolicyViolation]]:
        """Validate API request against security policies."""
        policy = self.policies["api_security_policy"]
        violations = []
        
        # Check content type
        content_type = request_data.get("content_type", "")
        if content_type and content_type not in policy["allowed_content_types"]:
            violations.append(PolicyViolation(
                policy_name="api_security_policy",
                description="Invalid content type",
                severity=PolicyViolationSeverity.MEDIUM,
                details={"content_type": content_type, "allowed": policy["allowed_content_types"]},
                remediation="Use allowed content types only"
            ))
        
        # Check request size
        request_size = request_data.get("size_mb", 0)
        if request_size > policy["max_request_size_mb"]:
            violations.append(PolicyViolation(
                policy_name="api_security_policy",
                description="Request size exceeds limit",
                severity=PolicyViolationSeverity.HIGH,
                details={"size_mb": request_size, "max_allowed": policy["max_request_size_mb"]},
                remediation="Reduce request size or split into multiple requests"
            ))
        
        # Check HTTPS requirement
        if policy["require_https"] and not request_data.get("is_https", False):
            violations.append(PolicyViolation(
                policy_name="api_security_policy",
                description="HTTPS required for all requests",
                severity=PolicyViolationSeverity.HIGH,
                details={"protocol": request_data.get("protocol", "unknown")},
                remediation="Use HTTPS protocol for all API requests"
            ))
        
        # Check authentication requirement
        if policy["require_authentication"] and not request_data.get("authenticated", False):
            violations.append(PolicyViolation(
                policy_name="api_security_policy",
                description="Authentication required",
                severity=PolicyViolationSeverity.CRITICAL,
                details={"endpoint": request_data.get("endpoint", "unknown")},
                remediation="Provide valid authentication credentials"
            ))
        
        return len(violations) == 0, violations
    
    def validate_data_access(self, data_classification: str, user_role: str, 
                           access_type: str) -> Tuple[bool, List[PolicyViolation]]:
        """Validate data access against classification policies."""
        if data_classification not in self.policies["data_classification_policy"]:
            return False, [PolicyViolation(
                policy_name="data_classification_policy",
                description="Unknown data classification",
                severity=PolicyViolationSeverity.HIGH,
                details={"classification": data_classification},
                remediation="Use valid data classification levels"
            )]
        
        policy = self.policies["data_classification_policy"][data_classification]
        violations = []
        
        # Check if approval is required for confidential/restricted data
        if policy.get("require_approval", False) and access_type == "read":
            if user_role not in ["admin", "security_officer", "approved_user"]:
                violations.append(PolicyViolation(
                    policy_name="data_classification_policy",
                    description="Approval required for accessing classified data",
                    severity=PolicyViolationSeverity.HIGH,
                    details={"classification": data_classification, "user_role": user_role},
                    remediation="Obtain approval from data owner or security officer"
                ))
        
        # Check if MFA is required for restricted data
        if policy.get("require_mfa", False) and access_type in ["read", "write", "delete"]:
            # This would be checked against actual MFA status in real implementation
            violations.append(PolicyViolation(
                policy_name="data_classification_policy",
                description="Multi-factor authentication required for restricted data",
                severity=PolicyViolationSeverity.CRITICAL,
                details={"classification": data_classification},
                remediation="Complete multi-factor authentication"
            ))
        
        return len(violations) == 0, violations
    
    def validate_network_access(self, source_ip: str, destination_port: int, 
                              protocol: str) -> Tuple[bool, List[PolicyViolation]]:
        """Validate network access against network policies."""
        policy = self.policies["network_access_policy"]
        violations = []
        
        # Check blocked ports
        if destination_port in policy["blocked_ports"]:
            violations.append(PolicyViolation(
                policy_name="network_access_policy",
                description="Access to blocked port",
                severity=PolicyViolationSeverity.HIGH,
                details={"port": destination_port, "source_ip": source_ip},
                remediation="Use allowed ports only"
            ))
        
        # Check allowed protocols
        if protocol.lower() not in policy["allowed_protocols"]:
            violations.append(PolicyViolation(
                policy_name="network_access_policy",
                description="Disallowed protocol",
                severity=PolicyViolationSeverity.MEDIUM,
                details={"protocol": protocol, "allowed": policy["allowed_protocols"]},
                remediation="Use allowed protocols only"
            ))
        
        return len(violations) == 0, violations
    
    def check_compliance_requirements(self, data_type: str, 
                                    processing_purpose: str) -> List[str]:
        """Check compliance requirements for data processing."""
        policy = self.policies["compliance_policy"]
        requirements = []
        
        # GDPR requirements
        if policy["gdpr_compliance"]:
            if data_type in ["personal_data", "sensitive_personal_data"]:
                requirements.extend([
                    "Obtain explicit consent for data processing",
                    "Provide privacy notice to data subjects",
                    "Implement data subject rights (access, rectification, erasure)",
                    "Conduct privacy impact assessment if high risk",
                    "Implement data protection by design and by default"
                ])
        
        # FERPA requirements
        if policy["ferpa_compliance"]:
            if data_type == "educational_records":
                requirements.extend([
                    "Obtain written consent before disclosure",
                    "Maintain directory of disclosures",
                    "Provide annual notification of rights",
                    "Implement appropriate safeguards for electronic records"
                ])
        
        # General data protection requirements
        if data_type in ["personal_data", "sensitive_personal_data", "educational_records"]:
            requirements.extend([
                "Encrypt data at rest and in transit",
                "Implement access controls and audit logging",
                "Establish data retention and deletion procedures",
                "Conduct regular security assessments"
            ])
        
        return requirements
    
    def generate_policy_report(self) -> Dict[str, Any]:
        """Generate comprehensive policy compliance report."""
        report = {
            "generated_at": "2024-01-01T00:00:00Z",
            "policies": {
                "total_policies": len(self.policies),
                "policy_categories": list(self.policies.keys())
            },
            "violations": {
                "total_violations": len(self.violations),
                "by_severity": {
                    "critical": len([v for v in self.violations if v.severity == PolicyViolationSeverity.CRITICAL]),
                    "high": len([v for v in self.violations if v.severity == PolicyViolationSeverity.HIGH]),
                    "medium": len([v for v in self.violations if v.severity == PolicyViolationSeverity.MEDIUM]),
                    "low": len([v for v in self.violations if v.severity == PolicyViolationSeverity.LOW])
                }
            },
            "compliance_status": {
                "gdpr_compliant": self.policies["compliance_policy"]["gdpr_compliance"],
                "ferpa_compliant": self.policies["compliance_policy"]["ferpa_compliance"],
                "security_controls_implemented": True,
                "audit_logging_enabled": self.policies["audit_policy"]["log_all_access"]
            },
            "recommendations": [
                "Regularly review and update security policies",
                "Conduct periodic policy compliance audits",
                "Provide security awareness training to all users",
                "Implement automated policy enforcement where possible",
                "Establish incident response procedures for policy violations"
            ]
        }
        
        return report
    
    def update_policy(self, policy_name: str, policy_config: Dict[str, Any]) -> bool:
        """Update security policy configuration."""
        if policy_name not in self.policies:
            logger.error("Unknown policy", policy_name=policy_name)
            return False
        
        try:
            self.policies[policy_name].update(policy_config)
            logger.info("Security policy updated", policy_name=policy_name)
            return True
        except Exception as e:
            logger.error("Failed to update policy", policy_name=policy_name, error=str(e))
            return False
    
    def add_violation(self, violation: PolicyViolation):
        """Add policy violation to tracking."""
        self.violations.append(violation)
        logger.warning("Security policy violation recorded",
                      policy=violation.policy_name,
                      severity=violation.severity.value,
                      description=violation.description)


# Global security policy engine instance
security_policy_engine = SecurityPolicyEngine()