"""
Security audit and compliance checking for campus security system.
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
import aiofiles
import os
import subprocess

logger = structlog.get_logger()


class AuditSeverity(Enum):
    """Security audit finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    FERPA = "ferpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist"


@dataclass
class SecurityFinding:
    """Security audit finding."""
    id: str
    title: str
    description: str
    severity: AuditSeverity
    category: str
    compliance_frameworks: List[ComplianceFramework]
    remediation: str
    affected_components: List[str]
    evidence: Dict[str, Any]
    discovered_at: datetime
    status: str = "open"  # open, acknowledged, remediated, false_positive
    
    def __post_init__(self):
        if isinstance(self.discovered_at, str):
            self.discovered_at = datetime.fromisoformat(self.discovered_at)


@dataclass
class AuditReport:
    """Security audit report."""
    audit_id: str
    audit_date: datetime
    audit_type: str
    scope: List[str]
    findings: List[SecurityFinding]
    summary: Dict[str, Any]
    compliance_status: Dict[ComplianceFramework, Dict[str, Any]]
    recommendations: List[str]
    auditor: str
    
    def get_findings_by_severity(self) -> Dict[AuditSeverity, List[SecurityFinding]]:
        """Group findings by severity."""
        grouped = {severity: [] for severity in AuditSeverity}
        for finding in self.findings:
            grouped[finding.severity].append(finding)
        return grouped


class SecurityAuditor:
    """Security audit and compliance checker."""
    
    def __init__(self):
        self.audit_rules: Dict[str, Dict] = {}
        self.compliance_mappings: Dict[ComplianceFramework, Dict] = {}
        self.audit_history: List[AuditReport] = []
        self.initialize_audit_rules()
        self.initialize_compliance_mappings()
    
    def initialize_audit_rules(self):
        """Initialize security audit rules."""
        self.audit_rules = {
            "password_policy": {
                "title": "Password Policy Compliance",
                "description": "Check password policy configuration",
                "severity": AuditSeverity.HIGH,
                "category": "authentication",
                "check_function": self._check_password_policy
            },
            "encryption_at_rest": {
                "title": "Data Encryption at Rest",
                "description": "Verify data is encrypted at rest",
                "severity": AuditSeverity.CRITICAL,
                "category": "data_protection",
                "check_function": self._check_encryption_at_rest
            },
            "encryption_in_transit": {
                "title": "Data Encryption in Transit",
                "description": "Verify data is encrypted in transit",
                "severity": AuditSeverity.CRITICAL,
                "category": "data_protection",
                "check_function": self._check_encryption_in_transit
            },
            "access_controls": {
                "title": "Access Control Configuration",
                "description": "Review access control settings",
                "severity": AuditSeverity.HIGH,
                "category": "access_control",
                "check_function": self._check_access_controls
            },
            "audit_logging": {
                "title": "Audit Logging Configuration",
                "description": "Verify comprehensive audit logging",
                "severity": AuditSeverity.MEDIUM,
                "category": "logging",
                "check_function": self._check_audit_logging
            },
            "vulnerability_management": {
                "title": "Vulnerability Management Process",
                "description": "Check vulnerability scanning and patching",
                "severity": AuditSeverity.HIGH,
                "category": "vulnerability_management",
                "check_function": self._check_vulnerability_management
            },
            "backup_recovery": {
                "title": "Backup and Recovery Procedures",
                "description": "Verify backup and recovery capabilities",
                "severity": AuditSeverity.MEDIUM,
                "category": "business_continuity",
                "check_function": self._check_backup_recovery
            },
            "network_security": {
                "title": "Network Security Controls",
                "description": "Review network security configuration",
                "severity": AuditSeverity.HIGH,
                "category": "network_security",
                "check_function": self._check_network_security
            },
            "incident_response": {
                "title": "Incident Response Procedures",
                "description": "Verify incident response capabilities",
                "severity": AuditSeverity.MEDIUM,
                "category": "incident_response",
                "check_function": self._check_incident_response
            },
            "data_retention": {
                "title": "Data Retention Policies",
                "description": "Check data retention and deletion policies",
                "severity": AuditSeverity.MEDIUM,
                "category": "data_governance",
                "check_function": self._check_data_retention
            }
        }
        logger.info("Security audit rules initialized", rules_count=len(self.audit_rules))
    
    def initialize_compliance_mappings(self):
        """Initialize compliance framework mappings."""
        self.compliance_mappings = {
            ComplianceFramework.GDPR: {
                "name": "General Data Protection Regulation",
                "required_controls": [
                    "encryption_at_rest", "encryption_in_transit", "access_controls",
                    "audit_logging", "data_retention", "incident_response"
                ],
                "data_subject_rights": ["access", "rectification", "erasure", "portability"],
                "breach_notification_hours": 72
            },
            ComplianceFramework.FERPA: {
                "name": "Family Educational Rights and Privacy Act",
                "required_controls": [
                    "access_controls", "audit_logging", "data_retention"
                ],
                "educational_record_protection": True,
                "disclosure_restrictions": True
            },
            ComplianceFramework.SOC2: {
                "name": "Service Organization Control 2",
                "trust_principles": ["security", "availability", "processing_integrity", 
                                   "confidentiality", "privacy"],
                "required_controls": [
                    "access_controls", "encryption_at_rest", "encryption_in_transit",
                    "vulnerability_management", "incident_response", "backup_recovery"
                ]
            },
            ComplianceFramework.ISO27001: {
                "name": "ISO/IEC 27001 Information Security Management",
                "control_domains": [
                    "information_security_policies", "organization_of_information_security",
                    "human_resource_security", "asset_management", "access_control",
                    "cryptography", "physical_and_environmental_security",
                    "operations_security", "communications_security",
                    "system_acquisition_development_maintenance", "supplier_relationships",
                    "information_security_incident_management",
                    "information_security_aspects_business_continuity_management",
                    "compliance"
                ],
                "required_controls": [
                    "password_policy", "encryption_at_rest", "encryption_in_transit",
                    "access_controls", "audit_logging", "vulnerability_management",
                    "backup_recovery", "network_security", "incident_response"
                ]
            },
            ComplianceFramework.NIST: {
                "name": "NIST Cybersecurity Framework",
                "functions": ["identify", "protect", "detect", "respond", "recover"],
                "required_controls": [
                    "password_policy", "encryption_at_rest", "encryption_in_transit",
                    "access_controls", "audit_logging", "vulnerability_management",
                    "network_security", "incident_response", "backup_recovery"
                ]
            }
        }
        logger.info("Compliance mappings initialized", frameworks=len(self.compliance_mappings))
    
    async def run_security_audit(self, scope: List[str] = None, 
                               audit_type: str = "comprehensive") -> AuditReport:
        """Run comprehensive security audit."""
        audit_id = f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        audit_date = datetime.utcnow()
        
        if scope is None:
            scope = list(self.audit_rules.keys())
        
        logger.info("Starting security audit", audit_id=audit_id, scope=scope)
        
        findings = []
        
        # Run audit checks
        for rule_id in scope:
            if rule_id not in self.audit_rules:
                logger.warning("Unknown audit rule", rule_id=rule_id)
                continue
            
            rule = self.audit_rules[rule_id]
            
            try:
                logger.debug("Running audit check", rule_id=rule_id)
                check_result = await rule["check_function"]()
                
                if not check_result["passed"]:
                    finding = SecurityFinding(
                        id=f"{audit_id}_{rule_id}",
                        title=rule["title"],
                        description=rule["description"],
                        severity=rule["severity"],
                        category=rule["category"],
                        compliance_frameworks=self._get_compliance_frameworks_for_control(rule_id),
                        remediation=check_result.get("remediation", "No remediation provided"),
                        affected_components=check_result.get("affected_components", []),
                        evidence=check_result.get("evidence", {}),
                        discovered_at=audit_date
                    )
                    findings.append(finding)
                    
                logger.info("Audit check completed", rule_id=rule_id, passed=check_result["passed"])
                
            except Exception as e:
                logger.error("Audit check failed", rule_id=rule_id, error=str(e))
                
                # Create finding for failed check
                finding = SecurityFinding(
                    id=f"{audit_id}_{rule_id}_error",
                    title=f"Audit Check Failed: {rule['title']}",
                    description=f"Failed to execute audit check: {str(e)}",
                    severity=AuditSeverity.MEDIUM,
                    category="audit_system",
                    compliance_frameworks=[],
                    remediation="Fix audit system or check configuration",
                    affected_components=["audit_system"],
                    evidence={"error": str(e)},
                    discovered_at=audit_date
                )
                findings.append(finding)
        
        # Generate summary
        summary = self._generate_audit_summary(findings)
        
        # Check compliance status
        compliance_status = self._check_compliance_status(findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings)
        
        audit_report = AuditReport(
            audit_id=audit_id,
            audit_date=audit_date,
            audit_type=audit_type,
            scope=scope,
            findings=findings,
            summary=summary,
            compliance_status=compliance_status,
            recommendations=recommendations,
            auditor="automated_security_auditor"
        )
        
        self.audit_history.append(audit_report)
        
        logger.info("Security audit completed", 
                   audit_id=audit_id, 
                   findings_count=len(findings),
                   critical_findings=summary["critical_count"])
        
        return audit_report
    
    def _get_compliance_frameworks_for_control(self, control_id: str) -> List[ComplianceFramework]:
        """Get compliance frameworks that require this control."""
        frameworks = []
        for framework, mapping in self.compliance_mappings.items():
            if control_id in mapping.get("required_controls", []):
                frameworks.append(framework)
        return frameworks
    
    def _generate_audit_summary(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Generate audit summary statistics."""
        severity_counts = {severity.value: 0 for severity in AuditSeverity}
        category_counts = {}
        
        for finding in findings:
            severity_counts[finding.severity.value] += 1
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1
        
        return {
            "total_findings": len(findings),
            "critical_count": severity_counts["critical"],
            "high_count": severity_counts["high"],
            "medium_count": severity_counts["medium"],
            "low_count": severity_counts["low"],
            "info_count": severity_counts["info"],
            "categories": category_counts,
            "risk_score": self._calculate_risk_score(findings)
        }
    
    def _calculate_risk_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate overall risk score based on findings."""
        score = 0.0
        for finding in findings:
            if finding.severity == AuditSeverity.CRITICAL:
                score += 10.0
            elif finding.severity == AuditSeverity.HIGH:
                score += 7.0
            elif finding.severity == AuditSeverity.MEDIUM:
                score += 4.0
            elif finding.severity == AuditSeverity.LOW:
                score += 1.0
        return score
    
    def _check_compliance_status(self, findings: List[SecurityFinding]) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Check compliance status for each framework."""
        compliance_status = {}
        
        for framework, mapping in self.compliance_mappings.items():
            required_controls = mapping.get("required_controls", [])
            
            # Check which required controls have findings
            failed_controls = set()
            for finding in findings:
                if framework in finding.compliance_frameworks:
                    # Find which control this finding relates to
                    for control in required_controls:
                        if control in finding.id:
                            failed_controls.add(control)
            
            passed_controls = set(required_controls) - failed_controls
            compliance_percentage = len(passed_controls) / len(required_controls) * 100 if required_controls else 100
            
            compliance_status[framework] = {
                "name": mapping["name"],
                "compliance_percentage": compliance_percentage,
                "passed_controls": list(passed_controls),
                "failed_controls": list(failed_controls),
                "status": "compliant" if compliance_percentage >= 95 else "non_compliant"
            }
        
        return compliance_status
    
    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Priority recommendations based on critical/high findings
        critical_high_findings = [f for f in findings if f.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]]
        
        if critical_high_findings:
            recommendations.append("Immediately address all critical and high severity findings")
        
        # Category-specific recommendations
        categories = {}
        for finding in findings:
            if finding.category not in categories:
                categories[finding.category] = []
            categories[finding.category].append(finding)
        
        if "data_protection" in categories:
            recommendations.append("Review and strengthen data protection controls")
        
        if "access_control" in categories:
            recommendations.append("Implement stronger access control measures")
        
        if "vulnerability_management" in categories:
            recommendations.append("Enhance vulnerability management processes")
        
        if "network_security" in categories:
            recommendations.append("Review network security configuration")
        
        # Compliance-specific recommendations
        for framework, status in self._check_compliance_status(findings).items():
            if status["compliance_percentage"] < 95:
                recommendations.append(f"Address {framework.value.upper()} compliance gaps")
        
        return recommendations
    
    # Audit check functions
    async def _check_password_policy(self) -> Dict[str, Any]:
        """Check password policy configuration."""
        from .config import security_settings
        
        issues = []
        evidence = {}
        
        # Check minimum password length
        if security_settings.PASSWORD_MIN_LENGTH < 12:
            issues.append(f"Password minimum length is {security_settings.PASSWORD_MIN_LENGTH}, should be at least 12")
        evidence["min_length"] = security_settings.PASSWORD_MIN_LENGTH
        
        # Check complexity requirements
        complexity_checks = [
            ("uppercase", security_settings.PASSWORD_REQUIRE_UPPERCASE),
            ("lowercase", security_settings.PASSWORD_REQUIRE_LOWERCASE),
            ("numbers", security_settings.PASSWORD_REQUIRE_NUMBERS),
            ("symbols", security_settings.PASSWORD_REQUIRE_SYMBOLS)
        ]
        
        for check_name, enabled in complexity_checks:
            evidence[f"require_{check_name}"] = enabled
            if not enabled:
                issues.append(f"Password complexity requirement missing: {check_name}")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Configure strong password policy with minimum 12 characters and complexity requirements",
            "affected_components": ["authentication_system"]
        }
    
    async def _check_encryption_at_rest(self) -> Dict[str, Any]:
        """Check data encryption at rest."""
        issues = []
        evidence = {}
        
        # Check database encryption (this would need to be implemented based on actual DB config)
        # For now, we'll check if encryption settings are configured
        from .config import security_settings
        
        evidence["encryption_algorithm"] = security_settings.ENCRYPTION_ALGORITHM
        evidence["audit_log_encryption"] = security_settings.ENABLE_AUDIT_LOG_ENCRYPTION
        
        if not security_settings.ENABLE_AUDIT_LOG_ENCRYPTION:
            issues.append("Audit log encryption is disabled")
        
        # Check if encryption algorithm is strong
        weak_algorithms = ["DES", "3DES", "RC4", "MD5"]
        if any(weak in security_settings.ENCRYPTION_ALGORITHM for weak in weak_algorithms):
            issues.append(f"Weak encryption algorithm: {security_settings.ENCRYPTION_ALGORITHM}")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Enable encryption at rest for all sensitive data stores",
            "affected_components": ["database", "file_storage", "audit_logs"]
        }
    
    async def _check_encryption_in_transit(self) -> Dict[str, Any]:
        """Check data encryption in transit."""
        issues = []
        evidence = {}
        
        from .config import security_settings
        
        # Check HSTS configuration
        evidence["hsts_enabled"] = security_settings.ENABLE_HSTS
        evidence["hsts_max_age"] = security_settings.HSTS_MAX_AGE
        
        if not security_settings.ENABLE_HSTS:
            issues.append("HTTP Strict Transport Security (HSTS) is disabled")
        
        if security_settings.HSTS_MAX_AGE < 31536000:  # 1 year
            issues.append(f"HSTS max age is too short: {security_settings.HSTS_MAX_AGE} seconds")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Enable HTTPS with strong TLS configuration and HSTS",
            "affected_components": ["web_server", "api_endpoints"]
        }
    
    async def _check_access_controls(self) -> Dict[str, Any]:
        """Check access control configuration."""
        issues = []
        evidence = {}
        
        from .config import security_settings
        
        # Check session configuration
        evidence["session_timeout"] = security_settings.SESSION_TIMEOUT_MINUTES
        evidence["max_concurrent_sessions"] = security_settings.MAX_CONCURRENT_SESSIONS
        
        if security_settings.SESSION_TIMEOUT_MINUTES > 60:
            issues.append(f"Session timeout is too long: {security_settings.SESSION_TIMEOUT_MINUTES} minutes")
        
        if security_settings.MAX_CONCURRENT_SESSIONS > 5:
            issues.append(f"Maximum concurrent sessions is too high: {security_settings.MAX_CONCURRENT_SESSIONS}")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Configure appropriate session timeouts and concurrent session limits",
            "affected_components": ["authentication_system", "session_management"]
        }
    
    async def _check_audit_logging(self) -> Dict[str, Any]:
        """Check audit logging configuration."""
        issues = []
        evidence = {}
        
        from .config import security_settings
        
        evidence["audit_log_retention_days"] = security_settings.AUDIT_LOG_RETENTION_DAYS
        evidence["audit_log_encryption"] = security_settings.ENABLE_AUDIT_LOG_ENCRYPTION
        
        # Check retention period (should be at least 1 year for compliance)
        if security_settings.AUDIT_LOG_RETENTION_DAYS < 365:
            issues.append(f"Audit log retention period is too short: {security_settings.AUDIT_LOG_RETENTION_DAYS} days")
        
        if not security_settings.ENABLE_AUDIT_LOG_ENCRYPTION:
            issues.append("Audit log encryption is disabled")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Configure comprehensive audit logging with appropriate retention and encryption",
            "affected_components": ["audit_system", "logging_infrastructure"]
        }
    
    async def _check_vulnerability_management(self) -> Dict[str, Any]:
        """Check vulnerability management process."""
        issues = []
        evidence = {}
        
        from .config import security_settings
        from .vulnerability_scanner import vulnerability_scanner
        
        evidence["vulnerability_scanning_enabled"] = security_settings.ENABLE_VULNERABILITY_SCANNING
        evidence["scan_results_count"] = len(vulnerability_scanner.scan_results)
        
        if not security_settings.ENABLE_VULNERABILITY_SCANNING:
            issues.append("Vulnerability scanning is disabled")
        
        # Check for unaddressed critical vulnerabilities
        total_critical = sum(result.critical_count for result in vulnerability_scanner.scan_results.values())
        evidence["total_critical_vulnerabilities"] = total_critical
        
        if total_critical > security_settings.MAX_CRITICAL_VULNERABILITIES:
            issues.append(f"Too many critical vulnerabilities: {total_critical}")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Enable regular vulnerability scanning and address critical vulnerabilities promptly",
            "affected_components": ["vulnerability_scanner", "container_images"]
        }
    
    async def _check_backup_recovery(self) -> Dict[str, Any]:
        """Check backup and recovery procedures."""
        issues = []
        evidence = {}
        
        # This would need to be implemented based on actual backup configuration
        # For now, we'll check if backup procedures are documented
        
        # Check if backup directories exist
        backup_paths = ["/backups", "./backups", "/var/backups"]
        backup_exists = any(os.path.exists(path) for path in backup_paths)
        
        evidence["backup_directory_exists"] = backup_exists
        
        if not backup_exists:
            issues.append("No backup directory found")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Implement automated backup and recovery procedures",
            "affected_components": ["backup_system", "database", "file_storage"]
        }
    
    async def _check_network_security(self) -> Dict[str, Any]:
        """Check network security controls."""
        issues = []
        evidence = {}
        
        from .network_policies import network_policies
        
        config = network_policies.export_config()
        evidence["firewall_rules_count"] = len(config["firewall_rules"])
        evidence["trusted_networks_count"] = len(config["trusted_networks"])
        evidence["blocked_networks_count"] = len(config["blocked_networks"])
        
        if len(config["firewall_rules"]) == 0:
            issues.append("No firewall rules configured")
        
        # Check for default deny rule
        has_default_deny = any(rule["name"] == "default_deny" for rule in config["firewall_rules"])
        evidence["has_default_deny_rule"] = has_default_deny
        
        if not has_default_deny:
            issues.append("No default deny firewall rule found")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Configure comprehensive network security policies and firewall rules",
            "affected_components": ["firewall", "network_policies"]
        }
    
    async def _check_incident_response(self) -> Dict[str, Any]:
        """Check incident response procedures."""
        issues = []
        evidence = {}
        
        from .incident_response import incident_response
        from .config import security_settings
        
        evidence["auto_incident_response_enabled"] = security_settings.ENABLE_AUTO_INCIDENT_RESPONSE
        evidence["incident_notification_configured"] = bool(security_settings.INCIDENT_NOTIFICATION_WEBHOOK)
        evidence["total_incidents"] = len(incident_response.incidents)
        
        if not security_settings.ENABLE_AUTO_INCIDENT_RESPONSE:
            issues.append("Automatic incident response is disabled")
        
        if not security_settings.INCIDENT_NOTIFICATION_WEBHOOK:
            issues.append("Incident notification webhook is not configured")
        
        # Check if there are unresolved critical incidents
        from .incident_response import IncidentStatus, IncidentSeverity
        critical_open = [
            incident for incident in incident_response.incidents.values()
            if incident.severity == IncidentSeverity.CRITICAL and 
               incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]
        ]
        
        evidence["critical_open_incidents"] = len(critical_open)
        
        if len(critical_open) > 0:
            issues.append(f"Unresolved critical incidents: {len(critical_open)}")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Configure comprehensive incident response procedures and notifications",
            "affected_components": ["incident_response_system", "notification_system"]
        }
    
    async def _check_data_retention(self) -> Dict[str, Any]:
        """Check data retention policies."""
        issues = []
        evidence = {}
        
        from .config import security_settings
        
        evidence["data_retention_days"] = security_settings.DATA_RETENTION_DAYS
        evidence["gdpr_compliance_enabled"] = security_settings.ENABLE_GDPR_COMPLIANCE
        evidence["ferpa_compliance_enabled"] = security_settings.ENABLE_FERPA_COMPLIANCE
        
        # Check if retention period is reasonable
        if security_settings.DATA_RETENTION_DAYS > 2555:  # 7 years
            issues.append(f"Data retention period may be too long: {security_settings.DATA_RETENTION_DAYS} days")
        
        if security_settings.ENABLE_GDPR_COMPLIANCE and security_settings.DATA_RETENTION_DAYS > 2190:  # 6 years
            issues.append("GDPR compliance enabled but retention period exceeds typical requirements")
        
        return {
            "passed": len(issues) == 0,
            "evidence": evidence,
            "issues": issues,
            "remediation": "Configure appropriate data retention and deletion policies",
            "affected_components": ["data_management", "compliance_system"]
        }
    
    async def export_audit_report(self, audit_report: AuditReport, output_file: str):
        """Export audit report to file."""
        report_data = {
            "audit_id": audit_report.audit_id,
            "audit_date": audit_report.audit_date.isoformat(),
            "audit_type": audit_report.audit_type,
            "scope": audit_report.scope,
            "auditor": audit_report.auditor,
            "summary": audit_report.summary,
            "compliance_status": {
                framework.value: status for framework, status in audit_report.compliance_status.items()
            },
            "recommendations": audit_report.recommendations,
            "findings": [
                {
                    "id": finding.id,
                    "title": finding.title,
                    "description": finding.description,
                    "severity": finding.severity.value,
                    "category": finding.category,
                    "compliance_frameworks": [f.value for f in finding.compliance_frameworks],
                    "remediation": finding.remediation,
                    "affected_components": finding.affected_components,
                    "evidence": finding.evidence,
                    "discovered_at": finding.discovered_at.isoformat(),
                    "status": finding.status
                }
                for finding in audit_report.findings
            ]
        }
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(report_data, indent=2, default=str))
        
        logger.info("Audit report exported", output_file=output_file, audit_id=audit_report.audit_id)


# Global security auditor instance
security_auditor = SecurityAuditor()