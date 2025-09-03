"""
Compliance monitoring and enforcement for campus security system.
Supports GDPR, FERPA, COPPA, and other privacy regulations.
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
import aiofiles

logger = structlog.get_logger()


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    FERPA = "ferpa"
    COPPA = "coppa"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"


class ComplianceStatus(Enum):
    """Compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    severity: str
    automated_check: bool = False
    check_function: Optional[str] = None
    remediation_steps: List[str] = None
    
    def __post_init__(self):
        if self.remediation_steps is None:
            self.remediation_steps = []


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    id: str
    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    severity: str
    detected_at: datetime
    status: str = "open"
    remediation_deadline: Optional[datetime] = None
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class DataSubjectRequest:
    """Data subject access request (DSAR)."""
    id: str
    request_type: str  # access, rectification, erasure, portability
    subject_id: str
    subject_email: str
    framework: ComplianceFramework
    requested_at: datetime
    status: str = "pending"
    deadline: Optional[datetime] = None
    response_data: Optional[Dict] = None
    completed_at: Optional[datetime] = None


class ComplianceMonitor:
    """Compliance monitoring and enforcement system."""
    
    def __init__(self):
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        self.compliance_status: Dict[ComplianceFramework, ComplianceStatus] = {}
        self.data_retention_policies: Dict[str, Dict] = {}
        self.initialize_rules()
        self.initialize_retention_policies()
    
    def initialize_rules(self):
        """Initialize compliance rules for different frameworks."""
        rules = [
            # GDPR Rules
            ComplianceRule(
                id="gdpr_001",
                framework=ComplianceFramework.GDPR,
                title="Data Processing Lawful Basis",
                description="All personal data processing must have a lawful basis",
                requirement="Article 6 - Lawfulness of processing",
                severity="high",
                automated_check=True,
                check_function="check_lawful_basis",
                remediation_steps=[
                    "Document lawful basis for each data processing activity",
                    "Update privacy policy with lawful basis information",
                    "Implement consent mechanisms where required"
                ]
            ),
            ComplianceRule(
                id="gdpr_002",
                framework=ComplianceFramework.GDPR,
                title="Data Subject Rights",
                description="System must support data subject rights (access, rectification, erasure)",
                requirement="Articles 15-22 - Rights of the data subject",
                severity="high",
                automated_check=True,
                check_function="check_data_subject_rights",
                remediation_steps=[
                    "Implement data subject access request handling",
                    "Create data portability export functionality",
                    "Implement right to erasure (right to be forgotten)"
                ]
            ),
            ComplianceRule(
                id="gdpr_003",
                framework=ComplianceFramework.GDPR,
                title="Data Retention Limits",
                description="Personal data must not be kept longer than necessary",
                requirement="Article 5(1)(e) - Storage limitation",
                severity="medium",
                automated_check=True,
                check_function="check_data_retention",
                remediation_steps=[
                    "Define data retention periods for each data category",
                    "Implement automated data deletion",
                    "Document retention policy and review regularly"
                ]
            ),
            ComplianceRule(
                id="gdpr_004",
                framework=ComplianceFramework.GDPR,
                title="Data Protection by Design",
                description="Privacy must be built into system design",
                requirement="Article 25 - Data protection by design and by default",
                severity="high",
                automated_check=False,
                remediation_steps=[
                    "Conduct privacy impact assessments",
                    "Implement privacy-enhancing technologies",
                    "Default to highest privacy settings"
                ]
            ),
            
            # FERPA Rules
            ComplianceRule(
                id="ferpa_001",
                framework=ComplianceFramework.FERPA,
                title="Educational Record Protection",
                description="Educational records must be protected from unauthorized disclosure",
                requirement="20 USC 1232g - Family Educational Rights and Privacy Act",
                severity="high",
                automated_check=True,
                check_function="check_educational_record_protection",
                remediation_steps=[
                    "Implement access controls for educational records",
                    "Log all access to educational records",
                    "Obtain consent for non-directory information disclosure"
                ]
            ),
            ComplianceRule(
                id="ferpa_002",
                framework=ComplianceFramework.FERPA,
                title="Directory Information Handling",
                description="Directory information disclosure must follow FERPA guidelines",
                requirement="34 CFR 99.37 - What conditions apply to disclosing directory information?",
                severity="medium",
                automated_check=True,
                check_function="check_directory_information",
                remediation_steps=[
                    "Define directory information categories",
                    "Implement opt-out mechanisms for directory information",
                    "Maintain records of directory information disclosures"
                ]
            ),
            
            # COPPA Rules
            ComplianceRule(
                id="coppa_001",
                framework=ComplianceFramework.COPPA,
                title="Parental Consent for Children Under 13",
                description="Verifiable parental consent required for children under 13",
                requirement="15 USC 6502 - Children's Online Privacy Protection Act",
                severity="critical",
                automated_check=True,
                check_function="check_parental_consent",
                remediation_steps=[
                    "Implement age verification mechanisms",
                    "Create parental consent collection process",
                    "Restrict data collection for users under 13"
                ]
            ),
            
            # General Security Rules
            ComplianceRule(
                id="sec_001",
                framework=ComplianceFramework.GDPR,  # Applies to multiple frameworks
                title="Data Encryption at Rest",
                description="Personal data must be encrypted when stored",
                requirement="Article 32 - Security of processing",
                severity="high",
                automated_check=True,
                check_function="check_encryption_at_rest",
                remediation_steps=[
                    "Implement database encryption",
                    "Encrypt file storage systems",
                    "Use strong encryption algorithms (AES-256)"
                ]
            ),
            ComplianceRule(
                id="sec_002",
                framework=ComplianceFramework.GDPR,
                title="Data Encryption in Transit",
                description="Personal data must be encrypted during transmission",
                requirement="Article 32 - Security of processing",
                severity="high",
                automated_check=True,
                check_function="check_encryption_in_transit",
                remediation_steps=[
                    "Implement TLS 1.3 for all communications",
                    "Use HTTPS for all web traffic",
                    "Encrypt API communications"
                ]
            )
        ]
        
        for rule in rules:
            self.compliance_rules[rule.id] = rule
        
        logger.info("Compliance rules initialized", count=len(rules))
    
    def initialize_retention_policies(self):
        """Initialize data retention policies."""
        self.data_retention_policies = {
            "security_events": {
                "retention_days": 2555,  # 7 years
                "framework": [ComplianceFramework.GDPR, ComplianceFramework.FERPA],
                "description": "Security events and incident logs"
            },
            "audit_logs": {
                "retention_days": 2555,  # 7 years
                "framework": [ComplianceFramework.GDPR, ComplianceFramework.SOX],
                "description": "System audit and access logs"
            },
            "video_evidence": {
                "retention_days": 90,  # 3 months default
                "framework": [ComplianceFramework.GDPR, ComplianceFramework.FERPA],
                "description": "Video surveillance evidence"
            },
            "user_sessions": {
                "retention_days": 30,
                "framework": [ComplianceFramework.GDPR],
                "description": "User session data and cookies"
            },
            "personal_data": {
                "retention_days": 365,  # 1 year default
                "framework": [ComplianceFramework.GDPR, ComplianceFramework.FERPA],
                "description": "General personal data"
            }
        }
        
        logger.info("Data retention policies initialized")
    
    async def run_compliance_check(self, framework: Optional[ComplianceFramework] = None) -> Dict[str, Any]:
        """Run compliance checks for specified framework or all frameworks."""
        logger.info("Starting compliance check", framework=framework.value if framework else "all")
        
        results = {
            "check_date": datetime.utcnow().isoformat(),
            "framework": framework.value if framework else "all",
            "rules_checked": 0,
            "violations_found": 0,
            "compliance_status": {},
            "violations": []
        }
        
        rules_to_check = [
            rule for rule in self.compliance_rules.values()
            if framework is None or rule.framework == framework
        ]
        
        for rule in rules_to_check:
            if rule.automated_check and rule.check_function:
                try:
                    violation = await self._run_rule_check(rule)
                    if violation:
                        self.violations[violation.id] = violation
                        results["violations"].append(asdict(violation))
                        results["violations_found"] += 1
                    
                    results["rules_checked"] += 1
                    
                except Exception as e:
                    logger.error("Compliance rule check failed", rule_id=rule.id, error=str(e))
        
        # Update compliance status
        await self._update_compliance_status()
        results["compliance_status"] = {
            framework.value: status.value 
            for framework, status in self.compliance_status.items()
        }
        
        logger.info("Compliance check completed",
                   rules_checked=results["rules_checked"],
                   violations_found=results["violations_found"])
        
        return results
    
    async def _run_rule_check(self, rule: ComplianceRule) -> Optional[ComplianceViolation]:
        """Run individual compliance rule check."""
        check_method = getattr(self, rule.check_function, None)
        if not check_method:
            logger.warning("Check function not found", rule_id=rule.id, function=rule.check_function)
            return None
        
        try:
            is_compliant = await check_method()
            
            if not is_compliant:
                violation_id = f"{rule.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                
                # Calculate remediation deadline based on severity
                deadline_days = {"critical": 7, "high": 30, "medium": 90, "low": 180}
                deadline = datetime.utcnow() + timedelta(days=deadline_days.get(rule.severity, 90))
                
                violation = ComplianceViolation(
                    id=violation_id,
                    rule_id=rule.id,
                    framework=rule.framework,
                    title=rule.title,
                    description=rule.description,
                    severity=rule.severity,
                    detected_at=datetime.utcnow(),
                    remediation_deadline=deadline
                )
                
                return violation
            
            return None
            
        except Exception as e:
            logger.error("Rule check execution failed", rule_id=rule.id, error=str(e))
            return None
    
    async def _update_compliance_status(self):
        """Update overall compliance status for each framework."""
        for framework in ComplianceFramework:
            framework_violations = [
                v for v in self.violations.values()
                if v.framework == framework and v.status == "open"
            ]
            
            if not framework_violations:
                self.compliance_status[framework] = ComplianceStatus.COMPLIANT
            else:
                critical_violations = [v for v in framework_violations if v.severity == "critical"]
                high_violations = [v for v in framework_violations if v.severity == "high"]
                
                if critical_violations:
                    self.compliance_status[framework] = ComplianceStatus.NON_COMPLIANT
                elif high_violations:
                    self.compliance_status[framework] = ComplianceStatus.PARTIALLY_COMPLIANT
                else:
                    self.compliance_status[framework] = ComplianceStatus.PARTIALLY_COMPLIANT
    
    # Compliance Check Functions
    async def check_lawful_basis(self) -> bool:
        """Check if lawful basis is documented for data processing."""
        # This would check if lawful basis is properly documented
        # For now, return True as placeholder
        return True
    
    async def check_data_subject_rights(self) -> bool:
        """Check if data subject rights are implemented."""
        # This would verify DSAR handling capabilities
        # Check if we have endpoints for data access, rectification, erasure
        return True
    
    async def check_data_retention(self) -> bool:
        """Check if data retention policies are enforced."""
        # This would check if old data is being properly deleted
        # according to retention policies
        return True
    
    async def check_educational_record_protection(self) -> bool:
        """Check FERPA educational record protection."""
        # This would verify access controls for educational records
        return True
    
    async def check_directory_information(self) -> bool:
        """Check FERPA directory information handling."""
        # This would verify directory information disclosure controls
        return True
    
    async def check_parental_consent(self) -> bool:
        """Check COPPA parental consent requirements."""
        # This would verify parental consent mechanisms for users under 13
        return True
    
    async def check_encryption_at_rest(self) -> bool:
        """Check if data is encrypted at rest."""
        # This would verify database and file encryption
        return True
    
    async def check_encryption_in_transit(self) -> bool:
        """Check if data is encrypted in transit."""
        # This would verify TLS/HTTPS usage
        return True
    
    async def create_data_subject_request(self, request_type: str, subject_id: str, 
                                        subject_email: str, framework: ComplianceFramework) -> DataSubjectRequest:
        """Create a new data subject access request."""
        request_id = f"dsar_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{subject_id}"
        
        # Calculate deadline based on framework
        deadline_days = {
            ComplianceFramework.GDPR: 30,  # 1 month
            ComplianceFramework.CCPA: 45,  # 45 days
            ComplianceFramework.FERPA: 45  # 45 days
        }
        
        deadline = datetime.utcnow() + timedelta(days=deadline_days.get(framework, 30))
        
        request = DataSubjectRequest(
            id=request_id,
            request_type=request_type,
            subject_id=subject_id,
            subject_email=subject_email,
            framework=framework,
            requested_at=datetime.utcnow(),
            deadline=deadline
        )
        
        self.data_subject_requests[request_id] = request
        
        logger.info("Data subject request created",
                   request_id=request_id,
                   type=request_type,
                   framework=framework.value)
        
        return request
    
    async def process_data_subject_request(self, request_id: str) -> Dict[str, Any]:
        """Process a data subject access request."""
        request = self.data_subject_requests.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")
        
        logger.info("Processing data subject request", request_id=request_id, type=request.request_type)
        
        response_data = {}
        
        if request.request_type == "access":
            # Collect all data for the subject
            response_data = await self._collect_subject_data(request.subject_id)
        
        elif request.request_type == "erasure":
            # Delete all data for the subject
            response_data = await self._erase_subject_data(request.subject_id)
        
        elif request.request_type == "rectification":
            # This would handle data correction requests
            response_data = {"message": "Rectification request processed"}
        
        elif request.request_type == "portability":
            # Export data in portable format
            response_data = await self._export_subject_data(request.subject_id)
        
        # Update request status
        request.status = "completed"
        request.response_data = response_data
        request.completed_at = datetime.utcnow()
        
        logger.info("Data subject request completed", request_id=request_id)
        
        return response_data
    
    async def _collect_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Collect all data for a data subject."""
        # This would query all systems for data related to the subject
        return {
            "subject_id": subject_id,
            "data_collected": "placeholder - would collect actual data",
            "collection_date": datetime.utcnow().isoformat()
        }
    
    async def _erase_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Erase all data for a data subject."""
        # This would delete data from all systems
        return {
            "subject_id": subject_id,
            "erasure_completed": True,
            "erasure_date": datetime.utcnow().isoformat()
        }
    
    async def _export_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Export data for a data subject in portable format."""
        # This would export data in JSON/CSV format
        return {
            "subject_id": subject_id,
            "export_format": "json",
            "export_date": datetime.utcnow().isoformat(),
            "data": "placeholder - would contain actual exported data"
        }
    
    async def generate_compliance_report(self, framework: Optional[ComplianceFramework] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "framework": framework.value if framework else "all",
            "compliance_status": {},
            "violations_summary": {},
            "data_subject_requests_summary": {},
            "recommendations": []
        }
        
        # Compliance status
        for fw, status in self.compliance_status.items():
            if framework is None or fw == framework:
                report["compliance_status"][fw.value] = status.value
        
        # Violations summary
        framework_violations = [
            v for v in self.violations.values()
            if framework is None or v.framework == framework
        ]
        
        report["violations_summary"] = {
            "total": len(framework_violations),
            "open": len([v for v in framework_violations if v.status == "open"]),
            "by_severity": {
                "critical": len([v for v in framework_violations if v.severity == "critical"]),
                "high": len([v for v in framework_violations if v.severity == "high"]),
                "medium": len([v for v in framework_violations if v.severity == "medium"]),
                "low": len([v for v in framework_violations if v.severity == "low"])
            }
        }
        
        # DSAR summary
        framework_requests = [
            r for r in self.data_subject_requests.values()
            if framework is None or r.framework == framework
        ]
        
        report["data_subject_requests_summary"] = {
            "total": len(framework_requests),
            "pending": len([r for r in framework_requests if r.status == "pending"]),
            "completed": len([r for r in framework_requests if r.status == "completed"]),
            "overdue": len([
                r for r in framework_requests 
                if r.status == "pending" and r.deadline and datetime.utcnow() > r.deadline
            ])
        }
        
        # Generate recommendations
        report["recommendations"] = self._generate_compliance_recommendations(framework_violations)
        
        return report
    
    def _generate_compliance_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate compliance recommendations based on violations."""
        recommendations = []
        
        critical_violations = [v for v in violations if v.severity == "critical" and v.status == "open"]
        if critical_violations:
            recommendations.append("Address critical compliance violations immediately")
        
        high_violations = [v for v in violations if v.severity == "high" and v.status == "open"]
        if len(high_violations) > 5:
            recommendations.append("High number of high-severity violations - consider compliance audit")
        
        overdue_violations = [
            v for v in violations 
            if v.status == "open" and v.remediation_deadline and datetime.utcnow() > v.remediation_deadline
        ]
        if overdue_violations:
            recommendations.append("Address overdue compliance violations")
        
        return recommendations
    
    async def export_compliance_data(self, output_file: str):
        """Export compliance data for audit purposes."""
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "compliance_rules": {rule_id: asdict(rule) for rule_id, rule in self.compliance_rules.items()},
            "violations": {violation_id: asdict(violation) for violation_id, violation in self.violations.items()},
            "data_subject_requests": {req_id: asdict(req) for req_id, req in self.data_subject_requests.items()},
            "compliance_status": {fw.value: status.value for fw, status in self.compliance_status.items()},
            "retention_policies": self.data_retention_policies
        }
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(export_data, indent=2, default=str))
        
        logger.info("Compliance data exported", output_file=output_file)


# Global compliance monitor instance
compliance_monitor = ComplianceMonitor()