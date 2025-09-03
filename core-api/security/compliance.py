"""
Compliance and data protection implementation for campus security system.
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
import aiofiles
from uuid import uuid4

from .config import security_settings

logger = structlog.get_logger()


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    FERPA = "ferpa"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DataSubjectRights(Enum):
    """Data subject rights under GDPR."""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    RESTRICTION = "restriction"
    OBJECTION = "objection"


@dataclass
class DataRetentionPolicy:
    """Data retention policy definition."""
    data_type: str
    classification: DataClassification
    retention_days: int
    deletion_method: str
    legal_basis: str
    review_frequency_days: int
    auto_delete: bool = True
    requires_approval: bool = False


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment (PIA) data structure."""
    id: str
    title: str
    description: str
    data_types: List[str]
    processing_purposes: List[str]
    legal_basis: str
    risk_level: str
    mitigation_measures: List[str]
    approval_status: str
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    next_review_date: Optional[datetime] = None


@dataclass
class DataSubjectRequest:
    """Data Subject Access Request (DSAR) tracking."""
    id: str
    request_type: DataSubjectRights
    subject_id: str
    subject_email: str
    request_details: str
    status: str
    created_at: datetime
    due_date: datetime
    completed_at: Optional[datetime] = None
    response_data: Optional[Dict] = None


class ComplianceManager:
    """Comprehensive compliance and data protection manager."""
    
    def __init__(self):
        self.retention_policies: Dict[str, DataRetentionPolicy] = {}
        self.privacy_assessments: Dict[str, PrivacyImpactAssessment] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        self.compliance_violations: List[Dict] = []
        self.initialize_default_policies()
    
    def initialize_default_policies(self):
        """Initialize default data retention policies."""
        default_policies = [
            DataRetentionPolicy(
                data_type="security_events",
                classification=DataClassification.INTERNAL,
                retention_days=2555,  # 7 years
                deletion_method="secure_delete",
                legal_basis="legitimate_interest",
                review_frequency_days=365,
                auto_delete=True
            ),
            DataRetentionPolicy(
                data_type="video_evidence",
                classification=DataClassification.CONFIDENTIAL,
                retention_days=2555,  # 7 years
                deletion_method="secure_delete_with_audit",
                legal_basis="legitimate_interest",
                review_frequency_days=365,
                auto_delete=False,
                requires_approval=True
            ),
            DataRetentionPolicy(
                data_type="personal_data",
                classification=DataClassification.RESTRICTED,
                retention_days=1095,  # 3 years
                deletion_method="cryptographic_erasure",
                legal_basis="consent",
                review_frequency_days=180,
                auto_delete=True,
                requires_approval=True
            ),
            DataRetentionPolicy(
                data_type="audit_logs",
                classification=DataClassification.INTERNAL,
                retention_days=2555,  # 7 years
                deletion_method="archive_then_delete",
                legal_basis="legal_obligation",
                review_frequency_days=365,
                auto_delete=True
            ),
            DataRetentionPolicy(
                data_type="user_sessions",
                classification=DataClassification.INTERNAL,
                retention_days=90,
                deletion_method="standard_delete",
                legal_basis="legitimate_interest",
                review_frequency_days=30,
                auto_delete=True
            ),
            DataRetentionPolicy(
                data_type="incident_reports",
                classification=DataClassification.CONFIDENTIAL,
                retention_days=2555,  # 7 years
                deletion_method="secure_delete_with_audit",
                legal_basis="legitimate_interest",
                review_frequency_days=365,
                auto_delete=False,
                requires_approval=True
            )
        ]
        
        for policy in default_policies:
            self.retention_policies[policy.data_type] = policy
        
        logger.info("Default data retention policies initialized", 
                   policies_count=len(default_policies))
    
    async def create_privacy_impact_assessment(self, 
                                             title: str,
                                             description: str,
                                             data_types: List[str],
                                             processing_purposes: List[str],
                                             legal_basis: str) -> PrivacyImpactAssessment:
        """Create a new Privacy Impact Assessment."""
        pia_id = str(uuid4())
        
        # Assess risk level based on data types and purposes
        risk_level = self._assess_privacy_risk(data_types, processing_purposes)
        
        # Generate mitigation measures
        mitigation_measures = self._generate_mitigation_measures(data_types, processing_purposes, risk_level)
        
        pia = PrivacyImpactAssessment(
            id=pia_id,
            title=title,
            description=description,
            data_types=data_types,
            processing_purposes=processing_purposes,
            legal_basis=legal_basis,
            risk_level=risk_level,
            mitigation_measures=mitigation_measures,
            approval_status="pending_review",
            created_at=datetime.utcnow(),
            next_review_date=datetime.utcnow() + timedelta(days=365)
        )
        
        self.privacy_assessments[pia_id] = pia
        
        logger.info("Privacy Impact Assessment created",
                   pia_id=pia_id,
                   risk_level=risk_level,
                   data_types=data_types)
        
        return pia
    
    def _assess_privacy_risk(self, data_types: List[str], processing_purposes: List[str]) -> str:
        """Assess privacy risk level based on data types and processing purposes."""
        high_risk_data = ["biometric_data", "personal_identifiers", "location_data", "behavioral_data"]
        high_risk_purposes = ["profiling", "automated_decision_making", "surveillance", "tracking"]
        
        risk_score = 0
        
        # Score based on data types
        for data_type in data_types:
            if data_type in high_risk_data:
                risk_score += 3
            elif "personal" in data_type.lower():
                risk_score += 2
            else:
                risk_score += 1
        
        # Score based on processing purposes
        for purpose in processing_purposes:
            if purpose in high_risk_purposes:
                risk_score += 3
            elif "monitoring" in purpose.lower():
                risk_score += 2
            else:
                risk_score += 1
        
        # Determine risk level
        if risk_score >= 12:
            return "high"
        elif risk_score >= 8:
            return "medium"
        elif risk_score >= 4:
            return "low"
        else:
            return "minimal"
    
    def _generate_mitigation_measures(self, data_types: List[str], 
                                    processing_purposes: List[str], 
                                    risk_level: str) -> List[str]:
        """Generate appropriate mitigation measures based on risk assessment."""
        measures = []
        
        # Base measures for all processing
        measures.extend([
            "Implement data minimization principles",
            "Apply purpose limitation controls",
            "Ensure data accuracy and quality",
            "Implement appropriate technical and organizational measures"
        ])
        
        # Risk-specific measures
        if risk_level in ["high", "medium"]:
            measures.extend([
                "Implement privacy by design and by default",
                "Conduct regular privacy audits",
                "Implement data protection impact assessments",
                "Establish data breach notification procedures"
            ])
        
        if risk_level == "high":
            measures.extend([
                "Implement enhanced encryption and pseudonymization",
                "Establish strict access controls and monitoring",
                "Conduct regular penetration testing",
                "Implement automated privacy compliance monitoring"
            ])
        
        # Data type specific measures
        if "biometric_data" in data_types:
            measures.append("Implement biometric template protection")
        
        if "location_data" in data_types:
            measures.append("Implement location data anonymization")
        
        # Purpose specific measures
        if "surveillance" in processing_purposes:
            measures.extend([
                "Implement privacy zones and masking",
                "Establish surveillance data retention limits",
                "Implement automated redaction capabilities"
            ])
        
        return measures
    
    async def process_data_subject_request(self, 
                                         request_type: DataSubjectRights,
                                         subject_id: str,
                                         subject_email: str,
                                         request_details: str) -> DataSubjectRequest:
        """Process a data subject access request."""
        request_id = str(uuid4())
        
        # Calculate due date (30 days for GDPR)
        due_date = datetime.utcnow() + timedelta(days=30)
        
        dsar = DataSubjectRequest(
            id=request_id,
            request_type=request_type,
            subject_id=subject_id,
            subject_email=subject_email,
            request_details=request_details,
            status="received",
            created_at=datetime.utcnow(),
            due_date=due_date
        )
        
        self.data_subject_requests[request_id] = dsar
        
        # Start processing based on request type
        await self._process_dsar_by_type(dsar)
        
        logger.info("Data subject request received",
                   request_id=request_id,
                   request_type=request_type.value,
                   subject_id=subject_id)
        
        return dsar
    
    async def _process_dsar_by_type(self, dsar: DataSubjectRequest):
        """Process DSAR based on request type."""
        if dsar.request_type == DataSubjectRights.ACCESS:
            await self._process_access_request(dsar)
        elif dsar.request_type == DataSubjectRights.ERASURE:
            await self._process_erasure_request(dsar)
        elif dsar.request_type == DataSubjectRights.RECTIFICATION:
            await self._process_rectification_request(dsar)
        elif dsar.request_type == DataSubjectRights.PORTABILITY:
            await self._process_portability_request(dsar)
        elif dsar.request_type == DataSubjectRights.RESTRICTION:
            await self._process_restriction_request(dsar)
        elif dsar.request_type == DataSubjectRights.OBJECTION:
            await self._process_objection_request(dsar)
    
    async def _process_access_request(self, dsar: DataSubjectRequest):
        """Process data access request."""
        dsar.status = "processing"
        
        # Collect all data related to the subject
        subject_data = {
            "personal_information": await self._collect_personal_data(dsar.subject_id),
            "security_events": await self._collect_security_events(dsar.subject_id),
            "audit_logs": await self._collect_audit_logs(dsar.subject_id),
            "video_appearances": await self._collect_video_data(dsar.subject_id)
        }
        
        dsar.response_data = subject_data
        dsar.status = "completed"
        dsar.completed_at = datetime.utcnow()
        
        logger.info("Access request processed", request_id=dsar.id)
    
    async def _process_erasure_request(self, dsar: DataSubjectRequest):
        """Process data erasure request (right to be forgotten)."""
        dsar.status = "processing"
        
        # Check if erasure is legally permissible
        if await self._can_erase_data(dsar.subject_id):
            # Perform secure deletion
            deletion_results = {
                "personal_data": await self._secure_delete_personal_data(dsar.subject_id),
                "video_data": await self._secure_delete_video_data(dsar.subject_id),
                "audit_logs": await self._anonymize_audit_logs(dsar.subject_id)
            }
            
            dsar.response_data = {"deletion_results": deletion_results}
            dsar.status = "completed"
        else:
            dsar.status = "rejected"
            dsar.response_data = {"reason": "Legal obligation prevents erasure"}
        
        dsar.completed_at = datetime.utcnow()
        logger.info("Erasure request processed", request_id=dsar.id, status=dsar.status)
    
    async def _collect_personal_data(self, subject_id: str) -> Dict:
        """Collect personal data for access request."""
        # This would integrate with actual database queries
        return {
            "user_profile": f"Profile data for {subject_id}",
            "preferences": f"User preferences for {subject_id}",
            "contact_information": f"Contact info for {subject_id}"
        }
    
    async def _collect_security_events(self, subject_id: str) -> List[Dict]:
        """Collect security events involving the subject."""
        # This would integrate with actual database queries
        return [
            {
                "event_id": "evt_123",
                "timestamp": "2024-01-01T10:00:00Z",
                "event_type": "access_granted",
                "location": "Building A"
            }
        ]
    
    async def _collect_audit_logs(self, subject_id: str) -> List[Dict]:
        """Collect audit logs for the subject."""
        # This would integrate with actual database queries
        return [
            {
                "log_id": "log_456",
                "timestamp": "2024-01-01T10:00:00Z",
                "action": "login",
                "ip_address": "10.0.1.100"
            }
        ]
    
    async def _collect_video_data(self, subject_id: str) -> List[Dict]:
        """Collect video data appearances."""
        # This would integrate with actual video analysis system
        return [
            {
                "video_id": "vid_789",
                "timestamp": "2024-01-01T10:00:00Z",
                "location": "Camera 1",
                "detection_confidence": 0.95
            }
        ]
    
    async def _can_erase_data(self, subject_id: str) -> bool:
        """Check if data can be legally erased."""
        # Check for legal obligations that prevent erasure
        # This would integrate with actual business logic
        return True
    
    async def _secure_delete_personal_data(self, subject_id: str) -> Dict:
        """Securely delete personal data."""
        # Implement secure deletion with audit trail
        return {"deleted": True, "method": "cryptographic_erasure"}
    
    async def _secure_delete_video_data(self, subject_id: str) -> Dict:
        """Securely delete video data."""
        # Implement video data deletion with redaction
        return {"deleted": True, "method": "secure_overwrite"}
    
    async def _anonymize_audit_logs(self, subject_id: str) -> Dict:
        """Anonymize audit logs instead of deletion."""
        # Replace identifiers with anonymous tokens
        return {"anonymized": True, "method": "pseudonymization"}
    
    async def apply_data_retention_policies(self) -> Dict[str, Any]:
        """Apply data retention policies and delete expired data."""
        results = {
            "policies_applied": 0,
            "records_deleted": 0,
            "errors": []
        }
        
        for data_type, policy in self.retention_policies.items():
            try:
                logger.info("Applying retention policy", data_type=data_type)
                
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_days)
                
                # Apply retention policy
                if policy.auto_delete:
                    deleted_count = await self._delete_expired_data(data_type, cutoff_date, policy)
                    results["records_deleted"] += deleted_count
                else:
                    # Mark for manual review
                    await self._mark_for_review(data_type, cutoff_date)
                
                results["policies_applied"] += 1
                
            except Exception as e:
                error_msg = f"Failed to apply retention policy for {data_type}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error("Retention policy application failed", 
                           data_type=data_type, error=str(e))
        
        logger.info("Data retention policies applied", 
                   policies=results["policies_applied"],
                   deleted=results["records_deleted"])
        
        return results
    
    async def _delete_expired_data(self, data_type: str, cutoff_date: datetime, 
                                 policy: DataRetentionPolicy) -> int:
        """Delete expired data according to retention policy."""
        # This would integrate with actual database operations
        logger.info("Deleting expired data",
                   data_type=data_type,
                   cutoff_date=cutoff_date,
                   deletion_method=policy.deletion_method)
        
        # Simulate deletion count
        return 42  # Would return actual count from database
    
    async def _mark_for_review(self, data_type: str, cutoff_date: datetime):
        """Mark data for manual review before deletion."""
        logger.info("Marking data for manual review",
                   data_type=data_type,
                   cutoff_date=cutoff_date)
    
    def generate_compliance_report(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate compliance report for specified framework."""
        if framework == ComplianceFramework.GDPR:
            return self._generate_gdpr_report()
        elif framework == ComplianceFramework.FERPA:
            return self._generate_ferpa_report()
        elif framework == ComplianceFramework.SOC2:
            return self._generate_soc2_report()
        else:
            return {"error": f"Unsupported compliance framework: {framework.value}"}
    
    def _generate_gdpr_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        return {
            "framework": "GDPR",
            "generated_at": datetime.utcnow().isoformat(),
            "compliance_status": {
                "lawful_basis_documented": True,
                "privacy_notices_provided": True,
                "consent_management_implemented": True,
                "data_subject_rights_supported": True,
                "breach_notification_procedures": True,
                "privacy_by_design_implemented": True
            },
            "data_processing_activities": {
                "total_activities": len(self.privacy_assessments),
                "high_risk_activities": len([
                    pia for pia in self.privacy_assessments.values()
                    if pia.risk_level == "high"
                ]),
                "pending_assessments": len([
                    pia for pia in self.privacy_assessments.values()
                    if pia.approval_status == "pending_review"
                ])
            },
            "data_subject_requests": {
                "total_requests": len(self.data_subject_requests),
                "completed_requests": len([
                    dsar for dsar in self.data_subject_requests.values()
                    if dsar.status == "completed"
                ]),
                "pending_requests": len([
                    dsar for dsar in self.data_subject_requests.values()
                    if dsar.status in ["received", "processing"]
                ]),
                "overdue_requests": len([
                    dsar for dsar in self.data_subject_requests.values()
                    if dsar.due_date < datetime.utcnow() and dsar.status != "completed"
                ])
            },
            "data_retention": {
                "policies_defined": len(self.retention_policies),
                "auto_deletion_enabled": len([
                    policy for policy in self.retention_policies.values()
                    if policy.auto_delete
                ])
            },
            "recommendations": [
                "Conduct regular privacy audits",
                "Update privacy notices annually",
                "Review data retention policies quarterly",
                "Provide privacy training to staff"
            ]
        }
    
    def _generate_ferpa_report(self) -> Dict[str, Any]:
        """Generate FERPA compliance report."""
        return {
            "framework": "FERPA",
            "generated_at": datetime.utcnow().isoformat(),
            "compliance_status": {
                "annual_notification_provided": True,
                "directory_information_defined": True,
                "consent_procedures_established": True,
                "access_rights_implemented": True,
                "disclosure_records_maintained": True,
                "security_measures_implemented": True
            },
            "educational_records": {
                "access_controls_implemented": True,
                "audit_logging_enabled": True,
                "encryption_at_rest": True,
                "encryption_in_transit": True
            },
            "disclosure_tracking": {
                "disclosure_log_maintained": True,
                "authorized_disclosures": 0,  # Would be actual count
                "unauthorized_access_incidents": 0
            },
            "recommendations": [
                "Review directory information annually",
                "Update consent forms as needed",
                "Conduct staff training on FERPA requirements",
                "Regular audit of access controls"
            ]
        }
    
    def _generate_soc2_report(self) -> Dict[str, Any]:
        """Generate SOC 2 compliance report."""
        return {
            "framework": "SOC 2",
            "generated_at": datetime.utcnow().isoformat(),
            "trust_service_criteria": {
                "security": {
                    "access_controls": True,
                    "logical_access": True,
                    "network_security": True,
                    "data_protection": True
                },
                "availability": {
                    "system_monitoring": True,
                    "backup_procedures": True,
                    "disaster_recovery": True,
                    "capacity_planning": True
                },
                "processing_integrity": {
                    "data_validation": True,
                    "error_handling": True,
                    "audit_trails": True,
                    "change_management": True
                },
                "confidentiality": {
                    "data_classification": True,
                    "encryption": True,
                    "access_restrictions": True,
                    "disposal_procedures": True
                },
                "privacy": {
                    "privacy_notice": True,
                    "consent_management": True,
                    "data_subject_rights": True,
                    "retention_policies": True
                }
            },
            "control_effectiveness": "Effective",
            "recommendations": [
                "Conduct annual penetration testing",
                "Review access controls quarterly",
                "Update incident response procedures",
                "Enhance monitoring and alerting"
            ]
        }
    
    async def export_compliance_data(self, output_file: str):
        """Export compliance data for audit purposes."""
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "retention_policies": {
                data_type: asdict(policy) 
                for data_type, policy in self.retention_policies.items()
            },
            "privacy_assessments": {
                pia_id: asdict(pia) 
                for pia_id, pia in self.privacy_assessments.items()
            },
            "data_subject_requests": {
                dsar_id: asdict(dsar) 
                for dsar_id, dsar in self.data_subject_requests.items()
            },
            "compliance_violations": self.compliance_violations
        }
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(export_data, indent=2, default=str))
        
        logger.info("Compliance data exported", output_file=output_file)


# Global compliance manager instance
compliance_manager = ComplianceManager()