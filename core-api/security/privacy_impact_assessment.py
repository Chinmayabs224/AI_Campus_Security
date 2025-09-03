"""
Privacy Impact Assessment (PIA) service for campus security system.
Supports GDPR, FERPA, COPPA compliance requirements.
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

logger = structlog.get_logger()


class PIAStatus(Enum):
    """Privacy Impact Assessment status."""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"
    EXPIRED = "expired"


class RiskLevel(Enum):
    """Privacy risk levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataType(Enum):
    """Types of personal data processed."""
    BASIC_PERSONAL = "basic_personal"
    SENSITIVE_PERSONAL = "sensitive_personal"
    BIOMETRIC = "biometric"
    LOCATION = "location"
    BEHAVIORAL = "behavioral"
    HEALTH = "health"
    FINANCIAL = "financial"
    EDUCATIONAL = "educational"
    EMPLOYMENT = "employment"
    CRIMINAL = "criminal"


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    SECURITY_MONITORING = "security_monitoring"
    ACCESS_CONTROL = "access_control"
    INCIDENT_INVESTIGATION = "incident_investigation"
    COMPLIANCE_REPORTING = "compliance_reporting"
    SYSTEM_ADMINISTRATION = "system_administration"
    ANALYTICS = "analytics"
    TRAINING = "training"
    RESEARCH = "research"


@dataclass
class DataProcessingActivity:
    """Data processing activity definition."""
    id: str
    name: str
    description: str
    data_types: List[DataType]
    processing_purposes: List[ProcessingPurpose]
    legal_basis: str
    data_subjects: List[str]
    data_sources: List[str]
    data_recipients: List[str]
    retention_period: str
    cross_border_transfers: bool
    automated_decision_making: bool
    profiling: bool
    created_at: datetime
    updated_at: Optional[datetime] = None


@dataclass
class RiskAssessment:
    """Privacy risk assessment."""
    id: str
    activity_id: str
    risk_category: str
    risk_description: str
    likelihood: str  # very_low, low, medium, high, very_high
    impact: str  # negligible, minor, moderate, major, severe
    risk_level: RiskLevel
    mitigation_measures: List[str]
    residual_risk_level: RiskLevel
    created_at: datetime
    assessed_by: Optional[str] = None


@dataclass
class PrivacyImpactAssessment:
    """Complete Privacy Impact Assessment."""
    id: str
    title: str
    description: str
    version: str
    status: PIAStatus
    processing_activities: List[DataProcessingActivity]
    risk_assessments: List[RiskAssessment]
    overall_risk_level: RiskLevel
    mitigation_plan: List[str]
    monitoring_measures: List[str]
    review_date: datetime
    approval_required: bool
    created_at: datetime
    created_by: str
    reviewed_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    compliance_frameworks: List[str] = None
    
    def __post_init__(self):
        if self.compliance_frameworks is None:
            self.compliance_frameworks = []


class PrivacyImpactAssessmentService:
    """Service for managing Privacy Impact Assessments."""
    
    def __init__(self):
        self.assessments: Dict[str, PrivacyImpactAssessment] = {}
        self.templates: Dict[str, Dict] = {}
        self.initialize_templates()
    
    def initialize_templates(self):
        """Initialize PIA templates for different scenarios."""
        self.templates = {
            "video_surveillance": {
                "title": "Video Surveillance System PIA",
                "description": "Privacy impact assessment for video surveillance implementation",
                "data_types": [DataType.BIOMETRIC, DataType.LOCATION, DataType.BEHAVIORAL],
                "processing_purposes": [ProcessingPurpose.SECURITY_MONITORING, ProcessingPurpose.INCIDENT_INVESTIGATION],
                "legal_basis": "Legitimate interest for security purposes",
                "high_risk_factors": [
                    "Biometric data processing",
                    "Continuous monitoring",
                    "Public space surveillance",
                    "Potential for function creep"
                ]
            },
            "access_control": {
                "title": "Access Control System PIA",
                "description": "Privacy impact assessment for access control implementation",
                "data_types": [DataType.BASIC_PERSONAL, DataType.BIOMETRIC, DataType.LOCATION],
                "processing_purposes": [ProcessingPurpose.ACCESS_CONTROL, ProcessingPurpose.SECURITY_MONITORING],
                "legal_basis": "Legitimate interest and contractual necessity",
                "high_risk_factors": [
                    "Biometric authentication",
                    "Location tracking",
                    "Access pattern analysis"
                ]
            },
            "incident_management": {
                "title": "Incident Management System PIA",
                "description": "Privacy impact assessment for incident management processes",
                "data_types": [DataType.BASIC_PERSONAL, DataType.SENSITIVE_PERSONAL, DataType.BEHAVIORAL],
                "processing_purposes": [ProcessingPurpose.INCIDENT_INVESTIGATION, ProcessingPurpose.COMPLIANCE_REPORTING],
                "legal_basis": "Legitimate interest and legal obligation",
                "high_risk_factors": [
                    "Sensitive personal data",
                    "Detailed behavioral analysis",
                    "Long-term data retention"
                ]
            }
        }
        
        logger.info("PIA templates initialized", count=len(self.templates))
    
    async def create_assessment(self, title: str, description: str, created_by: str,
                              template_name: Optional[str] = None) -> PrivacyImpactAssessment:
        """Create a new Privacy Impact Assessment."""
        assessment_id = str(uuid4())
        
        # Use template if specified
        template = self.templates.get(template_name, {}) if template_name else {}
        
        assessment = PrivacyImpactAssessment(
            id=assessment_id,
            title=template.get("title", title),
            description=template.get("description", description),
            version="1.0",
            status=PIAStatus.DRAFT,
            processing_activities=[],
            risk_assessments=[],
            overall_risk_level=RiskLevel.MEDIUM,  # Default, will be calculated
            mitigation_plan=[],
            monitoring_measures=[],
            review_date=datetime.utcnow() + timedelta(days=365),  # Annual review
            approval_required=True,
            created_at=datetime.utcnow(),
            created_by=created_by,
            compliance_frameworks=["GDPR", "FERPA"]
        )
        
        self.assessments[assessment_id] = assessment
        
        logger.info("Privacy Impact Assessment created",
                   assessment_id=assessment_id,
                   title=title,
                   created_by=created_by)
        
        return assessment
    
    async def add_processing_activity(self, assessment_id: str, 
                                    activity_data: Dict[str, Any]) -> DataProcessingActivity:
        """Add a data processing activity to an assessment."""
        assessment = self.assessments.get(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        activity_id = str(uuid4())
        
        activity = DataProcessingActivity(
            id=activity_id,
            name=activity_data["name"],
            description=activity_data["description"],
            data_types=[DataType(dt) for dt in activity_data["data_types"]],
            processing_purposes=[ProcessingPurpose(pp) for pp in activity_data["processing_purposes"]],
            legal_basis=activity_data["legal_basis"],
            data_subjects=activity_data["data_subjects"],
            data_sources=activity_data["data_sources"],
            data_recipients=activity_data["data_recipients"],
            retention_period=activity_data["retention_period"],
            cross_border_transfers=activity_data.get("cross_border_transfers", False),
            automated_decision_making=activity_data.get("automated_decision_making", False),
            profiling=activity_data.get("profiling", False),
            created_at=datetime.utcnow()
        )
        
        assessment.processing_activities.append(activity)
        
        # Recalculate risk level
        await self._calculate_overall_risk(assessment)
        
        logger.info("Processing activity added to PIA",
                   assessment_id=assessment_id,
                   activity_id=activity_id,
                   activity_name=activity.name)
        
        return activity
    
    async def conduct_risk_assessment(self, assessment_id: str, activity_id: str,
                                    assessed_by: str) -> List[RiskAssessment]:
        """Conduct risk assessment for a processing activity."""
        assessment = self.assessments.get(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        activity = next((a for a in assessment.processing_activities if a.id == activity_id), None)
        if not activity:
            raise ValueError(f"Activity {activity_id} not found")
        
        # Generate risk assessments based on activity characteristics
        risk_assessments = []
        
        # Risk categories to assess
        risk_categories = [
            "data_breach",
            "unauthorized_access",
            "function_creep",
            "discrimination",
            "surveillance_overreach",
            "data_quality",
            "vendor_risks",
            "technical_failures"
        ]
        
        for category in risk_categories:
            risk_assessment = await self._assess_risk_category(activity, category, assessed_by)
            if risk_assessment:
                risk_assessments.append(risk_assessment)
                assessment.risk_assessments.append(risk_assessment)
        
        # Recalculate overall risk
        await self._calculate_overall_risk(assessment)
        
        logger.info("Risk assessment completed",
                   assessment_id=assessment_id,
                   activity_id=activity_id,
                   risks_identified=len(risk_assessments))
        
        return risk_assessments
    
    async def _assess_risk_category(self, activity: DataProcessingActivity, 
                                  category: str, assessed_by: str) -> Optional[RiskAssessment]:
        """Assess risk for a specific category."""
        risk_id = str(uuid4())
        
        # Risk assessment logic based on category and activity characteristics
        risk_factors = self._identify_risk_factors(activity, category)
        
        if not risk_factors["applicable"]:
            return None
        
        likelihood = self._calculate_likelihood(activity, category, risk_factors)
        impact = self._calculate_impact(activity, category, risk_factors)
        risk_level = self._calculate_risk_level(likelihood, impact)
        
        mitigation_measures = self._generate_mitigation_measures(activity, category, risk_level)
        residual_risk_level = self._calculate_residual_risk(risk_level, mitigation_measures)
        
        return RiskAssessment(
            id=risk_id,
            activity_id=activity.id,
            risk_category=category,
            risk_description=risk_factors["description"],
            likelihood=likelihood,
            impact=impact,
            risk_level=risk_level,
            mitigation_measures=mitigation_measures,
            residual_risk_level=residual_risk_level,
            created_at=datetime.utcnow(),
            assessed_by=assessed_by
        )
    
    def _identify_risk_factors(self, activity: DataProcessingActivity, category: str) -> Dict[str, Any]:
        """Identify risk factors for a specific category."""
        risk_factors = {"applicable": False, "description": "", "factors": []}
        
        if category == "data_breach":
            if DataType.BIOMETRIC in activity.data_types or DataType.SENSITIVE_PERSONAL in activity.data_types:
                risk_factors["applicable"] = True
                risk_factors["description"] = "Risk of unauthorized access to sensitive biometric or personal data"
                risk_factors["factors"] = ["sensitive_data", "high_value_target"]
        
        elif category == "unauthorized_access":
            if ProcessingPurpose.SECURITY_MONITORING in activity.processing_purposes:
                risk_factors["applicable"] = True
                risk_factors["description"] = "Risk of unauthorized access to security monitoring data"
                risk_factors["factors"] = ["privileged_access", "monitoring_data"]
        
        elif category == "function_creep":
            if ProcessingPurpose.ANALYTICS in activity.processing_purposes:
                risk_factors["applicable"] = True
                risk_factors["description"] = "Risk of using data beyond original security purposes"
                risk_factors["factors"] = ["analytics_capability", "broad_purposes"]
        
        elif category == "discrimination":
            if activity.automated_decision_making or activity.profiling:
                risk_factors["applicable"] = True
                risk_factors["description"] = "Risk of discriminatory automated decision-making"
                risk_factors["factors"] = ["automated_decisions", "profiling"]
        
        elif category == "surveillance_overreach":
            if DataType.LOCATION in activity.data_types or DataType.BEHAVIORAL in activity.data_types:
                risk_factors["applicable"] = True
                risk_factors["description"] = "Risk of excessive surveillance and privacy intrusion"
                risk_factors["factors"] = ["location_tracking", "behavioral_monitoring"]
        
        return risk_factors
    
    def _calculate_likelihood(self, activity: DataProcessingActivity, category: str, 
                            risk_factors: Dict[str, Any]) -> str:
        """Calculate likelihood of risk occurrence."""
        score = 0
        
        # Base score from risk factors
        score += len(risk_factors["factors"])
        
        # Adjust based on data types
        if DataType.BIOMETRIC in activity.data_types:
            score += 2
        if DataType.SENSITIVE_PERSONAL in activity.data_types:
            score += 1
        
        # Adjust based on processing characteristics
        if activity.automated_decision_making:
            score += 2
        if activity.cross_border_transfers:
            score += 1
        
        # Convert score to likelihood
        if score >= 6:
            return "very_high"
        elif score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        elif score >= 1:
            return "low"
        else:
            return "very_low"
    
    def _calculate_impact(self, activity: DataProcessingActivity, category: str,
                        risk_factors: Dict[str, Any]) -> str:
        """Calculate impact of risk occurrence."""
        score = 0
        
        # Base score from data sensitivity
        if DataType.BIOMETRIC in activity.data_types:
            score += 3
        if DataType.SENSITIVE_PERSONAL in activity.data_types:
            score += 2
        if DataType.HEALTH in activity.data_types:
            score += 3
        
        # Adjust based on data subjects
        if "children" in activity.data_subjects:
            score += 2
        if "employees" in activity.data_subjects:
            score += 1
        
        # Adjust based on scale
        if len(activity.data_subjects) > 1000:  # Assuming large scale
            score += 2
        
        # Convert score to impact
        if score >= 8:
            return "severe"
        elif score >= 6:
            return "major"
        elif score >= 4:
            return "moderate"
        elif score >= 2:
            return "minor"
        else:
            return "negligible"
    
    def _calculate_risk_level(self, likelihood: str, impact: str) -> RiskLevel:
        """Calculate overall risk level from likelihood and impact."""
        likelihood_scores = {"very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}
        impact_scores = {"negligible": 1, "minor": 2, "moderate": 3, "major": 4, "severe": 5}
        
        total_score = likelihood_scores[likelihood] * impact_scores[impact]
        
        if total_score >= 20:
            return RiskLevel.CRITICAL
        elif total_score >= 12:
            return RiskLevel.HIGH
        elif total_score >= 6:
            return RiskLevel.MEDIUM
        elif total_score >= 3:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _generate_mitigation_measures(self, activity: DataProcessingActivity, 
                                    category: str, risk_level: RiskLevel) -> List[str]:
        """Generate appropriate mitigation measures."""
        measures = []
        
        # Base measures for all risks
        measures.extend([
            "Implement appropriate technical and organizational measures",
            "Conduct regular security assessments",
            "Provide staff training on privacy and security"
        ])
        
        # Category-specific measures
        if category == "data_breach":
            measures.extend([
                "Implement encryption at rest and in transit",
                "Use strong access controls and authentication",
                "Implement data loss prevention (DLP) tools",
                "Establish incident response procedures"
            ])
        
        elif category == "unauthorized_access":
            measures.extend([
                "Implement role-based access control (RBAC)",
                "Use multi-factor authentication",
                "Implement audit logging and monitoring",
                "Regular access reviews and deprovisioning"
            ])
        
        elif category == "function_creep":
            measures.extend([
                "Clearly define and document processing purposes",
                "Implement purpose limitation controls",
                "Regular review of data usage",
                "Obtain additional consent for new purposes"
            ])
        
        elif category == "discrimination":
            measures.extend([
                "Implement algorithmic fairness testing",
                "Provide human oversight for automated decisions",
                "Implement bias detection and mitigation",
                "Regular algorithm auditing"
            ])
        
        elif category == "surveillance_overreach":
            measures.extend([
                "Implement privacy zones and masking",
                "Use data minimization principles",
                "Implement proportionality assessments",
                "Regular necessity and proportionality reviews"
            ])
        
        # Risk level specific measures
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            measures.extend([
                "Conduct Data Protection Impact Assessment",
                "Consult with Data Protection Authority if required",
                "Implement enhanced monitoring and controls",
                "Consider privacy-enhancing technologies"
            ])
        
        return measures
    
    def _calculate_residual_risk(self, original_risk: RiskLevel, 
                               mitigation_measures: List[str]) -> RiskLevel:
        """Calculate residual risk after mitigation measures."""
        # Simple reduction based on number and quality of measures
        reduction_factor = min(len(mitigation_measures) * 0.1, 0.5)  # Max 50% reduction
        
        risk_values = {
            RiskLevel.MINIMAL: 1,
            RiskLevel.LOW: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.HIGH: 4,
            RiskLevel.CRITICAL: 5
        }
        
        original_value = risk_values[original_risk]
        reduced_value = max(1, int(original_value * (1 - reduction_factor)))
        
        value_to_risk = {v: k for k, v in risk_values.items()}
        return value_to_risk[reduced_value]
    
    async def _calculate_overall_risk(self, assessment: PrivacyImpactAssessment):
        """Calculate overall risk level for the assessment."""
        if not assessment.risk_assessments:
            assessment.overall_risk_level = RiskLevel.LOW
            return
        
        # Find highest risk level
        risk_levels = [ra.risk_level for ra in assessment.risk_assessments]
        
        if RiskLevel.CRITICAL in risk_levels:
            assessment.overall_risk_level = RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            assessment.overall_risk_level = RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            assessment.overall_risk_level = RiskLevel.MEDIUM
        elif RiskLevel.LOW in risk_levels:
            assessment.overall_risk_level = RiskLevel.LOW
        else:
            assessment.overall_risk_level = RiskLevel.MINIMAL
        
        # Generate mitigation plan
        assessment.mitigation_plan = await self._generate_mitigation_plan(assessment)
        assessment.monitoring_measures = await self._generate_monitoring_measures(assessment)
    
    async def _generate_mitigation_plan(self, assessment: PrivacyImpactAssessment) -> List[str]:
        """Generate comprehensive mitigation plan."""
        all_measures = []
        
        for risk_assessment in assessment.risk_assessments:
            all_measures.extend(risk_assessment.mitigation_measures)
        
        # Remove duplicates and prioritize
        unique_measures = list(set(all_measures))
        
        # Add assessment-level measures
        if assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            unique_measures.extend([
                "Establish privacy governance framework",
                "Implement privacy by design principles",
                "Conduct regular privacy audits",
                "Establish data protection officer oversight"
            ])
        
        return unique_measures
    
    async def _generate_monitoring_measures(self, assessment: PrivacyImpactAssessment) -> List[str]:
        """Generate monitoring measures for ongoing compliance."""
        measures = [
            "Regular review of processing activities",
            "Monitor data subject complaints and requests",
            "Track privacy metrics and KPIs",
            "Conduct periodic risk reassessments"
        ]
        
        if assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            measures.extend([
                "Continuous monitoring of data processing",
                "Automated privacy compliance checking",
                "Regular third-party privacy audits",
                "Real-time privacy violation detection"
            ])
        
        return measures
    
    async def submit_for_review(self, assessment_id: str, submitted_by: str) -> bool:
        """Submit assessment for review."""
        assessment = self.assessments.get(assessment_id)
        if not assessment:
            return False
        
        if assessment.status != PIAStatus.DRAFT:
            return False
        
        assessment.status = PIAStatus.UNDER_REVIEW
        
        logger.info("PIA submitted for review",
                   assessment_id=assessment_id,
                   submitted_by=submitted_by)
        
        return True
    
    async def approve_assessment(self, assessment_id: str, approved_by: str) -> bool:
        """Approve a Privacy Impact Assessment."""
        assessment = self.assessments.get(assessment_id)
        if not assessment:
            return False
        
        if assessment.status != PIAStatus.UNDER_REVIEW:
            return False
        
        assessment.status = PIAStatus.APPROVED
        assessment.approved_by = approved_by
        assessment.approved_at = datetime.utcnow()
        
        logger.info("PIA approved",
                   assessment_id=assessment_id,
                   approved_by=approved_by)
        
        return True
    
    async def generate_pia_report(self, assessment_id: str) -> Dict[str, Any]:
        """Generate comprehensive PIA report."""
        assessment = self.assessments.get(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        return {
            "assessment_id": assessment.id,
            "title": assessment.title,
            "description": assessment.description,
            "version": assessment.version,
            "status": assessment.status.value,
            "overall_risk_level": assessment.overall_risk_level.value,
            "created_at": assessment.created_at.isoformat(),
            "created_by": assessment.created_by,
            "approved_by": assessment.approved_by,
            "approved_at": assessment.approved_at.isoformat() if assessment.approved_at else None,
            "review_date": assessment.review_date.isoformat(),
            "compliance_frameworks": assessment.compliance_frameworks,
            "processing_activities": [
                {
                    "id": activity.id,
                    "name": activity.name,
                    "description": activity.description,
                    "data_types": [dt.value for dt in activity.data_types],
                    "processing_purposes": [pp.value for pp in activity.processing_purposes],
                    "legal_basis": activity.legal_basis,
                    "data_subjects": activity.data_subjects,
                    "retention_period": activity.retention_period,
                    "cross_border_transfers": activity.cross_border_transfers,
                    "automated_decision_making": activity.automated_decision_making,
                    "profiling": activity.profiling
                }
                for activity in assessment.processing_activities
            ],
            "risk_assessments": [
                {
                    "id": risk.id,
                    "risk_category": risk.risk_category,
                    "risk_description": risk.risk_description,
                    "likelihood": risk.likelihood,
                    "impact": risk.impact,
                    "risk_level": risk.risk_level.value,
                    "mitigation_measures": risk.mitigation_measures,
                    "residual_risk_level": risk.residual_risk_level.value
                }
                for risk in assessment.risk_assessments
            ],
            "mitigation_plan": assessment.mitigation_plan,
            "monitoring_measures": assessment.monitoring_measures
        }
    
    async def export_assessment(self, assessment_id: str, output_file: str):
        """Export PIA to file."""
        report = await self.generate_pia_report(assessment_id)
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        logger.info("PIA exported", assessment_id=assessment_id, output_file=output_file)


# Global PIA service instance
pia_service = PrivacyImpactAssessmentService()