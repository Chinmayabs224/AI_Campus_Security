"""
FastAPI router for compliance and data protection endpoints.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
import structlog

from auth.dependencies import get_current_user, require_permissions
from .compliance import compliance_manager
from .compliance_monitor import compliance_monitor, ComplianceFramework, DataSubjectRights
from .data_retention import data_retention_service, DataCategory, DeletionMethod
from .privacy_impact_assessment import pia_service, PIAStatus, RiskLevel
from .policy_enforcement import policy_enforcement_engine, PolicyType, ViolationSeverity
from .backup_recovery import backup_recovery_manager

logger = structlog.get_logger()

router = APIRouter()


# Pydantic models for request/response
class DataRetentionPolicyRequest(BaseModel):
    name: str
    data_category: str
    retention_days: int
    deletion_method: str
    legal_basis: str
    auto_delete: bool = True
    requires_approval: bool = False
    review_frequency_days: int = 365
    compliance_frameworks: List[str] = []


class PIARequest(BaseModel):
    title: str
    description: str
    template_name: Optional[str] = None


class ProcessingActivityRequest(BaseModel):
    name: str
    description: str
    data_types: List[str]
    processing_purposes: List[str]
    legal_basis: str
    data_subjects: List[str]
    data_sources: List[str]
    data_recipients: List[str]
    retention_period: str
    cross_border_transfers: bool = False
    automated_decision_making: bool = False
    profiling: bool = False


class DSARRequest(BaseModel):
    request_type: str
    subject_id: str
    subject_email: str
    request_details: str
    framework: str = "gdpr"


class PolicyEvaluationRequest(BaseModel):
    context: Dict[str, Any]


class BackupJobRequest(BaseModel):
    job_id: str


class RecoveryPlanRequest(BaseModel):
    plan_id: str
    recovery_point: Optional[datetime] = None


# Data Retention Endpoints
@router.get("/data-retention/policies")
async def get_retention_policies(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get all data retention policies."""
    return {
        "policies": [
            {
                "id": policy.id,
                "name": policy.name,
                "data_category": policy.data_category.value,
                "retention_days": policy.retention_days,
                "deletion_method": policy.deletion_method.value,
                "auto_delete": policy.auto_delete,
                "requires_approval": policy.requires_approval,
                "compliance_frameworks": policy.compliance_frameworks
            }
            for policy in data_retention_service.policies.values()
        ]
    }


@router.post("/data-retention/apply-policies")
async def apply_retention_policies(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Apply all data retention policies."""
    try:
        results = await data_retention_service.apply_retention_policies()
        return results
    except Exception as e:
        logger.error("Failed to apply retention policies", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-retention/status/{data_id}")
async def get_retention_status(
    data_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get retention status for specific data."""
    status = await data_retention_service.get_retention_status(data_id)
    if not status:
        raise HTTPException(status_code=404, detail="Retention record not found")
    
    return {
        "id": status.id,
        "policy_id": status.policy_id,
        "data_id": status.data_id,
        "data_category": status.data_category.value,
        "created_at": status.created_at.isoformat(),
        "expires_at": status.expires_at.isoformat(),
        "status": status.status.value,
        "deletion_scheduled_at": status.deletion_scheduled_at.isoformat() if status.deletion_scheduled_at else None
    }


@router.post("/data-retention/extend/{record_id}")
async def extend_retention(
    record_id: str,
    additional_days: int = Body(...),
    reason: str = Body(...),
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Extend retention period for specific data."""
    success = await data_retention_service.extend_retention(record_id, additional_days, reason)
    if not success:
        raise HTTPException(status_code=404, detail="Retention record not found")
    
    return {"message": "Retention period extended successfully"}


@router.get("/data-retention/report")
async def get_retention_report(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get comprehensive retention report."""
    return data_retention_service.generate_retention_report()


# Privacy Impact Assessment Endpoints
@router.post("/pia/create")
async def create_pia(
    request: PIARequest,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Create a new Privacy Impact Assessment."""
    try:
        pia = await pia_service.create_assessment(
            title=request.title,
            description=request.description,
            created_by=current_user["user_id"],
            template_name=request.template_name
        )
        
        return {
            "assessment_id": pia.id,
            "title": pia.title,
            "status": pia.status.value,
            "created_at": pia.created_at.isoformat()
        }
    except Exception as e:
        logger.error("Failed to create PIA", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pia/{assessment_id}/activity")
async def add_processing_activity(
    assessment_id: str,
    request: ProcessingActivityRequest,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Add processing activity to PIA."""
    try:
        activity = await pia_service.add_processing_activity(
            assessment_id=assessment_id,
            activity_data=request.dict()
        )
        
        return {
            "activity_id": activity.id,
            "name": activity.name,
            "data_types": [dt.value for dt in activity.data_types],
            "processing_purposes": [pp.value for pp in activity.processing_purposes]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to add processing activity", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pia/{assessment_id}/risk-assessment/{activity_id}")
async def conduct_risk_assessment(
    assessment_id: str,
    activity_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Conduct risk assessment for processing activity."""
    try:
        risks = await pia_service.conduct_risk_assessment(
            assessment_id=assessment_id,
            activity_id=activity_id,
            assessed_by=current_user["user_id"]
        )
        
        return {
            "risks_identified": len(risks),
            "risk_assessments": [
                {
                    "id": risk.id,
                    "risk_category": risk.risk_category,
                    "risk_level": risk.risk_level.value,
                    "likelihood": risk.likelihood,
                    "impact": risk.impact,
                    "mitigation_measures": risk.mitigation_measures
                }
                for risk in risks
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to conduct risk assessment", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pia/{assessment_id}/submit")
async def submit_pia_for_review(
    assessment_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Submit PIA for review."""
    success = await pia_service.submit_for_review(assessment_id, current_user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Assessment not found or not in draft status")
    
    return {"message": "PIA submitted for review successfully"}


@router.post("/pia/{assessment_id}/approve")
async def approve_pia(
    assessment_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:approve"]))
):
    """Approve a Privacy Impact Assessment."""
    success = await pia_service.approve_assessment(assessment_id, current_user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Assessment not found or not under review")
    
    return {"message": "PIA approved successfully"}


@router.get("/pia/{assessment_id}/report")
async def get_pia_report(
    assessment_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get comprehensive PIA report."""
    try:
        return await pia_service.generate_pia_report(assessment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate PIA report", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Data Subject Rights Endpoints
@router.post("/dsar/create")
async def create_dsar(
    request: DSARRequest,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Create a Data Subject Access Request."""
    try:
        framework = ComplianceFramework(request.framework.lower())
        request_type = DataSubjectRights(request.request_type.lower())
        
        dsar = await compliance_manager.process_data_subject_request(
            request_type=request_type,
            subject_id=request.subject_id,
            subject_email=request.subject_email,
            request_details=request.request_details
        )
        
        return {
            "request_id": dsar.id,
            "request_type": dsar.request_type.value,
            "status": dsar.status,
            "due_date": dsar.due_date.isoformat(),
            "created_at": dsar.created_at.isoformat()
        }
    except Exception as e:
        logger.error("Failed to create DSAR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dsar/{request_id}")
async def get_dsar_status(
    request_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get DSAR status and response."""
    dsar = compliance_manager.data_subject_requests.get(request_id)
    if not dsar:
        raise HTTPException(status_code=404, detail="DSAR not found")
    
    return {
        "request_id": dsar.id,
        "request_type": dsar.request_type.value,
        "subject_id": dsar.subject_id,
        "status": dsar.status,
        "created_at": dsar.created_at.isoformat(),
        "due_date": dsar.due_date.isoformat(),
        "completed_at": dsar.completed_at.isoformat() if dsar.completed_at else None,
        "response_data": dsar.response_data if dsar.status == "completed" else None
    }


# Compliance Monitoring Endpoints
@router.post("/compliance/check")
async def run_compliance_check(
    framework: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Run compliance checks."""
    try:
        framework_enum = ComplianceFramework(framework.lower()) if framework else None
        results = await compliance_monitor.run_compliance_check(framework_enum)
        return results
    except Exception as e:
        logger.error("Failed to run compliance check", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/report")
async def get_compliance_report(
    framework: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get comprehensive compliance report."""
    try:
        framework_enum = ComplianceFramework(framework.lower()) if framework else None
        return await compliance_monitor.generate_compliance_report(framework_enum)
    except Exception as e:
        logger.error("Failed to generate compliance report", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/violations")
async def get_compliance_violations(
    severity: Optional[str] = Query(None),
    framework: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get compliance violations."""
    violations = list(compliance_monitor.violations.values())
    
    if severity:
        violations = [v for v in violations if v.severity == severity]
    
    if framework:
        framework_enum = ComplianceFramework(framework.lower())
        violations = [v for v in violations if v.framework == framework_enum]
    
    return {
        "violations": [
            {
                "id": v.id,
                "rule_id": v.rule_id,
                "framework": v.framework.value,
                "title": v.title,
                "severity": v.severity,
                "detected_at": v.detected_at.isoformat(),
                "status": v.status,
                "remediation_deadline": v.remediation_deadline.isoformat() if v.remediation_deadline else None
            }
            for v in violations
        ]
    }


# Policy Enforcement Endpoints
@router.post("/policy/evaluate")
async def evaluate_policies(
    request: PolicyEvaluationRequest,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Evaluate policies against given context."""
    try:
        violations = await policy_enforcement_engine.evaluate_policies(request.context)
        return {
            "violations_detected": len(violations),
            "violations": [
                {
                    "id": v.id,
                    "rule_id": v.rule_id,
                    "policy_type": v.policy_type.value,
                    "severity": v.severity.value,
                    "action_taken": v.action_taken.value,
                    "resource_id": v.resource_id
                }
                for v in violations
            ]
        }
    except Exception as e:
        logger.error("Failed to evaluate policies", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policy/metrics")
async def get_policy_metrics(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get policy enforcement metrics."""
    metrics = policy_enforcement_engine.get_metrics()
    return {
        "total_rules": metrics.total_rules,
        "active_rules": metrics.active_rules,
        "violations_detected": metrics.violations_detected,
        "violations_resolved": metrics.violations_resolved,
        "violations_by_severity": metrics.violations_by_severity,
        "violations_by_type": metrics.violations_by_type,
        "enforcement_actions": metrics.enforcement_actions,
        "last_updated": metrics.last_updated.isoformat()
    }


@router.post("/policy/violations/{violation_id}/resolve")
async def resolve_policy_violation(
    violation_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:write"]))
):
    """Resolve a policy violation."""
    success = await policy_enforcement_engine.resolve_violation(violation_id, current_user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    return {"message": "Violation resolved successfully"}


@router.get("/policy/enforcement-report")
async def get_enforcement_report(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get comprehensive policy enforcement report."""
    try:
        return await policy_enforcement_engine.generate_enforcement_report()
    except Exception as e:
        logger.error("Failed to generate enforcement report", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Backup and Recovery Endpoints
@router.post("/backup/execute")
async def execute_backup(
    request: BackupJobRequest,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["system:backup"]))
):
    """Execute a backup job."""
    try:
        result = await backup_recovery_manager.execute_backup_job(request.job_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to execute backup", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backup/status")
async def get_backup_status(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["system:read"]))
):
    """Get backup system status."""
    return backup_recovery_manager.get_backup_status()


@router.post("/recovery/execute")
async def execute_recovery(
    request: RecoveryPlanRequest,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["system:recovery"]))
):
    """Execute a disaster recovery plan."""
    try:
        result = await backup_recovery_manager.execute_recovery_plan(
            request.plan_id,
            request.recovery_point
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to execute recovery plan", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recovery/test/{plan_id}")
async def test_recovery_plan(
    plan_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["system:test"]))
):
    """Test a disaster recovery plan."""
    try:
        result = await backup_recovery_manager.test_recovery_plan(plan_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to test recovery plan", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recovery/status")
async def get_recovery_status(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["system:read"]))
):
    """Get recovery system status."""
    return backup_recovery_manager.get_recovery_status()


# General Compliance Endpoints
@router.get("/compliance/frameworks")
async def get_supported_frameworks(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get list of supported compliance frameworks."""
    return {
        "frameworks": [
            {
                "code": framework.value,
                "name": framework.name,
                "description": f"{framework.name} compliance framework"
            }
            for framework in ComplianceFramework
        ]
    }


@router.get("/compliance/dashboard")
async def get_compliance_dashboard(
    current_user: dict = Depends(get_current_user),
    _: None = Depends(require_permissions(["compliance:read"]))
):
    """Get compliance dashboard data."""
    try:
        # Get data from all compliance services
        retention_report = data_retention_service.generate_retention_report()
        policy_metrics = policy_enforcement_engine.get_metrics()
        backup_status = backup_recovery_manager.get_backup_status()
        recovery_status = backup_recovery_manager.get_recovery_status()
        
        # Get recent violations
        recent_violations = [
            {
                "id": v.id,
                "policy_type": v.policy_type.value,
                "severity": v.severity.value,
                "detected_at": v.detected_at.isoformat(),
                "resolved": v.resolved
            }
            for v in list(policy_enforcement_engine.violations.values())[-10:]
        ]
        
        return {
            "data_retention": {
                "total_records": retention_report["total_records"],
                "expiring_soon": retention_report["expiring_soon"],
                "overdue_records": retention_report["overdue_records"]
            },
            "policy_enforcement": {
                "active_rules": policy_metrics.active_rules,
                "violations_detected": policy_metrics.violations_detected,
                "violations_resolved": policy_metrics.violations_resolved
            },
            "backup_recovery": {
                "backup_jobs": backup_status["total_jobs"],
                "failed_backups": backup_status["failed_jobs"],
                "recovery_plans": recovery_status["total_plans"],
                "tested_plans": recovery_status["tested_plans"]
            },
            "recent_violations": recent_violations,
            "compliance_score": min(100, max(0, 100 - len([v for v in recent_violations if not v["resolved"]])))
        }
    except Exception as e:
        logger.error("Failed to get compliance dashboard", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))