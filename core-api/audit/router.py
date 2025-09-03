"""
Audit logging API endpoints.
"""
from datetime import datetime, timedelta
from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db_session
from auth.dependencies import get_current_user, require_role
from auth.models import User, UserRole
from .service import audit_service
from .models import (
    AuditLogFilter, AuditLogResponse, AuditStats, ComplianceReport,
    ComplianceTag, DSARRequest, DataRetentionPolicy
)

router = APIRouter()


@router.get("/logs", response_model=List[AuditLogResponse])
async def search_audit_logs(
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by user ID"),
    username: Optional[str] = Query(None, description="Filter by username"),
    action: Optional[str] = Query(None, description="Filter by action"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    start_time: Optional[datetime] = Query(None, description="Start time for filtering"),
    end_time: Optional[datetime] = Query(None, description="End time for filtering"),
    ip_address: Optional[str] = Query(None, description="Filter by IP address"),
    success: Optional[bool] = Query(None, description="Filter by success status"),
    compliance_tag: Optional[str] = Query(None, description="Filter by compliance tag"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    contains_pii: Optional[bool] = Query(None, description="Filter by PII involvement"),
    data_classification: Optional[str] = Query(None, description="Filter by data classification"),
    limit: int = Query(100, ge=1, le=1000, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Search audit logs with filtering and pagination.
    Requires admin or security supervisor role.
    """
    # Check permissions
    if current_user.role not in [UserRole.ADMIN, UserRole.SECURITY_SUPERVISOR]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to access audit logs"
        )
    
    # Create filter object
    filters = AuditLogFilter(
        user_id=user_id,
        username=username,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        start_time=start_time,
        end_time=end_time,
        ip_address=ip_address,
        success=success,
        compliance_tag=compliance_tag,
        risk_level=risk_level,
        contains_pii=contains_pii,
        data_classification=data_classification,
        limit=limit,
        offset=offset
    )
    
    # Log the audit log access
    await audit_service.log_action(
        action="audit_log_search",
        user_id=current_user.id,
        username=current_user.username,
        resource_type="audit_log",
        compliance_tags=[ComplianceTag.GDPR],
        risk_level="medium",
        business_justification="Security audit and compliance review",
        metadata={
            "filters": filters.model_dump(exclude_none=True)
        }
    )
    
    return await audit_service.search_audit_logs(filters, session)


@router.get("/stats", response_model=AuditStats)
async def get_audit_statistics(
    start_time: Optional[datetime] = Query(None, description="Start time for statistics"),
    end_time: Optional[datetime] = Query(None, description="End time for statistics"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Get audit statistics for a time period.
    Requires admin or security supervisor role.
    """
    # Check permissions
    if current_user.role not in [UserRole.ADMIN, UserRole.SECURITY_SUPERVISOR]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to access audit statistics"
        )
    
    # Default to last 30 days if no time range specified
    if not start_time:
        start_time = datetime.utcnow() - timedelta(days=30)
    if not end_time:
        end_time = datetime.utcnow()
    
    # Log the statistics access
    await audit_service.log_action(
        action="audit_stats_access",
        user_id=current_user.id,
        username=current_user.username,
        resource_type="audit_log",
        compliance_tags=[ComplianceTag.GDPR],
        risk_level="low",
        business_justification="Security monitoring and reporting",
        metadata={
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
    )
    
    return await audit_service.get_audit_stats(start_time, end_time, session)


@router.post("/compliance-report", response_model=ComplianceReport)
async def generate_compliance_report(
    framework: ComplianceTag,
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Generate a compliance report for a specific framework.
    Requires admin role.
    """
    # Check permissions
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to generate compliance reports"
        )
    
    # Validate date range
    if end_date <= start_date:
        raise HTTPException(
            status_code=400,
            detail="End date must be after start date"
        )
    
    # Validate date range is not too large (max 1 year)
    if (end_date - start_date).days > 365:
        raise HTTPException(
            status_code=400,
            detail="Date range cannot exceed 365 days"
        )
    
    # Log the compliance report generation
    await audit_service.log_action(
        action="compliance_report_generate",
        user_id=current_user.id,
        username=current_user.username,
        resource_type="audit_log",
        compliance_tags=[framework],
        risk_level="medium",
        business_justification="Compliance reporting and audit requirements",
        metadata={
            "framework": str(framework),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    )
    
    return await audit_service.generate_compliance_report(
        framework, start_date, end_date, current_user.username, session
    )


@router.post("/dsar-request")
async def process_dsar_request(
    dsar_request: DSARRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Process a Data Subject Access Request (DSAR).
    Requires admin or security supervisor role.
    """
    # Check permissions
    if current_user.role not in [UserRole.ADMIN, UserRole.SECURITY_SUPERVISOR]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to process DSAR requests"
        )
    
    # Log the DSAR request processing
    await audit_service.log_action(
        action="dsar_request_process",
        user_id=current_user.id,
        username=current_user.username,
        resource_type="user",
        resource_id=dsar_request.subject_identifier,
        compliance_tags=[ComplianceTag.GDPR],
        risk_level="high",
        business_justification="Legal compliance - GDPR data subject rights",
        contains_pii=True,
        metadata={
            "request_id": str(dsar_request.request_id),
            "request_type": dsar_request.request_type,
            "requested_data_types": dsar_request.requested_data_types
        }
    )
    
    result = await audit_service.process_dsar_request(dsar_request, session)
    
    return {
        "message": "DSAR request processed successfully",
        "request_id": str(dsar_request.request_id),
        "result": result
    }


@router.get("/my-activity", response_model=List[AuditLogResponse])
async def get_my_audit_activity(
    start_time: Optional[datetime] = Query(None, description="Start time for filtering"),
    end_time: Optional[datetime] = Query(None, description="End time for filtering"),
    limit: int = Query(50, ge=1, le=500, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Get audit logs for the current user's activity.
    Users can view their own audit trail.
    """
    # Default to last 30 days if no time range specified
    if not start_time:
        start_time = datetime.utcnow() - timedelta(days=30)
    if not end_time:
        end_time = datetime.utcnow()
    
    # Create filter for current user only
    filters = AuditLogFilter(
        user_id=current_user.id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset
    )
    
    # Log the self-audit access
    await audit_service.log_action(
        action="self_audit_access",
        user_id=current_user.id,
        username=current_user.username,
        resource_type="audit_log",
        compliance_tags=[ComplianceTag.GDPR],
        risk_level="low",
        business_justification="User accessing their own audit trail",
        metadata={
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
    )
    
    return await audit_service.search_audit_logs(filters, session)


@router.post("/cleanup")
async def cleanup_expired_logs(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Trigger cleanup of expired audit logs.
    Requires admin role.
    """
    # Check permissions
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to cleanup audit logs"
        )
    
    # Log the cleanup initiation
    await audit_service.log_action(
        action="audit_cleanup_initiate",
        user_id=current_user.id,
        username=current_user.username,
        resource_type="audit_log",
        compliance_tags=[ComplianceTag.GDPR],
        risk_level="medium",
        business_justification="Data retention policy enforcement"
    )
    
    # Run cleanup in background
    background_tasks.add_task(audit_service.cleanup_expired_logs)
    
    return {
        "message": "Audit log cleanup initiated",
        "status": "running"
    }


@router.get("/export")
async def export_audit_logs(
    format: str = Query("csv", description="Export format (csv, json)"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by user ID"),
    start_time: Optional[datetime] = Query(None, description="Start time for filtering"),
    end_time: Optional[datetime] = Query(None, description="End time for filtering"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session)
):
    """
    Export audit logs in specified format.
    Requires admin or security supervisor role.
    """
    # Check permissions
    if current_user.role not in [UserRole.ADMIN, UserRole.SECURITY_SUPERVISOR]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to export audit logs"
        )
    
    # Validate format
    if format not in ["csv", "json"]:
        raise HTTPException(
            status_code=400,
            detail="Format must be 'csv' or 'json'"
        )
    
    # Default to last 30 days if no time range specified
    if not start_time:
        start_time = datetime.utcnow() - timedelta(days=30)
    if not end_time:
        end_time = datetime.utcnow()
    
    # Create filter
    filters = AuditLogFilter(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        limit=10000  # Large limit for export
    )
    
    # Log the export action
    await audit_service.log_action(
        action="audit_log_export",
        user_id=current_user.id,
        username=current_user.username,
        resource_type="audit_log",
        compliance_tags=[ComplianceTag.GDPR],
        risk_level="high",
        business_justification="Audit log export for compliance or investigation",
        metadata={
            "format": format,
            "filters": filters.model_dump(exclude_none=True)
        }
    )
    
    # Get audit logs
    logs = await audit_service.search_audit_logs(filters, session)
    
    if format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=[log.model_dump() for log in logs],
            headers={
                "Content-Disposition": f"attachment; filename=audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )
    
    elif format == "csv":
        import csv
        import io
        from fastapi.responses import StreamingResponse
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        if logs:
            writer.writerow(logs[0].model_dump().keys())
            
            # Write data
            for log in logs:
                writer.writerow(log.model_dump().values())
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )