"""
Incident management router.
"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
import structlog

from core.database import get_db_session, database_manager
from core.redis import redis_manager
from .models import (
    IncidentCreate, IncidentUpdate, IncidentResponse, IncidentFilter,
    IncidentNoteCreate, IncidentNoteResponse, IncidentAssignment,
    IncidentEscalation, IncidentStats, IncidentSummary,
    IncidentStatus, IncidentSeverity, IncidentPriority
)
from .service import IncidentService

logger = structlog.get_logger()

router = APIRouter()


@router.get("/health")
async def incidents_health():
    """Incidents service health check."""
    try:
        # Check database connectivity
        db_healthy = await database_manager.health_check()
        
        # Check Redis connectivity
        redis_healthy = await redis_manager.health_check()
        
        return {
            "service": "incidents",
            "status": "healthy" if db_healthy and redis_healthy else "degraded",
            "database": "connected" if db_healthy else "disconnected",
            "redis": "connected" if redis_healthy else "disconnected"
        }
    except Exception as e:
        logger.error("Incidents health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"service": "incidents", "status": "unhealthy", "error": str(e)}
        )


@router.post("/", response_model=IncidentResponse)
async def create_incident(
    incident_data: IncidentCreate = Body(...),
    # TODO: Add authentication dependency to get current user
    current_user: str = "system"  # Placeholder until auth is implemented
):
    """
    Create a new security incident.
    
    Creates an incident from security events or manual reporting.
    """
    try:
        incident = await IncidentService.create_incident(incident_data, current_user)
        
        logger.info(
            "Incident created via API",
            incident_id=str(incident.id),
            title=incident_data.title,
            severity=incident_data.severity.value,
            created_by=current_user
        )
        
        return incident
        
    except Exception as e:
        logger.error("Failed to create incident via API", error=str(e), created_by=current_user)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create incident"
        )


@router.get("/", response_model=List[IncidentResponse])
async def list_incidents(
    status_filter: Optional[IncidentStatus] = Query(None, alias="status", description="Filter by status"),
    severity: Optional[IncidentSeverity] = Query(None, description="Filter by severity"),
    priority: Optional[IncidentPriority] = Query(None, description="Filter by priority"),
    assigned_to: Optional[str] = Query(None, description="Filter by assigned user"),
    location: Optional[str] = Query(None, description="Filter by location"),
    escalated: Optional[bool] = Query(None, description="Filter by escalation status"),
    start_date: Optional[datetime] = Query(None, description="Filter incidents after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter incidents before this date"),
    search: Optional[str] = Query(None, description="Search in title and description"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of incidents to return"),
    offset: int = Query(0, ge=0, description="Number of incidents to skip")
):
    """
    List security incidents with optional filtering.
    
    Returns a paginated list of incidents based on the provided filters.
    """
    try:
        filters = IncidentFilter(
            status=status_filter,
            severity=severity,
            priority=priority,
            assigned_to=assigned_to,
            location=location,
            escalated=escalated,
            start_date=start_date,
            end_date=end_date,
            search=search,
            limit=limit,
            offset=offset
        )
        
        incidents = await IncidentService.get_incidents(filters)
        
        logger.info(
            "Incidents retrieved via API",
            count=len(incidents),
            filters=filters.dict(exclude_none=True)
        )
        
        return incidents
        
    except Exception as e:
        logger.error("Failed to list incidents", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve incidents"
        )


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: UUID):
    """
    Get a specific security incident by ID.
    """
    try:
        incident = await IncidentService.get_incident_by_id(incident_id)
        
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found"
            )
        
        return incident
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get incident", error=str(e), incident_id=str(incident_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve incident"
        )


@router.put("/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    incident_id: UUID,
    update_data: IncidentUpdate = Body(...),
    # TODO: Add authentication dependency to get current user
    current_user: str = "system"  # Placeholder until auth is implemented
):
    """
    Update an existing security incident.
    """
    try:
        incident = await IncidentService.update_incident(incident_id, update_data, current_user)
        
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found"
            )
        
        logger.info(
            "Incident updated via API",
            incident_id=str(incident_id),
            updated_by=current_user
        )
        
        return incident
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update incident", error=str(e), incident_id=str(incident_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update incident"
        )


@router.post("/{incident_id}/assign", response_model=IncidentResponse)
async def assign_incident(
    incident_id: UUID,
    assignment: IncidentAssignment = Body(...),
    # TODO: Add authentication dependency to get current user
    current_user: str = "system"  # Placeholder until auth is implemented
):
    """
    Assign an incident to a user.
    """
    try:
        incident = await IncidentService.assign_incident(incident_id, assignment, current_user)
        
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found"
            )
        
        logger.info(
            "Incident assigned via API",
            incident_id=str(incident_id),
            assigned_to=assignment.assigned_to,
            assigned_by=current_user
        )
        
        return incident
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to assign incident", error=str(e), incident_id=str(incident_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign incident"
        )


@router.post("/{incident_id}/escalate", response_model=IncidentResponse)
async def escalate_incident(
    incident_id: UUID,
    escalation: IncidentEscalation = Body(...),
    # TODO: Add authentication dependency to get current user
    current_user: str = "system"  # Placeholder until auth is implemented
):
    """
    Escalate an incident.
    """
    try:
        incident = await IncidentService.escalate_incident(incident_id, escalation, current_user)
        
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Incident not found"
            )
        
        logger.info(
            "Incident escalated via API",
            incident_id=str(incident_id),
            escalated_by=current_user,
            reason=escalation.escalation_reason
        )
        
        return incident
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to escalate incident", error=str(e), incident_id=str(incident_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to escalate incident"
        )


@router.post("/{incident_id}/notes", response_model=IncidentNoteResponse)
async def add_incident_note(
    incident_id: UUID,
    note_data: IncidentNoteCreate = Body(...),
    # TODO: Add authentication dependency to get current user
    current_user: str = "system"  # Placeholder until auth is implemented
):
    """
    Add a note to an incident.
    """
    try:
        note = await IncidentService.add_incident_note(incident_id, note_data, current_user)
        
        logger.info(
            "Note added to incident via API",
            incident_id=str(incident_id),
            note_id=str(note.id),
            author=current_user
        )
        
        return note
        
    except Exception as e:
        logger.error("Failed to add incident note", error=str(e), incident_id=str(incident_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add note"
        )


@router.get("/{incident_id}/notes", response_model=List[IncidentNoteResponse])
async def get_incident_notes(incident_id: UUID):
    """
    Get all notes for an incident.
    """
    try:
        notes = await IncidentService.get_incident_notes(incident_id)
        
        return notes
        
    except Exception as e:
        logger.error("Failed to get incident notes", error=str(e), incident_id=str(incident_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve notes"
        )


@router.get("/stats/summary", response_model=IncidentStats)
async def get_incident_statistics(
    start_date: Optional[datetime] = Query(None, description="Statistics start date"),
    end_date: Optional[datetime] = Query(None, description="Statistics end date")
):
    """
    Get incident statistics and analytics.
    
    Returns aggregated statistics about incidents for the specified time period.
    """
    try:
        stats = await IncidentService.get_incident_stats(start_date, end_date)
        
        logger.info(
            "Incident statistics retrieved via API",
            total_incidents=stats.total_incidents,
            start_date=start_date,
            end_date=end_date
        )
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get incident statistics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )