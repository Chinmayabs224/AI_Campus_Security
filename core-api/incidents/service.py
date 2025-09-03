"""
Incident management service with business logic.
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

import structlog
from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import database_manager
from core.redis import redis_manager, cache_manager
from .models import (
    Incident, IncidentEvent, IncidentNote, IncidentStatusHistory,
    IncidentCreate, IncidentUpdate, IncidentResponse, IncidentFilter,
    IncidentNoteCreate, IncidentNoteResponse, IncidentAssignment,
    IncidentEscalation, IncidentStats, IncidentSummary,
    IncidentStatus, IncidentSeverity, IncidentPriority
)

logger = structlog.get_logger()


class IncidentService:
    """Service for managing security incidents."""
    
    @staticmethod
    async def create_incident(
        incident_data: IncidentCreate,
        created_by: str
    ) -> IncidentResponse:
        """Create a new security incident."""
        try:
            async with database_manager.get_session() as session:
                # Create incident record
                incident = Incident(
                    title=incident_data.title,
                    description=incident_data.description,
                    severity=incident_data.severity.value,
                    priority=incident_data.priority.value,
                    location=incident_data.location,
                    camera_ids=",".join(incident_data.camera_ids) if incident_data.camera_ids else None,
                    assigned_to=incident_data.assigned_to,
                    created_by=created_by
                )
                
                session.add(incident)
                await session.flush()
                await session.refresh(incident)
                
                # Create status history entry
                status_history = IncidentStatusHistory(
                    incident_id=incident.id,
                    previous_status=None,
                    new_status=IncidentStatus.OPEN.value,
                    changed_by=created_by,
                    change_reason="Incident created"
                )
                session.add(status_history)
                
                # Link events if provided
                if incident_data.event_ids:
                    for event_id in incident_data.event_ids:
                        incident_event = IncidentEvent(
                            incident_id=incident.id,
                            event_id=event_id
                        )
                        session.add(incident_event)
                
                # Set assignment timestamp if assigned
                if incident_data.assigned_to:
                    incident.assigned_at = datetime.utcnow()
                    incident.status = IncidentStatus.ASSIGNED.value
                    
                    # Add assignment history
                    assignment_history = IncidentStatusHistory(
                        incident_id=incident.id,
                        previous_status=IncidentStatus.OPEN.value,
                        new_status=IncidentStatus.ASSIGNED.value,
                        changed_by=created_by,
                        change_reason=f"Assigned to {incident_data.assigned_to}"
                    )
                    session.add(assignment_history)
                
                # Update incident counters
                await IncidentService._update_incident_counters(incident_data.severity, incident_data.priority)
                
                # Check for auto-escalation rules
                await IncidentService._check_escalation_rules(incident)
                
                logger.info(
                    "Incident created",
                    incident_id=str(incident.id),
                    title=incident_data.title,
                    severity=incident_data.severity.value,
                    created_by=created_by
                )
                
                return IncidentResponse.from_orm(incident)
                
        except Exception as e:
            logger.error("Failed to create incident", error=str(e), created_by=created_by)
            raise
    
    @staticmethod
    async def get_incidents(
        filters: IncidentFilter,
        user_id: Optional[str] = None
    ) -> List[IncidentResponse]:
        """Get incidents with filtering."""
        try:
            async with database_manager.get_session() as session:
                query = select(Incident)
                
                # Apply filters
                conditions = []
                
                if filters.status:
                    conditions.append(Incident.status == filters.status.value)
                
                if filters.severity:
                    conditions.append(Incident.severity == filters.severity.value)
                
                if filters.priority:
                    conditions.append(Incident.priority == filters.priority.value)
                
                if filters.assigned_to:
                    conditions.append(Incident.assigned_to == filters.assigned_to)
                
                if filters.created_by:
                    conditions.append(Incident.created_by == filters.created_by)
                
                if filters.location:
                    conditions.append(Incident.location.ilike(f"%{filters.location}%"))
                
                if filters.escalated is not None:
                    conditions.append(Incident.escalated == filters.escalated)
                
                if filters.start_date:
                    conditions.append(Incident.created_at >= filters.start_date)
                
                if filters.end_date:
                    conditions.append(Incident.created_at <= filters.end_date)
                
                if filters.search:
                    search_condition = or_(
                        Incident.title.ilike(f"%{filters.search}%"),
                        Incident.description.ilike(f"%{filters.search}%")
                    )
                    conditions.append(search_condition)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                # Order by priority, severity, and creation time
                query = query.order_by(
                    Incident.priority.desc(),
                    Incident.severity.desc(),
                    Incident.created_at.desc()
                )
                
                # Apply pagination
                query = query.offset(filters.offset).limit(filters.limit)
                
                result = await session.execute(query)
                incidents = result.scalars().all()
                
                return [IncidentResponse.from_orm(incident) for incident in incidents]
                
        except Exception as e:
            logger.error("Failed to get incidents", error=str(e), filters=filters.dict())
            raise
    
    @staticmethod
    async def get_incident_by_id(incident_id: UUID) -> Optional[IncidentResponse]:
        """Get a specific incident by ID."""
        try:
            async with database_manager.get_session() as session:
                query = select(Incident).where(Incident.id == incident_id)
                result = await session.execute(query)
                incident = result.scalar_one_or_none()
                
                if incident:
                    return IncidentResponse.from_orm(incident)
                return None
                
        except Exception as e:
            logger.error("Failed to get incident by ID", error=str(e), incident_id=str(incident_id))
            raise
    
    @staticmethod
    async def update_incident(
        incident_id: UUID,
        update_data: IncidentUpdate,
        updated_by: str
    ) -> Optional[IncidentResponse]:
        """Update an existing incident."""
        try:
            async with database_manager.get_session() as session:
                query = select(Incident).where(Incident.id == incident_id)
                result = await session.execute(query)
                incident = result.scalar_one_or_none()
                
                if not incident:
                    return None
                
                # Track changes for history
                changes = []
                previous_status = incident.status
                
                # Update fields
                if update_data.title is not None:
                    incident.title = update_data.title
                    changes.append(f"Title updated")
                
                if update_data.description is not None:
                    incident.description = update_data.description
                    changes.append(f"Description updated")
                
                if update_data.status is not None:
                    incident.status = update_data.status.value
                    changes.append(f"Status changed from {previous_status} to {update_data.status.value}")
                    
                    # Set timestamps based on status
                    if update_data.status == IncidentStatus.RESOLVED:
                        incident.resolved_at = datetime.utcnow()
                    elif update_data.status == IncidentStatus.CLOSED:
                        incident.closed_at = datetime.utcnow()
                
                if update_data.severity is not None:
                    incident.severity = update_data.severity.value
                    changes.append(f"Severity updated to {update_data.severity.value}")
                
                if update_data.priority is not None:
                    incident.priority = update_data.priority.value
                    changes.append(f"Priority updated to {update_data.priority.value}")
                
                if update_data.assigned_to is not None:
                    incident.assigned_to = update_data.assigned_to
                    incident.assigned_at = datetime.utcnow()
                    if incident.status == IncidentStatus.OPEN.value:
                        incident.status = IncidentStatus.ASSIGNED.value
                    changes.append(f"Assigned to {update_data.assigned_to}")
                
                if update_data.location is not None:
                    incident.location = update_data.location
                    changes.append(f"Location updated")
                
                if update_data.escalation_reason is not None:
                    incident.escalation_reason = update_data.escalation_reason
                    incident.escalated = True
                    incident.escalated_at = datetime.utcnow()
                    changes.append(f"Escalated: {update_data.escalation_reason}")
                
                incident.updated_at = datetime.utcnow()
                
                # Create status history if status changed
                if update_data.status and update_data.status.value != previous_status:
                    status_history = IncidentStatusHistory(
                        incident_id=incident.id,
                        previous_status=previous_status,
                        new_status=update_data.status.value,
                        changed_by=updated_by,
                        change_reason="; ".join(changes)
                    )
                    session.add(status_history)
                
                logger.info(
                    "Incident updated",
                    incident_id=str(incident.id),
                    changes=changes,
                    updated_by=updated_by
                )
                
                return IncidentResponse.from_orm(incident)
                
        except Exception as e:
            logger.error("Failed to update incident", error=str(e), incident_id=str(incident_id))
            raise
    
    @staticmethod
    async def assign_incident(
        incident_id: UUID,
        assignment: IncidentAssignment,
        assigned_by: str
    ) -> Optional[IncidentResponse]:
        """Assign an incident to a user."""
        try:
            update_data = IncidentUpdate(
                assigned_to=assignment.assigned_to,
                status=IncidentStatus.ASSIGNED
            )
            
            incident = await IncidentService.update_incident(incident_id, update_data, assigned_by)
            
            if incident:
                # Add assignment note
                note = IncidentNoteCreate(
                    content=f"Incident assigned to {assignment.assigned_to}. {assignment.assignment_reason or ''}",
                    note_type="assignment"
                )
                await IncidentService.add_incident_note(incident_id, note, assigned_by)
            
            return incident
            
        except Exception as e:
            logger.error("Failed to assign incident", error=str(e), incident_id=str(incident_id))
            raise
    
    @staticmethod
    async def escalate_incident(
        incident_id: UUID,
        escalation: IncidentEscalation,
        escalated_by: str
    ) -> Optional[IncidentResponse]:
        """Escalate an incident."""
        try:
            update_data = IncidentUpdate(
                escalation_reason=escalation.escalation_reason,
                priority=escalation.new_priority,
                severity=escalation.new_severity
            )
            
            incident = await IncidentService.update_incident(incident_id, update_data, escalated_by)
            
            if incident:
                # Add escalation note
                note = IncidentNoteCreate(
                    content=f"Incident escalated: {escalation.escalation_reason}",
                    note_type="escalation"
                )
                await IncidentService.add_incident_note(incident_id, note, escalated_by)
            
            return incident
            
        except Exception as e:
            logger.error("Failed to escalate incident", error=str(e), incident_id=str(incident_id))
            raise
    
    @staticmethod
    async def add_incident_note(
        incident_id: UUID,
        note_data: IncidentNoteCreate,
        author: str
    ) -> IncidentNoteResponse:
        """Add a note to an incident."""
        try:
            async with database_manager.get_session() as session:
                note = IncidentNote(
                    incident_id=incident_id,
                    author=author,
                    content=note_data.content,
                    note_type=note_data.note_type
                )
                
                session.add(note)
                await session.flush()
                await session.refresh(note)
                
                logger.info(
                    "Incident note added",
                    incident_id=str(incident_id),
                    note_id=str(note.id),
                    author=author
                )
                
                return IncidentNoteResponse.from_orm(note)
                
        except Exception as e:
            logger.error("Failed to add incident note", error=str(e), incident_id=str(incident_id))
            raise
    
    @staticmethod
    async def get_incident_notes(incident_id: UUID) -> List[IncidentNoteResponse]:
        """Get all notes for an incident."""
        try:
            async with database_manager.get_session() as session:
                query = select(IncidentNote).where(
                    IncidentNote.incident_id == incident_id
                ).order_by(IncidentNote.created_at.desc())
                
                result = await session.execute(query)
                notes = result.scalars().all()
                
                return [IncidentNoteResponse.from_orm(note) for note in notes]
                
        except Exception as e:
            logger.error("Failed to get incident notes", error=str(e), incident_id=str(incident_id))
            raise
    
    @staticmethod
    async def get_incident_stats(
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> IncidentStats:
        """Get incident statistics."""
        try:
            # Use cache for recent stats
            cache_key = f"incident_stats:{start_date}:{end_date}"
            cached_stats = await redis_manager.get_json(cache_key)
            if cached_stats:
                return IncidentStats(**cached_stats)
            
            async with database_manager.get_session() as session:
                query = select(Incident)
                
                # Apply time filters
                if start_date:
                    query = query.where(Incident.created_at >= start_date)
                if end_date:
                    query = query.where(Incident.created_at <= end_date)
                
                # Get total count
                count_query = select(func.count(Incident.id))
                if start_date:
                    count_query = count_query.where(Incident.created_at >= start_date)
                if end_date:
                    count_query = count_query.where(Incident.created_at <= end_date)
                
                total_result = await session.execute(count_query)
                total_incidents = total_result.scalar()
                
                # Get incidents by status
                status_query = select(
                    Incident.status,
                    func.count(Incident.id).label('count')
                ).group_by(Incident.status)
                
                if start_date:
                    status_query = status_query.where(Incident.created_at >= start_date)
                if end_date:
                    status_query = status_query.where(Incident.created_at <= end_date)
                
                status_result = await session.execute(status_query)
                incidents_by_status = {row.status: row.count for row in status_result}
                
                # Get incidents by severity
                severity_query = select(
                    Incident.severity,
                    func.count(Incident.id).label('count')
                ).group_by(Incident.severity)
                
                if start_date:
                    severity_query = severity_query.where(Incident.created_at >= start_date)
                if end_date:
                    severity_query = severity_query.where(Incident.created_at <= end_date)
                
                severity_result = await session.execute(severity_query)
                incidents_by_severity = {row.severity: row.count for row in severity_result}
                
                # Get incidents by priority
                priority_query = select(
                    Incident.priority,
                    func.count(Incident.id).label('count')
                ).group_by(Incident.priority)
                
                if start_date:
                    priority_query = priority_query.where(Incident.created_at >= start_date)
                if end_date:
                    priority_query = priority_query.where(Incident.created_at <= end_date)
                
                priority_result = await session.execute(priority_query)
                incidents_by_priority = {row.priority: row.count for row in priority_result}
                
                # Get average resolution time
                resolution_query = select(
                    func.avg(
                        func.extract('epoch', Incident.resolved_at - Incident.created_at) / 3600
                    ).label('avg_hours')
                ).where(Incident.resolved_at.isnot(None))
                
                if start_date:
                    resolution_query = resolution_query.where(Incident.created_at >= start_date)
                if end_date:
                    resolution_query = resolution_query.where(Incident.created_at <= end_date)
                
                resolution_result = await session.execute(resolution_query)
                avg_resolution_time = resolution_result.scalar()
                
                # Get escalated incidents count
                escalated_query = select(func.count(Incident.id)).where(Incident.escalated == True)
                if start_date:
                    escalated_query = escalated_query.where(Incident.created_at >= start_date)
                if end_date:
                    escalated_query = escalated_query.where(Incident.created_at <= end_date)
                
                escalated_result = await session.execute(escalated_query)
                escalated_incidents = escalated_result.scalar()
                
                # Get unassigned incidents count
                unassigned_query = select(func.count(Incident.id)).where(Incident.assigned_to.is_(None))
                if start_date:
                    unassigned_query = unassigned_query.where(Incident.created_at >= start_date)
                if end_date:
                    unassigned_query = unassigned_query.where(Incident.created_at <= end_date)
                
                unassigned_result = await session.execute(unassigned_query)
                unassigned_incidents = unassigned_result.scalar()
                
                stats = IncidentStats(
                    total_incidents=total_incidents,
                    incidents_by_status=incidents_by_status,
                    incidents_by_severity=incidents_by_severity,
                    incidents_by_priority=incidents_by_priority,
                    avg_resolution_time_hours=float(avg_resolution_time) if avg_resolution_time else None,
                    escalated_incidents=escalated_incidents,
                    unassigned_incidents=unassigned_incidents
                )
                
                # Cache for 5 minutes
                await redis_manager.set_json(cache_key, stats.dict(), expire=300)
                
                return stats
                
        except Exception as e:
            logger.error("Failed to get incident statistics", error=str(e))
            raise
    
    @staticmethod
    async def _update_incident_counters(severity: IncidentSeverity, priority: IncidentPriority):
        """Update incident counters in Redis."""
        try:
            async with redis_manager.pipeline() as pipe:
                # Daily counters
                today = datetime.utcnow().strftime("%Y-%m-%d")
                pipe.incr(f"incidents:daily:{today}")
                pipe.incr(f"incidents:daily:{today}:severity:{severity.value}")
                pipe.incr(f"incidents:daily:{today}:priority:{priority.value}")
                
                # Set expiration for counters (30 days)
                for key in [
                    f"incidents:daily:{today}",
                    f"incidents:daily:{today}:severity:{severity.value}",
                    f"incidents:daily:{today}:priority:{priority.value}"
                ]:
                    pipe.expire(key, 30 * 24 * 3600)  # 30 days
                
        except Exception as e:
            logger.error("Failed to update incident counters", error=str(e))
    
    @staticmethod
    async def _check_escalation_rules(incident: Incident):
        """Check and apply automatic escalation rules."""
        try:
            # Rule 1: Critical severity incidents should be escalated immediately
            if incident.severity == IncidentSeverity.CRITICAL.value and not incident.escalated:
                incident.escalated = True
                incident.escalated_at = datetime.utcnow()
                incident.escalation_reason = "Automatic escalation: Critical severity incident"
                
                logger.info(
                    "Incident auto-escalated",
                    incident_id=str(incident.id),
                    reason="Critical severity"
                )
            
            # Rule 2: High severity incidents unassigned for more than 15 minutes
            # This would be implemented in a background task
            
        except Exception as e:
            logger.error("Failed to check escalation rules", error=str(e), incident_id=str(incident.id))