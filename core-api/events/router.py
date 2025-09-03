"""
Event ingestion and processing router.
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
    SecurityEventCreate, SecurityEventResponse, EventFilter, EventStats,
    EventType, ThreatLevel
)
from .service import EventService
from .auth import get_authenticated_device, check_device_rate_limit

logger = structlog.get_logger()

router = APIRouter()


@router.get("/health")
async def events_health():
    """Events service health check."""
    try:
        # Check database connectivity
        db_healthy = await database_manager.health_check()
        
        # Check Redis connectivity
        redis_healthy = await redis_manager.health_check()
        
        return {
            "service": "events",
            "status": "healthy" if db_healthy and redis_healthy else "degraded",
            "database": "connected" if db_healthy else "disconnected",
            "redis": "connected" if redis_healthy else "disconnected"
        }
    except Exception as e:
        logger.error("Events health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"service": "events", "status": "unhealthy", "error": str(e)}
        )


@router.post("/ingest", response_model=SecurityEventResponse)
async def ingest_event(
    event_data: SecurityEventCreate = Body(...),
    device_id: str = Depends(get_authenticated_device),
    _: None = Depends(check_device_rate_limit)
):
    """
    Ingest a security event from an edge device.
    
    This endpoint receives security events from edge nodes and processes them
    for incident creation and alerting.
    """
    try:
        # Check for duplicate events
        is_duplicate = await EventService.check_duplicate_event(
            event_data.camera_id,
            event_data.event_type,
            event_data.timestamp,
            event_data.confidence_score
        )
        
        if is_duplicate:
            logger.info(
                "Duplicate event detected, skipping",
                camera_id=event_data.camera_id,
                event_type=event_data.event_type.value,
                device_id=device_id
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Duplicate event detected"
            )
        
        # Create the event
        event = await EventService.create_event(event_data, device_id)
        
        logger.info(
            "Event ingested successfully",
            event_id=str(event.id),
            device_id=device_id,
            event_type=event_data.event_type.value
        )
        
        return event
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to ingest event", error=str(e), device_id=device_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process event"
        )


@router.get("/", response_model=List[SecurityEventResponse])
async def list_events(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    event_type: Optional[EventType] = Query(None, description="Filter by event type"),
    threat_level: Optional[ThreatLevel] = Query(None, description="Filter by threat level"),
    start_time: Optional[datetime] = Query(None, description="Filter events after this time"),
    end_time: Optional[datetime] = Query(None, description="Filter events before this time"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence score"),
    processed: Optional[str] = Query(None, description="Filter by processing status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return"),
    offset: int = Query(0, ge=0, description="Number of events to skip")
):
    """
    List security events with optional filtering.
    
    Returns a paginated list of security events based on the provided filters.
    """
    try:
        filters = EventFilter(
            camera_id=camera_id,
            event_type=event_type,
            threat_level=threat_level,
            start_time=start_time,
            end_time=end_time,
            min_confidence=min_confidence,
            processed=processed,
            limit=limit,
            offset=offset
        )
        
        events = await EventService.get_events(filters)
        
        logger.info(
            "Events retrieved",
            count=len(events),
            filters=filters.dict(exclude_none=True)
        )
        
        return events
        
    except Exception as e:
        logger.error("Failed to list events", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve events"
        )


@router.get("/{event_id}", response_model=SecurityEventResponse)
async def get_event(event_id: UUID):
    """
    Get a specific security event by ID.
    """
    try:
        event = await EventService.get_event_by_id(event_id)
        
        if not event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event not found"
            )
        
        return event
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get event", error=str(e), event_id=str(event_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve event"
        )


@router.get("/stats/summary", response_model=EventStats)
async def get_event_statistics(
    start_time: Optional[datetime] = Query(None, description="Statistics start time"),
    end_time: Optional[datetime] = Query(None, description="Statistics end time")
):
    """
    Get event statistics and analytics.
    
    Returns aggregated statistics about security events for the specified time period.
    """
    try:
        stats = await EventService.get_event_stats(start_time, end_time)
        
        logger.info(
            "Event statistics retrieved",
            total_events=stats.total_events,
            start_time=start_time,
            end_time=end_time
        )
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get event statistics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@router.post("/batch", response_model=List[SecurityEventResponse])
async def ingest_batch_events(
    events: List[SecurityEventCreate] = Body(..., max_items=50),
    device_id: str = Depends(get_authenticated_device),
    _: None = Depends(check_device_rate_limit)
):
    """
    Ingest multiple security events in a batch.
    
    Allows edge devices to send multiple events in a single request for efficiency.
    Maximum 50 events per batch.
    """
    try:
        if len(events) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 events per batch"
            )
        
        created_events = []
        failed_events = []
        
        for i, event_data in enumerate(events):
            try:
                # Check for duplicates
                is_duplicate = await EventService.check_duplicate_event(
                    event_data.camera_id,
                    event_data.event_type,
                    event_data.timestamp,
                    event_data.confidence_score
                )
                
                if not is_duplicate:
                    event = await EventService.create_event(event_data, device_id)
                    created_events.append(event)
                else:
                    logger.info(
                        "Duplicate event in batch, skipping",
                        batch_index=i,
                        camera_id=event_data.camera_id,
                        device_id=device_id
                    )
                    
            except Exception as e:
                logger.error(
                    "Failed to process event in batch",
                    batch_index=i,
                    error=str(e),
                    device_id=device_id
                )
                failed_events.append({"index": i, "error": str(e)})
        
        logger.info(
            "Batch event ingestion completed",
            total_events=len(events),
            created_events=len(created_events),
            failed_events=len(failed_events),
            device_id=device_id
        )
        
        if failed_events:
            # Return partial success with details about failures
            return JSONResponse(
                status_code=207,  # Multi-Status
                content={
                    "created_events": [event.dict() for event in created_events],
                    "failed_events": failed_events,
                    "message": f"Processed {len(created_events)}/{len(events)} events successfully"
                }
            )
        
        return created_events
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process batch events", error=str(e), device_id=device_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch events"
        )