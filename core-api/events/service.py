"""
Event processing service with business logic.
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

import structlog
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import database_manager
from core.redis import redis_manager, cache_manager
from .models import (
    SecurityEvent, SecurityEventCreate, SecurityEventResponse,
    EventFilter, EventStats, EventType, ThreatLevel
)

logger = structlog.get_logger()


class EventService:
    """Service for managing security events."""
    
    @staticmethod
    async def create_event(
        event_data: SecurityEventCreate,
        device_id: str
    ) -> SecurityEventResponse:
        """Create a new security event."""
        try:
            async with database_manager.get_session() as session:
                # Create event record
                event = SecurityEvent(
                    camera_id=event_data.camera_id,
                    timestamp=event_data.timestamp,
                    event_type=event_data.event_type.value,
                    threat_level=event_data.threat_level.value,
                    confidence_score=event_data.confidence_score,
                    bounding_boxes=[box.dict() for box in event_data.bounding_boxes] if event_data.bounding_boxes else None,
                    event_metadata=event_data.metadata.dict() if event_data.metadata else None
                )
                
                session.add(event)
                await session.flush()
                await session.refresh(event)
                
                # Cache recent event for deduplication
                cache_key = f"recent_event:{event_data.camera_id}:{event_data.event_type.value}"
                await redis_manager.set_json(cache_key, {
                    "timestamp": event_data.timestamp.isoformat(),
                    "confidence": event_data.confidence_score,
                    "event_id": str(event.id)
                }, expire=300)  # 5 minutes
                
                # Increment event counters
                await EventService._update_event_counters(event_data.event_type, event_data.threat_level)
                
                # Trigger incident creation for high-confidence events
                if event_data.confidence_score >= 0.8 and event_data.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    await EventService._trigger_incident_creation(event)
                
                logger.info(
                    "Security event created",
                    event_id=str(event.id),
                    camera_id=event_data.camera_id,
                    event_type=event_data.event_type.value,
                    threat_level=event_data.threat_level.value,
                    confidence=event_data.confidence_score,
                    device_id=device_id
                )
                
                return SecurityEventResponse.from_orm(event)
                
        except Exception as e:
            logger.error("Failed to create security event", error=str(e), device_id=device_id)
            raise
    
    @staticmethod
    async def get_events(
        filters: EventFilter,
        user_id: Optional[str] = None
    ) -> List[SecurityEventResponse]:
        """Get events with filtering."""
        try:
            async with database_manager.get_session() as session:
                query = select(SecurityEvent)
                
                # Apply filters
                conditions = []
                
                if filters.camera_id:
                    conditions.append(SecurityEvent.camera_id == filters.camera_id)
                
                if filters.event_type:
                    conditions.append(SecurityEvent.event_type == filters.event_type.value)
                
                if filters.threat_level:
                    conditions.append(SecurityEvent.threat_level == filters.threat_level.value)
                
                if filters.start_time:
                    conditions.append(SecurityEvent.timestamp >= filters.start_time)
                
                if filters.end_time:
                    conditions.append(SecurityEvent.timestamp <= filters.end_time)
                
                if filters.min_confidence:
                    conditions.append(SecurityEvent.confidence_score >= filters.min_confidence)
                
                if filters.processed:
                    conditions.append(SecurityEvent.processed == filters.processed)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                # Order by timestamp descending
                query = query.order_by(SecurityEvent.timestamp.desc())
                
                # Apply pagination
                query = query.offset(filters.offset).limit(filters.limit)
                
                result = await session.execute(query)
                events = result.scalars().all()
                
                return [SecurityEventResponse.from_orm(event) for event in events]
                
        except Exception as e:
            logger.error("Failed to get events", error=str(e), filters=filters.dict())
            raise
    
    @staticmethod
    async def get_event_by_id(event_id: UUID) -> Optional[SecurityEventResponse]:
        """Get a specific event by ID."""
        try:
            async with database_manager.get_session() as session:
                query = select(SecurityEvent).where(SecurityEvent.id == event_id)
                result = await session.execute(query)
                event = result.scalar_one_or_none()
                
                if event:
                    return SecurityEventResponse.from_orm(event)
                return None
                
        except Exception as e:
            logger.error("Failed to get event by ID", error=str(e), event_id=str(event_id))
            raise
    
    @staticmethod
    async def get_event_stats(
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> EventStats:
        """Get event statistics."""
        try:
            # Use cache for recent stats
            cache_key = f"event_stats:{start_time}:{end_time}"
            cached_stats = await redis_manager.get_json(cache_key)
            if cached_stats:
                return EventStats(**cached_stats)
            
            async with database_manager.get_session() as session:
                query = select(SecurityEvent)
                
                # Apply time filters
                if start_time:
                    query = query.where(SecurityEvent.timestamp >= start_time)
                if end_time:
                    query = query.where(SecurityEvent.timestamp <= end_time)
                
                # Get total count
                count_query = select(func.count(SecurityEvent.id))
                if start_time:
                    count_query = count_query.where(SecurityEvent.timestamp >= start_time)
                if end_time:
                    count_query = count_query.where(SecurityEvent.timestamp <= end_time)
                
                total_result = await session.execute(count_query)
                total_events = total_result.scalar()
                
                # Get events by type
                type_query = select(
                    SecurityEvent.event_type,
                    func.count(SecurityEvent.id).label('count')
                ).group_by(SecurityEvent.event_type)
                
                if start_time:
                    type_query = type_query.where(SecurityEvent.timestamp >= start_time)
                if end_time:
                    type_query = type_query.where(SecurityEvent.timestamp <= end_time)
                
                type_result = await session.execute(type_query)
                events_by_type = {row.event_type: row.count for row in type_result}
                
                # Get events by threat level
                threat_query = select(
                    SecurityEvent.threat_level,
                    func.count(SecurityEvent.id).label('count')
                ).group_by(SecurityEvent.threat_level)
                
                if start_time:
                    threat_query = threat_query.where(SecurityEvent.timestamp >= start_time)
                if end_time:
                    threat_query = threat_query.where(SecurityEvent.timestamp <= end_time)
                
                threat_result = await session.execute(threat_query)
                events_by_threat_level = {row.threat_level: row.count for row in threat_result}
                
                # Get average confidence
                avg_query = select(func.avg(SecurityEvent.confidence_score))
                if start_time:
                    avg_query = avg_query.where(SecurityEvent.timestamp >= start_time)
                if end_time:
                    avg_query = avg_query.where(SecurityEvent.timestamp <= end_time)
                
                avg_result = await session.execute(avg_query)
                avg_confidence = float(avg_result.scalar() or 0.0)
                
                # Get time range
                time_query = select(
                    func.min(SecurityEvent.timestamp).label('min_time'),
                    func.max(SecurityEvent.timestamp).label('max_time')
                )
                if start_time:
                    time_query = time_query.where(SecurityEvent.timestamp >= start_time)
                if end_time:
                    time_query = time_query.where(SecurityEvent.timestamp <= end_time)
                
                time_result = await session.execute(time_query)
                time_row = time_result.first()
                
                stats = EventStats(
                    total_events=total_events,
                    events_by_type=events_by_type,
                    events_by_threat_level=events_by_threat_level,
                    avg_confidence=avg_confidence,
                    time_range={
                        "start": time_row.min_time or datetime.utcnow(),
                        "end": time_row.max_time or datetime.utcnow()
                    }
                )
                
                # Cache for 5 minutes
                await redis_manager.set_json(cache_key, stats.dict(), expire=300)
                
                return stats
                
        except Exception as e:
            logger.error("Failed to get event statistics", error=str(e))
            raise
    
    @staticmethod
    async def check_duplicate_event(
        camera_id: str,
        event_type: EventType,
        timestamp: datetime,
        confidence_score: float
    ) -> bool:
        """Check if this is a duplicate event."""
        try:
            cache_key = f"recent_event:{camera_id}:{event_type.value}"
            recent_event = await redis_manager.get_json(cache_key)
            
            if not recent_event:
                return False
            
            # Check if within deduplication window (30 seconds)
            recent_timestamp = datetime.fromisoformat(recent_event["timestamp"])
            time_diff = abs((timestamp - recent_timestamp).total_seconds())
            
            if time_diff <= 30:
                # Check if confidence is significantly higher
                confidence_diff = confidence_score - recent_event["confidence"]
                if confidence_diff > 0.1:  # 10% improvement
                    return False  # Allow higher confidence event
                return True  # Duplicate
            
            return False
            
        except Exception as e:
            logger.error("Failed to check duplicate event", error=str(e))
            return False  # Allow event if check fails
    
    @staticmethod
    async def _update_event_counters(event_type: EventType, threat_level: ThreatLevel):
        """Update event counters in Redis."""
        try:
            async with redis_manager.pipeline() as pipe:
                # Daily counters
                today = datetime.utcnow().strftime("%Y-%m-%d")
                pipe.incr(f"events:daily:{today}")
                pipe.incr(f"events:daily:{today}:type:{event_type.value}")
                pipe.incr(f"events:daily:{today}:threat:{threat_level.value}")
                
                # Hourly counters
                hour = datetime.utcnow().strftime("%Y-%m-%d:%H")
                pipe.incr(f"events:hourly:{hour}")
                pipe.incr(f"events:hourly:{hour}:type:{event_type.value}")
                pipe.incr(f"events:hourly:{hour}:threat:{threat_level.value}")
                
                # Set expiration for counters (30 days)
                for key in [
                    f"events:daily:{today}",
                    f"events:daily:{today}:type:{event_type.value}",
                    f"events:daily:{today}:threat:{threat_level.value}",
                    f"events:hourly:{hour}",
                    f"events:hourly:{hour}:type:{event_type.value}",
                    f"events:hourly:{hour}:threat:{threat_level.value}"
                ]:
                    pipe.expire(key, 30 * 24 * 3600)  # 30 days
                
        except Exception as e:
            logger.error("Failed to update event counters", error=str(e))
    
    @staticmethod
    async def _trigger_incident_creation(event: SecurityEvent):
        """Trigger incident creation for high-priority events."""
        try:
            # Add to incident creation queue
            incident_data = {
                "event_id": str(event.id),
                "camera_id": event.camera_id,
                "event_type": event.event_type,
                "threat_level": event.threat_level,
                "confidence_score": event.confidence_score,
                "timestamp": event.timestamp.isoformat()
            }
            
            await redis_manager.set_json(
                f"incident_queue:{event.id}",
                incident_data,
                expire=3600  # 1 hour
            )
            
            logger.info(
                "High-priority event queued for incident creation",
                event_id=str(event.id),
                threat_level=event.threat_level
            )
            
        except Exception as e:
            logger.error("Failed to trigger incident creation", error=str(e), event_id=str(event.id))