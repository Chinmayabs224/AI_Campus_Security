"""
WebSocket manager for real-time alert distribution
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
import redis
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from models import NotificationRequest, NotificationPriority, IncidentType

logger = logging.getLogger(__name__)

class ConnectionType(Enum):
    """Types of WebSocket connections"""
    DASHBOARD = "dashboard"
    MOBILE = "mobile"
    ADMIN = "admin"

@dataclass
class WebSocketConnection:
    """WebSocket connection information"""
    connection_id: str
    user_id: str
    connection_type: ConnectionType
    websocket: Any  # WebSocket object
    connected_at: datetime
    last_ping: datetime
    subscribed_locations: List[str] = None
    subscribed_incident_types: List[IncidentType] = None
    
    def __post_init__(self):
        if self.subscribed_locations is None:
            self.subscribed_locations = []
        if self.subscribed_incident_types is None:
            self.subscribed_incident_types = list(IncidentType)

@dataclass
class AlertMessage:
    """Real-time alert message"""
    alert_id: str
    incident_id: str
    user_id: str
    title: str
    message: str
    priority: NotificationPriority
    incident_type: IncidentType
    location: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

class WebSocketManager:
    """Manages WebSocket connections and real-time alert distribution"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.location_subscriptions: Dict[str, Set[str]] = {}  # location -> set of connection_ids
        self.active_alerts: Dict[str, AlertMessage] = {}
        self.escalation_tasks: Dict[str, asyncio.Task] = {}
        
    async def connect(self, websocket, user_id: str, connection_type: ConnectionType, 
                     subscriptions: Dict[str, Any] = None) -> str:
        """Register a new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        
        connection = WebSocketConnection(
            connection_id=connection_id,
            user_id=user_id,
            connection_type=connection_type,
            websocket=websocket,
            connected_at=datetime.utcnow(),
            last_ping=datetime.utcnow(),
            subscribed_locations=subscriptions.get('locations', []) if subscriptions else [],
            subscribed_incident_types=[
                IncidentType(t) for t in subscriptions.get('incident_types', [])
            ] if subscriptions and subscriptions.get('incident_types') else list(IncidentType)
        )
        
        # Store connection
        self.connections[connection_id] = connection
        
        # Update user connections mapping
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Update location subscriptions
        for location in connection.subscribed_locations:
            if location not in self.location_subscriptions:
                self.location_subscriptions[location] = set()
            self.location_subscriptions[location].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
        
        # Send initial connection confirmation
        await self._send_to_connection(connection_id, {
            'type': 'connection_established',
            'connection_id': connection_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Send any active alerts for this user
        await self._send_active_alerts_to_connection(connection_id)
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Unregister a WebSocket connection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        user_id = connection.user_id
        
        # Remove from user connections
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove from location subscriptions
        for location in connection.subscribed_locations:
            if location in self.location_subscriptions:
                self.location_subscriptions[location].discard(connection_id)
                if not self.location_subscriptions[location]:
                    del self.location_subscriptions[location]
        
        # Remove connection
        del self.connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id} for user {user_id}")
    
    async def distribute_alert(self, alert: AlertMessage) -> Dict[str, Any]:
        """Distribute alert to relevant WebSocket connections"""
        try:
            # Store active alert
            self.active_alerts[alert.alert_id] = alert
            
            # Find relevant connections
            target_connections = await self._find_target_connections(alert)
            
            # Prepare alert message
            alert_data = {
                'type': 'security_alert',
                'alert': asdict(alert),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send to all target connections
            sent_count = 0
            failed_count = 0
            
            for connection_id in target_connections:
                try:
                    await self._send_to_connection(connection_id, alert_data)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send alert to connection {connection_id}: {str(e)}")
                    failed_count += 1
            
            # Store alert in Redis for persistence
            await self._store_alert_in_redis(alert)
            
            # Schedule escalation if needed
            await self._schedule_escalation(alert)
            
            logger.info(f"Alert {alert.alert_id} distributed to {sent_count} connections")
            
            return {
                'alert_id': alert.alert_id,
                'sent_count': sent_count,
                'failed_count': failed_count,
                'target_connections': len(target_connections)
            }
            
        except Exception as e:
            logger.error(f"Error distributing alert {alert.alert_id}: {str(e)}")
            return {
                'alert_id': alert.alert_id,
                'error': str(e),
                'sent_count': 0,
                'failed_count': 0
            }
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = user_id
            alert.acknowledged_at = datetime.utcnow()
            
            # Cancel escalation if scheduled
            if alert_id in self.escalation_tasks:
                self.escalation_tasks[alert_id].cancel()
                del self.escalation_tasks[alert_id]
            
            # Broadcast acknowledgment to all relevant connections
            ack_message = {
                'type': 'alert_acknowledged',
                'alert_id': alert_id,
                'acknowledged_by': user_id,
                'acknowledged_at': alert.acknowledged_at.isoformat(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            target_connections = await self._find_target_connections(alert)
            for connection_id in target_connections:
                try:
                    await self._send_to_connection(connection_id, ack_message)
                except Exception as e:
                    logger.error(f"Failed to send acknowledgment to connection {connection_id}: {str(e)}")
            
            # Update in Redis
            await self._update_alert_in_redis(alert)
            
            logger.info(f"Alert {alert_id} acknowledged by user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
            return False
    
    async def get_active_alerts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active alerts for a user"""
        try:
            user_alerts = []
            
            for alert in self.active_alerts.values():
                # Check if user should see this alert based on location/type subscriptions
                if await self._should_user_receive_alert(user_id, alert):
                    user_alerts.append(asdict(alert))
            
            # Sort by priority and timestamp
            user_alerts.sort(key=lambda x: (
                self._get_priority_weight(x['priority']),
                x['timestamp']
            ), reverse=True)
            
            return user_alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts for user {user_id}: {str(e)}")
            return []
    
    async def update_connection_subscriptions(self, connection_id: str, 
                                           subscriptions: Dict[str, Any]) -> bool:
        """Update connection subscriptions"""
        try:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            
            # Remove from old location subscriptions
            for location in connection.subscribed_locations:
                if location in self.location_subscriptions:
                    self.location_subscriptions[location].discard(connection_id)
            
            # Update subscriptions
            connection.subscribed_locations = subscriptions.get('locations', [])
            connection.subscribed_incident_types = [
                IncidentType(t) for t in subscriptions.get('incident_types', [])
            ] if subscriptions.get('incident_types') else list(IncidentType)
            
            # Add to new location subscriptions
            for location in connection.subscribed_locations:
                if location not in self.location_subscriptions:
                    self.location_subscriptions[location] = set()
                self.location_subscriptions[location].add(connection_id)
            
            logger.info(f"Updated subscriptions for connection {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating subscriptions for connection {connection_id}: {str(e)}")
            return False
    
    async def _find_target_connections(self, alert: AlertMessage) -> Set[str]:
        """Find connections that should receive this alert"""
        target_connections = set()
        
        # Find connections by location
        if alert.location in self.location_subscriptions:
            target_connections.update(self.location_subscriptions[alert.location])
        
        # Find connections by user (direct assignment)
        if alert.user_id in self.user_connections:
            target_connections.update(self.user_connections[alert.user_id])
        
        # Filter by incident type subscriptions
        filtered_connections = set()
        for connection_id in target_connections:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                if alert.incident_type in connection.subscribed_incident_types:
                    filtered_connections.add(connection_id)
        
        return filtered_connections
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to a specific WebSocket connection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        try:
            await connection.websocket.send(json.dumps(message))
            connection.last_ping = datetime.utcnow()
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {str(e)}")
            # Remove dead connection
            await self.disconnect(connection_id)
    
    async def _send_active_alerts_to_connection(self, connection_id: str):
        """Send all active alerts to a newly connected client"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        user_alerts = await self.get_active_alerts(connection.user_id)
        
        if user_alerts:
            message = {
                'type': 'active_alerts',
                'alerts': user_alerts,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self._send_to_connection(connection_id, message)
    
    async def _should_user_receive_alert(self, user_id: str, alert: AlertMessage) -> bool:
        """Check if user should receive this alert"""
        # Direct assignment
        if alert.user_id == user_id:
            return True
        
        # Check user's connections for location/type subscriptions
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                if connection_id in self.connections:
                    connection = self.connections[connection_id]
                    
                    # Check location subscription
                    if (not connection.subscribed_locations or 
                        alert.location in connection.subscribed_locations):
                        
                        # Check incident type subscription
                        if alert.incident_type in connection.subscribed_incident_types:
                            return True
        
        return False
    
    async def _store_alert_in_redis(self, alert: AlertMessage):
        """Store alert in Redis for persistence"""
        try:
            alert_data = asdict(alert)
            # Convert datetime objects to ISO strings for JSON serialization
            alert_data['timestamp'] = alert.timestamp.isoformat()
            if alert.acknowledged_at:
                alert_data['acknowledged_at'] = alert.acknowledged_at.isoformat()
            
            self.redis.setex(
                f"active_alert:{alert.alert_id}",
                timedelta(hours=24).total_seconds(),  # Keep for 24 hours
                json.dumps(alert_data)
            )
        except Exception as e:
            logger.error(f"Error storing alert in Redis: {str(e)}")
    
    async def _update_alert_in_redis(self, alert: AlertMessage):
        """Update alert in Redis"""
        await self._store_alert_in_redis(alert)
    
    async def _schedule_escalation(self, alert: AlertMessage):
        """Schedule alert escalation if not acknowledged"""
        if alert.priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
            escalation_delay = 300 if alert.priority == NotificationPriority.HIGH else 120  # 5 min or 2 min
            
            async def escalate():
                await asyncio.sleep(escalation_delay)
                
                if (alert.alert_id in self.active_alerts and 
                    not self.active_alerts[alert.alert_id].acknowledged):
                    
                    await self._escalate_alert(alert.alert_id)
            
            task = asyncio.create_task(escalate())
            self.escalation_tasks[alert.alert_id] = task
    
    async def _escalate_alert(self, alert_id: str):
        """Escalate unacknowledged alert"""
        try:
            if alert_id not in self.active_alerts:
                return
            
            alert = self.active_alerts[alert_id]
            
            # Create escalation message
            escalation_message = {
                'type': 'alert_escalation',
                'alert_id': alert_id,
                'original_alert': asdict(alert),
                'escalation_level': 1,  # Could be incremented for multiple escalations
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send to all admin connections
            admin_connections = [
                conn_id for conn_id, conn in self.connections.items()
                if conn.connection_type == ConnectionType.ADMIN
            ]
            
            for connection_id in admin_connections:
                await self._send_to_connection(connection_id, escalation_message)
            
            logger.warning(f"Alert {alert_id} escalated due to no acknowledgment")
            
            # Clean up escalation task
            if alert_id in self.escalation_tasks:
                del self.escalation_tasks[alert_id]
                
        except Exception as e:
            logger.error(f"Error escalating alert {alert_id}: {str(e)}")
    
    def _get_priority_weight(self, priority: str) -> int:
        """Get numeric weight for priority sorting"""
        weights = {
            'LOW': 1,
            'MEDIUM': 2,
            'HIGH': 3,
            'CRITICAL': 4
        }
        return weights.get(priority, 1)
    
    async def cleanup_expired_alerts(self):
        """Clean up expired alerts (should be called periodically)"""
        try:
            current_time = datetime.utcnow()
            expired_alerts = []
            
            for alert_id, alert in self.active_alerts.items():
                # Remove alerts older than 24 hours
                if (current_time - alert.timestamp).total_seconds() > 86400:
                    expired_alerts.append(alert_id)
            
            for alert_id in expired_alerts:
                del self.active_alerts[alert_id]
                
                # Cancel any pending escalation
                if alert_id in self.escalation_tasks:
                    self.escalation_tasks[alert_id].cancel()
                    del self.escalation_tasks[alert_id]
            
            if expired_alerts:
                logger.info(f"Cleaned up {len(expired_alerts)} expired alerts")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired alerts: {str(e)}")
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        try:
            stats = {
                'total_connections': len(self.connections),
                'connections_by_type': {},
                'active_alerts': len(self.active_alerts),
                'pending_escalations': len(self.escalation_tasks),
                'location_subscriptions': len(self.location_subscriptions)
            }
            
            # Count by connection type
            for connection in self.connections.values():
                conn_type = connection.connection_type.value
                stats['connections_by_type'][conn_type] = stats['connections_by_type'].get(conn_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {str(e)}")
            return {}