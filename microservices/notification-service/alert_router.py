"""
Alert prioritization and routing logic
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import json

from models import NotificationRequest, NotificationPriority, IncidentType, AlertEscalation
from websocket_manager import AlertMessage, WebSocketManager

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels for routing"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RoutingRule:
    """Alert routing rule configuration"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]  # Conditions to match (incident_type, location, priority, etc.)
    target_users: List[str]
    target_roles: List[str]
    channels: List[str]
    escalation_delay_minutes: int = 15
    max_escalation_levels: int = 3
    enabled: bool = True

@dataclass
class EscalationLevel:
    """Escalation level configuration"""
    level: int
    delay_minutes: int
    target_users: List[str]
    target_roles: List[str]
    channels: List[str]
    notification_template: Optional[str] = None

class AlertRouter:
    """Handles alert prioritization, routing, and escalation"""
    
    def __init__(self, redis_client: redis.Redis, websocket_manager: WebSocketManager):
        self.redis = redis_client
        self.websocket_manager = websocket_manager
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.escalation_configs: Dict[str, List[EscalationLevel]] = {}
        self.active_escalations: Dict[str, Dict[str, Any]] = {}
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default routing rules"""
        # Critical incidents - immediate escalation
        self.routing_rules["critical_incidents"] = RoutingRule(
            rule_id="critical_incidents",
            name="Critical Security Incidents",
            conditions={
                "incident_types": [IncidentType.VIOLENCE.value, IncidentType.FIRE.value, IncidentType.MEDICAL_EMERGENCY.value],
                "priority": [NotificationPriority.CRITICAL.value]
            },
            target_users=["security_supervisor", "emergency_coordinator"],
            target_roles=["security_admin", "emergency_response"],
            channels=["push", "sms", "whatsapp", "email"],
            escalation_delay_minutes=2,
            max_escalation_levels=3
        )
        
        # High priority incidents
        self.routing_rules["high_priority"] = RoutingRule(
            rule_id="high_priority",
            name="High Priority Security Events",
            conditions={
                "incident_types": [IncidentType.INTRUSION.value, IncidentType.ABANDONED_OBJECT.value],
                "priority": [NotificationPriority.HIGH.value]
            },
            target_users=["security_guard_1", "security_guard_2"],
            target_roles=["security_personnel"],
            channels=["push", "sms"],
            escalation_delay_minutes=5,
            max_escalation_levels=2
        )
        
        # Standard incidents
        self.routing_rules["standard_incidents"] = RoutingRule(
            rule_id="standard_incidents",
            name="Standard Security Events",
            conditions={
                "incident_types": [IncidentType.LOITERING.value, IncidentType.CROWDING.value],
                "priority": [NotificationPriority.MEDIUM.value, NotificationPriority.LOW.value]
            },
            target_users=["security_guard_1"],
            target_roles=["security_personnel"],
            channels=["push"],
            escalation_delay_minutes=15,
            max_escalation_levels=1
        )
        
        # System alerts
        self.routing_rules["system_alerts"] = RoutingRule(
            rule_id="system_alerts",
            name="System and Technical Alerts",
            conditions={
                "incident_types": [IncidentType.SYSTEM_ALERT.value]
            },
            target_users=["system_admin"],
            target_roles=["technical_support"],
            channels=["email", "push"],
            escalation_delay_minutes=30,
            max_escalation_levels=1
        )
    
    async def route_alert(self, notification: NotificationRequest) -> Dict[str, Any]:
        """Route alert based on priority and rules"""
        try:
            # Create alert message
            alert = AlertMessage(
                alert_id=notification.notification_id,
                incident_id=notification.incident_id or "",
                user_id=notification.user_id,
                title=notification.title,
                message=notification.message,
                priority=notification.priority,
                incident_type=self._extract_incident_type(notification),
                location=notification.metadata.get('location', 'Unknown') if notification.metadata else 'Unknown',
                timestamp=datetime.utcnow(),
                metadata=notification.metadata or {}
            )
            
            # Find matching routing rules
            matching_rules = await self._find_matching_rules(alert)
            
            if not matching_rules:
                # Use default routing if no rules match
                matching_rules = [self.routing_rules["standard_incidents"]]
            
            # Apply routing rules
            routing_results = []
            for rule in matching_rules:
                result = await self._apply_routing_rule(alert, rule)
                routing_results.append(result)
            
            # Distribute via WebSocket
            ws_result = await self.websocket_manager.distribute_alert(alert)
            
            # Schedule escalation if needed
            await self._schedule_alert_escalation(alert, matching_rules)
            
            return {
                'alert_id': alert.alert_id,
                'routing_results': routing_results,
                'websocket_result': ws_result,
                'matching_rules': [rule.rule_id for rule in matching_rules]
            }
            
        except Exception as e:
            logger.error(f"Error routing alert: {str(e)}")
            return {
                'alert_id': notification.notification_id,
                'error': str(e),
                'routing_results': [],
                'websocket_result': {}
            }
    
    async def _find_matching_rules(self, alert: AlertMessage) -> List[RoutingRule]:
        """Find routing rules that match the alert"""
        matching_rules = []
        
        for rule in self.routing_rules.values():
            if not rule.enabled:
                continue
            
            if await self._rule_matches_alert(rule, alert):
                matching_rules.append(rule)
        
        # Sort by priority (critical rules first)
        matching_rules.sort(key=lambda r: self._get_rule_priority(r), reverse=True)
        
        return matching_rules
    
    async def _rule_matches_alert(self, rule: RoutingRule, alert: AlertMessage) -> bool:
        """Check if a routing rule matches an alert"""
        conditions = rule.conditions
        
        # Check incident type
        if 'incident_types' in conditions:
            if alert.incident_type.value not in conditions['incident_types']:
                return False
        
        # Check priority
        if 'priority' in conditions:
            if alert.priority.value not in conditions['priority']:
                return False
        
        # Check location (if specified)
        if 'locations' in conditions:
            if alert.location not in conditions['locations']:
                return False
        
        # Check time-based conditions
        if 'time_range' in conditions:
            current_hour = datetime.utcnow().hour
            time_range = conditions['time_range']
            if not (time_range['start'] <= current_hour <= time_range['end']):
                return False
        
        # Check confidence threshold (if specified)
        if 'min_confidence' in conditions and alert.metadata:
            confidence = alert.metadata.get('confidence', 0)
            if confidence < conditions['min_confidence']:
                return False
        
        return True
    
    async def _apply_routing_rule(self, alert: AlertMessage, rule: RoutingRule) -> Dict[str, Any]:
        """Apply a routing rule to an alert"""
        try:
            # Get target users (direct users + users from roles)
            target_users = set(rule.target_users)
            
            # Add users from roles (this would typically query a user service)
            for role in rule.target_roles:
                role_users = await self._get_users_by_role(role)
                target_users.update(role_users)
            
            # Send notifications to target users
            notification_results = []
            for user_id in target_users:
                # Create notification request for each user
                user_notification = NotificationRequest(
                    notification_id=f"{alert.alert_id}_{user_id}",
                    user_id=user_id,
                    incident_id=alert.incident_id,
                    title=alert.title,
                    message=alert.message,
                    channels=[ch for ch in rule.channels],  # Convert to NotificationChannel enum
                    priority=alert.priority,
                    metadata=alert.metadata
                )
                
                # This would integrate with the notification service
                # For now, we'll just log the routing decision
                notification_results.append({
                    'user_id': user_id,
                    'channels': rule.channels,
                    'status': 'routed'
                })
            
            logger.info(f"Applied routing rule '{rule.name}' to alert {alert.alert_id}, "
                       f"targeting {len(target_users)} users")
            
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.name,
                'target_users': list(target_users),
                'channels': rule.channels,
                'notification_results': notification_results
            }
            
        except Exception as e:
            logger.error(f"Error applying routing rule {rule.rule_id}: {str(e)}")
            return {
                'rule_id': rule.rule_id,
                'error': str(e),
                'target_users': [],
                'notification_results': []
            }
    
    async def _schedule_alert_escalation(self, alert: AlertMessage, rules: List[RoutingRule]):
        """Schedule alert escalation based on rules"""
        try:
            # Use the most restrictive escalation timing
            min_delay = min(rule.escalation_delay_minutes for rule in rules)
            max_levels = max(rule.max_escalation_levels for rule in rules)
            
            # Create escalation configuration
            escalation_config = {
                'alert_id': alert.alert_id,
                'levels': [],
                'current_level': 0,
                'max_levels': max_levels,
                'started_at': datetime.utcnow()
            }
            
            # Define escalation levels
            for level in range(1, max_levels + 1):
                escalation_level = {
                    'level': level,
                    'delay_minutes': min_delay * level,
                    'target_users': self._get_escalation_users(level),
                    'channels': self._get_escalation_channels(level),
                    'executed': False
                }
                escalation_config['levels'].append(escalation_level)
            
            # Store escalation config
            self.active_escalations[alert.alert_id] = escalation_config
            
            # Schedule first escalation
            await self._schedule_next_escalation(alert.alert_id)
            
        except Exception as e:
            logger.error(f"Error scheduling escalation for alert {alert.alert_id}: {str(e)}")
    
    async def _schedule_next_escalation(self, alert_id: str):
        """Schedule the next escalation level"""
        if alert_id not in self.active_escalations:
            return
        
        escalation = self.active_escalations[alert_id]
        current_level = escalation['current_level']
        
        if current_level >= len(escalation['levels']):
            return  # No more escalation levels
        
        level_config = escalation['levels'][current_level]
        delay_seconds = level_config['delay_minutes'] * 60
        
        async def escalate():
            await asyncio.sleep(delay_seconds)
            await self._execute_escalation(alert_id, current_level)
        
        # Create escalation task
        task = asyncio.create_task(escalate())
        escalation[f'task_level_{current_level}'] = task
    
    async def _execute_escalation(self, alert_id: str, level: int):
        """Execute an escalation level"""
        try:
            if alert_id not in self.active_escalations:
                return
            
            escalation = self.active_escalations[alert_id]
            
            # Check if alert was acknowledged
            if alert_id in self.websocket_manager.active_alerts:
                alert = self.websocket_manager.active_alerts[alert_id]
                if alert.acknowledged:
                    logger.info(f"Alert {alert_id} was acknowledged, canceling escalation")
                    return
            
            level_config = escalation['levels'][level]
            
            # Create escalation notification
            escalation_message = {
                'type': 'alert_escalation',
                'alert_id': alert_id,
                'escalation_level': level + 1,
                'target_users': level_config['target_users'],
                'channels': level_config['channels'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send escalation notifications
            for user_id in level_config['target_users']:
                # This would integrate with the notification service
                logger.warning(f"ESCALATION Level {level + 1}: Alert {alert_id} to user {user_id}")
            
            # Mark level as executed
            level_config['executed'] = True
            escalation['current_level'] = level + 1
            
            # Schedule next level if available
            if escalation['current_level'] < len(escalation['levels']):
                await self._schedule_next_escalation(alert_id)
            else:
                logger.critical(f"Alert {alert_id} reached maximum escalation level")
            
        except Exception as e:
            logger.error(f"Error executing escalation for alert {alert_id}, level {level}: {str(e)}")
    
    async def acknowledge_alert_escalation(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert and cancel escalation"""
        try:
            # Cancel escalation
            if alert_id in self.active_escalations:
                escalation = self.active_escalations[alert_id]
                
                # Cancel any pending escalation tasks
                for key, value in escalation.items():
                    if key.startswith('task_level_') and hasattr(value, 'cancel'):
                        value.cancel()
                
                # Remove from active escalations
                del self.active_escalations[alert_id]
                
                logger.info(f"Escalation canceled for alert {alert_id} by user {user_id}")
            
            # Acknowledge in WebSocket manager
            return await self.websocket_manager.acknowledge_alert(alert_id, user_id)
            
        except Exception as e:
            logger.error(f"Error acknowledging alert escalation {alert_id}: {str(e)}")
            return False
    
    def _extract_incident_type(self, notification: NotificationRequest) -> IncidentType:
        """Extract incident type from notification"""
        if notification.metadata and 'incident_type' in notification.metadata:
            try:
                return IncidentType(notification.metadata['incident_type'])
            except ValueError:
                pass
        
        # Default to system alert if not specified
        return IncidentType.SYSTEM_ALERT
    
    def _get_rule_priority(self, rule: RoutingRule) -> int:
        """Get priority weight for rule sorting"""
        # Rules with critical conditions get higher priority
        conditions = rule.conditions
        
        if 'priority' in conditions:
            if NotificationPriority.CRITICAL.value in conditions['priority']:
                return 4
            elif NotificationPriority.HIGH.value in conditions['priority']:
                return 3
            elif NotificationPriority.MEDIUM.value in conditions['priority']:
                return 2
        
        return 1
    
    async def _get_users_by_role(self, role: str) -> List[str]:
        """Get users by role (placeholder - would integrate with user service)"""
        # This is a placeholder implementation
        # In a real system, this would query a user service or database
        role_mappings = {
            'security_admin': ['security_supervisor', 'security_manager'],
            'security_personnel': ['security_guard_1', 'security_guard_2', 'security_guard_3'],
            'emergency_response': ['emergency_coordinator', 'fire_chief', 'medical_coordinator'],
            'technical_support': ['system_admin', 'it_support']
        }
        
        return role_mappings.get(role, [])
    
    def _get_escalation_users(self, level: int) -> List[str]:
        """Get users for escalation level"""
        escalation_users = {
            1: ['security_supervisor'],
            2: ['security_manager', 'emergency_coordinator'],
            3: ['campus_director', 'police_liaison']
        }
        
        return escalation_users.get(level, [])
    
    def _get_escalation_channels(self, level: int) -> List[str]:
        """Get notification channels for escalation level"""
        escalation_channels = {
            1: ['push', 'sms'],
            2: ['push', 'sms', 'whatsapp'],
            3: ['push', 'sms', 'whatsapp', 'email']
        }
        
        return escalation_channels.get(level, ['push'])
    
    async def add_routing_rule(self, rule: RoutingRule) -> bool:
        """Add a new routing rule"""
        try:
            self.routing_rules[rule.rule_id] = rule
            
            # Store in Redis for persistence
            rule_data = asdict(rule)
            self.redis.setex(
                f"routing_rule:{rule.rule_id}",
                timedelta(days=365).total_seconds(),
                json.dumps(rule_data)
            )
            
            logger.info(f"Added routing rule: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding routing rule {rule.rule_id}: {str(e)}")
            return False
    
    async def remove_routing_rule(self, rule_id: str) -> bool:
        """Remove a routing rule"""
        try:
            if rule_id in self.routing_rules:
                del self.routing_rules[rule_id]
                
                # Remove from Redis
                self.redis.delete(f"routing_rule:{rule_id}")
                
                logger.info(f"Removed routing rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing routing rule {rule_id}: {str(e)}")
            return False
    
    async def get_escalation_stats(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        try:
            stats = {
                'active_escalations': len(self.active_escalations),
                'escalations_by_level': {},
                'routing_rules': len(self.routing_rules)
            }
            
            # Count escalations by level
            for escalation in self.active_escalations.values():
                level = escalation['current_level']
                stats['escalations_by_level'][level] = stats['escalations_by_level'].get(level, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting escalation stats: {str(e)}")
            return {}