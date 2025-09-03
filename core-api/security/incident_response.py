"""
Security incident response procedures for campus security system.
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
import aiofiles
from uuid import uuid4

logger = structlog.get_logger()


class IncidentSeverity(Enum):
    """Security incident severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentStatus(Enum):
    """Security incident status."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentType(Enum):
    """Types of security incidents."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTION = "malware_detection"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSIDER_THREAT = "insider_threat"
    PHYSICAL_SECURITY = "physical_security"
    SYSTEM_COMPROMISE = "system_compromise"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class SecurityIncident:
    """Security incident data structure."""
    id: str
    title: str
    description: str
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus
    detected_at: datetime
    reported_by: str
    assigned_to: Optional[str] = None
    affected_systems: List[str] = None
    indicators_of_compromise: List[str] = None
    response_actions: List[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.affected_systems is None:
            self.affected_systems = []
        if self.indicators_of_compromise is None:
            self.indicators_of_compromise = []
        if self.response_actions is None:
            self.response_actions = []


@dataclass
class ResponseAction:
    """Security response action."""
    id: str
    incident_id: str
    action_type: str
    description: str
    assigned_to: str
    status: str
    priority: int
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class IncidentResponseProcedures:
    """Security incident response procedures manager."""
    
    def __init__(self):
        self.incidents: Dict[str, SecurityIncident] = {}
        self.response_actions: Dict[str, ResponseAction] = {}
        self.response_playbooks: Dict[IncidentType, List[Dict]] = {}
        self.notification_handlers: List[Callable] = []
        self.escalation_rules: Dict[IncidentSeverity, Dict] = {}
        self.initialize_playbooks()
        self.initialize_escalation_rules()
    
    def initialize_playbooks(self):
        """Initialize incident response playbooks."""
        self.response_playbooks = {
            IncidentType.UNAUTHORIZED_ACCESS: [
                {
                    "action": "Isolate affected systems",
                    "priority": 1,
                    "timeout_minutes": 15,
                    "description": "Immediately isolate systems showing signs of unauthorized access"
                },
                {
                    "action": "Preserve evidence",
                    "priority": 2,
                    "timeout_minutes": 30,
                    "description": "Create forensic images and preserve log files"
                },
                {
                    "action": "Reset credentials",
                    "priority": 3,
                    "timeout_minutes": 60,
                    "description": "Reset all potentially compromised credentials"
                },
                {
                    "action": "Analyze attack vectors",
                    "priority": 4,
                    "timeout_minutes": 120,
                    "description": "Determine how unauthorized access was gained"
                }
            ],
            IncidentType.DATA_BREACH: [
                {
                    "action": "Contain the breach",
                    "priority": 1,
                    "timeout_minutes": 10,
                    "description": "Stop ongoing data exfiltration"
                },
                {
                    "action": "Assess data exposure",
                    "priority": 2,
                    "timeout_minutes": 60,
                    "description": "Determine what data was accessed or stolen"
                },
                {
                    "action": "Notify stakeholders",
                    "priority": 3,
                    "timeout_minutes": 120,
                    "description": "Inform management and prepare breach notifications"
                },
                {
                    "action": "Legal compliance review",
                    "priority": 4,
                    "timeout_minutes": 240,
                    "description": "Review legal obligations for breach notification"
                }
            ],
            IncidentType.MALWARE_DETECTION: [
                {
                    "action": "Isolate infected systems",
                    "priority": 1,
                    "timeout_minutes": 5,
                    "description": "Immediately disconnect infected systems from network"
                },
                {
                    "action": "Identify malware type",
                    "priority": 2,
                    "timeout_minutes": 30,
                    "description": "Analyze malware samples to understand capabilities"
                },
                {
                    "action": "Scan all systems",
                    "priority": 3,
                    "timeout_minutes": 120,
                    "description": "Perform comprehensive malware scan across infrastructure"
                },
                {
                    "action": "Update security controls",
                    "priority": 4,
                    "timeout_minutes": 180,
                    "description": "Update antivirus signatures and security rules"
                }
            ],
            IncidentType.VULNERABILITY_EXPLOIT: [
                {
                    "action": "Apply emergency patches",
                    "priority": 1,
                    "timeout_minutes": 30,
                    "description": "Apply patches for exploited vulnerabilities"
                },
                {
                    "action": "Block exploit attempts",
                    "priority": 2,
                    "timeout_minutes": 15,
                    "description": "Update firewall rules to block known exploit patterns"
                },
                {
                    "action": "Scan for indicators",
                    "priority": 3,
                    "timeout_minutes": 60,
                    "description": "Search for indicators of successful exploitation"
                },
                {
                    "action": "Vulnerability assessment",
                    "priority": 4,
                    "timeout_minutes": 240,
                    "description": "Conduct comprehensive vulnerability assessment"
                }
            ],
            IncidentType.SYSTEM_COMPROMISE: [
                {
                    "action": "Isolate compromised systems",
                    "priority": 1,
                    "timeout_minutes": 10,
                    "description": "Immediately isolate compromised systems"
                },
                {
                    "action": "Preserve system state",
                    "priority": 2,
                    "timeout_minutes": 30,
                    "description": "Create forensic images before cleanup"
                },
                {
                    "action": "Analyze compromise",
                    "priority": 3,
                    "timeout_minutes": 120,
                    "description": "Determine extent and method of compromise"
                },
                {
                    "action": "Rebuild systems",
                    "priority": 4,
                    "timeout_minutes": 480,
                    "description": "Rebuild compromised systems from clean backups"
                }
            ]
        }
        logger.info("Incident response playbooks initialized")
    
    def initialize_escalation_rules(self):
        """Initialize incident escalation rules."""
        self.escalation_rules = {
            IncidentSeverity.CRITICAL: {
                "immediate_notification": ["security_team", "management", "legal"],
                "escalation_timeout_minutes": 15,
                "auto_escalate_to": ["ciso", "ceo"],
                "external_notification_required": True
            },
            IncidentSeverity.HIGH: {
                "immediate_notification": ["security_team", "management"],
                "escalation_timeout_minutes": 60,
                "auto_escalate_to": ["security_manager"],
                "external_notification_required": False
            },
            IncidentSeverity.MEDIUM: {
                "immediate_notification": ["security_team"],
                "escalation_timeout_minutes": 240,
                "auto_escalate_to": ["security_lead"],
                "external_notification_required": False
            },
            IncidentSeverity.LOW: {
                "immediate_notification": ["security_analyst"],
                "escalation_timeout_minutes": 1440,  # 24 hours
                "auto_escalate_to": [],
                "external_notification_required": False
            }
        }
        logger.info("Incident escalation rules initialized")
    
    async def create_incident(self, 
                            title: str,
                            description: str,
                            incident_type: IncidentType,
                            severity: IncidentSeverity,
                            reported_by: str,
                            affected_systems: List[str] = None,
                            indicators_of_compromise: List[str] = None) -> SecurityIncident:
        """Create a new security incident."""
        incident_id = str(uuid4())
        
        incident = SecurityIncident(
            id=incident_id,
            title=title,
            description=description,
            incident_type=incident_type,
            severity=severity,
            status=IncidentStatus.DETECTED,
            detected_at=datetime.utcnow(),
            reported_by=reported_by,
            affected_systems=affected_systems or [],
            indicators_of_compromise=indicators_of_compromise or []
        )
        
        self.incidents[incident_id] = incident
        
        # Trigger immediate response
        await self._trigger_immediate_response(incident)
        
        logger.info("Security incident created",
                   incident_id=incident_id,
                   type=incident_type.value,
                   severity=severity.value)
        
        return incident
    
    async def _trigger_immediate_response(self, incident: SecurityIncident):
        """Trigger immediate response actions for an incident."""
        # Send notifications based on severity
        await self._send_incident_notifications(incident)
        
        # Create response actions from playbook
        await self._create_response_actions(incident)
        
        # Start escalation timer
        await self._start_escalation_timer(incident)
        
        logger.info("Immediate response triggered", incident_id=incident.id)
    
    async def _send_incident_notifications(self, incident: SecurityIncident):
        """Send incident notifications based on severity."""
        escalation_rule = self.escalation_rules.get(incident.severity)
        if not escalation_rule:
            return
        
        notification_groups = escalation_rule.get("immediate_notification", [])
        
        notification_data = {
            "incident_id": incident.id,
            "title": incident.title,
            "severity": incident.severity.value,
            "type": incident.incident_type.value,
            "detected_at": incident.detected_at.isoformat(),
            "affected_systems": incident.affected_systems
        }
        
        # Send notifications to each group
        for group in notification_groups:
            for handler in self.notification_handlers:
                try:
                    await handler(group, notification_data)
                except Exception as e:
                    logger.error("Failed to send incident notification",
                               group=group, incident_id=incident.id, error=str(e))
        
        logger.info("Incident notifications sent",
                   incident_id=incident.id,
                   groups=notification_groups)
    
    async def _create_response_actions(self, incident: SecurityIncident):
        """Create response actions from incident playbook."""
        playbook = self.response_playbooks.get(incident.incident_type, [])
        
        for action_template in playbook:
            action_id = str(uuid4())
            
            due_date = None
            if action_template.get("timeout_minutes"):
                due_date = datetime.utcnow() + timedelta(minutes=action_template["timeout_minutes"])
            
            action = ResponseAction(
                id=action_id,
                incident_id=incident.id,
                action_type=action_template["action"],
                description=action_template["description"],
                assigned_to="security_team",  # Default assignment
                status="pending",
                priority=action_template["priority"],
                due_date=due_date
            )
            
            self.response_actions[action_id] = action
            incident.response_actions.append(action_id)
        
        logger.info("Response actions created",
                   incident_id=incident.id,
                   actions_count=len(playbook))
    
    async def _start_escalation_timer(self, incident: SecurityIncident):
        """Start escalation timer for incident."""
        escalation_rule = self.escalation_rules.get(incident.severity)
        if not escalation_rule:
            return
        
        timeout_minutes = escalation_rule.get("escalation_timeout_minutes", 60)
        
        # Schedule escalation check
        asyncio.create_task(self._check_escalation(incident.id, timeout_minutes))
        
        logger.info("Escalation timer started",
                   incident_id=incident.id,
                   timeout_minutes=timeout_minutes)
    
    async def _check_escalation(self, incident_id: str, timeout_minutes: int):
        """Check if incident needs escalation after timeout."""
        await asyncio.sleep(timeout_minutes * 60)  # Convert to seconds
        
        incident = self.incidents.get(incident_id)
        if not incident:
            return
        
        # Check if incident is still open
        if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            return
        
        # Escalate incident
        await self._escalate_incident(incident)
    
    async def _escalate_incident(self, incident: SecurityIncident):
        """Escalate incident to higher authority."""
        escalation_rule = self.escalation_rules.get(incident.severity)
        if not escalation_rule:
            return
        
        escalate_to = escalation_rule.get("auto_escalate_to", [])
        
        if escalate_to:
            notification_data = {
                "incident_id": incident.id,
                "title": f"ESCALATED: {incident.title}",
                "severity": incident.severity.value,
                "type": incident.incident_type.value,
                "detected_at": incident.detected_at.isoformat(),
                "escalation_reason": "Timeout without resolution"
            }
            
            for recipient in escalate_to:
                for handler in self.notification_handlers:
                    try:
                        await handler(recipient, notification_data)
                    except Exception as e:
                        logger.error("Failed to send escalation notification",
                                   recipient=recipient, incident_id=incident.id, error=str(e))
        
        logger.warning("Incident escalated",
                      incident_id=incident.id,
                      escalated_to=escalate_to)
    
    async def update_incident_status(self, incident_id: str, status: IncidentStatus, notes: str = None):
        """Update incident status."""
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        old_status = incident.status
        incident.status = status
        incident.updated_at = datetime.utcnow()
        
        if status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.utcnow()
            incident.resolution_notes = notes
        
        logger.info("Incident status updated",
                   incident_id=incident_id,
                   old_status=old_status.value,
                   new_status=status.value)
    
    async def assign_incident(self, incident_id: str, assigned_to: str):
        """Assign incident to a person or team."""
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident.assigned_to = assigned_to
        incident.updated_at = datetime.utcnow()
        
        logger.info("Incident assigned",
                   incident_id=incident_id,
                   assigned_to=assigned_to)
    
    async def complete_response_action(self, action_id: str, notes: str = None):
        """Mark response action as completed."""
        action = self.response_actions.get(action_id)
        if not action:
            raise ValueError(f"Response action {action_id} not found")
        
        action.status = "completed"
        action.completed_at = datetime.utcnow()
        action.notes = notes
        
        logger.info("Response action completed",
                   action_id=action_id,
                   incident_id=action.incident_id)
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler for incident alerts."""
        self.notification_handlers.append(handler)
        logger.info("Notification handler added")
    
    async def generate_incident_report(self, incident_id: str) -> Dict:
        """Generate comprehensive incident report."""
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        # Get related response actions
        related_actions = [
            asdict(action) for action in self.response_actions.values()
            if action.incident_id == incident_id
        ]
        
        # Calculate response metrics
        response_time = None
        if incident.resolved_at:
            response_time = (incident.resolved_at - incident.detected_at).total_seconds()
        
        completed_actions = len([a for a in related_actions if a["status"] == "completed"])
        total_actions = len(related_actions)
        
        report = {
            "incident": asdict(incident),
            "response_actions": related_actions,
            "metrics": {
                "response_time_seconds": response_time,
                "actions_completed": completed_actions,
                "total_actions": total_actions,
                "completion_rate": completed_actions / total_actions if total_actions > 0 else 0
            },
            "timeline": self._generate_incident_timeline(incident, related_actions),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return report
    
    def _generate_incident_timeline(self, incident: SecurityIncident, actions: List[Dict]) -> List[Dict]:
        """Generate incident timeline."""
        timeline = []
        
        # Add incident creation
        timeline.append({
            "timestamp": incident.detected_at.isoformat(),
            "event": "Incident detected",
            "description": incident.title,
            "type": "incident"
        })
        
        # Add status changes
        if incident.resolved_at:
            timeline.append({
                "timestamp": incident.resolved_at.isoformat(),
                "event": "Incident resolved",
                "description": incident.resolution_notes or "Incident resolved",
                "type": "resolution"
            })
        
        # Add completed actions
        for action in actions:
            if action["completed_at"]:
                timeline.append({
                    "timestamp": action["completed_at"],
                    "event": "Action completed",
                    "description": action["description"],
                    "type": "action"
                })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        return timeline
    
    async def export_incidents(self, output_file: str, status_filter: List[IncidentStatus] = None):
        """Export incidents to file."""
        incidents_to_export = []
        
        for incident in self.incidents.values():
            if status_filter and incident.status not in status_filter:
                continue
            
            incident_data = asdict(incident)
            incident_data["response_actions"] = [
                asdict(action) for action in self.response_actions.values()
                if action.incident_id == incident.id
            ]
            incidents_to_export.append(incident_data)
        
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "incidents": incidents_to_export,
            "total_incidents": len(incidents_to_export)
        }
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(export_data, indent=2, default=str))
        
        logger.info("Incidents exported", output_file=output_file, count=len(incidents_to_export))


# Global incident response procedures instance
incident_response = IncidentResponseProcedures()