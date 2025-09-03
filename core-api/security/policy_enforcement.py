"""
Automated policy enforcement for compliance and data protection.
Monitors and enforces privacy, security, and compliance policies in real-time.
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
from uuid import uuid4

from core.database import get_db_session
from core.redis import redis_manager

logger = structlog.get_logger()


class PolicyType(Enum):
    """Types of policies that can be enforced."""
    DATA_RETENTION = "data_retention"
    ACCESS_CONTROL = "access_control"
    DATA_CLASSIFICATION = "data_classification"
    PRIVACY_PROTECTION = "privacy_protection"
    AUDIT_LOGGING = "audit_logging"
    ENCRYPTION = "encryption"
    DATA_TRANSFER = "data_transfer"
    CONSENT_MANAGEMENT = "consent_management"


class PolicyStatus(Enum):
    """Policy enforcement status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    UNDER_REVIEW = "under_review"


class ViolationSeverity(Enum):
    """Severity levels for policy violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EnforcementAction(Enum):
    """Actions that can be taken when policies are violated."""
    LOG_ONLY = "log_only"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    DELETE = "delete"
    ENCRYPT = "encrypt"
    ANONYMIZE = "anonymize"
    NOTIFY_ADMIN = "notify_admin"
    ESCALATE = "escalate"


@dataclass
class PolicyRule:
    """Individual policy rule definition."""
    id: str
    name: str
    description: str
    policy_type: PolicyType
    condition: str  # JSON string representing the condition
    action: EnforcementAction
    severity: ViolationSeverity
    enabled: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PolicyViolation:
    """Policy violation record."""
    id: str
    rule_id: str
    policy_type: PolicyType
    violation_description: str
    severity: ViolationSeverity
    detected_at: datetime
    resource_id: str
    resource_type: str
    action_taken: EnforcementAction
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PolicyEnforcementMetrics:
    """Metrics for policy enforcement."""
    total_rules: int
    active_rules: int
    violations_detected: int
    violations_resolved: int
    violations_by_severity: Dict[str, int]
    violations_by_type: Dict[str, int]
    enforcement_actions: Dict[str, int]
    last_updated: datetime


class PolicyEnforcementEngine:
    """Automated policy enforcement engine."""
    
    def __init__(self):
        self.rules: Dict[str, PolicyRule] = {}
        self.violations: Dict[str, PolicyViolation] = {}
        self.enforcement_handlers: Dict[PolicyType, Callable] = {}
        self.metrics = PolicyEnforcementMetrics(
            total_rules=0,
            active_rules=0,
            violations_detected=0,
            violations_resolved=0,
            violations_by_severity={},
            violations_by_type={},
            enforcement_actions={},
            last_updated=datetime.utcnow()
        )
        self.initialize_default_rules()
        self.register_enforcement_handlers()
    
    def initialize_default_rules(self):
        """Initialize default policy enforcement rules."""
        default_rules = [
            # Data Retention Rules
            PolicyRule(
                id="data_retention_001",
                name="Expired Personal Data Detection",
                description="Detect personal data that has exceeded retention period",
                policy_type=PolicyType.DATA_RETENTION,
                condition=json.dumps({
                    "data_category": "personal_data",
                    "retention_expired": True,
                    "auto_delete_enabled": False
                }),
                action=EnforcementAction.NOTIFY_ADMIN,
                severity=ViolationSeverity.MEDIUM,
                enabled=True,
                created_at=datetime.utcnow()
            ),
            PolicyRule(
                id="data_retention_002",
                name="Video Evidence Retention Violation",
                description="Video evidence stored beyond legal retention period",
                policy_type=PolicyType.DATA_RETENTION,
                condition=json.dumps({
                    "data_category": "video_evidence",
                    "retention_days_exceeded": 90,
                    "requires_approval": True
                }),
                action=EnforcementAction.QUARANTINE,
                severity=ViolationSeverity.HIGH,
                enabled=True,
                created_at=datetime.utcnow()
            ),
            
            # Access Control Rules
            PolicyRule(
                id="access_control_001",
                name="Unauthorized Sensitive Data Access",
                description="Detect unauthorized access to sensitive data",
                policy_type=PolicyType.ACCESS_CONTROL,
                condition=json.dumps({
                    "data_classification": "sensitive",
                    "user_clearance_level": "insufficient",
                    "access_attempted": True
                }),
                action=EnforcementAction.BLOCK,
                severity=ViolationSeverity.HIGH,
                enabled=True,
                created_at=datetime.utcnow()
            ),
            PolicyRule(
                id="access_control_002",
                name="Excessive Privilege Usage",
                description="Detect users with excessive privileges",
                policy_type=PolicyType.ACCESS_CONTROL,
                condition=json.dumps({
                    "privilege_level": "admin",
                    "last_used_days": 30,
                    "regular_usage": False
                }),
                action=EnforcementAction.WARN,
                severity=ViolationSeverity.MEDIUM,
                enabled=True,
                created_at=datetime.utcnow()
            ),
            
            # Privacy Protection Rules
            PolicyRule(
                id="privacy_001",
                name="Unredacted Personal Data in Logs",
                description="Detect personal data in system logs without redaction",
                policy_type=PolicyType.PRIVACY_PROTECTION,
                condition=json.dumps({
                    "log_contains_pii": True,
                    "redaction_applied": False,
                    "log_level": ["info", "debug"]
                }),
                action=EnforcementAction.ANONYMIZE,
                severity=ViolationSeverity.HIGH,
                enabled=True,
                created_at=datetime.utcnow()
            ),
            PolicyRule(
                id="privacy_002",
                name="Biometric Data Without Consent",
                description="Biometric data processing without explicit consent",
                policy_type=PolicyType.PRIVACY_PROTECTION,
                condition=json.dumps({
                    "data_type": "biometric",
                    "consent_status": "missing",
                    "processing_active": True
                }),
                action=EnforcementAction.BLOCK,
                severity=ViolationSeverity.CRITICAL,
                enabled=True,
                created_at=datetime.utcnow()
            ),
            
            # Encryption Rules
            PolicyRule(
                id="encryption_001",
                name="Unencrypted Sensitive Data Storage",
                description="Sensitive data stored without encryption",
                policy_type=PolicyType.ENCRYPTION,
                condition=json.dumps({
                    "data_classification": ["sensitive", "confidential"],
                    "encryption_status": "unencrypted",
                    "storage_type": "persistent"
                }),
                action=EnforcementAction.ENCRYPT,
                severity=ViolationSeverity.HIGH,
                enabled=True,
                created_at=datetime.utcnow()
            ),
            
            # Data Transfer Rules
            PolicyRule(
                id="data_transfer_001",
                name="Cross-Border Transfer Without Adequacy",
                description="Personal data transfer to countries without adequacy decision",
                policy_type=PolicyType.DATA_TRANSFER,
                condition=json.dumps({
                    "transfer_type": "cross_border",
                    "destination_adequacy": False,
                    "safeguards_in_place": False
                }),
                action=EnforcementAction.BLOCK,
                severity=ViolationSeverity.CRITICAL,
                enabled=True,
                created_at=datetime.utcnow()
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
        
        self._update_metrics()
        logger.info("Default policy enforcement rules initialized", count=len(default_rules))
    
    def register_enforcement_handlers(self):
        """Register handlers for different policy types."""
        self.enforcement_handlers = {
            PolicyType.DATA_RETENTION: self._handle_data_retention_violation,
            PolicyType.ACCESS_CONTROL: self._handle_access_control_violation,
            PolicyType.PRIVACY_PROTECTION: self._handle_privacy_violation,
            PolicyType.ENCRYPTION: self._handle_encryption_violation,
            PolicyType.DATA_TRANSFER: self._handle_data_transfer_violation,
            PolicyType.AUDIT_LOGGING: self._handle_audit_logging_violation,
            PolicyType.DATA_CLASSIFICATION: self._handle_data_classification_violation,
            PolicyType.CONSENT_MANAGEMENT: self._handle_consent_violation
        }
        
        logger.info("Policy enforcement handlers registered")
    
    async def evaluate_policies(self, context: Dict[str, Any]) -> List[PolicyViolation]:
        """Evaluate all active policies against the given context."""
        violations = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            try:
                if await self._evaluate_rule(rule, context):
                    violation = await self._create_violation(rule, context)
                    violations.append(violation)
                    await self._enforce_policy(violation, context)
                    
            except Exception as e:
                logger.error("Policy evaluation failed",
                           rule_id=rule.id,
                           error=str(e))
        
        if violations:
            self._update_metrics()
            logger.info("Policy violations detected",
                       count=len(violations),
                       context_type=context.get("type", "unknown"))
        
        return violations
    
    async def _evaluate_rule(self, rule: PolicyRule, context: Dict[str, Any]) -> bool:
        """Evaluate a single policy rule against context."""
        try:
            condition = json.loads(rule.condition)
            return self._match_condition(condition, context)
        except Exception as e:
            logger.error("Rule condition evaluation failed",
                        rule_id=rule.id,
                        error=str(e))
            return False
    
    def _match_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if context matches the rule condition."""
        for key, expected_value in condition.items():
            context_value = context.get(key)
            
            if isinstance(expected_value, list):
                if context_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Handle nested conditions
                if not isinstance(context_value, dict):
                    return False
                if not self._match_condition(expected_value, context_value):
                    return False
            else:
                if context_value != expected_value:
                    return False
        
        return True
    
    async def _create_violation(self, rule: PolicyRule, context: Dict[str, Any]) -> PolicyViolation:
        """Create a policy violation record."""
        violation_id = str(uuid4())
        
        violation = PolicyViolation(
            id=violation_id,
            rule_id=rule.id,
            policy_type=rule.policy_type,
            violation_description=f"Policy violation: {rule.name}",
            severity=rule.severity,
            detected_at=datetime.utcnow(),
            resource_id=context.get("resource_id", "unknown"),
            resource_type=context.get("resource_type", "unknown"),
            action_taken=rule.action,
            metadata=context
        )
        
        self.violations[violation_id] = violation
        
        # Store in database
        await self._store_violation(violation)
        
        logger.warning("Policy violation detected",
                      violation_id=violation_id,
                      rule_id=rule.id,
                      severity=rule.severity.value,
                      resource_id=violation.resource_id)
        
        return violation
    
    async def _store_violation(self, violation: PolicyViolation):
        """Store violation in database."""
        async with get_db_session() as db:
            await db.execute("""
                INSERT INTO policy_violations 
                (id, rule_id, policy_type, violation_description, severity, 
                 detected_at, resource_id, resource_type, action_taken, resolved, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (id) DO UPDATE SET
                    resolved = EXCLUDED.resolved,
                    resolved_at = EXCLUDED.resolved_at,
                    resolved_by = EXCLUDED.resolved_by
            """, violation.id, violation.rule_id, violation.policy_type.value,
                violation.violation_description, violation.severity.value,
                violation.detected_at, violation.resource_id, violation.resource_type,
                violation.action_taken.value, violation.resolved,
                json.dumps(violation.metadata) if violation.metadata else None)
    
    async def _enforce_policy(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Enforce policy by taking the specified action."""
        handler = self.enforcement_handlers.get(violation.policy_type)
        if handler:
            try:
                await handler(violation, context)
                logger.info("Policy enforcement action completed",
                           violation_id=violation.id,
                           action=violation.action_taken.value)
            except Exception as e:
                logger.error("Policy enforcement failed",
                           violation_id=violation.id,
                           action=violation.action_taken.value,
                           error=str(e))
        else:
            logger.warning("No enforcement handler found",
                          policy_type=violation.policy_type.value)
    
    # Enforcement Handlers
    async def _handle_data_retention_violation(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Handle data retention policy violations."""
        if violation.action_taken == EnforcementAction.NOTIFY_ADMIN:
            await self._notify_administrators(violation, "Data retention policy violation detected")
        elif violation.action_taken == EnforcementAction.QUARANTINE:
            await self._quarantine_data(context.get("resource_id"))
        elif violation.action_taken == EnforcementAction.DELETE:
            await self._delete_expired_data(context.get("resource_id"))
    
    async def _handle_access_control_violation(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Handle access control policy violations."""
        if violation.action_taken == EnforcementAction.BLOCK:
            await self._block_access(context.get("user_id"), context.get("resource_id"))
        elif violation.action_taken == EnforcementAction.WARN:
            await self._warn_user(context.get("user_id"), violation.violation_description)
        elif violation.action_taken == EnforcementAction.ESCALATE:
            await self._escalate_to_security_team(violation)
    
    async def _handle_privacy_violation(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Handle privacy protection policy violations."""
        if violation.action_taken == EnforcementAction.ANONYMIZE:
            await self._anonymize_data(context.get("resource_id"))
        elif violation.action_taken == EnforcementAction.BLOCK:
            await self._block_processing(context.get("resource_id"))
        elif violation.action_taken == EnforcementAction.ENCRYPT:
            await self._encrypt_data(context.get("resource_id"))
    
    async def _handle_encryption_violation(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Handle encryption policy violations."""
        if violation.action_taken == EnforcementAction.ENCRYPT:
            await self._encrypt_data(context.get("resource_id"))
        elif violation.action_taken == EnforcementAction.QUARANTINE:
            await self._quarantine_data(context.get("resource_id"))
    
    async def _handle_data_transfer_violation(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Handle data transfer policy violations."""
        if violation.action_taken == EnforcementAction.BLOCK:
            await self._block_data_transfer(context.get("transfer_id"))
        elif violation.action_taken == EnforcementAction.ESCALATE:
            await self._escalate_to_dpo(violation)
    
    async def _handle_audit_logging_violation(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Handle audit logging policy violations."""
        if violation.action_taken == EnforcementAction.LOG_ONLY:
            await self._enhanced_logging(context)
        elif violation.action_taken == EnforcementAction.NOTIFY_ADMIN:
            await self._notify_administrators(violation, "Audit logging policy violation")
    
    async def _handle_data_classification_violation(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Handle data classification policy violations."""
        if violation.action_taken == EnforcementAction.QUARANTINE:
            await self._quarantine_data(context.get("resource_id"))
        elif violation.action_taken == EnforcementAction.ENCRYPT:
            await self._encrypt_data(context.get("resource_id"))
    
    async def _handle_consent_violation(self, violation: PolicyViolation, context: Dict[str, Any]):
        """Handle consent management policy violations."""
        if violation.action_taken == EnforcementAction.BLOCK:
            await self._block_processing(context.get("resource_id"))
        elif violation.action_taken == EnforcementAction.DELETE:
            await self._delete_data_without_consent(context.get("resource_id"))
    
    # Enforcement Action Implementations
    async def _notify_administrators(self, violation: PolicyViolation, message: str):
        """Notify system administrators of policy violation."""
        notification = {
            "type": "policy_violation",
            "violation_id": violation.id,
            "severity": violation.severity.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store notification in Redis for real-time delivery
        await redis_manager.lpush("admin_notifications", json.dumps(notification))
        logger.info("Administrator notification sent", violation_id=violation.id)
    
    async def _quarantine_data(self, resource_id: str):
        """Quarantine data by moving it to secure storage."""
        logger.info("Quarantining data", resource_id=resource_id)
        # Implementation would move data to quarantine storage
    
    async def _delete_expired_data(self, resource_id: str):
        """Delete expired data according to retention policies."""
        logger.info("Deleting expired data", resource_id=resource_id)
        # Implementation would securely delete the data
    
    async def _block_access(self, user_id: str, resource_id: str):
        """Block user access to specific resource."""
        logger.info("Blocking access", user_id=user_id, resource_id=resource_id)
        # Implementation would update access control lists
    
    async def _warn_user(self, user_id: str, message: str):
        """Send warning to user about policy violation."""
        logger.info("Warning user", user_id=user_id, message=message)
        # Implementation would send user notification
    
    async def _escalate_to_security_team(self, violation: PolicyViolation):
        """Escalate violation to security team."""
        logger.info("Escalating to security team", violation_id=violation.id)
        # Implementation would create security incident
    
    async def _anonymize_data(self, resource_id: str):
        """Anonymize data by removing identifying information."""
        logger.info("Anonymizing data", resource_id=resource_id)
        # Implementation would remove/hash identifying fields
    
    async def _block_processing(self, resource_id: str):
        """Block data processing for specific resource."""
        logger.info("Blocking data processing", resource_id=resource_id)
        # Implementation would halt processing pipelines
    
    async def _encrypt_data(self, resource_id: str):
        """Encrypt data that should be protected."""
        logger.info("Encrypting data", resource_id=resource_id)
        # Implementation would apply encryption
    
    async def _block_data_transfer(self, transfer_id: str):
        """Block unauthorized data transfer."""
        logger.info("Blocking data transfer", transfer_id=transfer_id)
        # Implementation would halt transfer process
    
    async def _escalate_to_dpo(self, violation: PolicyViolation):
        """Escalate to Data Protection Officer."""
        logger.info("Escalating to DPO", violation_id=violation.id)
        # Implementation would notify DPO
    
    async def _enhanced_logging(self, context: Dict[str, Any]):
        """Enable enhanced logging for audit purposes."""
        logger.info("Enhanced logging enabled", context=context)
        # Implementation would increase logging verbosity
    
    async def _delete_data_without_consent(self, resource_id: str):
        """Delete data processed without proper consent."""
        logger.info("Deleting data without consent", resource_id=resource_id)
        # Implementation would delete unconsented data
    
    def _update_metrics(self):
        """Update enforcement metrics."""
        self.metrics.total_rules = len(self.rules)
        self.metrics.active_rules = len([r for r in self.rules.values() if r.enabled])
        self.metrics.violations_detected = len(self.violations)
        self.metrics.violations_resolved = len([v for v in self.violations.values() if v.resolved])
        
        # Count by severity
        self.metrics.violations_by_severity = {}
        for severity in ViolationSeverity:
            count = len([v for v in self.violations.values() if v.severity == severity])
            self.metrics.violations_by_severity[severity.value] = count
        
        # Count by type
        self.metrics.violations_by_type = {}
        for policy_type in PolicyType:
            count = len([v for v in self.violations.values() if v.policy_type == policy_type])
            self.metrics.violations_by_type[policy_type.value] = count
        
        # Count by action
        self.metrics.enforcement_actions = {}
        for action in EnforcementAction:
            count = len([v for v in self.violations.values() if v.action_taken == action])
            self.metrics.enforcement_actions[action.value] = count
        
        self.metrics.last_updated = datetime.utcnow()
    
    async def resolve_violation(self, violation_id: str, resolved_by: str) -> bool:
        """Mark a violation as resolved."""
        violation = self.violations.get(violation_id)
        if not violation:
            return False
        
        violation.resolved = True
        violation.resolved_at = datetime.utcnow()
        violation.resolved_by = resolved_by
        
        await self._store_violation(violation)
        self._update_metrics()
        
        logger.info("Policy violation resolved",
                   violation_id=violation_id,
                   resolved_by=resolved_by)
        
        return True
    
    def get_metrics(self) -> PolicyEnforcementMetrics:
        """Get current enforcement metrics."""
        self._update_metrics()
        return self.metrics
    
    def get_violations_by_severity(self, severity: ViolationSeverity) -> List[PolicyViolation]:
        """Get violations by severity level."""
        return [v for v in self.violations.values() if v.severity == severity and not v.resolved]
    
    def get_violations_by_type(self, policy_type: PolicyType) -> List[PolicyViolation]:
        """Get violations by policy type."""
        return [v for v in self.violations.values() if v.policy_type == policy_type and not v.resolved]
    
    async def generate_enforcement_report(self) -> Dict[str, Any]:
        """Generate comprehensive enforcement report."""
        metrics = self.get_metrics()
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": asdict(metrics),
            "active_violations": [
                {
                    "id": v.id,
                    "rule_id": v.rule_id,
                    "policy_type": v.policy_type.value,
                    "severity": v.severity.value,
                    "detected_at": v.detected_at.isoformat(),
                    "resource_id": v.resource_id,
                    "action_taken": v.action_taken.value
                }
                for v in self.violations.values() if not v.resolved
            ],
            "policy_rules": [
                {
                    "id": r.id,
                    "name": r.name,
                    "policy_type": r.policy_type.value,
                    "enabled": r.enabled,
                    "severity": r.severity.value,
                    "action": r.action.value
                }
                for r in self.rules.values()
            ]
        }


# Global policy enforcement engine instance
policy_enforcement_engine = PolicyEnforcementEngine()