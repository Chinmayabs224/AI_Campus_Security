"""
Data retention and deletion policies implementation for campus security system.
Ensures compliance with GDPR, FERPA, and other privacy regulations.
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
import aiofiles
from uuid import uuid4

from core.database import get_db_session
from core.config import settings

logger = structlog.get_logger()


class DataCategory(Enum):
    """Data categories for retention policies."""
    SECURITY_EVENTS = "security_events"
    VIDEO_EVIDENCE = "video_evidence"
    PERSONAL_DATA = "personal_data"
    AUDIT_LOGS = "audit_logs"
    USER_SESSIONS = "user_sessions"
    INCIDENT_REPORTS = "incident_reports"
    BIOMETRIC_DATA = "biometric_data"
    LOCATION_DATA = "location_data"
    SYSTEM_LOGS = "system_logs"
    BACKUP_DATA = "backup_data"


class DeletionMethod(Enum):
    """Methods for data deletion."""
    STANDARD_DELETE = "standard_delete"
    SECURE_DELETE = "secure_delete"
    CRYPTOGRAPHIC_ERASURE = "cryptographic_erasure"
    ARCHIVE_THEN_DELETE = "archive_then_delete"
    ANONYMIZATION = "anonymization"
    PSEUDONYMIZATION = "pseudonymization"


class RetentionStatus(Enum):
    """Status of retention policy application."""
    ACTIVE = "active"
    EXPIRED = "expired"
    PENDING_DELETION = "pending_deletion"
    DELETED = "deleted"
    ARCHIVED = "archived"
    REVIEW_REQUIRED = "review_required"


@dataclass
class DataRetentionPolicy:
    """Data retention policy definition."""
    id: str
    name: str
    data_category: DataCategory
    retention_days: int
    deletion_method: DeletionMethod
    legal_basis: str
    auto_delete: bool
    requires_approval: bool
    review_frequency_days: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    approved_by: Optional[str] = None
    compliance_frameworks: List[str] = None
    
    def __post_init__(self):
        if self.compliance_frameworks is None:
            self.compliance_frameworks = []


@dataclass
class DataRetentionRecord:
    """Individual data retention record."""
    id: str
    policy_id: str
    data_id: str
    data_category: DataCategory
    created_at: datetime
    expires_at: datetime
    status: RetentionStatus
    deletion_scheduled_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    deletion_method_used: Optional[DeletionMethod] = None
    metadata: Optional[Dict[str, Any]] = None


class DataRetentionService:
    """Service for managing data retention and deletion policies."""
    
    def __init__(self):
        self.policies: Dict[str, DataRetentionPolicy] = {}
        self.retention_records: Dict[str, DataRetentionRecord] = {}
        self.deletion_queue: List[str] = []
        self.initialize_default_policies()
    
    def initialize_default_policies(self):
        """Initialize default data retention policies."""
        default_policies = [
            DataRetentionPolicy(
                id="security_events_policy",
                name="Security Events Retention",
                data_category=DataCategory.SECURITY_EVENTS,
                retention_days=2555,  # 7 years
                deletion_method=DeletionMethod.ARCHIVE_THEN_DELETE,
                legal_basis="Legitimate interest for security purposes",
                auto_delete=True,
                requires_approval=False,
                review_frequency_days=365,
                created_at=datetime.utcnow(),
                compliance_frameworks=["GDPR", "FERPA", "SOC2"]
            ),
            DataRetentionPolicy(
                id="video_evidence_policy",
                name="Video Evidence Retention",
                data_category=DataCategory.VIDEO_EVIDENCE,
                retention_days=90,  # 3 months default
                deletion_method=DeletionMethod.SECURE_DELETE,
                legal_basis="Legitimate interest for security purposes",
                auto_delete=False,
                requires_approval=True,
                review_frequency_days=30,
                created_at=datetime.utcnow(),
                compliance_frameworks=["GDPR", "FERPA", "COPPA"]
            ),
            DataRetentionPolicy(
                id="personal_data_policy",
                name="Personal Data Retention",
                data_category=DataCategory.PERSONAL_DATA,
                retention_days=1095,  # 3 years
                deletion_method=DeletionMethod.CRYPTOGRAPHIC_ERASURE,
                legal_basis="Consent or legitimate interest",
                auto_delete=True,
                requires_approval=True,
                review_frequency_days=180,
                created_at=datetime.utcnow(),
                compliance_frameworks=["GDPR", "FERPA", "COPPA"]
            ),
            DataRetentionPolicy(
                id="audit_logs_policy",
                name="Audit Logs Retention",
                data_category=DataCategory.AUDIT_LOGS,
                retention_days=2555,  # 7 years
                deletion_method=DeletionMethod.ARCHIVE_THEN_DELETE,
                legal_basis="Legal obligation",
                auto_delete=True,
                requires_approval=False,
                review_frequency_days=365,
                created_at=datetime.utcnow(),
                compliance_frameworks=["GDPR", "SOX", "SOC2"]
            ),
            DataRetentionPolicy(
                id="biometric_data_policy",
                name="Biometric Data Retention",
                data_category=DataCategory.BIOMETRIC_DATA,
                retention_days=365,  # 1 year
                deletion_method=DeletionMethod.CRYPTOGRAPHIC_ERASURE,
                legal_basis="Explicit consent",
                auto_delete=True,
                requires_approval=True,
                review_frequency_days=90,
                created_at=datetime.utcnow(),
                compliance_frameworks=["GDPR", "BIPA"]
            ),
            DataRetentionPolicy(
                id="user_sessions_policy",
                name="User Sessions Retention",
                data_category=DataCategory.USER_SESSIONS,
                retention_days=90,
                deletion_method=DeletionMethod.STANDARD_DELETE,
                legal_basis="Legitimate interest",
                auto_delete=True,
                requires_approval=False,
                review_frequency_days=30,
                created_at=datetime.utcnow(),
                compliance_frameworks=["GDPR"]
            )
        ]
        
        for policy in default_policies:
            self.policies[policy.id] = policy
        
        logger.info("Default data retention policies initialized", count=len(default_policies))
    
    async def create_retention_record(self, policy_id: str, data_id: str, 
                                    metadata: Optional[Dict[str, Any]] = None) -> DataRetentionRecord:
        """Create a new data retention record."""
        policy = self.policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")
        
        record_id = str(uuid4())
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(days=policy.retention_days)
        
        record = DataRetentionRecord(
            id=record_id,
            policy_id=policy_id,
            data_id=data_id,
            data_category=policy.data_category,
            created_at=created_at,
            expires_at=expires_at,
            status=RetentionStatus.ACTIVE,
            metadata=metadata or {}
        )
        
        self.retention_records[record_id] = record
        
        # Store in database
        await self._store_retention_record(record)
        
        logger.info("Retention record created",
                   record_id=record_id,
                   policy_id=policy_id,
                   data_id=data_id,
                   expires_at=expires_at)
        
        return record
    
    async def _store_retention_record(self, record: DataRetentionRecord):
        """Store retention record in database."""
        async with get_db_session() as db:
            await db.execute("""
                INSERT INTO data_retention_records 
                (id, policy_id, data_id, data_category, created_at, expires_at, status, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    deletion_scheduled_at = EXCLUDED.deletion_scheduled_at,
                    deleted_at = EXCLUDED.deleted_at,
                    deletion_method_used = EXCLUDED.deletion_method_used,
                    metadata = EXCLUDED.metadata
            """, record.id, record.policy_id, record.data_id, record.data_category.value,
                record.created_at, record.expires_at, record.status.value, 
                json.dumps(record.metadata) if record.metadata else None)
    
    async def apply_retention_policies(self) -> Dict[str, Any]:
        """Apply all active retention policies and process expired data."""
        logger.info("Starting retention policy application")
        
        results = {
            "started_at": datetime.utcnow().isoformat(),
            "policies_processed": 0,
            "records_processed": 0,
            "records_expired": 0,
            "records_deleted": 0,
            "records_archived": 0,
            "errors": []
        }
        
        try:
            # Load retention records from database
            await self._load_retention_records()
            
            # Process each policy
            for policy_id, policy in self.policies.items():
                try:
                    policy_results = await self._apply_policy(policy)
                    results["policies_processed"] += 1
                    results["records_processed"] += policy_results["processed"]
                    results["records_expired"] += policy_results["expired"]
                    results["records_deleted"] += policy_results["deleted"]
                    results["records_archived"] += policy_results["archived"]
                    
                except Exception as e:
                    error_msg = f"Failed to apply policy {policy_id}: {str(e)}"
                    results["errors"].append(error_msg)
                    logger.error("Policy application failed", policy_id=policy_id, error=str(e))
            
            # Process deletion queue
            await self._process_deletion_queue()
            
            results["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info("Retention policy application completed",
                       policies_processed=results["policies_processed"],
                       records_expired=results["records_expired"],
                       records_deleted=results["records_deleted"])
            
        except Exception as e:
            results["errors"].append(f"Retention policy application failed: {str(e)}")
            logger.error("Retention policy application failed", error=str(e))
        
        return results
    
    async def _load_retention_records(self):
        """Load retention records from database."""
        async with get_db_session() as db:
            records = await db.fetch("""
                SELECT id, policy_id, data_id, data_category, created_at, expires_at, 
                       status, deletion_scheduled_at, deleted_at, deletion_method_used, metadata
                FROM data_retention_records
                WHERE status IN ('active', 'expired', 'pending_deletion')
            """)
            
            for record_data in records:
                record = DataRetentionRecord(
                    id=record_data['id'],
                    policy_id=record_data['policy_id'],
                    data_id=record_data['data_id'],
                    data_category=DataCategory(record_data['data_category']),
                    created_at=record_data['created_at'],
                    expires_at=record_data['expires_at'],
                    status=RetentionStatus(record_data['status']),
                    deletion_scheduled_at=record_data['deletion_scheduled_at'],
                    deleted_at=record_data['deleted_at'],
                    deletion_method_used=DeletionMethod(record_data['deletion_method_used']) if record_data['deletion_method_used'] else None,
                    metadata=json.loads(record_data['metadata']) if record_data['metadata'] else {}
                )
                self.retention_records[record.id] = record
    
    async def _apply_policy(self, policy: DataRetentionPolicy) -> Dict[str, int]:
        """Apply a specific retention policy."""
        results = {"processed": 0, "expired": 0, "deleted": 0, "archived": 0}
        
        current_time = datetime.utcnow()
        
        # Find records for this policy
        policy_records = [
            record for record in self.retention_records.values()
            if record.policy_id == policy.id
        ]
        
        for record in policy_records:
            results["processed"] += 1
            
            # Check if record has expired
            if record.expires_at <= current_time and record.status == RetentionStatus.ACTIVE:
                record.status = RetentionStatus.EXPIRED
                results["expired"] += 1
                
                # Schedule for deletion if auto-delete is enabled
                if policy.auto_delete and not policy.requires_approval:
                    record.status = RetentionStatus.PENDING_DELETION
                    record.deletion_scheduled_at = current_time
                    self.deletion_queue.append(record.id)
                elif policy.requires_approval:
                    record.status = RetentionStatus.REVIEW_REQUIRED
                
                await self._store_retention_record(record)
            
            # Process pending deletions
            elif record.status == RetentionStatus.PENDING_DELETION:
                deletion_result = await self._delete_data(record, policy)
                if deletion_result["success"]:
                    if policy.deletion_method == DeletionMethod.ARCHIVE_THEN_DELETE:
                        results["archived"] += 1
                    else:
                        results["deleted"] += 1
        
        return results
    
    async def _delete_data(self, record: DataRetentionRecord, policy: DataRetentionPolicy) -> Dict[str, Any]:
        """Delete data according to the specified deletion method."""
        logger.info("Deleting data",
                   record_id=record.id,
                   data_id=record.data_id,
                   method=policy.deletion_method.value)
        
        try:
            if policy.deletion_method == DeletionMethod.STANDARD_DELETE:
                success = await self._standard_delete(record)
            elif policy.deletion_method == DeletionMethod.SECURE_DELETE:
                success = await self._secure_delete(record)
            elif policy.deletion_method == DeletionMethod.CRYPTOGRAPHIC_ERASURE:
                success = await self._cryptographic_erasure(record)
            elif policy.deletion_method == DeletionMethod.ARCHIVE_THEN_DELETE:
                success = await self._archive_then_delete(record)
            elif policy.deletion_method == DeletionMethod.ANONYMIZATION:
                success = await self._anonymize_data(record)
            elif policy.deletion_method == DeletionMethod.PSEUDONYMIZATION:
                success = await self._pseudonymize_data(record)
            else:
                success = False
                logger.error("Unknown deletion method", method=policy.deletion_method.value)
            
            if success:
                record.status = RetentionStatus.DELETED
                record.deleted_at = datetime.utcnow()
                record.deletion_method_used = policy.deletion_method
                await self._store_retention_record(record)
            
            return {"success": success}
            
        except Exception as e:
            logger.error("Data deletion failed",
                        record_id=record.id,
                        error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _standard_delete(self, record: DataRetentionRecord) -> bool:
        """Perform standard deletion."""
        # Implementation would depend on data category
        if record.data_category == DataCategory.USER_SESSIONS:
            return await self._delete_user_sessions(record.data_id)
        elif record.data_category == DataCategory.SYSTEM_LOGS:
            return await self._delete_system_logs(record.data_id)
        return True
    
    async def _secure_delete(self, record: DataRetentionRecord) -> bool:
        """Perform secure deletion with overwriting."""
        # Implementation would include secure overwriting
        if record.data_category == DataCategory.VIDEO_EVIDENCE:
            return await self._secure_delete_video(record.data_id)
        return True
    
    async def _cryptographic_erasure(self, record: DataRetentionRecord) -> bool:
        """Perform cryptographic erasure by deleting encryption keys."""
        # Implementation would delete encryption keys making data unrecoverable
        if record.data_category == DataCategory.PERSONAL_DATA:
            return await self._delete_encryption_keys(record.data_id)
        elif record.data_category == DataCategory.BIOMETRIC_DATA:
            return await self._delete_biometric_keys(record.data_id)
        return True
    
    async def _archive_then_delete(self, record: DataRetentionRecord) -> bool:
        """Archive data then delete from active storage."""
        # Implementation would move data to archive storage
        if record.data_category == DataCategory.AUDIT_LOGS:
            return await self._archive_audit_logs(record.data_id)
        elif record.data_category == DataCategory.SECURITY_EVENTS:
            return await self._archive_security_events(record.data_id)
        return True
    
    async def _anonymize_data(self, record: DataRetentionRecord) -> bool:
        """Anonymize data by removing identifying information."""
        # Implementation would remove or hash identifying fields
        return await self._remove_identifiers(record.data_id, record.data_category)
    
    async def _pseudonymize_data(self, record: DataRetentionRecord) -> bool:
        """Pseudonymize data by replacing identifiers with pseudonyms."""
        # Implementation would replace identifiers with pseudonyms
        return await self._replace_with_pseudonyms(record.data_id, record.data_category)
    
    # Placeholder implementations for specific deletion methods
    async def _delete_user_sessions(self, data_id: str) -> bool:
        """Delete user session data."""
        async with get_db_session() as db:
            await db.execute("DELETE FROM user_sessions WHERE id = $1", data_id)
        return True
    
    async def _delete_system_logs(self, data_id: str) -> bool:
        """Delete system log data."""
        async with get_db_session() as db:
            await db.execute("DELETE FROM system_logs WHERE id = $1", data_id)
        return True
    
    async def _secure_delete_video(self, data_id: str) -> bool:
        """Securely delete video evidence."""
        # Implementation would securely overwrite video files
        logger.info("Securely deleting video evidence", data_id=data_id)
        return True
    
    async def _delete_encryption_keys(self, data_id: str) -> bool:
        """Delete encryption keys for cryptographic erasure."""
        # Implementation would delete encryption keys from key management system
        logger.info("Deleting encryption keys", data_id=data_id)
        return True
    
    async def _delete_biometric_keys(self, data_id: str) -> bool:
        """Delete biometric data encryption keys."""
        logger.info("Deleting biometric encryption keys", data_id=data_id)
        return True
    
    async def _archive_audit_logs(self, data_id: str) -> bool:
        """Archive audit logs to long-term storage."""
        logger.info("Archiving audit logs", data_id=data_id)
        return True
    
    async def _archive_security_events(self, data_id: str) -> bool:
        """Archive security events to long-term storage."""
        logger.info("Archiving security events", data_id=data_id)
        return True
    
    async def _remove_identifiers(self, data_id: str, category: DataCategory) -> bool:
        """Remove identifying information from data."""
        logger.info("Anonymizing data", data_id=data_id, category=category.value)
        return True
    
    async def _replace_with_pseudonyms(self, data_id: str, category: DataCategory) -> bool:
        """Replace identifiers with pseudonyms."""
        logger.info("Pseudonymizing data", data_id=data_id, category=category.value)
        return True
    
    async def _process_deletion_queue(self):
        """Process the deletion queue."""
        logger.info("Processing deletion queue", queue_size=len(self.deletion_queue))
        
        for record_id in self.deletion_queue.copy():
            record = self.retention_records.get(record_id)
            if record and record.status == RetentionStatus.PENDING_DELETION:
                policy = self.policies.get(record.policy_id)
                if policy:
                    await self._delete_data(record, policy)
                self.deletion_queue.remove(record_id)
    
    async def get_retention_status(self, data_id: str) -> Optional[DataRetentionRecord]:
        """Get retention status for specific data."""
        for record in self.retention_records.values():
            if record.data_id == data_id:
                return record
        return None
    
    async def extend_retention(self, record_id: str, additional_days: int, reason: str) -> bool:
        """Extend retention period for specific data."""
        record = self.retention_records.get(record_id)
        if not record:
            return False
        
        old_expires_at = record.expires_at
        record.expires_at = record.expires_at + timedelta(days=additional_days)
        record.status = RetentionStatus.ACTIVE
        
        if not record.metadata:
            record.metadata = {}
        record.metadata["retention_extended"] = {
            "extended_at": datetime.utcnow().isoformat(),
            "additional_days": additional_days,
            "reason": reason,
            "old_expires_at": old_expires_at.isoformat()
        }
        
        await self._store_retention_record(record)
        
        logger.info("Retention period extended",
                   record_id=record_id,
                   additional_days=additional_days,
                   new_expires_at=record.expires_at)
        
        return True
    
    def generate_retention_report(self) -> Dict[str, Any]:
        """Generate comprehensive retention report."""
        current_time = datetime.utcnow()
        
        # Count records by status
        status_counts = {}
        for status in RetentionStatus:
            status_counts[status.value] = len([
                r for r in self.retention_records.values() 
                if r.status == status
            ])
        
        # Count records by category
        category_counts = {}
        for category in DataCategory:
            category_counts[category.value] = len([
                r for r in self.retention_records.values() 
                if r.data_category == category
            ])
        
        # Find expiring records (next 30 days)
        expiring_soon = [
            r for r in self.retention_records.values()
            if r.status == RetentionStatus.ACTIVE and 
            r.expires_at <= current_time + timedelta(days=30)
        ]
        
        # Find overdue records
        overdue_records = [
            r for r in self.retention_records.values()
            if r.status == RetentionStatus.EXPIRED and
            r.expires_at < current_time - timedelta(days=7)
        ]
        
        return {
            "generated_at": current_time.isoformat(),
            "total_policies": len(self.policies),
            "total_records": len(self.retention_records),
            "status_breakdown": status_counts,
            "category_breakdown": category_counts,
            "expiring_soon": len(expiring_soon),
            "overdue_records": len(overdue_records),
            "deletion_queue_size": len(self.deletion_queue),
            "policies": [
                {
                    "id": policy.id,
                    "name": policy.name,
                    "category": policy.data_category.value,
                    "retention_days": policy.retention_days,
                    "auto_delete": policy.auto_delete,
                    "requires_approval": policy.requires_approval
                }
                for policy in self.policies.values()
            ]
        }


# Global data retention service instance
data_retention_service = DataRetentionService()