"""
Audit logging service implementation.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import uuid
import json
import hashlib

from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from core.database import database_manager
from auth.models import AuditLog, User
from .models import (
    AuditLogExtended, AuditLogCreate, AuditLogFilter, AuditLogResponse,
    AuditAction, ResourceType, ComplianceTag, ComplianceReport,
    AuditStats, DataRetentionPolicy, DSARRequest
)

logger = structlog.get_logger()


class AuditService:
    """Service for managing audit logs and compliance."""
    
    def __init__(self):
        self.retention_policies: Dict[str, DataRetentionPolicy] = {}
        self._load_default_retention_policies()
    
    def _load_default_retention_policies(self):
        """Load default retention policies for different resource types."""
        # GDPR compliance - 7 years for security logs
        self.retention_policies["security_logs"] = DataRetentionPolicy(
            resource_type=ResourceType.AUDIT_LOG,
            retention_days=2555,  # 7 years
            compliance_framework=ComplianceTag.GDPR,
            auto_archive=True,
            auto_delete=False
        )
        
        # FERPA compliance - 5 years for educational records
        self.retention_policies["educational_records"] = DataRetentionPolicy(
            resource_type=ResourceType.USER,
            retention_days=1825,  # 5 years
            compliance_framework=ComplianceTag.FERPA,
            auto_archive=True,
            auto_delete=False
        )
        
        # Evidence retention - 3 years default
        self.retention_policies["evidence"] = DataRetentionPolicy(
            resource_type=ResourceType.EVIDENCE,
            retention_days=1095,  # 3 years
            compliance_framework=ComplianceTag.GDPR,
            auto_archive=True,
            auto_delete=False
        )
    
    async def log_action(
        self,
        action: Union[AuditAction, str],
        user_id: Optional[uuid.UUID] = None,
        username: Optional[str] = None,
        session_id: Optional[uuid.UUID] = None,
        api_key_id: Optional[uuid.UUID] = None,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        compliance_tags: Optional[List[ComplianceTag]] = None,
        risk_level: str = "low",
        business_justification: Optional[str] = None,
        contains_pii: bool = False,
        data_classification: Optional[str] = None,
        event_metadata: Optional[Dict[str, Any]] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> uuid.UUID:
        """
        Log an audit event with comprehensive details.
        
        Args:
            action: The action being performed
            user_id: ID of the user performing the action
            username: Username (for performance, to avoid joins)
            session_id: Session ID if applicable
            api_key_id: API key ID if applicable
            resource_type: Type of resource being accessed
            resource_id: ID of the specific resource
            endpoint: API endpoint being accessed
            method: HTTP method
            ip_address: Client IP address
            user_agent: Client user agent
            compliance_tags: Relevant compliance frameworks
            risk_level: Risk level of the action
            business_justification: Justification for the action
            contains_pii: Whether the action involves PII
            data_classification: Data classification level
            event_metadata: Additional metadata
            before_state: State before the action
            after_state: State after the action
            success: Whether the action was successful
            error_code: Error code if failed
            error_message: Error message if failed
            duration_ms: Duration of the action in milliseconds
            
        Returns:
            UUID of the created audit log entry
        """
        try:
            # Calculate retention date based on resource type and compliance
            retention_date = self._calculate_retention_date(resource_type, compliance_tags)
            
            # Create audit log entry
            audit_log = AuditLogExtended(
                user_id=user_id,
                username=username,
                session_id=session_id,
                api_key_id=api_key_id,
                action=str(action),
                resource_type=str(resource_type) if resource_type else None,
                resource_id=resource_id,
                endpoint=endpoint,
                method=method,
                ip_address=ip_address,
                user_agent=user_agent,
                compliance_tags=[str(tag) for tag in compliance_tags] if compliance_tags else None,
                risk_level=risk_level,
                business_justification=business_justification,
                contains_pii=contains_pii,
                data_classification=data_classification,
                event_metadata=event_metadata,
                before_state=before_state,
                after_state=after_state,
                success=success,
                error_code=error_code,
                error_message=error_message,
                duration_ms=duration_ms,
                retention_date=retention_date
            )
            
            # Also create entry in the original audit_logs table for backward compatibility
            legacy_audit_log = AuditLog(
                user_id=user_id,
                session_id=session_id,
                api_key_id=api_key_id,
                action=str(action),
                resource_type=str(resource_type) if resource_type else None,
                resource_id=resource_id,
                endpoint=endpoint,
                method=method,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=event_metadata,
                success=success,
                error_message=error_message
            )
            
            async with database_manager.get_session() as session:
                session.add(audit_log)
                session.add(legacy_audit_log)
                await session.commit()
                
                logger.info(
                    "Audit log created",
                    audit_id=str(audit_log.id),
                    action=action,
                    user_id=str(user_id) if user_id else None,
                    resource_type=resource_type,
                    success=success
                )
                
                return audit_log.id
                
        except Exception as e:
            logger.error("Failed to create audit log", error=str(e))
            # Don't raise exception to avoid breaking the main operation
            return uuid.uuid4()  # Return a dummy ID
    
    def _calculate_retention_date(
        self,
        resource_type: Optional[ResourceType],
        compliance_tags: Optional[List[ComplianceTag]]
    ) -> Optional[datetime]:
        """Calculate retention date based on resource type and compliance requirements."""
        if not resource_type:
            return None
        
        # Find the longest retention period required
        max_retention_days = 0
        
        # Check compliance-specific requirements
        if compliance_tags:
            for tag in compliance_tags:
                if tag == ComplianceTag.GDPR:
                    max_retention_days = max(max_retention_days, 2555)  # 7 years
                elif tag == ComplianceTag.FERPA:
                    max_retention_days = max(max_retention_days, 1825)  # 5 years
                elif tag == ComplianceTag.SOX:
                    max_retention_days = max(max_retention_days, 2555)  # 7 years
        
        # Check resource-specific policies
        for policy in self.retention_policies.values():
            if policy.resource_type == resource_type:
                max_retention_days = max(max_retention_days, policy.retention_days)
        
        # Default to 3 years if no specific policy
        if max_retention_days == 0:
            max_retention_days = 1095  # 3 years
        
        return datetime.utcnow() + timedelta(days=max_retention_days)
    
    async def search_audit_logs(
        self,
        filters: AuditLogFilter,
        session: Optional[AsyncSession] = None
    ) -> List[AuditLogResponse]:
        """Search audit logs with filtering and pagination."""
        async def _search(db_session: AsyncSession):
            query = select(AuditLogExtended)
            
            # Apply filters
            conditions = []
            
            if filters.user_id:
                conditions.append(AuditLogExtended.user_id == filters.user_id)
            
            if filters.username:
                conditions.append(AuditLogExtended.username.ilike(f"%{filters.username}%"))
            
            if filters.action:
                conditions.append(AuditLogExtended.action == str(filters.action))
            
            if filters.resource_type:
                conditions.append(AuditLogExtended.resource_type == str(filters.resource_type))
            
            if filters.resource_id:
                conditions.append(AuditLogExtended.resource_id == filters.resource_id)
            
            if filters.start_time:
                conditions.append(AuditLogExtended.timestamp >= filters.start_time)
            
            if filters.end_time:
                conditions.append(AuditLogExtended.timestamp <= filters.end_time)
            
            if filters.ip_address:
                conditions.append(AuditLogExtended.ip_address == filters.ip_address)
            
            if filters.success is not None:
                conditions.append(AuditLogExtended.success == filters.success)
            
            if filters.compliance_tag:
                conditions.append(
                    AuditLogExtended.compliance_tags.contains([str(filters.compliance_tag)])
                )
            
            if filters.risk_level:
                conditions.append(AuditLogExtended.risk_level == filters.risk_level)
            
            if filters.contains_pii is not None:
                conditions.append(AuditLogExtended.contains_pii == filters.contains_pii)
            
            if filters.data_classification:
                conditions.append(AuditLogExtended.data_classification == filters.data_classification)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Order by timestamp descending (most recent first)
            query = query.order_by(desc(AuditLogExtended.timestamp))
            
            # Apply pagination
            query = query.offset(filters.offset).limit(filters.limit)
            
            result = await db_session.execute(query)
            audit_logs = result.scalars().all()
            
            return [AuditLogResponse.model_validate(log) for log in audit_logs]
        
        if session:
            return await _search(session)
        else:
            async with database_manager.get_session() as db_session:
                return await _search(db_session)
    
    async def get_audit_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session: Optional[AsyncSession] = None
    ) -> AuditStats:
        """Get audit statistics for a time period."""
        async def _get_stats(db_session: AsyncSession):
            query = select(AuditLogExtended)
            
            # Apply time filters
            conditions = []
            if start_time:
                conditions.append(AuditLogExtended.timestamp >= start_time)
            if end_time:
                conditions.append(AuditLogExtended.timestamp <= end_time)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Get all logs for analysis
            result = await db_session.execute(query)
            logs = result.scalars().all()
            
            # Calculate statistics
            total_logs = len(logs)
            
            logs_by_action = {}
            logs_by_resource_type = {}
            logs_by_risk_level = {}
            failed_actions = 0
            pii_access_count = 0
            unique_users = set()
            
            min_time = None
            max_time = None
            
            for log in logs:
                # Count by action
                logs_by_action[log.action] = logs_by_action.get(log.action, 0) + 1
                
                # Count by resource type
                if log.resource_type:
                    logs_by_resource_type[log.resource_type] = logs_by_resource_type.get(log.resource_type, 0) + 1
                
                # Count by risk level
                logs_by_risk_level[log.risk_level] = logs_by_risk_level.get(log.risk_level, 0) + 1
                
                # Count failed actions
                if not log.success:
                    failed_actions += 1
                
                # Count PII access
                if log.contains_pii:
                    pii_access_count += 1
                
                # Track unique users
                if log.user_id:
                    unique_users.add(str(log.user_id))
                
                # Track time range
                if min_time is None or log.timestamp < min_time:
                    min_time = log.timestamp
                if max_time is None or log.timestamp > max_time:
                    max_time = log.timestamp
            
            return AuditStats(
                total_logs=total_logs,
                logs_by_action=logs_by_action,
                logs_by_resource_type=logs_by_resource_type,
                logs_by_risk_level=logs_by_risk_level,
                failed_actions=failed_actions,
                pii_access_count=pii_access_count,
                unique_users=len(unique_users),
                time_range={
                    "start": min_time or datetime.utcnow(),
                    "end": max_time or datetime.utcnow()
                }
            )
        
        if session:
            return await _get_stats(session)
        else:
            async with database_manager.get_session() as db_session:
                return await _get_stats(db_session)
    
    async def generate_compliance_report(
        self,
        framework: ComplianceTag,
        start_date: datetime,
        end_date: datetime,
        generated_by: str,
        session: Optional[AsyncSession] = None
    ) -> ComplianceReport:
        """Generate a compliance report for a specific framework."""
        async def _generate_report(db_session: AsyncSession):
            # Query logs for the specified time period and compliance framework
            query = select(AuditLogExtended).where(
                and_(
                    AuditLogExtended.timestamp >= start_date,
                    AuditLogExtended.timestamp <= end_date,
                    AuditLogExtended.compliance_tags.contains([str(framework)])
                )
            )
            
            result = await db_session.execute(query)
            logs = result.scalars().all()
            
            # Analyze logs for compliance metrics
            total_events = len(logs)
            high_risk_events = sum(1 for log in logs if log.risk_level in ["high", "critical"])
            pii_access_events = sum(1 for log in logs if log.contains_pii)
            failed_access_attempts = sum(1 for log in logs if not log.success)
            
            # User activity summary
            user_activity = {}
            for log in logs:
                if log.username:
                    user_activity[log.username] = user_activity.get(log.username, 0) + 1
            
            # Resource access summary
            resource_access = {}
            for log in logs:
                if log.resource_type:
                    resource_access[log.resource_type] = resource_access.get(log.resource_type, 0) + 1
            
            return ComplianceReport(
                report_id=uuid.uuid4(),
                framework=framework,
                start_date=start_date,
                end_date=end_date,
                total_events=total_events,
                high_risk_events=high_risk_events,
                pii_access_events=pii_access_events,
                failed_access_attempts=failed_access_attempts,
                user_activity_summary=user_activity,
                resource_access_summary=resource_access,
                generated_at=datetime.utcnow(),
                generated_by=generated_by
            )
        
        if session:
            return await _generate_report(session)
        else:
            async with database_manager.get_session() as db_session:
                return await _generate_report(db_session)
    
    async def process_dsar_request(
        self,
        dsar_request: DSARRequest,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Process a Data Subject Access Request (DSAR)."""
        async def _process_dsar(db_session: AsyncSession):
            # Log the DSAR request
            await self.log_action(
                action=AuditAction.DSAR_REQUEST,
                resource_type=ResourceType.USER,
                resource_id=dsar_request.subject_identifier,
                compliance_tags=[ComplianceTag.GDPR],
                risk_level="medium",
                business_justification=dsar_request.business_justification,
                contains_pii=True,
                metadata={
                    "request_id": str(dsar_request.request_id),
                    "request_type": dsar_request.request_type,
                    "requested_data_types": dsar_request.requested_data_types,
                    "requested_by": dsar_request.requested_by
                }
            )
            
            # Find user by identifier
            user_query = select(User).where(
                or_(
                    User.email == dsar_request.subject_identifier,
                    User.username == dsar_request.subject_identifier,
                    User.saml_name_id == dsar_request.subject_identifier
                )
            )
            user_result = await db_session.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            if not user:
                return {
                    "status": "not_found",
                    "message": "No data found for the specified identifier"
                }
            
            # Collect data based on request type
            collected_data = {}
            
            if "audit_logs" in dsar_request.requested_data_types:
                # Get audit logs for the user
                audit_query = select(AuditLogExtended).where(
                    AuditLogExtended.user_id == user.id
                ).order_by(desc(AuditLogExtended.timestamp))
                
                audit_result = await db_session.execute(audit_query)
                audit_logs = audit_result.scalars().all()
                
                collected_data["audit_logs"] = [
                    {
                        "timestamp": log.timestamp.isoformat(),
                        "action": log.action,
                        "resource_type": log.resource_type,
                        "ip_address": log.ip_address,
                        "success": log.success
                    }
                    for log in audit_logs
                ]
            
            if "user_profile" in dsar_request.requested_data_types:
                collected_data["user_profile"] = {
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None
                }
            
            return {
                "status": "completed",
                "request_id": str(dsar_request.request_id),
                "data": collected_data,
                "processed_at": datetime.utcnow().isoformat()
            }
        
        if session:
            return await _process_dsar(session)
        else:
            async with database_manager.get_session() as db_session:
                return await _process_dsar(db_session)
    
    async def cleanup_expired_logs(self, session: Optional[AsyncSession] = None) -> int:
        """Clean up expired audit logs based on retention policies."""
        async def _cleanup(db_session: AsyncSession):
            # Find logs that have passed their retention date
            current_time = datetime.utcnow()
            
            # First, archive logs that are due for archival
            archive_query = select(AuditLogExtended).where(
                and_(
                    AuditLogExtended.retention_date <= current_time,
                    AuditLogExtended.archived == False
                )
            )
            
            archive_result = await db_session.execute(archive_query)
            logs_to_archive = archive_result.scalars().all()
            
            archived_count = 0
            for log in logs_to_archive:
                log.archived = True
                archived_count += 1
            
            await db_session.commit()
            
            logger.info(f"Archived {archived_count} audit logs")
            return archived_count
        
        if session:
            return await _cleanup(session)
        else:
            async with database_manager.get_session() as db_session:
                return await _cleanup(db_session)


# Global audit service instance
audit_service = AuditService()