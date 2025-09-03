"""
Background tasks for audit log management and compliance.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import uuid

import structlog
from sqlalchemy import select, func, and_

from core.database import database_manager
from .service import audit_service
from .models import AuditLogExtended, ComplianceTag, AuditAction

logger = structlog.get_logger()


class AuditTaskService:
    """Service for managing audit-related background tasks."""
    
    def __init__(self):
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_periodic_tasks(self):
        """Start all periodic audit tasks."""
        logger.info("Starting audit background tasks")
        
        # Start cleanup task (runs daily)
        self.running_tasks["cleanup"] = asyncio.create_task(
            self._periodic_cleanup_task()
        )
        
        # Start compliance monitoring (runs hourly)
        self.running_tasks["compliance_monitor"] = asyncio.create_task(
            self._periodic_compliance_monitoring()
        )
        
        # Start retention policy enforcement (runs daily)
        self.running_tasks["retention_enforcement"] = asyncio.create_task(
            self._periodic_retention_enforcement()
        )
        
        # Start anomaly detection (runs every 15 minutes)
        self.running_tasks["anomaly_detection"] = asyncio.create_task(
            self._periodic_anomaly_detection()
        )
    
    async def stop_periodic_tasks(self):
        """Stop all periodic audit tasks."""
        logger.info("Stopping audit background tasks")
        
        for task_name, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Cancelled audit task: {task_name}")
        
        self.running_tasks.clear()
    
    async def _periodic_cleanup_task(self):
        """Periodic task to clean up expired audit logs."""
        while True:
            try:
                logger.info("Starting audit log cleanup task")
                
                archived_count = await audit_service.cleanup_expired_logs()
                
                # Log the cleanup activity
                await audit_service.log_action(
                    action=AuditAction.SYSTEM_BACKUP,
                    resource_type="audit_log",
                    compliance_tags=[ComplianceTag.GDPR],
                    risk_level="low",
                    business_justification="Automated audit log cleanup and archival",
                    metadata={
                        "archived_count": archived_count,
                        "cleanup_type": "automated"
                    }
                )
                
                logger.info(f"Audit log cleanup completed, archived {archived_count} logs")
                
                # Wait 24 hours before next cleanup
                await asyncio.sleep(24 * 60 * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in audit cleanup task", error=str(e))
                # Wait 1 hour before retrying on error
                await asyncio.sleep(60 * 60)    
    
async def _periodic_compliance_monitoring(self):
        """Periodic task to monitor compliance violations."""
        while True:
            try:
                logger.info("Starting compliance monitoring task")
                
                # Check for compliance violations in the last hour
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=1)
                
                violations = await self._detect_compliance_violations(start_time, end_time)
                
                if violations:
                    # Log compliance violations
                    await audit_service.log_action(
                        action="compliance_violation_detected",
                        resource_type="audit_log",
                        compliance_tags=[ComplianceTag.GDPR, ComplianceTag.FERPA],
                        risk_level="high",
                        business_justification="Automated compliance monitoring",
                        metadata={
                            "violations": violations,
                            "monitoring_period": {
                                "start": start_time.isoformat(),
                                "end": end_time.isoformat()
                            }
                        }
                    )
                    
                    logger.warning(f"Detected {len(violations)} compliance violations")
                
                # Wait 1 hour before next check
                await asyncio.sleep(60 * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in compliance monitoring task", error=str(e))
                await asyncio.sleep(60 * 60)
    
    async def _periodic_retention_enforcement(self):
        """Periodic task to enforce data retention policies."""
        while True:
            try:
                logger.info("Starting retention policy enforcement task")
                
                # Find logs that should be deleted based on retention policies
                current_time = datetime.utcnow()
                
                async with database_manager.get_session() as session:
                    # Find logs past their retention date that are already archived
                    query = select(func.count(AuditLogExtended.id)).where(
                        and_(
                            AuditLogExtended.retention_date <= current_time,
                            AuditLogExtended.archived == True
                        )
                    )
                    
                    result = await session.execute(query)
                    eligible_for_deletion = result.scalar()
                    
                    if eligible_for_deletion > 0:
                        # Log the retention enforcement
                        await audit_service.log_action(
                            action=AuditAction.DATA_RETENTION_POLICY,
                            resource_type="audit_log",
                            compliance_tags=[ComplianceTag.GDPR],
                            risk_level="medium",
                            business_justification="Automated data retention policy enforcement",
                            metadata={
                                "eligible_for_deletion": eligible_for_deletion,
                                "enforcement_date": current_time.isoformat()
                            }
                        )
                        
                        logger.info(f"Found {eligible_for_deletion} logs eligible for deletion")
                
                # Wait 24 hours before next enforcement
                await asyncio.sleep(24 * 60 * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in retention enforcement task", error=str(e))
                await asyncio.sleep(60 * 60)
    
    async def _periodic_anomaly_detection(self):
        """Periodic task to detect anomalous audit patterns."""
        while True:
            try:
                logger.info("Starting audit anomaly detection task")
                
                # Check for anomalies in the last 15 minutes
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=15)
                
                anomalies = await self._detect_audit_anomalies(start_time, end_time)
                
                if anomalies:
                    # Log detected anomalies
                    await audit_service.log_action(
                        action="security_anomaly_detected",
                        resource_type="audit_log",
                        compliance_tags=[ComplianceTag.GDPR],
                        risk_level="high",
                        business_justification="Automated security anomaly detection",
                        metadata={
                            "anomalies": anomalies,
                            "detection_period": {
                                "start": start_time.isoformat(),
                                "end": end_time.isoformat()
                            }
                        }
                    )
                    
                    logger.warning(f"Detected {len(anomalies)} audit anomalies")
                
                # Wait 15 minutes before next check
                await asyncio.sleep(15 * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in anomaly detection task", error=str(e))
                await asyncio.sleep(15 * 60)   
 
    async def _detect_compliance_violations(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        """Detect potential compliance violations in audit logs."""
        violations = []
        
        async with database_manager.get_session() as session:
            # Check for excessive failed login attempts
            failed_login_query = select(func.count(AuditLogExtended.id)).where(
                and_(
                    AuditLogExtended.timestamp >= start_time,
                    AuditLogExtended.timestamp <= end_time,
                    AuditLogExtended.action == AuditAction.LOGIN_FAILED.value
                )
            )
            
            result = await session.execute(failed_login_query)
            failed_logins = result.scalar()
            
            if failed_logins > 50:  # Threshold for suspicious activity
                violations.append({
                    "type": "excessive_failed_logins",
                    "count": failed_logins,
                    "threshold": 50,
                    "severity": "high"
                })
            
            # Check for unauthorized evidence access
            evidence_access_query = select(func.count(AuditLogExtended.id)).where(
                and_(
                    AuditLogExtended.timestamp >= start_time,
                    AuditLogExtended.timestamp <= end_time,
                    AuditLogExtended.action == AuditAction.EVIDENCE_ACCESS.value,
                    AuditLogExtended.success == False
                )
            )
            
            result = await session.execute(evidence_access_query)
            failed_evidence_access = result.scalar()
            
            if failed_evidence_access > 10:  # Threshold for suspicious activity
                violations.append({
                    "type": "unauthorized_evidence_access_attempts",
                    "count": failed_evidence_access,
                    "threshold": 10,
                    "severity": "critical"
                })
            
            # Check for PII access without business justification
            pii_access_query = select(func.count(AuditLogExtended.id)).where(
                and_(
                    AuditLogExtended.timestamp >= start_time,
                    AuditLogExtended.timestamp <= end_time,
                    AuditLogExtended.contains_pii == True,
                    AuditLogExtended.business_justification.is_(None)
                )
            )
            
            result = await session.execute(pii_access_query)
            unjustified_pii_access = result.scalar()
            
            if unjustified_pii_access > 0:
                violations.append({
                    "type": "pii_access_without_justification",
                    "count": unjustified_pii_access,
                    "threshold": 0,
                    "severity": "medium"
                })
        
        return violations
    
    async def _detect_audit_anomalies(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        """Detect anomalous patterns in audit logs."""
        anomalies = []
        
        async with database_manager.get_session() as session:
            # Check for unusual IP address activity
            ip_activity_query = select(
                AuditLogExtended.ip_address,
                func.count(AuditLogExtended.id).label('activity_count')
            ).where(
                and_(
                    AuditLogExtended.timestamp >= start_time,
                    AuditLogExtended.timestamp <= end_time,
                    AuditLogExtended.ip_address.isnot(None)
                )
            ).group_by(AuditLogExtended.ip_address).having(
                func.count(AuditLogExtended.id) > 100  # Threshold for unusual activity
            )
            
            result = await session.execute(ip_activity_query)
            high_activity_ips = result.all()
            
            for ip, count in high_activity_ips:
                anomalies.append({
                    "type": "high_activity_ip",
                    "ip_address": ip,
                    "activity_count": count,
                    "threshold": 100,
                    "severity": "medium"
                })
            
            # Check for off-hours administrative activity
            off_hours_query = select(func.count(AuditLogExtended.id)).where(
                and_(
                    AuditLogExtended.timestamp >= start_time,
                    AuditLogExtended.timestamp <= end_time,
                    AuditLogExtended.risk_level == "high",
                    func.extract('hour', AuditLogExtended.timestamp).between(22, 6)  # 10 PM to 6 AM
                )
            )
            
            result = await session.execute(off_hours_query)
            off_hours_activity = result.scalar()
            
            if off_hours_activity > 5:  # Threshold for suspicious off-hours activity
                anomalies.append({
                    "type": "off_hours_high_risk_activity",
                    "count": off_hours_activity,
                    "threshold": 5,
                    "severity": "high"
                })
        
        return anomalies


# Global audit task service instance
audit_task_service = AuditTaskService()