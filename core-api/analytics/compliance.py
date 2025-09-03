"""
Compliance and audit reporting functionality.
"""
import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4
import json
from enum import Enum
import structlog

from core.database import get_db_session
from .models import ComplianceReport, ChainOfCustody

logger = structlog.get_logger()


class ComplianceReportType(str, Enum):
    """Types of compliance reports."""
    GDPR = "gdpr"
    FERPA = "ferpa"
    COPPA = "coppa"
    GENERAL = "general"
    PRIVACY_IMPACT = "privacy_impact"
    DATA_RETENTION = "data_retention"


class AuditReportType(str, Enum):
    """Types of audit reports."""
    ACCESS_LOG = "access_log"
    SYSTEM_CHANGES = "system_changes"
    SECURITY_EVENTS = "security_events"
    USER_ACTIVITY = "user_activity"
    DATA_PROCESSING = "data_processing"


class ComplianceReportGenerator:
    """Generate various compliance reports."""
    
    async def generate_gdpr_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        async with get_db_session() as db:
            # Data processing activities
            processing_activities = await db.fetch("""
                SELECT 
                    action,
                    COUNT(*) as count,
                    COUNT(DISTINCT user_id) as unique_users
                FROM audit_log
                WHERE timestamp::date BETWEEN $1 AND $2
                AND action IN ('data_access', 'data_export', 'data_deletion', 'data_modification')
                GROUP BY action
            """, start_date, end_date)
            
            # Data subject requests
            dsar_requests = await db.fetch("""
                SELECT 
                    request_type,
                    status,
                    COUNT(*) as count,
                    AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))/3600) as avg_resolution_hours
                FROM data_subject_requests
                WHERE created_at::date BETWEEN $1 AND $2
                GROUP BY request_type, status
            """, start_date, end_date)
            
            # Privacy violations
            privacy_violations = await db.fetch("""
                SELECT 
                    violation_type,
                    severity,
                    COUNT(*) as count
                FROM privacy_violations
                WHERE created_at::date BETWEEN $1 AND $2
                GROUP BY violation_type, severity
            """, start_date, end_date)
            
            # Data retention compliance
            retention_stats = await db.fetchrow("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN retention_expires > NOW() THEN 1 END) as compliant_records,
                    COUNT(CASE WHEN retention_expires < NOW() THEN 1 END) as expired_records
                FROM evidence
                WHERE created_at::date BETWEEN $1 AND $2
            """, start_date, end_date)
            
            return {
                "report_type": "GDPR Compliance Report",
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "processing_activities": [dict(row) for row in processing_activities],
                "dsar_requests": [dict(row) for row in dsar_requests],
                "privacy_violations": [dict(row) for row in privacy_violations],
                "retention_compliance": {
                    "total_records": retention_stats['total_records'] or 0,
                    "compliant_records": retention_stats['compliant_records'] or 0,
                    "expired_records": retention_stats['expired_records'] or 0,
                    "compliance_rate": (
                        (retention_stats['compliant_records'] or 0) / 
                        (retention_stats['total_records'] or 1) * 100
                    )  
              }
            }


class AuditReportGenerator:
    """Generate various audit reports."""
    
    async def generate_access_log_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate access log audit report."""
        async with get_db_session() as db:
            access_logs = await db.fetch("""
                SELECT 
                    user_id,
                    action,
                    resource_type,
                    COUNT(*) as access_count,
                    MIN(timestamp) as first_access,
                    MAX(timestamp) as last_access
                FROM audit_log
                WHERE timestamp::date BETWEEN $1 AND $2
                GROUP BY user_id, action, resource_type
                ORDER BY access_count DESC
            """, start_date, end_date)
            
            return {
                "report_type": "Access Log Audit Report",
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "access_summary": [dict(row) for row in access_logs],
                "total_accesses": sum(row['access_count'] for row in access_logs),
                "unique_users": len(set(row['user_id'] for row in access_logs))
            }
    
    async def generate_system_changes_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate system changes audit report."""
        async with get_db_session() as db:
            system_changes = await db.fetch("""
                SELECT 
                    action,
                    resource_type,
                    user_id,
                    timestamp,
                    ip_address
                FROM audit_log
                WHERE timestamp::date BETWEEN $1 AND $2
                AND action IN ('create', 'update', 'delete', 'configure')
                ORDER BY timestamp DESC
            """, start_date, end_date)
            
            return {
                "report_type": "System Changes Audit Report",
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "changes": [dict(row) for row in system_changes],
                "total_changes": len(system_changes)
            }


# Service instances
compliance_report_generator = ComplianceReportGenerator()
audit_report_generator = AuditReportGenerator()