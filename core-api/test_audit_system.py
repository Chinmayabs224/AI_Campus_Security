"""
Test script for the audit logging system.
"""
import asyncio
from datetime import datetime, timedelta
import uuid

from audit.service import audit_service
from audit.models import AuditAction, ResourceType, ComplianceTag
from core.database import database_manager


async def test_audit_logging():
    """Test the audit logging functionality."""
    print("Testing audit logging system...")
    
    # Initialize database connection
    await database_manager.connect()
    
    try:
        # Test 1: Basic audit log creation
        print("\n1. Testing basic audit log creation...")
        
        audit_id = await audit_service.log_action(
            action=AuditAction.LOGIN,
            user_id=uuid.uuid4(),
            username="test_user",
            resource_type=ResourceType.USER,
            endpoint="/api/v1/auth/login",
            method="POST",
            ip_address="192.168.1.100",
            compliance_tags=[ComplianceTag.GDPR],
            risk_level="low",
            business_justification="User authentication",
            contains_pii=False,
            metadata={"test": "basic_login"}
        )
        
        print(f"✓ Created audit log with ID: {audit_id}")
        
        # Test 2: High-risk audit log with PII
        print("\n2. Testing high-risk audit log with PII...")
        
        audit_id = await audit_service.log_action(
            action=AuditAction.EVIDENCE_ACCESS,
            user_id=uuid.uuid4(),
            username="security_admin",
            resource_type=ResourceType.EVIDENCE,
            resource_id="evidence_123",
            endpoint="/api/v1/evidence/evidence_123",
            method="GET",
            ip_address="10.0.0.50",
            compliance_tags=[ComplianceTag.GDPR, ComplianceTag.FERPA],
            risk_level="high",
            business_justification="Investigation of security incident #456",
            contains_pii=True,
            data_classification="restricted",
            metadata={
                "incident_id": "incident_456",
                "evidence_type": "video_clip",
                "duration_seconds": 30
            }
        )
        
        print(f"✓ Created high-risk audit log with ID: {audit_id}")
        
        # Test 3: Failed action audit log
        print("\n3. Testing failed action audit log...")
        
        audit_id = await audit_service.log_action(
            action=AuditAction.EVIDENCE_ACCESS,
            user_id=uuid.uuid4(),
            username="unauthorized_user",
            resource_type=ResourceType.EVIDENCE,
            resource_id="evidence_456",
            endpoint="/api/v1/evidence/evidence_456",
            method="GET",
            ip_address="203.0.113.15",
            compliance_tags=[ComplianceTag.GDPR],
            risk_level="high",
            business_justification="Attempted unauthorized access",
            contains_pii=True,
            success=False,
            error_code="403",
            error_message="Insufficient permissions to access evidence",
            metadata={"attempted_access": "unauthorized"}
        )
        
        print(f"✓ Created failed action audit log with ID: {audit_id}")
        
        # Test 4: Search audit logs
        print("\n4. Testing audit log search...")
        
        from audit.models import AuditLogFilter
        
        # Search for recent logs
        filters = AuditLogFilter(
            start_time=datetime.utcnow() - timedelta(minutes=5),
            end_time=datetime.utcnow(),
            limit=10
        )
        
        logs = await audit_service.search_audit_logs(filters)
        print(f"✓ Found {len(logs)} recent audit logs")
        
        for log in logs[:3]:  # Show first 3 logs
            print(f"  - {log.timestamp}: {log.action} by {log.username or 'system'} ({log.success})")
        
        # Test 5: Get audit statistics
        print("\n5. Testing audit statistics...")
        
        stats = await audit_service.get_audit_stats(
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow()
        )
        
        print(f"✓ Audit statistics:")
        print(f"  - Total logs: {stats.total_logs}")
        print(f"  - Failed actions: {stats.failed_actions}")
        print(f"  - PII access count: {stats.pii_access_count}")
        print(f"  - Unique users: {stats.unique_users}")
        
        # Test 6: Generate compliance report
        print("\n6. Testing compliance report generation...")
        
        report = await audit_service.generate_compliance_report(
            framework=ComplianceTag.GDPR,
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow(),
            generated_by="test_system"
        )
        
        print(f"✓ Generated compliance report:")
        print(f"  - Report ID: {report.report_id}")
        print(f"  - Framework: {report.framework}")
        print(f"  - Total events: {report.total_events}")
        print(f"  - High risk events: {report.high_risk_events}")
        print(f"  - PII access events: {report.pii_access_events}")
        
        # Test 7: Test DSAR request processing
        print("\n7. Testing DSAR request processing...")
        
        from audit.models import DSARRequest
        
        dsar_request = DSARRequest(
            subject_identifier="test_user",
            request_type="access",
            requested_data_types=["audit_logs", "user_profile"],
            business_justification="User requested their data under GDPR Article 15",
            requested_by="data_protection_officer",
            due_date=datetime.utcnow() + timedelta(days=30)
        )
        
        dsar_result = await audit_service.process_dsar_request(dsar_request)
        print(f"✓ Processed DSAR request:")
        print(f"  - Status: {dsar_result['status']}")
        print(f"  - Request ID: {dsar_result.get('request_id', 'N/A')}")
        
        print("\n✅ All audit logging tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
    
    finally:
        # Close database connection
        await database_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(test_audit_logging())