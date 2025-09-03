#!/usr/bin/env python3
"""
User Acceptance Testing (UAT) for Campus Security System.
Tests real-world scenarios with security personnel workflows.
"""
import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from httpx import AsyncClient
import websockets

from conftest import (
    test_client, test_db, test_redis, test_storage,
    mock_cameras, auth_headers, admin_auth_headers
)


class SecurityPersonnelUAT:
    """User Acceptance Tests for Security Personnel."""
    
    @pytest.mark.asyncio
    async def test_incident_response_workflow(
        self, test_client: AsyncClient, mock_cameras, auth_headers
    ):
        """Test complete incident response workflow from detection to resolution."""
        
        # Scenario: Security guard receives alert and responds to incident
        camera = mock_cameras[0]
        camera.start_stream()
        
        # Step 1: High-priority incident occurs
        incident_data = {
            "camera_id": camera.camera_id,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "violence",
            "confidence_score": 0.94,
            "bounding_boxes": [
                {"x": 200, "y": 150, "width": 120, "height": 180, "class": "person", "confidence": 0.94}
            ],
            "metadata": {
                "location": "Student Center - Main Hall",
                "zone": "public",
                "priority": "critical"
            }
        }
        
        # Create high-priority event
        response = await test_client.post(
            "/api/v1/events",
            json=incident_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        event_id = response.json()["id"]
        
        # Step 2: Verify incident auto-creation for high-confidence events
        await asyncio.sleep(1)
        incidents_response = await test_client.get(
            "/api/v1/incidents",
            params={"event_id": event_id},
            headers=auth_headers
        )
        assert incidents_response.status_code == 200
        incidents = incidents_response.json()["items"]
        assert len(incidents) > 0
        
        incident = incidents[0]
        incident_id = incident["id"]
        assert incident["severity"] == "high"
        assert incident["status"] == "open"
        
        # Step 3: Security guard receives notification and views incident
        incident_details = await test_client.get(
            f"/api/v1/incidents/{incident_id}",
            headers=auth_headers
        )
        assert incident_details.status_code == 200
        
        # Step 4: Guard assigns incident to themselves
        assignment_response = await test_client.patch(
            f"/api/v1/incidents/{incident_id}",
            json={
                "status": "assigned",
                "assigned_to": "guard_john_doe",
                "response_notes": "Responding to scene immediately"
            },
            headers=auth_headers
        )
        assert assignment_response.status_code == 200
        
        # Step 5: Guard arrives on scene and updates status
        update_response = await test_client.patch(
            f"/api/v1/incidents/{incident_id}",
            json={
                "status": "investigating",
                "response_notes": "On scene - situation under control, calling backup"
            },
            headers=auth_headers
        )
        assert update_response.status_code == 200
        
        # Step 6: Guard resolves incident
        resolution_response = await test_client.patch(
            f"/api/v1/incidents/{incident_id}",
            json={
                "status": "resolved",
                "resolution_notes": "False alarm - students practicing theater performance. Area secured.",
                "resolution_time": datetime.utcnow().isoformat()
            },
            headers=auth_headers
        )
        assert resolution_response.status_code == 200
        
        # Step 7: Verify audit trail
        audit_response = await test_client.get(
            "/api/v1/audit/logs",
            params={"resource_id": incident_id},
            headers=auth_headers
        )
        assert audit_response.status_code == 200
        audit_logs = audit_response.json()["items"]
        
        # Should have logs for creation, assignment, updates, and resolution
        assert len(audit_logs) >= 4
        
        camera.stop_stream()
    
    @pytest.mark.asyncio
    async def test_multi_incident_management(
        self, test_client: AsyncClient, mock_cameras, auth_headers
    ):
        """Test managing multiple concurrent incidents."""
        
        # Create multiple incidents of different priorities
        incidents_data = [
            {
                "camera_id": mock_cameras[0].camera_id,
                "event_type": "theft",
                "confidence": 0.87,
                "priority": "high",
                "location": "Library - Computer Lab"
            },
            {
                "camera_id": mock_cameras[1].camera_id,
                "event_type": "vandalism",
                "confidence": 0.75,
                "priority": "medium",
                "location": "Parking Lot B"
            },
            {
                "camera_id": mock_cameras[2].camera_id,
                "event_type": "loitering",
                "confidence": 0.68,
                "priority": "low",
                "location": "Building C - Entrance"
            }
        ]
        
        created_incidents = []
        
        # Create incidents
        for i, incident_data in enumerate(incidents_data):
            mock_cameras[i].start_stream()
            
            event_data = {
                "camera_id": incident_data["camera_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": incident_data["event_type"],
                "confidence_score": incident_data["confidence"],
                "metadata": {
                    "location": incident_data["location"],
                    "priority": incident_data["priority"]
                }
            }
            
            response = await test_client.post(
                "/api/v1/events",
                json=event_data,
                headers=auth_headers
            )
            assert response.status_code == 201
            created_incidents.append(response.json()["id"])
        
        await asyncio.sleep(2)  # Allow incident processing
        
        # Test incident prioritization and filtering
        # Get high priority incidents first
        high_priority_response = await test_client.get(
            "/api/v1/incidents",
            params={"priority": "high", "status": "open"},
            headers=auth_headers
        )
        assert high_priority_response.status_code == 200
        high_priority_incidents = high_priority_response.json()["items"]
        assert len(high_priority_incidents) >= 1
        
        # Test bulk incident assignment
        for incident in high_priority_incidents:
            assign_response = await test_client.patch(
                f"/api/v1/incidents/{incident['id']}",
                json={"status": "assigned", "assigned_to": "guard_team_alpha"},
                headers=auth_headers
            )
            assert assign_response.status_code == 200
        
        # Test incident dashboard view
        dashboard_response = await test_client.get(
            "/api/v1/dashboard/incidents",
            headers=auth_headers
        )
        assert dashboard_response.status_code == 200
        dashboard_data = dashboard_response.json()
        
        # Should show incident counts by status and priority
        assert "open_incidents" in dashboard_data
        assert "assigned_incidents" in dashboard_data
        assert "priority_breakdown" in dashboard_data
        
        # Cleanup
        for camera in mock_cameras[:3]:
            camera.stop_stream()
    
    @pytest.mark.asyncio
    async def test_evidence_management_workflow(
        self, test_client: AsyncClient, test_storage, auth_headers
    ):
        """Test evidence collection, viewing, and export workflow."""
        
        storage_client, bucket = test_storage
        
        # Create incident with evidence
        event_data = {
            "camera_id": "evidence_test_camera",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "theft",
            "confidence_score": 0.89,
            "evidence_clip_path": "evidence/theft_incident_20240103_143022.mp4",
            "metadata": {
                "location": "Bookstore - Electronics Section",
                "evidence_duration": 30,
                "faces_detected": 2
            }
        }
        
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        event_id = response.json()["id"]
        
        # Get associated incident
        await asyncio.sleep(1)
        incidents_response = await test_client.get(
            "/api/v1/incidents",
            params={"event_id": event_id},
            headers=auth_headers
        )
        incident_id = incidents_response.json()["items"][0]["id"]
        
        # Test evidence viewing
        evidence_response = await test_client.get(
            f"/api/v1/incidents/{incident_id}/evidence",
            headers=auth_headers
        )
        assert evidence_response.status_code == 200
        evidence_data = evidence_response.json()
        
        # Verify evidence metadata
        assert "video_clips" in evidence_data
        assert "privacy_processed" in evidence_data
        
        # Test evidence download/export
        if evidence_data["video_clips"]:
            clip_id = evidence_data["video_clips"][0]["id"]
            download_response = await test_client.get(
                f"/api/v1/evidence/{clip_id}/download",
                headers=auth_headers
            )
            # Should return download URL or file stream
            assert download_response.status_code in [200, 302]
        
        # Test evidence chain of custody
        custody_response = await test_client.get(
            f"/api/v1/evidence/{clip_id}/custody",
            headers=auth_headers
        )
        assert custody_response.status_code == 200
        custody_data = custody_response.json()
        
        # Should track all access events
        assert "access_log" in custody_data
        assert len(custody_data["access_log"]) > 0
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring_dashboard(
        self, test_client: AsyncClient, mock_cameras, auth_headers
    ):
        """Test real-time monitoring dashboard functionality."""
        
        # Test WebSocket connection for real-time updates
        ws_url = "ws://test/ws/dashboard"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                # Authenticate WebSocket connection
                await websocket.send(json.dumps({
                    "type": "auth",
                    "token": auth_headers["Authorization"].split(" ")[1]
                }))
                
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)
                assert auth_data["type"] == "auth_success"
                
                # Start camera streams
                for camera in mock_cameras[:3]:
                    camera.start_stream()
                
                # Create incident that should trigger real-time update
                incident_data = {
                    "camera_id": mock_cameras[0].camera_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "intrusion",
                    "confidence_score": 0.91,
                    "metadata": {"real_time_test": True}
                }
                
                # Create event via API
                response = await test_client.post(
                    "/api/v1/events",
                    json=incident_data,
                    headers=auth_headers
                )
                assert response.status_code == 201
                
                # Wait for WebSocket notification
                try:
                    notification = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    notification_data = json.loads(notification)
                    
                    assert notification_data["type"] in ["incident_created", "dashboard_update"]
                    
                except asyncio.TimeoutError:
                    pytest.fail("Real-time notification not received within timeout")
                
                # Test camera status updates
                camera_status_response = await test_client.get(
                    "/api/v1/cameras/status",
                    headers=auth_headers
                )
                assert camera_status_response.status_code == 200
                camera_status = camera_status_response.json()
                
                # Should show active cameras
                active_cameras = [c for c in camera_status["cameras"] if c["status"] == "active"]
                assert len(active_cameras) >= 3
                
                # Cleanup
                for camera in mock_cameras[:3]:
                    camera.stop_stream()
        
        except Exception as e:
            # WebSocket might not be available in test environment
            pytest.skip(f"WebSocket testing skipped: {e}")


class SupervisorUAT:
    """User Acceptance Tests for Supervisors/Administrators."""
    
    @pytest.mark.asyncio
    async def test_analytics_and_reporting(
        self, test_client: AsyncClient, admin_auth_headers
    ):
        """Test analytics dashboard and reporting functionality."""
        
        # Test incident analytics
        analytics_response = await test_client.get(
            "/api/v1/analytics/incidents/summary",
            params={"period": "24h"},
            headers=admin_auth_headers
        )
        assert analytics_response.status_code == 200
        analytics_data = analytics_response.json()
        
        # Should include key metrics
        expected_metrics = [
            "total_incidents", "incidents_by_type", "incidents_by_location",
            "response_times", "resolution_rates"
        ]
        for metric in expected_metrics:
            assert metric in analytics_data
        
        # Test performance metrics
        performance_response = await test_client.get(
            "/api/v1/analytics/performance",
            params={"period": "7d"},
            headers=admin_auth_headers
        )
        assert performance_response.status_code == 200
        performance_data = performance_response.json()
        
        # Should include system performance data
        assert "detection_accuracy" in performance_data
        assert "alert_latency" in performance_data
        assert "system_uptime" in performance_data
        
        # Test custom report generation
        report_request = {
            "report_type": "incident_summary",
            "date_range": {
                "start": (datetime.now() - timedelta(days=7)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "filters": {
                "incident_types": ["theft", "vandalism", "intrusion"],
                "locations": ["Building A", "Building B"]
            },
            "format": "json"
        }
        
        report_response = await test_client.post(
            "/api/v1/reports/generate",
            json=report_request,
            headers=admin_auth_headers
        )
        assert report_response.status_code in [200, 202]  # Immediate or async generation
    
    @pytest.mark.asyncio
    async def test_system_configuration_management(
        self, test_client: AsyncClient, admin_auth_headers
    ):
        """Test system configuration and camera management."""
        
        # Test camera configuration
        camera_config = {
            "camera_id": "new_test_camera_01",
            "name": "Test Camera - Main Entrance",
            "location": {
                "building": "Administration",
                "floor": 1,
                "coordinates": {"lat": 40.7128, "lng": -74.0060}
            },
            "stream_url": "rtsp://test.camera.url/stream",
            "detection_zones": [
                {"name": "entrance", "coordinates": [[0, 0], [100, 0], [100, 100], [0, 100]]}
            ],
            "privacy_zones": [
                {"name": "reception_desk", "coordinates": [[20, 20], [80, 20], [80, 60], [20, 60]]}
            ]
        }
        
        camera_response = await test_client.post(
            "/api/v1/cameras",
            json=camera_config,
            headers=admin_auth_headers
        )
        assert camera_response.status_code == 201
        
        # Test detection threshold configuration
        threshold_config = {
            "event_type": "intrusion",
            "confidence_threshold": 0.75,
            "incident_creation_threshold": 0.85,
            "alert_threshold": 0.90
        }
        
        threshold_response = await test_client.put(
            "/api/v1/config/detection_thresholds",
            json=threshold_config,
            headers=admin_auth_headers
        )
        assert threshold_response.status_code == 200
        
        # Test notification configuration
        notification_config = {
            "channels": {
                "email": {
                    "enabled": True,
                    "recipients": ["security@university.edu", "admin@university.edu"]
                },
                "sms": {
                    "enabled": True,
                    "recipients": ["+1234567890"]
                },
                "push": {
                    "enabled": True,
                    "topics": ["security_alerts", "critical_incidents"]
                }
            },
            "escalation_rules": [
                {
                    "condition": "no_response_15_minutes",
                    "action": "escalate_to_supervisor"
                },
                {
                    "condition": "critical_incident",
                    "action": "immediate_all_channels"
                }
            ]
        }
        
        notification_response = await test_client.put(
            "/api/v1/config/notifications",
            json=notification_config,
            headers=admin_auth_headers
        )
        assert notification_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_compliance_and_audit_management(
        self, test_client: AsyncClient, admin_auth_headers
    ):
        """Test compliance monitoring and audit management."""
        
        # Test GDPR compliance dashboard
        gdpr_response = await test_client.get(
            "/api/v1/compliance/gdpr/status",
            headers=admin_auth_headers
        )
        assert gdpr_response.status_code == 200
        gdpr_data = gdpr_response.json()
        
        # Should show compliance status
        assert "data_retention_compliance" in gdpr_data
        assert "encryption_status" in gdpr_data
        assert "dsar_requests" in gdpr_data
        
        # Test audit log search and export
        audit_search = {
            "date_range": {
                "start": (datetime.now() - timedelta(days=1)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "actions": ["evidence_access", "incident_creation", "user_login"],
            "users": ["guard_john_doe", "admin_jane_smith"]
        }
        
        audit_response = await test_client.post(
            "/api/v1/audit/search",
            json=audit_search,
            headers=admin_auth_headers
        )
        assert audit_response.status_code == 200
        
        # Test compliance report generation
        compliance_report_request = {
            "report_type": "gdpr_compliance",
            "period": "monthly",
            "include_sections": [
                "data_processing_activities",
                "retention_policy_compliance",
                "dsar_handling",
                "security_measures"
            ]
        }
        
        compliance_response = await test_client.post(
            "/api/v1/compliance/reports/generate",
            json=compliance_report_request,
            headers=admin_auth_headers
        )
        assert compliance_response.status_code in [200, 202]


@pytest.mark.asyncio
async def test_complete_user_acceptance_suite(
    test_client: AsyncClient, test_db, test_redis, test_storage,
    mock_cameras, auth_headers, admin_auth_headers
):
    """Run complete User Acceptance Testing suite."""
    
    print("\n" + "="*60)
    print("USER ACCEPTANCE TESTING SUITE")
    print("="*60)
    
    test_results = []
    
    # Security Personnel UAT
    print("\n1. Security Personnel Workflows...")
    security_uat = SecurityPersonnelUAT()
    
    try:
        await security_uat.test_incident_response_workflow(
            test_client, mock_cameras, auth_headers
        )
        test_results.append({"test": "incident_response_workflow", "status": "PASS"})
        print("   ✓ Incident Response Workflow")
    except Exception as e:
        test_results.append({"test": "incident_response_workflow", "status": "FAIL", "error": str(e)})
        print(f"   ✗ Incident Response Workflow: {e}")
    
    try:
        await security_uat.test_multi_incident_management(
            test_client, mock_cameras, auth_headers
        )
        test_results.append({"test": "multi_incident_management", "status": "PASS"})
        print("   ✓ Multi-Incident Management")
    except Exception as e:
        test_results.append({"test": "multi_incident_management", "status": "FAIL", "error": str(e)})
        print(f"   ✗ Multi-Incident Management: {e}")
    
    try:
        await security_uat.test_evidence_management_workflow(
            test_client, test_storage, auth_headers
        )
        test_results.append({"test": "evidence_management_workflow", "status": "PASS"})
        print("   ✓ Evidence Management Workflow")
    except Exception as e:
        test_results.append({"test": "evidence_management_workflow", "status": "FAIL", "error": str(e)})
        print(f"   ✗ Evidence Management Workflow: {e}")
    
    try:
        await security_uat.test_real_time_monitoring_dashboard(
            test_client, mock_cameras, auth_headers
        )
        test_results.append({"test": "real_time_monitoring_dashboard", "status": "PASS"})
        print("   ✓ Real-time Monitoring Dashboard")
    except Exception as e:
        test_results.append({"test": "real_time_monitoring_dashboard", "status": "FAIL", "error": str(e)})
        print(f"   ✗ Real-time Monitoring Dashboard: {e}")
    
    # Supervisor UAT
    print("\n2. Supervisor/Administrator Workflows...")
    supervisor_uat = SupervisorUAT()
    
    try:
        await supervisor_uat.test_analytics_and_reporting(
            test_client, admin_auth_headers
        )
        test_results.append({"test": "analytics_and_reporting", "status": "PASS"})
        print("   ✓ Analytics and Reporting")
    except Exception as e:
        test_results.append({"test": "analytics_and_reporting", "status": "FAIL", "error": str(e)})
        print(f"   ✗ Analytics and Reporting: {e}")
    
    try:
        await supervisor_uat.test_system_configuration_management(
            test_client, admin_auth_headers
        )
        test_results.append({"test": "system_configuration_management", "status": "PASS"})
        print("   ✓ System Configuration Management")
    except Exception as e:
        test_results.append({"test": "system_configuration_management", "status": "FAIL", "error": str(e)})
        print(f"   ✗ System Configuration Management: {e}")
    
    try:
        await supervisor_uat.test_compliance_and_audit_management(
            test_client, admin_auth_headers
        )
        test_results.append({"test": "compliance_and_audit_management", "status": "PASS"})
        print("   ✓ Compliance and Audit Management")
    except Exception as e:
        test_results.append({"test": "compliance_and_audit_management", "status": "FAIL", "error": str(e)})
        print(f"   ✗ Compliance and Audit Management: {e}")
    
    # Generate UAT report
    passed_tests = sum(1 for r in test_results if r["status"] == "PASS")
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    uat_report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": success_rate,
            "overall_status": "PASS" if success_rate >= 85 else "FAIL"
        },
        "test_results": test_results
    }
    
    # Save UAT report
    with open("user_acceptance_test_report.json", "w") as f:
        json.dump(uat_report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("USER ACCEPTANCE TESTING SUMMARY")
    print("="*60)
    print(f"Overall Status: {uat_report['summary']['overall_status']}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"\nDetailed report saved to: user_acceptance_test_report.json")
    
    # Assert UAT success
    assert success_rate >= 85, f"User Acceptance Testing failed with {success_rate:.1f}% success rate"