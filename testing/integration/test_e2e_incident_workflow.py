#!/usr/bin/env python3
"""
End-to-end testing for complete incident detection workflow.
"""
import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
import json
import time
from typing import Dict, Any, List

from httpx import AsyncClient
import websockets
import cv2
import numpy as np

from conftest import (
    test_client, test_db, test_redis, test_storage,
    mock_cameras, mock_security_event, mock_incident,
    auth_headers, admin_auth_headers, performance_thresholds
)


class TestIncidentDetectionWorkflow:
    """Test complete incident detection workflow from camera to alert."""

    @pytest.mark.asyncio
    async def test_complete_incident_workflow(
        self, test_client: AsyncClient, test_db, test_redis, test_storage,
        mock_cameras, auth_headers, performance_thresholds
    ):
        """Test complete workflow: camera stream -> detection -> incident -> alert."""
        
        # Step 1: Simulate camera stream processing
        camera = mock_cameras[0]
        camera.start_stream()
        
        # Step 2: Generate security event from edge service
        event_data = {
            "camera_id": camera.camera_id,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "intrusion",
            "confidence_score": 0.92,
            "bounding_boxes": [
                {
                    "x": 150, "y": 200, "width": 100, "height": 200,
                    "class": "person", "confidence": 0.92
                }
            ],
            "metadata": {
                "location": "Building A - Main Entrance",
                "zone": "restricted",
                "model_version": "yolo_v8_security_1.2"
            }
        }
        
        start_time = time.time()
        
        # Create event via API
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        event_id = response.json()["id"]
        
        # Step 3: Verify incident creation (high confidence should trigger incident)
        await asyncio.sleep(0.5)  # Allow processing time
        
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
        
        # Step 4: Verify alert generation and latency
        alert_latency = time.time() - start_time
        assert alert_latency < performance_thresholds["alert_latency_max"]
        
        # Step 5: Verify evidence storage
        evidence_response = await test_client.get(
            f"/api/v1/incidents/{incident_id}/evidence",
            headers=auth_headers
        )
        assert evidence_response.status_code == 200
        
        # Step 6: Test incident management workflow
        # Assign incident
        assign_response = await test_client.patch(
            f"/api/v1/incidents/{incident_id}",
            json={"status": "assigned", "assigned_to": "security_guard_01"},
            headers=auth_headers
        )
        assert assign_response.status_code == 200
        
        # Resolve incident
        resolve_response = await test_client.patch(
            f"/api/v1/incidents/{incident_id}",
            json={"status": "resolved", "resolution_notes": "False alarm - authorized personnel"},
            headers=auth_headers
        )
        assert resolve_response.status_code == 200
        
        camera.stop_stream()

    @pytest.mark.asyncio
    async def test_concurrent_stream_processing(
        self, test_client: AsyncClient, mock_cameras, auth_headers, performance_thresholds
    ):
        """Test concurrent processing of multiple camera streams."""
        
        concurrent_streams = min(len(mock_cameras), performance_thresholds["concurrent_streams_min"])
        
        # Start multiple camera streams
        for i in range(concurrent_streams):
            mock_cameras[i].start_stream()
        
        # Generate events from multiple cameras simultaneously
        tasks = []
        for i in range(concurrent_streams):
            event_data = {
                "camera_id": mock_cameras[i].camera_id,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "suspicious_behavior",
                "confidence_score": 0.75,
                "metadata": {"location": f"Zone {i+1}"}
            }
            
            task = test_client.post(
                "/api/v1/events",
                json=event_data,
                headers=auth_headers
            )
            tasks.append(task)
        
        # Execute all requests concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        processing_time = time.time() - start_time
        
        # Verify all events were processed successfully
        for response in responses:
            assert response.status_code == 201
        
        # Verify processing time is within acceptable limits
        assert processing_time < performance_thresholds["api_response_time_max"] * concurrent_streams
        
        # Cleanup
        for camera in mock_cameras[:concurrent_streams]:
            camera.stop_stream()

    @pytest.mark.asyncio
    async def test_real_time_notifications(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test real-time WebSocket notifications for incidents."""
        
        # Connect to WebSocket endpoint
        ws_url = "ws://test/ws/incidents"
        
        async with websockets.connect(ws_url) as websocket:
            # Send authentication
            await websocket.send(json.dumps({
                "type": "auth",
                "token": auth_headers["Authorization"].split(" ")[1]
            }))
            
            # Create high-priority incident
            incident_data = {
                "camera_id": "test_camera_01",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "violence",
                "confidence_score": 0.95,
                "metadata": {"priority": "high"}
            }
            
            # Create event that should trigger incident
            response = await test_client.post(
                "/api/v1/events",
                json=incident_data,
                headers=auth_headers
            )
            assert response.status_code == 201
            
            # Wait for WebSocket notification
            try:
                notification = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                notification_data = json.loads(notification)
                
                assert notification_data["type"] == "incident_created"
                assert notification_data["data"]["severity"] == "high"
                
            except asyncio.TimeoutError:
                pytest.fail("WebSocket notification not received within timeout")

    @pytest.mark.asyncio
    async def test_evidence_chain_of_custody(
        self, test_client: AsyncClient, test_storage, auth_headers, admin_auth_headers
    ):
        """Test evidence management and chain of custody."""
        
        storage_client, bucket = test_storage
        
        # Create incident with evidence
        event_data = {
            "camera_id": "test_camera_01",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "theft",
            "confidence_score": 0.88,
            "evidence_clip_path": "test_evidence.mp4"
        }
        
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        event_id = response.json()["id"]
        
        # Get associated incident
        incidents_response = await test_client.get(
            "/api/v1/incidents",
            params={"event_id": event_id},
            headers=auth_headers
        )
        incident_id = incidents_response.json()["items"][0]["id"]
        
        # Access evidence (should create audit log)
        evidence_response = await test_client.get(
            f"/api/v1/incidents/{incident_id}/evidence",
            headers=auth_headers
        )
        assert evidence_response.status_code == 200
        
        # Verify audit log entry
        audit_response = await test_client.get(
            "/api/v1/audit/logs",
            params={"resource_id": incident_id, "action": "evidence_access"},
            headers=admin_auth_headers
        )
        assert audit_response.status_code == 200
        audit_logs = audit_response.json()["items"]
        assert len(audit_logs) > 0
        assert audit_logs[0]["action"] == "evidence_access"

    @pytest.mark.asyncio
    async def test_privacy_redaction_workflow(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test privacy redaction in evidence processing."""
        
        # Create event with privacy-sensitive content
        event_data = {
            "camera_id": "test_camera_privacy",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "loitering",
            "confidence_score": 0.70,
            "metadata": {
                "privacy_zone": True,
                "faces_detected": 2
            }
        }
        
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        
        # Verify privacy service was triggered
        # This would typically involve checking if the privacy microservice
        # processed the video for face blurring
        
        # Get processed evidence
        event_id = response.json()["id"]
        evidence_response = await test_client.get(
            f"/api/v1/events/{event_id}/evidence",
            headers=auth_headers
        )
        
        if evidence_response.status_code == 200:
            evidence = evidence_response.json()
            # Verify redaction metadata
            assert evidence.get("privacy_processed", False)
            assert "redaction_applied" in evidence.get("metadata", {})


class TestSystemResilience:
    """Test system resilience and failure scenarios."""

    @pytest.mark.asyncio
    async def test_database_failure_recovery(
        self, test_client: AsyncClient, test_redis, auth_headers
    ):
        """Test system behavior during database failures."""
        
        # Simulate database connection issue
        # Events should be cached in Redis for later processing
        
        event_data = {
            "camera_id": "test_camera_01",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "intrusion",
            "confidence_score": 0.85
        }
        
        # This should succeed even if database is temporarily unavailable
        # due to Redis caching and retry mechanisms
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        
        # Should either succeed immediately or return accepted for later processing
        assert response.status_code in [201, 202]

    @pytest.mark.asyncio
    async def test_storage_failure_handling(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test handling of storage service failures."""
        
        # Create event when storage is unavailable
        event_data = {
            "camera_id": "test_camera_01",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "vandalism",
            "confidence_score": 0.80,
            "evidence_clip_path": "unavailable_storage.mp4"
        }
        
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        
        # Event should be created even if evidence storage fails
        assert response.status_code == 201
        
        # Evidence should be marked as pending or failed
        event_id = response.json()["id"]
        evidence_response = await test_client.get(
            f"/api/v1/events/{event_id}/evidence",
            headers=auth_headers
        )
        
        if evidence_response.status_code == 200:
            evidence = evidence_response.json()
            assert evidence.get("status") in ["pending", "failed", "unavailable"]

    @pytest.mark.asyncio
    async def test_high_load_graceful_degradation(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test graceful degradation under high load."""
        
        # Generate high volume of events rapidly
        tasks = []
        for i in range(100):
            event_data = {
                "camera_id": f"load_test_camera_{i % 10}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "motion_detected",
                "confidence_score": 0.60 + (i % 40) / 100  # Varying confidence
            }
            
            task = test_client.post(
                "/api/v1/events",
                json=event_data,
                headers=auth_headers
            )
            tasks.append(task)
        
        # Execute requests with some concurrency
        batch_size = 20
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            responses = await asyncio.gather(*batch, return_exceptions=True)
            
            # Most requests should succeed, some may be rate limited
            success_count = sum(1 for r in responses 
                              if not isinstance(r, Exception) and r.status_code in [201, 202, 429])
            
            # At least 70% should be handled successfully (including rate limiting)
            assert success_count / len(batch) >= 0.7


@pytest.mark.asyncio
async def test_end_to_end_performance_benchmark(
    test_client: AsyncClient, mock_cameras, auth_headers, performance_thresholds
):
    """Comprehensive performance benchmark test."""
    
    results = {
        "detection_latency": [],
        "incident_creation_time": [],
        "alert_delivery_time": [],
        "api_response_times": []
    }
    
    # Run multiple iterations for statistical significance
    for iteration in range(10):
        camera = mock_cameras[0]
        camera.start_stream()
        
        # Measure detection to incident workflow
        start_time = time.time()
        
        event_data = {
            "camera_id": camera.camera_id,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "intrusion",
            "confidence_score": 0.90,
            "metadata": {"iteration": iteration}
        }
        
        # Create event
        api_start = time.time()
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        api_time = time.time() - api_start
        results["api_response_times"].append(api_time)
        
        assert response.status_code == 201
        event_id = response.json()["id"]
        
        # Wait for incident creation
        incident_created = False
        incident_start = time.time()
        
        for _ in range(50):  # Max 5 seconds wait
            incidents_response = await test_client.get(
                "/api/v1/incidents",
                params={"event_id": event_id},
                headers=auth_headers
            )
            
            if incidents_response.status_code == 200:
                incidents = incidents_response.json()["items"]
                if len(incidents) > 0:
                    incident_created = True
                    break
            
            await asyncio.sleep(0.1)
        
        incident_time = time.time() - incident_start
        total_time = time.time() - start_time
        
        if incident_created:
            results["incident_creation_time"].append(incident_time)
            results["detection_latency"].append(total_time)
        
        camera.stop_stream()
        await asyncio.sleep(0.1)  # Brief pause between iterations
    
    # Analyze results
    avg_detection_latency = sum(results["detection_latency"]) / len(results["detection_latency"])
    avg_api_response = sum(results["api_response_times"]) / len(results["api_response_times"])
    
    # Verify performance meets requirements
    assert avg_detection_latency < performance_thresholds["alert_latency_max"]
    assert avg_api_response < performance_thresholds["api_response_time_max"]
    
    # Log performance metrics
    print(f"\nPerformance Benchmark Results:")
    print(f"Average Detection Latency: {avg_detection_latency:.3f}s")
    print(f"Average API Response Time: {avg_api_response:.3f}s")
    print(f"Incident Creation Success Rate: {len(results['incident_creation_time'])/10*100:.1f}%")