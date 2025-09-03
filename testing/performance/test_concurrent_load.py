#!/usr/bin/env python3
"""
Concurrent load testing for camera stream processing.
"""
import pytest
import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

from httpx import AsyncClient
import websockets

from conftest import (
    test_client, test_db, test_redis, mock_cameras,
    auth_headers, performance_thresholds
)


class TestConcurrentStreamProcessing:
    """Test concurrent processing of multiple camera streams."""

    @pytest.mark.asyncio
    async def test_multiple_camera_streams(
        self, test_client: AsyncClient, mock_cameras, auth_headers, performance_thresholds
    ):
        """Test processing multiple concurrent camera streams."""
        
        num_cameras = min(len(mock_cameras), 10)
        concurrent_events = []
        
        # Start all camera streams
        for i in range(num_cameras):
            mock_cameras[i].start_stream()
        
        # Generate events from all cameras simultaneously
        start_time = time.time()
        
        tasks = []
        for i in range(num_cameras):
            for event_num in range(5):  # 5 events per camera
                event_data = {
                    "camera_id": mock_cameras[i].camera_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": random.choice(["intrusion", "suspicious_behavior", "loitering"]),
                    "confidence_score": random.uniform(0.7, 0.95),
                    "metadata": {
                        "location": f"Zone {i+1}",
                        "event_sequence": event_num
                    }
                }
                
                task = test_client.post(
                    "/api/v1/events",
                    json=event_data,
                    headers=auth_headers
                )
                tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        # Analyze results
        successful_responses = [r for r in responses 
                              if not isinstance(r, Exception) and r.status_code == 201]
        
        success_rate = len(successful_responses) / len(tasks)
        throughput = len(successful_responses) / processing_time
        
        # Verify performance requirements
        assert success_rate >= 0.95  # 95% success rate
        assert throughput >= 10  # At least 10 events/second
        assert processing_time < performance_thresholds["api_response_time_max"] * len(tasks) / 10
        
        # Cleanup
        for camera in mock_cameras[:num_cameras]:
            camera.stop_stream()

    @pytest.mark.asyncio
    async def test_stream_processing_under_load(
        self, test_client: AsyncClient, mock_cameras, auth_headers
    ):
        """Test stream processing performance under high load."""
        
        camera = mock_cameras[0]
        camera.start_stream()
        
        # Simulate high-frequency events (30 FPS equivalent)
        events_per_second = 30
        test_duration = 10  # seconds
        total_events = events_per_second * test_duration
        
        event_times = []
        successful_events = 0
        
        start_time = time.time()
        
        for i in range(total_events):
            event_start = time.time()
            
            event_data = {
                "camera_id": camera.camera_id,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "motion_detected",
                "confidence_score": random.uniform(0.6, 0.9),
                "frame_number": i,
                "metadata": {"fps_test": True}
            }
            
            try:
                response = await test_client.post(
                    "/api/v1/events",
                    json=event_data,
                    headers=auth_headers
                )
                
                if response.status_code == 201:
                    successful_events += 1
                    event_times.append(time.time() - event_start)
                
            except Exception as e:
                print(f"Event {i} failed: {e}")
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            expected_time = i / events_per_second
            if elapsed < expected_time:
                await asyncio.sleep(expected_time - elapsed)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        success_rate = successful_events / total_events
        actual_fps = successful_events / total_time
        avg_event_time = sum(event_times) / len(event_times) if event_times else 0
        
        # Verify performance
        assert success_rate >= 0.8  # 80% success rate under load
        assert actual_fps >= 20  # At least 20 FPS processing
        assert avg_event_time < 0.1  # Less than 100ms per event
        
        camera.stop_stream()

    @pytest.mark.asyncio
    async def test_websocket_concurrent_connections(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test concurrent WebSocket connections for real-time updates."""
        
        num_connections = 50
        connections = []
        received_messages = []
        
        try:
            # Establish multiple WebSocket connections
            for i in range(num_connections):
                ws_url = f"ws://test/ws/incidents?token={auth_headers['Authorization'].split(' ')[1]}"
                
                try:
                    websocket = await websockets.connect(ws_url)
                    connections.append(websocket)
                except Exception as e:
                    print(f"Failed to connect WebSocket {i}: {e}")
            
            # Generate incident that should trigger notifications
            incident_data = {
                "camera_id": "test_camera_broadcast",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "violence",
                "confidence_score": 0.95,
                "metadata": {"broadcast_test": True}
            }
            
            # Create incident
            response = await test_client.post(
                "/api/v1/events",
                json=incident_data,
                headers=auth_headers
            )
            
            if response.status_code == 201:
                # Wait for notifications on all connections
                notification_tasks = []
                for ws in connections:
                    task = asyncio.create_task(self._wait_for_notification(ws))
                    notification_tasks.append(task)
                
                # Wait for notifications with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*notification_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                    
                    successful_notifications = sum(1 for r in results 
                                                 if not isinstance(r, Exception) and r is not None)
                    
                    # At least 80% of connections should receive notifications
                    notification_rate = successful_notifications / len(connections)
                    assert notification_rate >= 0.8
                    
                except asyncio.TimeoutError:
                    pytest.fail("WebSocket notifications timed out")
        
        finally:
            # Cleanup connections
            for ws in connections:
                try:
                    await ws.close()
                except:
                    pass

    async def _wait_for_notification(self, websocket):
        """Wait for notification on WebSocket connection."""
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            data = json.loads(message)
            return data if data.get("type") == "incident_created" else None
        except (asyncio.TimeoutError, json.JSONDecodeError, websockets.exceptions.ConnectionClosed):
            return None

    @pytest.mark.asyncio
    async def test_database_connection_pooling(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test database connection pooling under concurrent load."""
        
        # Generate many concurrent database operations
        num_concurrent = 100
        
        tasks = []
        for i in range(num_concurrent):
            # Mix of read and write operations
            if i % 3 == 0:
                # Write operation
                event_data = {
                    "camera_id": f"pool_test_camera_{i % 10}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "motion_detected",
                    "confidence_score": random.uniform(0.6, 0.9),
                    "metadata": {"pool_test": i}
                }
                task = test_client.post("/api/v1/events", json=event_data, headers=auth_headers)
            else:
                # Read operation
                task = test_client.get("/api/v1/incidents", 
                                     params={"limit": 10}, headers=auth_headers)
            
            tasks.append(task)
        
        # Execute all operations concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Analyze results
        successful_responses = [r for r in responses 
                              if not isinstance(r, Exception) and r.status_code in [200, 201]]
        
        success_rate = len(successful_responses) / len(tasks)
        throughput = len(successful_responses) / execution_time
        
        # Verify connection pooling effectiveness
        assert success_rate >= 0.95  # 95% success rate
        assert throughput >= 50  # At least 50 operations/second
        assert execution_time < 5.0  # Complete within 5 seconds

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(
        self, test_client: AsyncClient, mock_cameras, auth_headers
    ):
        """Test memory usage during sustained load."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        camera = mock_cameras[0]
        camera.start_stream()
        
        # Generate sustained load for memory testing
        num_events = 1000
        batch_size = 50
        
        for batch in range(0, num_events, batch_size):
            tasks = []
            
            for i in range(batch_size):
                event_data = {
                    "camera_id": camera.camera_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "motion_detected",
                    "confidence_score": random.uniform(0.6, 0.9),
                    "metadata": {
                        "batch": batch // batch_size,
                        "event": i,
                        "memory_test": True
                    }
                }
                
                task = test_client.post("/api/v1/events", json=event_data, headers=auth_headers)
                tasks.append(task)
            
            # Execute batch
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check memory usage periodically
            if batch % (batch_size * 5) == 0:  # Every 5 batches
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory should not increase excessively (allow 100MB increase)
                assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"
        
        camera.stop_stream()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        # Total memory increase should be reasonable
        assert total_increase < 150, f"Total memory increase: {total_increase:.1f}MB"


@pytest.mark.asyncio
async def test_stress_testing_scenario(
    test_client: AsyncClient, mock_cameras, auth_headers, performance_thresholds
):
    """Comprehensive stress testing scenario."""
    
    # Stress test parameters
    num_cameras = min(len(mock_cameras), 20)
    events_per_camera = 100
    concurrent_users = 10
    
    # Start camera streams
    for i in range(num_cameras):
        mock_cameras[i].start_stream()
    
    # Create multiple user sessions
    user_tasks = []
    
    for user_id in range(concurrent_users):
        user_task = asyncio.create_task(
            simulate_user_activity(test_client, mock_cameras[:num_cameras], 
                                 auth_headers, events_per_camera // concurrent_users)
        )
        user_tasks.append(user_task)
    
    # Execute stress test
    start_time = time.time()
    results = await asyncio.gather(*user_tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # Analyze stress test results
    successful_users = sum(1 for r in results if not isinstance(r, Exception))
    total_events = sum(r.get("events_processed", 0) for r in results 
                      if isinstance(r, dict))
    
    overall_throughput = total_events / total_time if total_time > 0 else 0
    
    # Verify stress test performance
    assert successful_users >= concurrent_users * 0.8  # 80% of users successful
    assert overall_throughput >= 50  # At least 50 events/second overall
    
    # Cleanup
    for camera in mock_cameras[:num_cameras]:
        camera.stop_stream()


async def simulate_user_activity(
    client: AsyncClient, cameras: List, auth_headers: Dict, num_events: int
) -> Dict[str, Any]:
    """Simulate realistic user activity during stress test."""
    
    events_processed = 0
    errors = 0
    
    try:
        for i in range(num_events):
            # Select random camera
            camera = random.choice(cameras)
            
            # Generate event
            event_data = {
                "camera_id": camera.camera_id,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": random.choice([
                    "motion_detected", "intrusion", "suspicious_behavior", 
                    "loitering", "vandalism"
                ]),
                "confidence_score": random.uniform(0.6, 0.95),
                "metadata": {
                    "stress_test": True,
                    "user_simulation": True
                }
            }
            
            try:
                response = await client.post(
                    "/api/v1/events",
                    json=event_data,
                    headers=auth_headers
                )
                
                if response.status_code == 201:
                    events_processed += 1
                else:
                    errors += 1
                
            except Exception:
                errors += 1
            
            # Simulate user think time
            await asyncio.sleep(random.uniform(0.01, 0.1))
        
        return {
            "events_processed": events_processed,
            "errors": errors,
            "success_rate": events_processed / num_events if num_events > 0 else 0
        }
        
    except Exception as e:
        return {"error": str(e), "events_processed": events_processed}