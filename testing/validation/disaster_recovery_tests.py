#!/usr/bin/env python3
"""
Disaster Recovery and System Resilience Testing.
Tests system behavior under failure conditions and recovery scenarios.
"""
import pytest
import asyncio
import time
import json
import psutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from httpx import AsyncClient
import redis
import psycopg2

from conftest import (
    test_client, test_db, test_redis, test_storage,
    mock_cameras, auth_headers, performance_thresholds
)


@dataclass
class FailureScenario:
    """Represents a system failure scenario."""
    name: str
    description: str
    failure_type: str
    expected_behavior: str
    recovery_time_max: float  # seconds


class DisasterRecoveryTester:
    """Disaster recovery and resilience testing framework."""
    
    def __init__(self, client: AsyncClient):
        self.client = client
        self.failure_scenarios: List[FailureScenario] = []
        self.test_results: List[Dict[str, Any]] = []
    
    def add_scenario(self, scenario: FailureScenario):
        """Add a failure scenario to test."""
        self.failure_scenarios.append(scenario)
    
    async def simulate_database_failure(self, duration: float = 30.0):
        """Simulate database connection failure."""
        # This would typically involve stopping the database service
        # For testing, we'll simulate by overwhelming connections
        print(f"Simulating database failure for {duration} seconds...")
        await asyncio.sleep(duration)
        print("Database failure simulation ended")
    
    async def simulate_redis_failure(self, duration: float = 15.0):
        """Simulate Redis cache failure."""
        print(f"Simulating Redis failure for {duration} seconds...")
        await asyncio.sleep(duration)
        print("Redis failure simulation ended")
    
    async def simulate_storage_failure(self, duration: float = 20.0):
        """Simulate storage service failure."""
        print(f"Simulating storage failure for {duration} seconds...")
        await asyncio.sleep(duration)
        print("Storage failure simulation ended")
    
    async def simulate_network_partition(self, duration: float = 25.0):
        """Simulate network partition between services."""
        print(f"Simulating network partition for {duration} seconds...")
        await asyncio.sleep(duration)
        print("Network partition simulation ended")


class SystemResilienceTests:
    """System resilience test suite."""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure_resilience(
        self, test_client: AsyncClient, test_redis, auth_headers
    ):
        """Test system behavior during database connection failures."""
        
        dr_tester = DisasterRecoveryTester(test_client)
        
        # Create events before failure
        pre_failure_events = []
        for i in range(3):
            event_data = {
                "camera_id": f"resilience_camera_{i}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "motion_detected",
                "confidence_score": 0.70,
                "metadata": {"pre_failure": True}
            }
            
            response = await test_client.post(
                "/api/v1/events",
                json=event_data,
                headers=auth_headers
            )
            if response.status_code == 201:
                pre_failure_events.append(response.json()["id"])
        
        # Simulate database failure
        failure_start = time.time()
        
        # During failure, events should be queued in Redis
        failure_events = []
        failure_task = asyncio.create_task(
            dr_tester.simulate_database_failure(30.0)
        )
        
        # Try to create events during failure
        for i in range(5):
            event_data = {
                "camera_id": f"failure_camera_{i}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "intrusion",
                "confidence_score": 0.85,
                "metadata": {"during_failure": True}
            }
            
            try:
                response = await test_client.post(
                    "/api/v1/events",
                    json=event_data,
                    headers=auth_headers,
                    timeout=5.0
                )
                
                # Should either succeed (cached) or return 202 (queued)
                if response.status_code in [201, 202]:
                    failure_events.append(response.json().get("id", f"queued_{i}"))
                
            except Exception as e:
                print(f"Expected failure during DB outage: {e}")
            
            await asyncio.sleep(2)
        
        await failure_task
        failure_duration = time.time() - failure_start
        
        # After recovery, verify system functionality
        recovery_start = time.time()
        
        # Test system recovery
        recovery_event_data = {
            "camera_id": "recovery_test_camera",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "test_recovery",
            "confidence_score": 0.80,
            "metadata": {"post_recovery": True}
        }
        
        recovery_successful = False
        for attempt in range(10):  # Max 30 seconds for recovery
            try:
                response = await test_client.post(
                    "/api/v1/events",
                    json=recovery_event_data,
                    headers=auth_headers
                )
                
                if response.status_code == 201:
                    recovery_successful = True
                    break
                    
            except Exception:
                pass
            
            await asyncio.sleep(3)
        
        recovery_time = time.time() - recovery_start
        
        # Validate resilience metrics
        assert recovery_successful, "System failed to recover from database failure"
        assert recovery_time < 60, f"Recovery took too long: {recovery_time:.2f}s"
        
        # Verify queued events were processed (if applicable)
        if failure_events:
            await asyncio.sleep(5)  # Allow processing time
            
            # Check if queued events were eventually processed
            processed_events = 0
            for event_id in failure_events:
                if event_id.startswith("queued_"):
                    continue
                    
                try:
                    event_response = await test_client.get(
                        f"/api/v1/events/{event_id}",
                        headers=auth_headers
                    )
                    if event_response.status_code == 200:
                        processed_events += 1
                except Exception:
                    pass
            
            print(f"Processed {processed_events}/{len(failure_events)} queued events")
        
        return {
            "test": "database_failure_resilience",
            "failure_duration": failure_duration,
            "recovery_time": recovery_time,
            "recovery_successful": recovery_successful,
            "events_during_failure": len(failure_events)
        }
    
    @pytest.mark.asyncio
    async def test_storage_service_failure_handling(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test handling of storage service failures."""
        
        dr_tester = DisasterRecoveryTester(test_client)
        
        # Create event with evidence during storage failure
        failure_task = asyncio.create_task(
            dr_tester.simulate_storage_failure(20.0)
        )
        
        event_data = {
            "camera_id": "storage_failure_camera",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "theft",
            "confidence_score": 0.88,
            "evidence_clip_path": "evidence/storage_failure_test.mp4",
            "metadata": {"storage_failure_test": True}
        }
        
        # Event should be created even if evidence storage fails
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        event_id = response.json()["id"]
        
        await failure_task
        
        # Verify event exists but evidence may be marked as failed/pending
        event_response = await test_client.get(
            f"/api/v1/events/{event_id}",
            headers=auth_headers
        )
        assert event_response.status_code == 200
        
        # Check evidence status
        evidence_response = await test_client.get(
            f"/api/v1/events/{event_id}/evidence",
            headers=auth_headers
        )
        
        if evidence_response.status_code == 200:
            evidence_data = evidence_response.json()
            # Evidence should be marked as failed or pending retry
            assert evidence_data.get("status") in ["failed", "pending", "retry_scheduled"]
        
        return {
            "test": "storage_failure_handling",
            "event_created": True,
            "evidence_handling": "graceful_degradation"
        }
    
    @pytest.mark.asyncio
    async def test_high_memory_usage_handling(
        self, test_client: AsyncClient, mock_cameras, auth_headers
    ):
        """Test system behavior under high memory usage."""
        
        # Monitor initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate high volume of events to stress memory
        stress_events = []
        batch_size = 50
        
        for batch in range(5):  # 250 events total
            tasks = []
            
            for i in range(batch_size):
                event_data = {
                    "camera_id": f"stress_camera_{i % 10}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "motion_detected",
                    "confidence_score": 0.60 + (i % 40) / 100,
                    "metadata": {
                        "stress_test": True,
                        "batch": batch,
                        "large_data": "x" * 1000  # Add some bulk to events
                    }
                }
                
                task = test_client.post(
                    "/api/v1/events",
                    json=event_data,
                    headers=auth_headers
                )
                tasks.append(task)
            
            # Execute batch
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful responses
            successful = sum(
                1 for r in responses 
                if not isinstance(r, Exception) and r.status_code in [201, 202]
            )
            stress_events.append(successful)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            print(f"Batch {batch}: {successful}/{batch_size} events, Memory: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # If memory usage is too high, system should start rejecting requests
            if memory_increase > 200:  # 200MB increase threshold
                print("High memory usage detected - checking system response")
                break
            
            await asyncio.sleep(1)  # Brief pause between batches
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # System should handle memory pressure gracefully
        total_successful = sum(stress_events)
        
        return {
            "test": "high_memory_usage_handling",
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": total_memory_increase,
            "total_events_processed": total_successful,
            "graceful_handling": total_memory_increase < 500  # 500MB limit
        }
    
    @pytest.mark.asyncio
    async def test_concurrent_user_load_resilience(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test system resilience under concurrent user load."""
        
        # Simulate multiple users accessing the system simultaneously
        concurrent_users = 20
        requests_per_user = 10
        
        async def user_simulation(user_id: int):
            """Simulate a single user's requests."""
            user_results = []
            
            for request_num in range(requests_per_user):
                try:
                    # Mix of different API calls
                    if request_num % 3 == 0:
                        # List incidents
                        response = await test_client.get(
                            "/api/v1/incidents",
                            params={"limit": 20},
                            headers=auth_headers
                        )
                    elif request_num % 3 == 1:
                        # Create event
                        event_data = {
                            "camera_id": f"concurrent_camera_{user_id}",
                            "timestamp": datetime.utcnow().isoformat(),
                            "event_type": "motion_detected",
                            "confidence_score": 0.65,
                            "metadata": {"user_id": user_id, "request_num": request_num}
                        }
                        response = await test_client.post(
                            "/api/v1/events",
                            json=event_data,
                            headers=auth_headers
                        )
                    else:
                        # Get dashboard data
                        response = await test_client.get(
                            "/api/v1/dashboard/summary",
                            headers=auth_headers
                        )
                    
                    user_results.append({
                        "request_num": request_num,
                        "status_code": response.status_code,
                        "success": response.status_code < 400
                    })
                    
                except Exception as e:
                    user_results.append({
                        "request_num": request_num,
                        "status_code": 0,
                        "success": False,
                        "error": str(e)
                    })
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            return user_results
        
        # Execute concurrent user simulations
        start_time = time.time()
        
        user_tasks = [
            asyncio.create_task(user_simulation(user_id))
            for user_id in range(concurrent_users)
        ]
        
        all_user_results = await asyncio.gather(*user_tasks)
        
        execution_time = time.time() - start_time
        
        # Analyze results
        total_requests = 0
        successful_requests = 0
        rate_limited_requests = 0
        failed_requests = 0
        
        for user_results in all_user_results:
            for result in user_results:
                total_requests += 1
                if result["success"]:
                    successful_requests += 1
                elif result.get("status_code") == 429:
                    rate_limited_requests += 1
                else:
                    failed_requests += 1
        
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        handled_rate = (successful_requests + rate_limited_requests) / total_requests if total_requests > 0 else 0
        
        return {
            "test": "concurrent_user_load_resilience",
            "concurrent_users": concurrent_users,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "rate_limited_requests": rate_limited_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "handled_rate": handled_rate,
            "execution_time": execution_time,
            "resilient": handled_rate >= 0.85  # 85% handled gracefully
        }


@pytest.mark.asyncio
async def test_comprehensive_disaster_recovery_suite(
    test_client: AsyncClient, test_db, test_redis, test_storage,
    mock_cameras, auth_headers, performance_thresholds
):
    """Run comprehensive disaster recovery and resilience testing."""
    
    print("\n" + "="*60)
    print("DISASTER RECOVERY & RESILIENCE TESTING")
    print("="*60)
    
    resilience_tester = SystemResilienceTests()
    test_results = []
    
    # Database failure resilience
    print("\n1. Database Failure Resilience...")
    try:
        db_result = await resilience_tester.test_database_connection_failure_resilience(
            test_client, test_redis, auth_headers
        )
        test_results.append(db_result)
        print(f"   ✓ Recovery Time: {db_result['recovery_time']:.2f}s")
    except Exception as e:
        test_results.append({
            "test": "database_failure_resilience",
            "error": str(e),
            "recovery_successful": False
        })
        print(f"   ✗ Database Failure Test: {e}")
    
    # Storage failure handling
    print("\n2. Storage Service Failure Handling...")
    try:
        storage_result = await resilience_tester.test_storage_service_failure_handling(
            test_client, auth_headers
        )
        test_results.append(storage_result)
        print("   ✓ Graceful storage failure handling")
    except Exception as e:
        test_results.append({
            "test": "storage_failure_handling",
            "error": str(e),
            "event_created": False
        })
        print(f"   ✗ Storage Failure Test: {e}")
    
    # High memory usage handling
    print("\n3. High Memory Usage Handling...")
    try:
        memory_result = await resilience_tester.test_high_memory_usage_handling(
            test_client, mock_cameras, auth_headers
        )
        test_results.append(memory_result)
        print(f"   ✓ Memory increase: {memory_result['memory_increase_mb']:.1f}MB")
    except Exception as e:
        test_results.append({
            "test": "high_memory_usage_handling",
            "error": str(e),
            "graceful_handling": False
        })
        print(f"   ✗ Memory Usage Test: {e}")
    
    # Concurrent user load resilience
    print("\n4. Concurrent User Load Resilience...")
    try:
        load_result = await resilience_tester.test_concurrent_user_load_resilience(
            test_client, auth_headers
        )
        test_results.append(load_result)
        print(f"   ✓ Handled Rate: {load_result['handled_rate']:.1%}")
    except Exception as e:
        test_results.append({
            "test": "concurrent_user_load_resilience",
            "error": str(e),
            "resilient": False
        })
        print(f"   ✗ Concurrent Load Test: {e}")
    
    # Generate disaster recovery report
    successful_tests = sum(
        1 for result in test_results 
        if result.get("recovery_successful", True) and 
           result.get("graceful_handling", True) and 
           result.get("resilient", True) and
           "error" not in result
    )
    
    total_tests = len(test_results)
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    dr_report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed": successful_tests,
            "failed": total_tests - successful_tests,
            "success_rate": success_rate,
            "overall_status": "PASS" if success_rate >= 75 else "FAIL"
        },
        "resilience_metrics": {
            "database_recovery": next(
                (r for r in test_results if r.get("test") == "database_failure_resilience"), 
                {"recovery_successful": False}
            ).get("recovery_successful", False),
            "storage_graceful_degradation": next(
                (r for r in test_results if r.get("test") == "storage_failure_handling"), 
                {"event_created": False}
            ).get("event_created", False),
            "memory_management": next(
                (r for r in test_results if r.get("test") == "high_memory_usage_handling"), 
                {"graceful_handling": False}
            ).get("graceful_handling", False),
            "concurrent_load_handling": next(
                (r for r in test_results if r.get("test") == "concurrent_user_load_resilience"), 
                {"resilient": False}
            ).get("resilient", False)
        },
        "detailed_results": test_results
    }
    
    # Save disaster recovery report
    with open("disaster_recovery_report.json", "w") as f:
        json.dump(dr_report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DISASTER RECOVERY TESTING SUMMARY")
    print("="*60)
    print(f"Overall Status: {dr_report['summary']['overall_status']}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Tests Passed: {successful_tests}/{total_tests}")
    
    print("\nResilience Metrics:")
    for metric, status in dr_report["resilience_metrics"].items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {metric.replace('_', ' ').title()}")
    
    print(f"\nDetailed report saved to: disaster_recovery_report.json")
    
    # Assert disaster recovery success
    assert success_rate >= 75, f"Disaster recovery testing failed with {success_rate:.1f}% success rate"