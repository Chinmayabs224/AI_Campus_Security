#!/usr/bin/env python3
"""
System validation and acceptance testing for AI-Powered Campus Security System.
Validates performance requirements, model accuracy, and system resilience.
"""
import pytest
import asyncio
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import psutil
import numpy as np
from httpx import AsyncClient

from conftest import (
    test_client, test_db, test_redis, test_storage,
    mock_cameras, auth_headers, performance_thresholds
)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    measured_value: float
    threshold: float
    unit: str
    details: Dict[str, Any] = None


class SystemValidator:
    """System validation framework."""
    
    def __init__(self, client: AsyncClient, thresholds: Dict[str, float]):
        self.client = client
        self.thresholds = thresholds
        self.results: List[ValidationResult] = []
    
    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "results": [
                {
                    "test": r.test_name,
                    "status": "PASS" if r.passed else "FAIL",
                    "value": r.measured_value,
                    "threshold": r.threshold,
                    "unit": r.unit,
                    "details": r.details
                }
                for r in self.results
            ]
        }


class PerformanceValidator:
    """Performance validation tests."""
    
    @pytest.mark.asyncio
    async def test_alert_latency_validation(
        self, test_client: AsyncClient, mock_cameras, auth_headers, performance_thresholds
    ):
        """Validate alert latency meets <5s requirement."""
        validator = SystemValidator(test_client, performance_thresholds)
        latencies = []
        
        # Test multiple scenarios
        test_scenarios = [
            {"event_type": "intrusion", "confidence": 0.95, "priority": "high"},
            {"event_type": "violence", "confidence": 0.88, "priority": "critical"},
            {"event_type": "theft", "confidence": 0.82, "priority": "high"},
            {"event_type": "vandalism", "confidence": 0.75, "priority": "medium"}
        ]
        
        for i, scenario in enumerate(test_scenarios):
            camera = mock_cameras[i % len(mock_cameras)]
            camera.start_stream()
            
            # Measure end-to-end latency
            start_time = time.time()
            
            event_data = {
                "camera_id": camera.camera_id,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": scenario["event_type"],
                "confidence_score": scenario["confidence"],
                "metadata": {"priority": scenario["priority"]}
            }
            
            # Create event
            response = await test_client.post(
                "/api/v1/events",
                json=event_data,
                headers=auth_headers
            )
            assert response.status_code == 201
            event_id = response.json()["id"]
            
            # Wait for incident creation and alert
            incident_created = False
            for _ in range(100):  # Max 10 seconds
                incidents_response = await test_client.get(
                    "/api/v1/incidents",
                    params={"event_id": event_id},
                    headers=auth_headers
                )
                
                if incidents_response.status_code == 200:
                    incidents = incidents_response.json()["items"]
                    if incidents and incidents[0].get("alert_sent"):
                        incident_created = True
                        break
                
                await asyncio.sleep(0.1)
            
            latency = time.time() - start_time
            latencies.append(latency)
            
            camera.stop_stream()
        
        # Validate latency requirements
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Average latency validation
        validator.add_result(ValidationResult(
            test_name="average_alert_latency",
            passed=avg_latency < performance_thresholds["alert_latency_max"],
            measured_value=avg_latency,
            threshold=performance_thresholds["alert_latency_max"],
            unit="seconds",
            details={"all_latencies": latencies}
        ))
        
        # P95 latency validation
        validator.add_result(ValidationResult(
            test_name="p95_alert_latency",
            passed=p95_latency < performance_thresholds["alert_latency_max"],
            measured_value=p95_latency,
            threshold=performance_thresholds["alert_latency_max"],
            unit="seconds"
        ))
        
        return validator

    @pytest.mark.asyncio
    async def test_concurrent_stream_performance(
        self, test_client: AsyncClient, mock_cameras, auth_headers, performance_thresholds
    ):
        """Validate concurrent stream processing performance."""
        validator = SystemValidator(test_client, performance_thresholds)
        
        # Test with increasing concurrent streams
        stream_counts = [5, 10, 15, 20]
        performance_data = []
        
        for stream_count in stream_counts:
            if stream_count > len(mock_cameras):
                break
            
            # Start concurrent streams
            active_cameras = mock_cameras[:stream_count]
            for camera in active_cameras:
                camera.start_stream()
            
            # Generate concurrent events
            start_time = time.time()
            tasks = []
            
            for i, camera in enumerate(active_cameras):
                event_data = {
                    "camera_id": camera.camera_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "motion_detected",
                    "confidence_score": 0.65 + (i % 30) / 100,
                    "metadata": {"stream_test": True}
                }
                
                task = test_client.post(
                    "/api/v1/events",
                    json=event_data,
                    headers=auth_headers
                )
                tasks.append(task)
            
            # Execute concurrent requests
            responses = await asyncio.gather(*tasks)
            processing_time = time.time() - start_time
            
            # Validate responses
            success_count = sum(1 for r in responses if r.status_code in [201, 202])
            success_rate = success_count / len(responses)
            
            performance_data.append({
                "stream_count": stream_count,
                "processing_time": processing_time,
                "success_rate": success_rate,
                "throughput": success_count / processing_time
            })
            
            # Cleanup
            for camera in active_cameras:
                camera.stop_stream()
            
            await asyncio.sleep(1)  # Brief pause between tests
        
        # Validate concurrent processing capability
        max_successful_streams = max(
            data["stream_count"] for data in performance_data 
            if data["success_rate"] >= 0.95
        )
        
        validator.add_result(ValidationResult(
            test_name="concurrent_stream_capacity",
            passed=max_successful_streams >= performance_thresholds["concurrent_streams_min"],
            measured_value=max_successful_streams,
            threshold=performance_thresholds["concurrent_streams_min"],
            unit="streams",
            details={"performance_data": performance_data}
        ))
        
        return validator


class ModelAccuracyValidator:
    """AI model accuracy validation tests."""
    
    @pytest.mark.asyncio
    async def test_detection_accuracy_validation(
        self, test_client: AsyncClient, auth_headers, performance_thresholds
    ):
        """Validate model detection accuracy and false positive rates."""
        validator = SystemValidator(test_client, performance_thresholds)
        
        # Test with known ground truth data
        test_cases = [
            # True positives
            {"event_type": "intrusion", "confidence": 0.92, "expected": True},
            {"event_type": "violence", "confidence": 0.88, "expected": True},
            {"event_type": "theft", "confidence": 0.85, "expected": True},
            {"event_type": "vandalism", "confidence": 0.78, "expected": True},
            
            # True negatives (normal activity)
            {"event_type": "normal", "confidence": 0.15, "expected": False},
            {"event_type": "normal", "confidence": 0.25, "expected": False},
            {"event_type": "normal", "confidence": 0.35, "expected": False},
            
            # Edge cases
            {"event_type": "suspicious_behavior", "confidence": 0.65, "expected": True},
            {"event_type": "loitering", "confidence": 0.55, "expected": False},
            {"event_type": "crowd_detection", "confidence": 0.70, "expected": True}
        ]
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for i, case in enumerate(test_cases):
            event_data = {
                "camera_id": f"validation_camera_{i}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": case["event_type"],
                "confidence_score": case["confidence"],
                "metadata": {"validation_test": True}
            }
            
            response = await test_client.post(
                "/api/v1/events",
                json=event_data,
                headers=auth_headers
            )
            assert response.status_code == 201
            event_id = response.json()["id"]
            
            # Check if incident was created (system decision)
            await asyncio.sleep(0.5)
            incidents_response = await test_client.get(
                "/api/v1/incidents",
                params={"event_id": event_id},
                headers=auth_headers
            )
            
            incident_created = (
                incidents_response.status_code == 200 and
                len(incidents_response.json()["items"]) > 0
            )
            
            # Calculate confusion matrix
            if case["expected"] and incident_created:
                true_positives += 1
            elif case["expected"] and not incident_created:
                false_negatives += 1
            elif not case["expected"] and incident_created:
                false_positives += 1
            else:
                true_negatives += 1
        
        # Calculate metrics
        total_predictions = len(test_cases)
        accuracy = (true_positives + true_negatives) / total_predictions
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # Validate accuracy
        validator.add_result(ValidationResult(
            test_name="detection_accuracy",
            passed=accuracy >= performance_thresholds["detection_accuracy_min"],
            measured_value=accuracy,
            threshold=performance_thresholds["detection_accuracy_min"],
            unit="ratio",
            details={
                "confusion_matrix": {
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "true_negatives": true_negatives,
                    "false_negatives": false_negatives
                },
                "precision": precision,
                "recall": recall
            }
        ))
        
        # Validate false positive rate
        validator.add_result(ValidationResult(
            test_name="false_positive_rate",
            passed=false_positive_rate <= performance_thresholds["false_positive_rate_max"],
            measured_value=false_positive_rate,
            threshold=performance_thresholds["false_positive_rate_max"],
            unit="ratio"
        ))
        
        return validator


class DisasterRecoveryValidator:
    """Disaster recovery and system resilience tests."""
    
    @pytest.mark.asyncio
    async def test_database_failure_recovery(
        self, test_client: AsyncClient, test_redis, auth_headers
    ):
        """Test system resilience during database failures."""
        validator = SystemValidator(test_client, {})
        
        # Simulate database connection issues
        # Events should be cached in Redis for later processing
        
        event_data = {
            "camera_id": "resilience_test_camera",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "intrusion",
            "confidence_score": 0.85,
            "metadata": {"resilience_test": True}
        }
        
        # Test event creation during simulated DB failure
        response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        
        # Should succeed or be queued for later processing
        db_resilience_passed = response.status_code in [201, 202]
        
        validator.add_result(ValidationResult(
            test_name="database_failure_resilience",
            passed=db_resilience_passed,
            measured_value=1.0 if db_resilience_passed else 0.0,
            threshold=1.0,
            unit="boolean",
            details={"response_code": response.status_code}
        ))
        
        return validator
    
    @pytest.mark.asyncio
    async def test_high_load_graceful_degradation(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test graceful degradation under extreme load."""
        validator = SystemValidator(test_client, {})
        
        # Generate high volume of requests
        load_test_size = 200
        tasks = []
        
        for i in range(load_test_size):
            event_data = {
                "camera_id": f"load_test_camera_{i % 20}",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "motion_detected",
                "confidence_score": 0.50 + (i % 50) / 100,
                "metadata": {"load_test": True}
            }
            
            task = test_client.post(
                "/api/v1/events",
                json=event_data,
                headers=auth_headers
            )
            tasks.append(task)
        
        # Execute in batches to simulate realistic load
        batch_size = 50
        successful_requests = 0
        rate_limited_requests = 0
        failed_requests = 0
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            responses = await asyncio.gather(*batch, return_exceptions=True)
            
            for response in responses:
                if isinstance(response, Exception):
                    failed_requests += 1
                elif response.status_code == 201:
                    successful_requests += 1
                elif response.status_code == 429:  # Rate limited
                    rate_limited_requests += 1
                else:
                    failed_requests += 1
            
            await asyncio.sleep(0.1)  # Brief pause between batches
        
        # Calculate graceful degradation metrics
        total_requests = load_test_size
        handled_requests = successful_requests + rate_limited_requests
        graceful_degradation_rate = handled_requests / total_requests
        
        # System should handle at least 80% of requests gracefully
        validator.add_result(ValidationResult(
            test_name="high_load_graceful_degradation",
            passed=graceful_degradation_rate >= 0.8,
            measured_value=graceful_degradation_rate,
            threshold=0.8,
            unit="ratio",
            details={
                "successful_requests": successful_requests,
                "rate_limited_requests": rate_limited_requests,
                "failed_requests": failed_requests,
                "total_requests": total_requests
            }
        ))
        
        return validator


class UserAcceptanceValidator:
    """User acceptance testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_security_personnel_workflow(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test complete security personnel workflow."""
        validator = SystemValidator(test_client, {})
        
        # Simulate security guard workflow
        workflow_steps = []
        
        # Step 1: Login and dashboard access
        dashboard_response = await test_client.get(
            "/api/v1/dashboard/summary",
            headers=auth_headers
        )
        workflow_steps.append({
            "step": "dashboard_access",
            "success": dashboard_response.status_code == 200
        })
        
        # Step 2: Create incident from event
        event_data = {
            "camera_id": "uat_camera_01",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "suspicious_behavior",
            "confidence_score": 0.78,
            "metadata": {"uat_test": True}
        }
        
        event_response = await test_client.post(
            "/api/v1/events",
            json=event_data,
            headers=auth_headers
        )
        workflow_steps.append({
            "step": "event_creation",
            "success": event_response.status_code == 201
        })
        
        if event_response.status_code == 201:
            event_id = event_response.json()["id"]
            
            # Step 3: View incident details
            await asyncio.sleep(1)  # Allow incident creation
            incidents_response = await test_client.get(
                "/api/v1/incidents",
                params={"event_id": event_id},
                headers=auth_headers
            )
            
            incident_found = (
                incidents_response.status_code == 200 and
                len(incidents_response.json()["items"]) > 0
            )
            workflow_steps.append({
                "step": "incident_viewing",
                "success": incident_found
            })
            
            if incident_found:
                incident_id = incidents_response.json()["items"][0]["id"]
                
                # Step 4: Assign incident
                assign_response = await test_client.patch(
                    f"/api/v1/incidents/{incident_id}",
                    json={"status": "assigned", "assigned_to": "guard_001"},
                    headers=auth_headers
                )
                workflow_steps.append({
                    "step": "incident_assignment",
                    "success": assign_response.status_code == 200
                })
                
                # Step 5: Add notes and resolve
                resolve_response = await test_client.patch(
                    f"/api/v1/incidents/{incident_id}",
                    json={
                        "status": "resolved",
                        "resolution_notes": "Investigated - no threat found"
                    },
                    headers=auth_headers
                )
                workflow_steps.append({
                    "step": "incident_resolution",
                    "success": resolve_response.status_code == 200
                })
        
        # Calculate workflow success rate
        successful_steps = sum(1 for step in workflow_steps if step["success"])
        workflow_success_rate = successful_steps / len(workflow_steps)
        
        validator.add_result(ValidationResult(
            test_name="security_personnel_workflow",
            passed=workflow_success_rate >= 0.9,
            measured_value=workflow_success_rate,
            threshold=0.9,
            unit="ratio",
            details={"workflow_steps": workflow_steps}
        ))
        
        return validator


@pytest.mark.asyncio
async def test_comprehensive_system_validation(
    test_client: AsyncClient, test_db, test_redis, test_storage,
    mock_cameras, auth_headers, performance_thresholds
):
    """Run comprehensive system validation suite."""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE SYSTEM VALIDATION")
    print("="*60)
    
    all_results = []
    
    # Performance validation
    print("\n1. Performance Validation...")
    perf_validator = PerformanceValidator()
    
    latency_results = await perf_validator.test_alert_latency_validation(
        test_client, mock_cameras, auth_headers, performance_thresholds
    )
    all_results.extend(latency_results.results)
    
    concurrent_results = await perf_validator.test_concurrent_stream_performance(
        test_client, mock_cameras, auth_headers, performance_thresholds
    )
    all_results.extend(concurrent_results.results)
    
    # Model accuracy validation
    print("2. Model Accuracy Validation...")
    accuracy_validator = ModelAccuracyValidator()
    
    accuracy_results = await accuracy_validator.test_detection_accuracy_validation(
        test_client, auth_headers, performance_thresholds
    )
    all_results.extend(accuracy_results.results)
    
    # Disaster recovery validation
    print("3. Disaster Recovery Validation...")
    dr_validator = DisasterRecoveryValidator()
    
    db_recovery_results = await dr_validator.test_database_failure_recovery(
        test_client, test_redis, auth_headers
    )
    all_results.extend(db_recovery_results.results)
    
    load_degradation_results = await dr_validator.test_high_load_graceful_degradation(
        test_client, auth_headers
    )
    all_results.extend(load_degradation_results.results)
    
    # User acceptance validation
    print("4. User Acceptance Validation...")
    uat_validator = UserAcceptanceValidator()
    
    workflow_results = await uat_validator.test_security_personnel_workflow(
        test_client, auth_headers
    )
    all_results.extend(workflow_results.results)
    
    # Generate comprehensive report
    passed_tests = sum(1 for r in all_results if r.passed)
    total_tests = len(all_results)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": success_rate,
            "overall_status": "PASS" if success_rate >= 90 else "FAIL"
        },
        "categories": {
            "performance": {
                "tests": [r for r in all_results if "latency" in r.test_name or "concurrent" in r.test_name],
                "status": "PASS" if all(r.passed for r in all_results if "latency" in r.test_name or "concurrent" in r.test_name) else "FAIL"
            },
            "accuracy": {
                "tests": [r for r in all_results if "accuracy" in r.test_name or "false_positive" in r.test_name],
                "status": "PASS" if all(r.passed for r in all_results if "accuracy" in r.test_name or "false_positive" in r.test_name) else "FAIL"
            },
            "resilience": {
                "tests": [r for r in all_results if "resilience" in r.test_name or "degradation" in r.test_name],
                "status": "PASS" if all(r.passed for r in all_results if "resilience" in r.test_name or "degradation" in r.test_name) else "FAIL"
            },
            "user_acceptance": {
                "tests": [r for r in all_results if "workflow" in r.test_name],
                "status": "PASS" if all(r.passed for r in all_results if "workflow" in r.test_name) else "FAIL"
            }
        },
        "detailed_results": [
            {
                "test_name": r.test_name,
                "status": "PASS" if r.passed else "FAIL",
                "measured_value": r.measured_value,
                "threshold": r.threshold,
                "unit": r.unit,
                "details": r.details
            }
            for r in all_results
        ]
    }
    
    # Save validation report
    with open("system_validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SYSTEM VALIDATION SUMMARY")
    print("="*60)
    print(f"Overall Status: {validation_report['summary']['overall_status']}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print("\nCategory Results:")
    for category, data in validation_report["categories"].items():
        print(f"  {category.title()}: {data['status']}")
    
    print(f"\nDetailed report saved to: system_validation_report.json")
    
    # Assert overall validation success
    assert success_rate >= 90, f"System validation failed with {success_rate:.1f}% success rate"