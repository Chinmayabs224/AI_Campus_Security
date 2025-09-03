#!/usr/bin/env python3
"""
Security and compliance testing suite.
"""
import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
import json
import hashlib
import base64
from typing import Dict, Any, List

from httpx import AsyncClient
import jwt
from cryptography.fernet import Fernet

from conftest import (
    test_client, test_db, test_redis, test_storage,
    test_user, test_admin_user, auth_headers, admin_auth_headers,
    compliance_requirements
)


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""

    @pytest.mark.asyncio
    async def test_jwt_token_validation(self, test_client: AsyncClient):
        """Test JWT token validation and security."""
        
        # Test with invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = await test_client.get("/api/v1/incidents", headers=invalid_headers)
        assert response.status_code == 401
        
        # Test with expired token (mock)
        expired_token = jwt.encode(
            {"sub": "test_user", "exp": datetime.utcnow() - timedelta(hours=1)},
            "secret", algorithm="HS256"
        )
        expired_headers = {"Authorization": f"Bearer {expired_token}"}
        response = await test_client.get("/api/v1/incidents", headers=expired_headers)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_role_based_access_control(
        self, test_client: AsyncClient, auth_headers, admin_auth_headers
    ):
        """Test RBAC enforcement."""
        
        # Regular user should not access admin endpoints
        response = await test_client.get("/api/v1/admin/users", headers=auth_headers)
        assert response.status_code == 403
        
        # Admin should access admin endpoints
        response = await test_client.get("/api/v1/admin/users", headers=admin_auth_headers)
        assert response.status_code in [200, 404]  # 404 if no users exist
        
        # Test incident access permissions
        incident_data = {
            "camera_id": "test_camera_01",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "intrusion",
            "confidence_score": 0.85
        }
        
        # Create incident
        response = await test_client.post(
            "/api/v1/events", json=incident_data, headers=auth_headers
        )
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_api_rate_limiting(self, test_client: AsyncClient, auth_headers):
        """Test API rate limiting protection."""
        
        # Make rapid requests to trigger rate limiting
        tasks = []
        for _ in range(100):
            task = test_client.get("/api/v1/incidents", headers=auth_headers)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should have some rate limited responses
        rate_limited = sum(1 for r in responses 
                          if not isinstance(r, Exception) and r.status_code == 429)
        
        # At least some requests should be rate limited
        assert rate_limited > 0

    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, test_client: AsyncClient, auth_headers):
        """Test SQL injection attack protection."""
        
        # Attempt SQL injection in query parameters
        malicious_params = {
            "camera_id": "'; DROP TABLE incidents; --",
            "limit": "1 OR 1=1",
            "status": "open' UNION SELECT * FROM users --"
        }
        
        response = await test_client.get(
            "/api/v1/incidents", 
            params=malicious_params, 
            headers=auth_headers
        )
        
        # Should not return 500 error (indicates SQL injection blocked)
        assert response.status_code in [200, 400, 422]  # Valid responses

    @pytest.mark.asyncio
    async def test_xss_protection(self, test_client: AsyncClient, auth_headers):
        """Test XSS attack protection."""
        
        # Attempt XSS in event data
        xss_payload = "<script>alert('xss')</script>"
        event_data = {
            "camera_id": xss_payload,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "intrusion",
            "confidence_score": 0.85,
            "metadata": {
                "description": xss_payload,
                "location": f"Building A {xss_payload}"
            }
        }
        
        response = await test_client.post(
            "/api/v1/events", json=event_data, headers=auth_headers
        )
        
        # Should either reject or sanitize the input
        if response.status_code == 201:
            # If accepted, verify data is sanitized
            event_id = response.json()["id"]
            get_response = await test_client.get(
                f"/api/v1/events/{event_id}", headers=auth_headers
            )
            
            if get_response.status_code == 200:
                event = get_response.json()
                # XSS payload should be sanitized
                assert "<script>" not in str(event)


class TestDataProtectionCompliance:
    """Test GDPR, FERPA, and other data protection compliance."""

    @pytest.mark.asyncio
    async def test_gdpr_data_subject_rights(
        self, test_client: AsyncClient, test_db, auth_headers, compliance_requirements
    ):
        """Test GDPR data subject access rights."""
        
        gdpr_reqs = compliance_requirements["gdpr"]
        
        # Create DSAR (Data Subject Access Request)
        dsar_data = {
            "subject_email": "john.doe@example.com",
            "request_type": "access",
            "subject_name": "John Doe",
            "description": "Request access to all personal data"
        }
        
        response = await test_client.post(
            "/api/v1/privacy/dsar", json=dsar_data, headers=auth_headers
        )
        assert response.status_code == 201
        
        dsar_id = response.json()["id"]
        
        # Verify DSAR processing
        dsar_response = await test_client.get(
            f"/api/v1/privacy/dsar/{dsar_id}", headers=auth_headers
        )
        assert dsar_response.status_code == 200
        
        dsar = dsar_response.json()
        assert dsar["status"] in ["pending", "processing", "completed"]

    @pytest.mark.asyncio
    async def test_audit_logging_compliance(
        self, test_client: AsyncClient, test_db, auth_headers, admin_auth_headers,
        compliance_requirements
    ):
        """Test comprehensive audit logging for compliance."""
        
        # Perform actions that should be audited
        actions = [
            ("GET", "/api/v1/incidents"),
            ("POST", "/api/v1/events", {
                "camera_id": "test_camera_01",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "intrusion",
                "confidence_score": 0.85
            })
        ]
        
        for method, endpoint, *data in actions:
            if method == "GET":
                response = await test_client.get(endpoint, headers=auth_headers)
            elif method == "POST":
                response = await test_client.post(
                    endpoint, json=data[0] if data else {}, headers=auth_headers
                )
            
            # Verify action was logged
            await asyncio.sleep(0.1)  # Allow audit log processing
        
        # Check audit logs
        audit_response = await test_client.get(
            "/api/v1/audit/logs", 
            params={"limit": 10}, 
            headers=admin_auth_headers
        )
        
        if audit_response.status_code == 200:
            logs = audit_response.json()["items"]
            
            # Verify required audit fields
            for log in logs:
                assert "timestamp" in log
                assert "user_id" in log
                assert "action" in log
                assert "resource_type" in log


class TestSecurityHardening:
    """Test security hardening measures."""

    @pytest.mark.asyncio
    async def test_input_validation_security(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test comprehensive input validation."""
        
        # Test various malicious inputs
        malicious_inputs = [
            {"camera_id": "A" * 1000},  # Buffer overflow attempt
            {"confidence_score": -1},   # Invalid range
            {"confidence_score": 2.0},  # Invalid range
            {"timestamp": "invalid_date"},  # Invalid format
            {"event_type": "../../../etc/passwd"},  # Path traversal
        ]
        
        for malicious_data in malicious_inputs:
            base_data = {
                "camera_id": "test_camera_01",
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "intrusion",
                "confidence_score": 0.85
            }
            base_data.update(malicious_data)
            
            response = await test_client.post(
                "/api/v1/events", json=base_data, headers=auth_headers
            )
            
            # Should reject malicious input
            assert response.status_code in [400, 422]


@pytest.mark.asyncio
async def test_penetration_testing_simulation(
    test_client: AsyncClient, auth_headers
):
    """Simulate basic penetration testing scenarios."""
    
    # Test common attack vectors
    attack_vectors = [
        # Directory traversal
        {"path": "/api/v1/../../../etc/passwd"},
        
        # Command injection
        {"params": {"cmd": "; cat /etc/passwd"}},
        
        # Large payload DoS
        {"json": {"data": "A" * 100000}},
    ]
    
    for vector in attack_vectors:
        try:
            if "path" in vector:
                response = await test_client.get(vector["path"], headers=auth_headers)
            elif "params" in vector:
                response = await test_client.get(
                    "/api/v1/incidents", 
                    params=vector["params"], 
                    headers=auth_headers
                )
            elif "json" in vector:
                response = await test_client.post(
                    "/api/v1/events", 
                    json=vector["json"], 
                    headers=auth_headers
                )
            
            # System should handle attacks gracefully
            assert response.status_code in [400, 401, 403, 404, 413, 422, 429]
            
        except Exception as e:
            # Should not cause unhandled exceptions
            pytest.fail(f"Attack vector caused unhandled exception: {e}")