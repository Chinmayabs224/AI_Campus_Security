"""
Audit logging middleware for automatic request tracking.
"""
import time
from typing import Callable, Optional
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from .service import audit_service
from .models import AuditAction, ResourceType, ComplianceTag

logger = structlog.get_logger()


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log API requests for audit purposes."""
    
    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs", 
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log audit information."""
        start_time = time.time()
        
        # Skip audit logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Extract request information
        method = request.method
        endpoint = str(request.url.path)
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")
        
        # Extract user information from request state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        username = getattr(request.state, "username", None)
        session_id = getattr(request.state, "session_id", None)
        api_key_id = getattr(request.state, "api_key_id", None)
        
        # Determine action based on HTTP method and endpoint
        action = self._determine_action(method, endpoint)
        resource_type = self._determine_resource_type(endpoint)
        
        # Determine compliance tags and risk level
        compliance_tags = self._determine_compliance_tags(endpoint, method)
        risk_level = self._determine_risk_level(endpoint, method)
        contains_pii = self._contains_pii(endpoint)
        data_classification = self._get_data_classification(endpoint)
        
        # Process the request
        response = None
        success = True
        error_code = None
        error_message = None
        
        try:
            response = await call_next(request)
            success = response.status_code < 400
            
            if not success:
                error_code = str(response.status_code)
                
        except Exception as e:
            success = False
            error_code = "500"
            error_message = str(e)
            logger.error("Request processing failed", error=str(e), endpoint=endpoint)
            raise
        
        finally:
            # Calculate request duration
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Extract resource ID from path if possible
            resource_id = self._extract_resource_id(endpoint)
            
            # Log the audit event asynchronously
            try:
                await audit_service.log_action(
                    action=action,
                    user_id=user_id,
                    username=username,
                    session_id=session_id,
                    api_key_id=api_key_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    endpoint=endpoint,
                    method=method,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    compliance_tags=compliance_tags,
                    risk_level=risk_level,
                    contains_pii=contains_pii,
                    data_classification=data_classification,
                    success=success,
                    error_code=error_code,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    event_metadata={
                        "query_params": dict(request.query_params),
                        "content_type": request.headers.get("content-type"),
                        "response_status": response.status_code if response else None
                    }
                )
            except Exception as audit_error:
                logger.error("Failed to log audit event", error=str(audit_error))
        
        return response
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request.client, "host"):
            return request.client.host
        
        return None    

    def _determine_action(self, method: str, endpoint: str) -> AuditAction:
        """Determine audit action based on HTTP method and endpoint."""
        # Authentication endpoints
        if "/auth/login" in endpoint:
            return AuditAction.LOGIN
        elif "/auth/logout" in endpoint:
            return AuditAction.LOGOUT
        elif "/auth/refresh" in endpoint:
            return AuditAction.LOGIN  # Token refresh is essentially re-authentication
        
        # Evidence endpoints
        elif "/evidence" in endpoint:
            if method == "GET":
                return AuditAction.EVIDENCE_ACCESS
            elif method == "DELETE":
                return AuditAction.EVIDENCE_DELETE
            elif method in ["POST", "PUT", "PATCH"]:
                return AuditAction.EVIDENCE_REDACT if "redact" in endpoint else AuditAction.UPDATE
        
        # User management endpoints
        elif "/users" in endpoint:
            if method == "POST":
                return AuditAction.USER_CREATE
            elif method in ["PUT", "PATCH"]:
                return AuditAction.USER_UPDATE
            elif method == "DELETE":
                return AuditAction.USER_DELETE
            else:
                return AuditAction.DATA_VIEW
        
        # Incident endpoints
        elif "/incidents" in endpoint:
            if method == "POST":
                return AuditAction.INCIDENT_CREATE
            elif method in ["PUT", "PATCH"]:
                return AuditAction.INCIDENT_UPDATE
            elif method == "GET":
                return AuditAction.DATA_VIEW
        
        # API key endpoints
        elif "/api-keys" in endpoint:
            if method == "POST":
                return AuditAction.API_KEY_CREATE
            elif method == "DELETE":
                return AuditAction.API_KEY_REVOKE
        
        # Generic actions based on HTTP method
        elif method == "POST":
            return AuditAction.CREATE
        elif method == "GET":
            return AuditAction.READ
        elif method in ["PUT", "PATCH"]:
            return AuditAction.UPDATE
        elif method == "DELETE":
            return AuditAction.DELETE
        
        return AuditAction.API_ACCESS
    
    def _determine_resource_type(self, endpoint: str) -> Optional[ResourceType]:
        """Determine resource type based on endpoint."""
        if "/users" in endpoint:
            return ResourceType.USER
        elif "/incidents" in endpoint:
            return ResourceType.INCIDENT
        elif "/events" in endpoint:
            return ResourceType.EVENT
        elif "/evidence" in endpoint:
            return ResourceType.EVIDENCE
        elif "/api-keys" in endpoint:
            return ResourceType.API_KEY
        elif "/audit" in endpoint:
            return ResourceType.AUDIT_LOG
        
        return None
    
    def _determine_compliance_tags(self, endpoint: str, method: str) -> List[ComplianceTag]:
        """Determine relevant compliance frameworks."""
        tags = []
        
        # All user data access requires GDPR compliance
        if "/users" in endpoint or "/evidence" in endpoint:
            tags.append(ComplianceTag.GDPR)
        
        # Educational institution data requires FERPA compliance
        if "/users" in endpoint or "/incidents" in endpoint:
            tags.append(ComplianceTag.FERPA)
        
        # Evidence and sensitive data may require additional compliance
        if "/evidence" in endpoint:
            tags.extend([ComplianceTag.GDPR, ComplianceTag.FERPA])
        
        return tags
    
    def _determine_risk_level(self, endpoint: str, method: str) -> str:
        """Determine risk level of the operation."""
        # High-risk operations
        if method == "DELETE":
            return "high"
        elif "/evidence" in endpoint and method in ["GET", "POST"]:
            return "high"  # Evidence access is always high risk
        elif "/users" in endpoint and method in ["POST", "PUT", "PATCH", "DELETE"]:
            return "medium"  # User management is medium risk
        elif "/admin" in endpoint:
            return "high"  # Admin operations are high risk
        
        # Medium-risk operations
        elif method in ["POST", "PUT", "PATCH"]:
            return "medium"
        elif "/incidents" in endpoint:
            return "medium"  # Incident management is medium risk
        
        # Low-risk operations (mostly read operations)
        return "low"
    
    def _contains_pii(self, endpoint: str) -> bool:
        """Determine if the endpoint typically involves PII."""
        pii_endpoints = ["/users", "/evidence", "/incidents"]
        return any(pii_endpoint in endpoint for pii_endpoint in pii_endpoints)
    
    def _get_data_classification(self, endpoint: str) -> Optional[str]:
        """Determine data classification level."""
        if "/evidence" in endpoint:
            return "restricted"  # Evidence is highly sensitive
        elif "/users" in endpoint:
            return "confidential"  # User data is confidential
        elif "/incidents" in endpoint:
            return "internal"  # Incident data is internal
        elif "/events" in endpoint:
            return "internal"  # Event data is internal
        
        return "public"  # Default to public for other endpoints
    
    def _extract_resource_id(self, endpoint: str) -> Optional[str]:
        """Extract resource ID from endpoint path."""
        # Look for UUID patterns in the path
        import re
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        match = re.search(uuid_pattern, endpoint)
        if match:
            return match.group(0)
        
        # Look for numeric IDs
        numeric_pattern = r'/(\d+)(?:/|$)'
        match = re.search(numeric_pattern, endpoint)
        if match:
            return match.group(1)
        
        return None