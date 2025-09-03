"""
Enhanced security middleware integrating all security controls.
"""
import time
import json
from typing import Optional, Dict, Any
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import structlog

from .network_policies import network_policies
from .incident_response import incident_response, IncidentType, IncidentSeverity

logger = structlog.get_logger()


class EnhancedSecurityMiddleware:
    """Enhanced security middleware with comprehensive security controls."""
    
    def __init__(self, app):
        self.app = app
        self.blocked_ips: set = set()
        self.suspicious_activity: Dict[str, Dict] = {}
        self.security_events: list = []
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Security checks
        security_check_result = await self._perform_security_checks(request, client_ip, user_agent)
        
        if not security_check_result["allowed"]:
            # Block the request
            response = JSONResponse(
                status_code=security_check_result["status_code"],
                content={
                    "error": "Access denied",
                    "message": security_check_result["message"],
                    "incident_id": security_check_result.get("incident_id")
                }
            )
            await response(scope, receive, send)
            return
        
        # Add security headers and continue
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                
                # Add comprehensive security headers
                security_headers = {
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"x-xss-protection": b"1; mode=block",
                    b"strict-transport-security": b"max-age=31536000; includeSubDomains; preload",
                    b"referrer-policy": b"strict-origin-when-cross-origin",
                    b"permissions-policy": b"geolocation=(), microphone=(), camera=(), payment=(), usb=()",
                    b"content-security-policy": b"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
                    b"x-permitted-cross-domain-policies": b"none",
                    b"x-download-options": b"noopen",
                    b"x-dns-prefetch-control": b"off"
                }
                
                headers.update(security_headers)
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address considering proxies."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    async def _perform_security_checks(self, request: Request, client_ip: str, user_agent: str) -> Dict[str, Any]:
        """Perform comprehensive security checks."""
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            await self._log_security_event("blocked_ip_access", {
                "ip": client_ip,
                "user_agent": user_agent,
                "path": request.url.path
            })
            return {
                "allowed": False,
                "status_code": 403,
                "message": "IP address is blocked"
            }
        
        # Check network policies
        if not network_policies.is_ip_allowed(client_ip):
            await self._log_security_event("network_policy_violation", {
                "ip": client_ip,
                "user_agent": user_agent,
                "path": request.url.path
            })
            return {
                "allowed": False,
                "status_code": 403,
                "message": "Access denied by network policy"
            }
        
        # Check for suspicious activity
        suspicious_result = await self._check_suspicious_activity(request, client_ip, user_agent)
        if not suspicious_result["allowed"]:
            return suspicious_result
        
        # Check for malicious patterns
        malicious_result = await self._check_malicious_patterns(request, client_ip)
        if not malicious_result["allowed"]:
            return malicious_result
        
        # All checks passed
        return {"allowed": True}
    
    async def _check_suspicious_activity(self, request: Request, client_ip: str, user_agent: str) -> Dict[str, Any]:
        """Check for suspicious activity patterns."""
        current_time = time.time()
        
        # Initialize tracking for new IPs
        if client_ip not in self.suspicious_activity:
            self.suspicious_activity[client_ip] = {
                "request_count": 0,
                "last_request": current_time,
                "failed_auth_attempts": 0,
                "suspicious_paths": [],
                "user_agents": set()
            }
        
        activity = self.suspicious_activity[client_ip]
        
        # Update activity tracking
        activity["request_count"] += 1
        activity["last_request"] = current_time
        activity["user_agents"].add(user_agent)
        
        # Check for rapid requests (potential DoS)
        time_window = 60  # 1 minute
        if activity["request_count"] > 100 and (current_time - activity["last_request"]) < time_window:
            await self._create_security_incident(
                "Potential DoS attack detected",
                f"IP {client_ip} made {activity['request_count']} requests in {time_window} seconds",
                IncidentType.DENIAL_OF_SERVICE,
                IncidentSeverity.HIGH,
                {"ip": client_ip, "request_count": activity["request_count"]}
            )
            
            # Temporarily block IP
            self.blocked_ips.add(client_ip)
            
            return {
                "allowed": False,
                "status_code": 429,
                "message": "Too many requests - potential DoS attack"
            }
        
        # Check for multiple user agents (potential bot)
        if len(activity["user_agents"]) > 5:
            await self._log_security_event("multiple_user_agents", {
                "ip": client_ip,
                "user_agents": list(activity["user_agents"]),
                "count": len(activity["user_agents"])
            })
        
        # Check for suspicious paths
        suspicious_paths = [
            "/admin", "/.env", "/config", "/backup", "/wp-admin",
            "/phpmyadmin", "/mysql", "/database", "/.git", "/api/v1/admin"
        ]
        
        if any(suspicious_path in request.url.path for suspicious_path in suspicious_paths):
            activity["suspicious_paths"].append(request.url.path)
            
            if len(activity["suspicious_paths"]) > 3:
                await self._create_security_incident(
                    "Suspicious path enumeration detected",
                    f"IP {client_ip} accessed multiple suspicious paths: {activity['suspicious_paths']}",
                    IncidentType.UNAUTHORIZED_ACCESS,
                    IncidentSeverity.MEDIUM,
                    {"ip": client_ip, "paths": activity["suspicious_paths"]}
                )
        
        return {"allowed": True}
    
    async def _check_malicious_patterns(self, request: Request, client_ip: str) -> Dict[str, Any]:
        """Check for malicious request patterns."""
        # Check request headers for injection attempts
        malicious_headers = ["<script", "javascript:", "data:text/html", "vbscript:"]
        
        for header_name, header_value in request.headers.items():
            if any(pattern in header_value.lower() for pattern in malicious_headers):
                await self._create_security_incident(
                    "Malicious header injection attempt",
                    f"IP {client_ip} sent malicious header: {header_name}={header_value}",
                    IncidentType.VULNERABILITY_EXPLOIT,
                    IncidentSeverity.HIGH,
                    {"ip": client_ip, "header": header_name, "value": header_value}
                )
                
                return {
                    "allowed": False,
                    "status_code": 400,
                    "message": "Malicious request detected"
                }
        
        # Check URL for SQL injection patterns
        sql_injection_patterns = [
            "union select", "drop table", "insert into", "delete from",
            "' or '1'='1", "' or 1=1", "admin'--", "' union select"
        ]
        
        url_path = request.url.path.lower()
        query_string = str(request.url.query).lower()
        
        for pattern in sql_injection_patterns:
            if pattern in url_path or pattern in query_string:
                await self._create_security_incident(
                    "SQL injection attempt detected",
                    f"IP {client_ip} attempted SQL injection: {request.url}",
                    IncidentType.VULNERABILITY_EXPLOIT,
                    IncidentSeverity.HIGH,
                    {"ip": client_ip, "url": str(request.url), "pattern": pattern}
                )
                
                return {
                    "allowed": False,
                    "status_code": 400,
                    "message": "Malicious request detected"
                }
        
        # Check for XSS patterns
        xss_patterns = [
            "<script", "javascript:", "onload=", "onerror=", "onclick=",
            "alert(", "document.cookie", "window.location"
        ]
        
        for pattern in xss_patterns:
            if pattern in url_path or pattern in query_string:
                await self._create_security_incident(
                    "XSS attempt detected",
                    f"IP {client_ip} attempted XSS: {request.url}",
                    IncidentType.VULNERABILITY_EXPLOIT,
                    IncidentSeverity.MEDIUM,
                    {"ip": client_ip, "url": str(request.url), "pattern": pattern}
                )
                
                return {
                    "allowed": False,
                    "status_code": 400,
                    "message": "Malicious request detected"
                }
        
        return {"allowed": True}
    
    async def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        logger.warning("Security event logged", event_type=event_type, details=details)
    
    async def _create_security_incident(self, title: str, description: str, 
                                      incident_type: IncidentType, severity: IncidentSeverity,
                                      indicators: Dict[str, Any]) -> str:
        """Create security incident."""
        try:
            incident = await incident_response.create_incident(
                title=title,
                description=description,
                incident_type=incident_type,
                severity=severity,
                reported_by="security_middleware",
                indicators_of_compromise=[json.dumps(indicators)]
            )
            
            logger.error("Security incident created",
                        incident_id=incident.id,
                        title=title,
                        severity=severity.value)
            
            return incident.id
            
        except Exception as e:
            logger.error("Failed to create security incident", error=str(e))
            return None
    
    def block_ip(self, ip_address: str, reason: str = "Manual block"):
        """Manually block an IP address."""
        self.blocked_ips.add(ip_address)
        logger.warning("IP address blocked", ip=ip_address, reason=reason)
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(ip_address)
        logger.info("IP address unblocked", ip=ip_address)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            "blocked_ips": len(self.blocked_ips),
            "tracked_ips": len(self.suspicious_activity),
            "security_events": len(self.security_events),
            "recent_events": self.security_events[-10:] if self.security_events else []
        }


# Global enhanced security middleware instance
enhanced_security_middleware = EnhancedSecurityMiddleware