"""
Custom middleware for the campus security API.
"""
import time
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import structlog

from .config import settings
from .redis import redis_manager

logger = structlog.get_logger()


class LoggingMiddleware:
    """Middleware for request/response logging."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Generate request ID
        request_id = str(uuid4())
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        # Add request ID to scope
        scope["request_id"] = request_id
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Log response
                process_time = time.time() - start_time
                logger.info(
                    "Request completed",
                    request_id=request_id,
                    status_code=message["status"],
                    process_time=f"{process_time:.3f}s"
                )
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


class SecurityHeadersMiddleware:
    """Middleware for adding security headers."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                
                # Add security headers
                security_headers = {
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"x-xss-protection": b"1; mode=block",
                    b"strict-transport-security": b"max-age=31536000; includeSubDomains",
                    b"referrer-policy": b"strict-origin-when-cross-origin",
                    b"permissions-policy": b"geolocation=(), microphone=(), camera=()"
                }
                
                headers.update(security_headers)
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


class RateLimitMiddleware:
    """Middleware for rate limiting requests."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            await self.app(scope, receive, send)
            return
        
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        client_id = f"rate_limit:{client_ip}:{hash(user_agent)}"
        
        try:
            # Check rate limit
            current_requests = await redis_manager.incr(client_id)
            
            if current_requests == 1:
                # First request in window, set expiration
                await redis_manager.expire(client_id, settings.RATE_LIMIT_WINDOW)
            
            if current_requests > settings.RATE_LIMIT_REQUESTS:
                # Rate limit exceeded
                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    requests=current_requests,
                    limit=settings.RATE_LIMIT_REQUESTS
                )
                
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Limit: {settings.RATE_LIMIT_REQUESTS} per {settings.RATE_LIMIT_WINDOW} seconds"
                    }
                )
                await response(scope, receive, send)
                return
        
        except Exception as e:
            # If Redis is unavailable, allow the request
            logger.error("Rate limiting failed", error=str(e))
        
        await self.app(scope, receive, send)


class AuthenticationMiddleware:
    """Middleware for authentication (will be implemented with auth service)."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        # For now, just pass through
        # Authentication logic will be implemented in the auth service
        await self.app(scope, receive, send)