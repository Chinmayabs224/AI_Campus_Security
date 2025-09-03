"""
Main FastAPI application for campus security system.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import structlog
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from core.config import settings
from core.database import database_manager
from core.redis import redis_manager
from core.middleware import (
    LoggingMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware
)
from security.middleware import enhanced_security_middleware
from security.init import security_manager
from auth.router import router as auth_router
from events.router import router as events_router
from incidents.router import router as incidents_router
from evidence.router import router as evidence_router
from analytics.router import router as analytics_router
from audit.router import router as audit_router
from audit.middleware import AuditMiddleware
from audit.tasks import audit_task_service
from security.compliance_router import router as compliance_router

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

SECURITY_INCIDENTS = Counter(
    'security_incidents_total',
    'Total security incidents detected',
    ['incident_type', 'severity']
)

FALSE_POSITIVE_RATE = Gauge(
    'security_false_positive_rate',
    'Current false positive rate for security detection'
)

INCIDENT_PROCESSING_DURATION = Histogram(
    'security_incident_processing_duration_seconds',
    'Time taken to process security incidents'
)

EVIDENCE_STORAGE_ERRORS = Counter(
    'evidence_storage_errors_total',
    'Total evidence storage errors'
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections'
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting campus security API server")
    
    # Initialize database connection pool
    await database_manager.connect()
    logger.info("Database connection pool initialized")
    
    # Initialize Redis connection
    await redis_manager.connect()
    logger.info("Redis connection initialized")
    
    # Start audit background tasks
    await audit_task_service.start_periodic_tasks()
    logger.info("Audit background tasks started")
    
    # Initialize security manager
    await security_manager.initialize()
    logger.info("Security manager initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down campus security API server")
    
    # Close database connections
    await database_manager.disconnect()
    logger.info("Database connections closed")
    
    # Stop audit background tasks
    await audit_task_service.stop_periodic_tasks()
    logger.info("Audit background tasks stopped")
    
    # Shutdown security manager
    await security_manager.shutdown()
    logger.info("Security manager shutdown")
    
    # Close Redis connection
    await redis_manager.disconnect()
    logger.info("Redis connection closed")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Campus Security API",
        description="AI-Powered Intelligent Security System for Campus Environments",
        version="1.0.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan
    )
    
    # Add enhanced security middleware
    app.add_middleware(enhanced_security_middleware)
    
    # Add basic security headers middleware (as fallback)
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add trusted host middleware
    if settings.ALLOWED_HOSTS:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    
    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)
    
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Add audit middleware
    app.add_middleware(AuditMiddleware)
    
    # Add Prometheus metrics middleware
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        """Middleware to collect Prometheus metrics."""
        start_time = time.time()
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            return response
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    # Include routers
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(events_router, prefix="/api/v1/events", tags=["Events"])
    app.include_router(incidents_router, prefix="/api/v1/incidents", tags=["Incidents"])
    app.include_router(evidence_router, prefix="/api/v1/evidence", tags=["Evidence"])
    app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])
    app.include_router(audit_router, prefix="/api/v1/audit", tags=["Audit"])
    app.include_router(compliance_router, prefix="/api/v1/compliance", tags=["Compliance"])
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        try:
            # Check database connection
            db_status = await database_manager.health_check()
            
            # Check Redis connection
            redis_status = await redis_manager.health_check()
            
            return {
                "status": "healthy",
                "database": "connected" if db_status else "disconnected",
                "redis": "connected" if redis_status else "disconnected",
                "version": "1.0.0"
            }
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e)
                }
            )
    
    # Prometheus metrics endpoint
    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    # Security status endpoint
    @app.get("/security/status", tags=["Security"])
    async def security_status():
        """Security status endpoint."""
        return {
            "security_manager": security_manager.get_security_status(),
            "middleware_stats": app.middleware_stack[0].cls.get_security_stats() if hasattr(app.middleware_stack[0].cls, 'get_security_stats') else {},
            "timestamp": time.time()
        }
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Campus Security API",
            "version": "1.0.0",
            "docs": "/docs" if settings.ENVIRONMENT != "production" else "disabled",
            "health": "/health",
            "metrics": "/metrics"
        }
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )