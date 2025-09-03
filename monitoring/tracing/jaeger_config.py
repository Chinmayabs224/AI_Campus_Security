"""
Jaeger distributed tracing configuration for campus security system.
"""
import os
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import structlog

logger = structlog.get_logger()

class JaegerConfig:
    """Configuration for Jaeger distributed tracing."""
    
    def __init__(self, service_name: str, service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.jaeger_host = os.getenv("JAEGER_HOST", "localhost")
        self.jaeger_port = int(os.getenv("JAEGER_PORT", "14268"))
        self.sampling_rate = float(os.getenv("TRACING_SAMPLING_RATE", "0.1"))
    
    def setup_tracing(self):
        """Set up distributed tracing with Jaeger."""
        
        # Create resource with service information
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": os.getenv("ENVIRONMENT", "development")
        })
        
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.jaeger_host,
            agent_port=self.jaeger_port,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        logger.info(
            "Jaeger tracing configured",
            service_name=self.service_name,
            jaeger_host=self.jaeger_host,
            jaeger_port=self.jaeger_port,
            sampling_rate=self.sampling_rate
        )
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application for tracing."""
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")
    
    def instrument_database(self):
        """Instrument database connections for tracing."""
        AsyncPGInstrumentor().instrument()
        logger.info("Database instrumentation enabled")
    
    def instrument_redis(self):
        """Instrument Redis connections for tracing."""
        RedisInstrumentor().instrument()
        logger.info("Redis instrumentation enabled")
    
    def instrument_http_requests(self):
        """Instrument HTTP requests for tracing."""
        RequestsInstrumentor().instrument()
        logger.info("HTTP requests instrumentation enabled")
    
    def get_tracer(self, name: str = None):
        """Get a tracer instance."""
        tracer_name = name or self.service_name
        return trace.get_tracer(tracer_name)


# Service-specific configurations
CORE_API_TRACER = JaegerConfig("campus-security-core-api")
EDGE_SERVICE_TRACER = JaegerConfig("campus-security-edge-service")
NOTIFICATION_SERVICE_TRACER = JaegerConfig("campus-security-notification-service")
PRIVACY_SERVICE_TRACER = JaegerConfig("campus-security-privacy-service")
COMPLIANCE_SERVICE_TRACER = JaegerConfig("campus-security-compliance-service")


def setup_service_tracing(service_name: str, app=None):
    """Set up tracing for a specific service."""
    
    tracer_configs = {
        "core-api": CORE_API_TRACER,
        "edge-service": EDGE_SERVICE_TRACER,
        "notification-service": NOTIFICATION_SERVICE_TRACER,
        "privacy-service": PRIVACY_SERVICE_TRACER,
        "compliance-service": COMPLIANCE_SERVICE_TRACER
    }
    
    if service_name not in tracer_configs:
        logger.warning(f"Unknown service name: {service_name}")
        return None
    
    tracer_config = tracer_configs[service_name]
    tracer_config.setup_tracing()
    
    # Instrument based on service type
    if service_name == "core-api" and app:
        tracer_config.instrument_fastapi(app)
        tracer_config.instrument_database()
        tracer_config.instrument_redis()
    
    tracer_config.instrument_http_requests()
    
    return tracer_config.get_tracer()


class SecurityOperationTracer:
    """Custom tracer for security-specific operations."""
    
    def __init__(self, service_name: str):
        self.tracer = setup_service_tracing(service_name)
    
    def trace_incident_detection(self, camera_id: str, detection_type: str):
        """Create a span for incident detection."""
        return self.tracer.start_span(
            "incident_detection",
            attributes={
                "camera.id": camera_id,
                "detection.type": detection_type,
                "operation.type": "security_detection"
            }
        )
    
    def trace_evidence_processing(self, incident_id: str, processing_type: str):
        """Create a span for evidence processing."""
        return self.tracer.start_span(
            "evidence_processing",
            attributes={
                "incident.id": incident_id,
                "processing.type": processing_type,
                "operation.type": "evidence_handling"
            }
        )
    
    def trace_alert_delivery(self, incident_id: str, delivery_method: str, recipient_count: int):
        """Create a span for alert delivery."""
        return self.tracer.start_span(
            "alert_delivery",
            attributes={
                "incident.id": incident_id,
                "delivery.method": delivery_method,
                "recipient.count": recipient_count,
                "operation.type": "notification"
            }
        )
    
    def trace_privacy_redaction(self, evidence_id: str, redaction_type: str):
        """Create a span for privacy redaction."""
        return self.tracer.start_span(
            "privacy_redaction",
            attributes={
                "evidence.id": evidence_id,
                "redaction.type": redaction_type,
                "operation.type": "privacy_protection"
            }
        )
    
    def trace_compliance_check(self, check_type: str, resource_id: str):
        """Create a span for compliance checking."""
        return self.tracer.start_span(
            "compliance_check",
            attributes={
                "check.type": check_type,
                "resource.id": resource_id,
                "operation.type": "compliance_validation"
            }
        )