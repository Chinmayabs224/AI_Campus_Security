"""
Monitoring and observability utilities for the campus security system.
"""
import time
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger()

# Initialize OpenTelemetry tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Jaeger exporter configuration
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=14268,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Custom metrics for security operations
SECURITY_METRICS = {
    'incident_detection_time': Histogram(
        'security_incident_detection_time_seconds',
        'Time taken to detect security incidents',
        ['detection_type']
    ),
    'model_inference_time': Histogram(
        'ai_model_inference_time_seconds',
        'Time taken for AI model inference',
        ['model_type']
    ),
    'evidence_processing_time': Histogram(
        'evidence_processing_time_seconds',
        'Time taken to process evidence',
        ['processing_type']
    ),
    'alert_delivery_time': Histogram(
        'alert_delivery_time_seconds',
        'Time taken to deliver alerts',
        ['delivery_method']
    ),
    'privacy_redaction_time': Histogram(
        'privacy_redaction_time_seconds',
        'Time taken for privacy redaction',
        ['redaction_type']
    )
}


class SecurityTracer:
    """Custom tracer for security operations."""
    
    def __init__(self):
        self.tracer = trace.get_tracer("campus-security")
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            start_time = time.time()
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
    
    def trace_incident_detection(self, detection_type: str):
        """Decorator for tracing incident detection operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with self.trace_operation(
                    f"incident_detection_{detection_type}",
                    {"detection_type": detection_type}
                ):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        SECURITY_METRICS['incident_detection_time'].labels(
                            detection_type=detection_type
                        ).observe(duration)
            return wrapper
        return decorator
    
    def trace_model_inference(self, model_type: str):
        """Decorator for tracing AI model inference."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with self.trace_operation(
                    f"model_inference_{model_type}",
                    {"model_type": model_type}
                ):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        SECURITY_METRICS['model_inference_time'].labels(
                            model_type=model_type
                        ).observe(duration)
            return wrapper
        return decorator
    
    def trace_evidence_processing(self, processing_type: str):
        """Decorator for tracing evidence processing."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with self.trace_operation(
                    f"evidence_processing_{processing_type}",
                    {"processing_type": processing_type}
                ):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        SECURITY_METRICS['evidence_processing_time'].labels(
                            processing_type=processing_type
                        ).observe(duration)
            return wrapper
        return decorator


# Global tracer instance
security_tracer = SecurityTracer()


def setup_instrumentation(app):
    """Set up automatic instrumentation for FastAPI and other libraries."""
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument database connections
    AsyncPGInstrumentor().instrument()
    
    # Instrument Redis
    RedisInstrumentor().instrument()
    
    logger.info("OpenTelemetry instrumentation configured")


class MetricsCollector:
    """Collector for custom security metrics."""
    
    @staticmethod
    def record_incident(incident_type: str, severity: str):
        """Record a security incident."""
        from main import SECURITY_INCIDENTS
        SECURITY_INCIDENTS.labels(
            incident_type=incident_type,
            severity=severity
        ).inc()
    
    @staticmethod
    def update_false_positive_rate(rate: float):
        """Update the false positive rate metric."""
        from main import FALSE_POSITIVE_RATE
        FALSE_POSITIVE_RATE.set(rate)
    
    @staticmethod
    def record_incident_processing_time(duration: float):
        """Record incident processing duration."""
        from main import INCIDENT_PROCESSING_DURATION
        INCIDENT_PROCESSING_DURATION.observe(duration)
    
    @staticmethod
    def record_evidence_storage_error():
        """Record an evidence storage error."""
        from main import EVIDENCE_STORAGE_ERRORS
        EVIDENCE_STORAGE_ERRORS.inc()
    
    @staticmethod
    def update_database_connections(count: int):
        """Update active database connections count."""
        from main import DATABASE_CONNECTIONS
        DATABASE_CONNECTIONS.set(count)


# Global metrics collector instance
metrics_collector = MetricsCollector()


class PerformanceMonitor:
    """Monitor performance of critical operations."""
    
    def __init__(self):
        self.operation_times = {}
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Monitor the performance of an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.operation_times[operation_name] = duration
            logger.info(
                "Operation completed",
                operation=operation_name,
                duration_seconds=duration
            )
    
    def get_operation_stats(self) -> Dict[str, float]:
        """Get statistics for monitored operations."""
        return self.operation_times.copy()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()