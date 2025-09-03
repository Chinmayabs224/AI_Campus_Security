"""
Prometheus metrics for Flask microservices.
"""
import time
import functools
from flask import Flask, request, g
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

logger = structlog.get_logger()

# Flask microservice metrics
FLASK_REQUEST_COUNT = Counter(
    'flask_http_requests_total',
    'Total Flask HTTP requests',
    ['method', 'endpoint', 'status_code', 'service']
)

FLASK_REQUEST_DURATION = Histogram(
    'flask_http_request_duration_seconds',
    'Flask HTTP request duration in seconds',
    ['method', 'endpoint', 'service']
)

NOTIFICATION_DELIVERY_COUNT = Counter(
    'notification_delivery_total',
    'Total notifications delivered',
    ['delivery_method', 'status']
)

NOTIFICATION_DELIVERY_DURATION = Histogram(
    'notification_delivery_duration_seconds',
    'Time taken to deliver notifications',
    ['delivery_method']
)

PRIVACY_REDACTION_COUNT = Counter(
    'privacy_redaction_total',
    'Total privacy redactions performed',
    ['redaction_type', 'status']
)

PRIVACY_REDACTION_DURATION = Histogram(
    'privacy_redaction_duration_seconds',
    'Time taken for privacy redaction',
    ['redaction_type']
)

COMPLIANCE_CHECKS = Counter(
    'compliance_checks_total',
    'Total compliance checks performed',
    ['check_type', 'result']
)

COMPLIANCE_VIOLATIONS = Counter(
    'compliance_violations_total',
    'Total compliance violations detected',
    ['violation_type', 'severity']
)


class FlaskMetricsCollector:
    """Metrics collector for Flask microservices."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
    
    def init_app(self, app: Flask):
        """Initialize metrics collection for Flask app."""
        
        @app.before_request
        def before_request():
            g.start_time = time.time()
        
        @app.after_request
        def after_request(response):
            if hasattr(g, 'start_time'):
                duration = time.time() - g.start_time
                
                FLASK_REQUEST_DURATION.labels(
                    method=request.method,
                    endpoint=request.endpoint or 'unknown',
                    service=self.service_name
                ).observe(duration)
                
                FLASK_REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.endpoint or 'unknown',
                    status_code=response.status_code,
                    service=self.service_name
                ).inc()
            
            return response
        
        @app.route('/metrics')
        def metrics():
            """Prometheus metrics endpoint."""
            return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
        
        logger.info(f"Metrics collection initialized for {self.service_name}")
    
    def record_notification_delivery(self, delivery_method: str, status: str, duration: float):
        """Record notification delivery metrics."""
        NOTIFICATION_DELIVERY_COUNT.labels(
            delivery_method=delivery_method,
            status=status
        ).inc()
        
        NOTIFICATION_DELIVERY_DURATION.labels(
            delivery_method=delivery_method
        ).observe(duration)
    
    def record_privacy_redaction(self, redaction_type: str, status: str, duration: float):
        """Record privacy redaction metrics."""
        PRIVACY_REDACTION_COUNT.labels(
            redaction_type=redaction_type,
            status=status
        ).inc()
        
        PRIVACY_REDACTION_DURATION.labels(
            redaction_type=redaction_type
        ).observe(duration)
    
    def record_compliance_check(self, check_type: str, result: str):
        """Record compliance check metrics."""
        COMPLIANCE_CHECKS.labels(
            check_type=check_type,
            result=result
        ).inc()
    
    def record_compliance_violation(self, violation_type: str, severity: str):
        """Record compliance violation metrics."""
        COMPLIANCE_VIOLATIONS.labels(
            violation_type=violation_type,
            severity=severity
        ).inc()


def monitor_operation(operation_name: str, metric_labels: dict = None):
    """Decorator to monitor operation performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(
                    "Operation failed",
                    operation=operation_name,
                    error=str(e),
                    **metric_labels or {}
                )
                raise
            finally:
                duration = time.time() - start_time
                logger.info(
                    "Operation completed",
                    operation=operation_name,
                    duration_seconds=duration,
                    **metric_labels or {}
                )
        return wrapper
    return decorator