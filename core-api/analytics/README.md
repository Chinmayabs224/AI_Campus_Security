# Analytics and Reporting System

## Overview

The Analytics and Reporting System provides comprehensive security analytics, compliance reporting, and predictive insights for the AI Campus Security platform. This system enables data-driven decision making through pattern analysis, trend detection, and automated compliance reporting.

## Features

### 🔍 Security Analytics Engine (Task 9.1)
- **Incident Pattern Analysis**: Identifies temporal, spatial, and categorical patterns in security incidents
- **Heat Map Generation**: Creates visual representations of security hotspots across campus locations
- **Performance Metrics Dashboard**: Real-time monitoring of system performance and detection accuracy
- **Predictive Analytics**: Resource planning insights based on historical data and trends

### 📋 Compliance and Audit Reporting (Task 9.2)
- **Automated Compliance Reports**: GDPR, FERPA, COPPA compliance reporting
- **Evidence Chain of Custody**: Complete audit trail for evidence handling
- **System Performance Reports**: Uptime, response times, and system health metrics
- **Custom Report Builder**: Flexible reporting for security analysis

## API Endpoints

### Analytics Endpoints
- `GET /api/v1/analytics/health` - Service health check
- `POST /api/v1/analytics/dashboard` - Comprehensive analytics dashboard
- `GET /api/v1/analytics/patterns` - Incident pattern analysis
- `GET /api/v1/analytics/heatmap` - Security incident heat map
- `GET /api/v1/analytics/trends` - Security trend analysis
- `GET /api/v1/analytics/performance` - System performance metrics
- `GET /api/v1/analytics/hotspots` - Security hotspot identification
- `GET /api/v1/analytics/predictions` - Predictive analytics insights

### Reporting Endpoints
- `POST /api/v1/analytics/reports/compliance` - Generate compliance reports
- `GET /api/v1/analytics/reports/chain-of-custody/{evidence_id}` - Evidence custody chain
- `GET /api/v1/analytics/reports/system-performance` - System performance reports

## Data Models

### Core Analytics Models
- `AnalyticsRequest` - Request parameters for analytics queries
- `AnalyticsResponse` - Comprehensive analytics response
- `IncidentPattern` - Pattern analysis results
- `HeatMapPoint` - Geographic incident data
- `TrendData` - Trend analysis with predictions
- `PerformanceMetrics` - System performance indicators
- `SecurityHotspot` - High-risk location identification
- `PredictiveInsight` - Future trend predictions

### Compliance Models
- `ComplianceReport` - Regulatory compliance reports
- `ChainOfCustody` - Evidence handling audit trail

## Database Schema

### Analytics Tables
- `locations` - Campus location data with coordinates
- `cameras` - Camera system status and performance
- `audit_findings` - Compliance audit results
- `data_subject_requests` - GDPR/FERPA data requests
- `privacy_violations` - Privacy incident tracking
- `compliance_reports` - Generated compliance reports
- `analytics_cache` - Performance optimization cache

## Usage Examples

### Generate Dashboard Analytics
```python
from analytics.models import AnalyticsRequest, TimeRange, MetricType

request = AnalyticsRequest(
    time_range=TimeRange.WEEK,
    location_ids=["building_a", "parking_lot"],
    metrics=[MetricType.INCIDENT_COUNT, MetricType.RESPONSE_TIME]
)

analytics_data = await analytics_service.generate_analytics(request)
```

### Create Compliance Report
```python
from datetime import date

report = await compliance_service.generate_compliance_report(
    report_type="gdpr",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31)
)
```

## Performance Features

### Caching
- Redis-based caching for analytics results (5-minute TTL)
- Database query optimization with strategic indexes
- Concurrent processing of analytics tasks

### Scalability
- Asynchronous processing for all analytics operations
- Efficient database queries with proper indexing
- Modular service architecture for easy scaling

## Security and Compliance

### Data Privacy
- Automatic PII redaction in reports
- Configurable data retention policies
- Audit logging for all data access

### Regulatory Compliance
- GDPR compliance reporting and data subject rights
- FERPA educational record protection
- COPPA child privacy compliance
- Automated evidence chain of custody

## Requirements Satisfied

This implementation satisfies the following requirements:

**Requirement 7.1**: Analytics dashboard with incident patterns, trends, and performance metrics
**Requirement 7.2**: Heat map visualization of security incidents across campus locations  
**Requirement 7.4**: Compliance reporting for GDPR, FERPA with automated report generation
**Requirement 4.2**: Evidence management with chain of custody and audit trails

## Installation and Setup

1. Ensure database migrations are applied:
   ```bash
   alembic upgrade head
   ```

2. Configure Redis for caching:
   ```bash
   # Redis should be running on default port 6379
   redis-server
   ```

3. The analytics system is automatically included in the main FastAPI application.

## Testing

The analytics system includes comprehensive test coverage for:
- Data model validation
- Service functionality
- API endpoint structure
- Compliance features

## Monitoring

The system provides built-in monitoring through:
- Health check endpoints
- Performance metrics tracking
- Error logging with structured logging
- Cache hit/miss statistics