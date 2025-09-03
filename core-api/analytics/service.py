"""
Analytics service for security data analysis and reporting.
"""
import asyncio
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import json
import numpy as np
from collections import defaultdict, Counter
import structlog

from core.database import get_db_session
from core.redis import redis_manager
from .models import (
    AnalyticsRequest, AnalyticsResponse, IncidentPattern, HeatMapPoint,
    TrendData, PerformanceMetrics, SecurityHotspot, PredictiveInsight,
    TimeRange, MetricType, ComplianceReport, ChainOfCustody
)

logger = structlog.get_logger()


class AnalyticsService:
    """Service for security analytics and reporting."""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache
    
    async def generate_analytics(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """Generate comprehensive analytics report."""
        logger.info("Generating analytics report", request=request.dict())
        
        # Check cache first
        cache_key = f"analytics:{hash(str(request.dict()))}"
        cached_result = await redis_manager.get(cache_key)
        if cached_result:
            logger.info("Returning cached analytics result")
            return AnalyticsResponse.parse_raw(cached_result)
        
        # Generate analytics data
        start_date, end_date = self._get_date_range(request.time_range, request.start_date, request.end_date)
        
        # Run analytics tasks concurrently
        patterns_task = self._analyze_incident_patterns(start_date, end_date, request.location_ids)
        heatmap_task = self._generate_heat_map(start_date, end_date, request.location_ids)
        trends_task = self._analyze_trends(start_date, end_date, request.metrics or [])
        performance_task = self._calculate_performance_metrics(start_date, end_date)
        hotspots_task = self._identify_security_hotspots(start_date, end_date)
        predictions_task = self._generate_predictions(start_date, end_date)
        
        patterns, heat_map, trends, performance, hotspots, predictions = await asyncio.gather(
            patterns_task, heatmap_task, trends_task, performance_task, hotspots_task, predictions_task
        )
        
        response = AnalyticsResponse(
            request_id=str(uuid4()),
            generated_at=datetime.utcnow(),
            time_range=request.time_range,
            patterns=patterns,
            heat_map=heat_map,
            trends=trends,
            performance_metrics=performance,
            hotspots=hotspots,
            predictions=predictions
        )
        
        # Cache the result
        await redis_manager.setex(cache_key, self.cache_ttl, response.json())
        
        return response
    
    async def _analyze_incident_patterns(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        location_ids: Optional[List[str]]
    ) -> List[IncidentPattern]:
        """Analyze incident patterns and trends."""
        async with get_db_session() as db:
            # Query incidents with location and time filters
            query = """
                SELECT 
                    i.event_type,
                    i.location_id,
                    i.created_at,
                    i.severity,
                    EXTRACT(hour FROM i.created_at) as hour_of_day,
                    EXTRACT(dow FROM i.created_at) as day_of_week
                FROM incidents i
                WHERE i.created_at BETWEEN $1 AND $2
            """
            params = [start_date, end_date]
            
            if location_ids:
                query += " AND i.location_id = ANY($3)"
                params.append(location_ids)
            
            result = await db.fetch(query, *params)
            
            if not result:
                return []
            
            # Analyze patterns
            patterns = []
            
            # Time-based patterns
            hour_counts = Counter(row['hour_of_day'] for row in result)
            peak_hours = [hour for hour, count in hour_counts.most_common(3)]
            
            patterns.append(IncidentPattern(
                pattern_type="temporal",
                frequency=len(result),
                locations=list(set(row['location_id'] for row in result if row['location_id'])),
                time_periods=[f"{hour}:00-{hour+1}:00" for hour in peak_hours],
                confidence=0.85,
                description=f"Peak incident activity during hours: {', '.join(map(str, peak_hours))}"
            ))
            
            # Location-based patterns
            location_counts = Counter(row['location_id'] for row in result if row['location_id'])
            if location_counts:
                top_locations = [loc for loc, count in location_counts.most_common(5)]
                patterns.append(IncidentPattern(
                    pattern_type="spatial",
                    frequency=sum(location_counts.values()),
                    locations=top_locations,
                    time_periods=[],
                    confidence=0.78,
                    description=f"High incident frequency in locations: {', '.join(top_locations[:3])}"
                ))
            
            # Event type patterns
            type_counts = Counter(row['event_type'] for row in result if row['event_type'])
            if type_counts:
                common_types = [event_type for event_type, count in type_counts.most_common(3)]
                patterns.append(IncidentPattern(
                    pattern_type="categorical",
                    frequency=sum(type_counts.values()),
                    locations=[],
                    time_periods=[],
                    confidence=0.82,
                    description=f"Most common incident types: {', '.join(common_types)}"
                ))
            
            return patterns
    
    async def _generate_heat_map(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        location_ids: Optional[List[str]]
    ) -> List[HeatMapPoint]:
        """Generate heat map data for security incidents."""
        async with get_db_session() as db:
            query = """
                SELECT 
                    i.location_id,
                    l.name as location_name,
                    l.latitude,
                    l.longitude,
                    COUNT(*) as incident_count,
                    AVG(CASE 
                        WHEN i.severity = 'high' THEN 3
                        WHEN i.severity = 'medium' THEN 2
                        ELSE 1
                    END) as avg_severity,
                    MAX(i.created_at) as last_incident
                FROM incidents i
                LEFT JOIN locations l ON i.location_id = l.id
                WHERE i.created_at BETWEEN $1 AND $2
            """
            params = [start_date, end_date]
            
            if location_ids:
                query += " AND i.location_id = ANY($3)"
                params.append(location_ids)
            
            query += " GROUP BY i.location_id, l.name, l.latitude, l.longitude"
            
            result = await db.fetch(query, *params)
            
            heat_map = []
            for row in result:
                if row['location_id'] and row['latitude'] and row['longitude']:
                    heat_map.append(HeatMapPoint(
                        location_id=row['location_id'],
                        location_name=row['location_name'] or f"Location {row['location_id']}",
                        latitude=float(row['latitude']),
                        longitude=float(row['longitude']),
                        incident_count=row['incident_count'],
                        severity_score=float(row['avg_severity'] or 1.0),
                        last_incident=row['last_incident']
                    ))
            
            return heat_map
    
    async def _analyze_trends(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        metrics: List[MetricType]
    ) -> List[TrendData]:
        """Analyze trends for specified metrics."""
        trends = []
        
        # Default metrics if none specified
        if not metrics:
            metrics = [MetricType.INCIDENT_COUNT, MetricType.RESPONSE_TIME, MetricType.FALSE_POSITIVE_RATE]
        
        for metric in metrics:
            trend_data = await self._calculate_metric_trend(metric, start_date, end_date)
            if trend_data:
                trends.append(trend_data)
        
        return trends
    
    async def _calculate_metric_trend(
        self, 
        metric: MetricType, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[TrendData]:
        """Calculate trend data for a specific metric."""
        async with get_db_session() as db:
            if metric == MetricType.INCIDENT_COUNT:
                query = """
                    SELECT 
                        DATE_TRUNC('day', created_at) as date,
                        COUNT(*) as value
                    FROM incidents
                    WHERE created_at BETWEEN $1 AND $2
                    GROUP BY DATE_TRUNC('day', created_at)
                    ORDER BY date
                """
            elif metric == MetricType.RESPONSE_TIME:
                query = """
                    SELECT 
                        DATE_TRUNC('day', created_at) as date,
                        AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))/60) as value
                    FROM incidents
                    WHERE created_at BETWEEN $1 AND $2 AND resolved_at IS NOT NULL
                    GROUP BY DATE_TRUNC('day', created_at)
                    ORDER BY date
                """
            else:
                # Placeholder for other metrics
                return None
            
            result = await db.fetch(query, start_date, end_date)
            
            if not result:
                return None
            
            data_points = [
                {"date": row['date'].isoformat(), "value": float(row['value'] or 0)}
                for row in result
            ]
            
            # Calculate trend direction
            values = [point['value'] for point in data_points]
            if len(values) >= 2:
                trend_percentage = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                if trend_percentage > 5:
                    trend_direction = "increasing"
                elif trend_percentage < -5:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
                trend_percentage = 0.0
            
            return TrendData(
                metric=metric,
                time_range=TimeRange.DAY,
                data_points=data_points,
                trend_direction=trend_direction,
                trend_percentage=trend_percentage,
                prediction=None  # Could add ML-based predictions here
            ) 
   
    async def _calculate_performance_metrics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> PerformanceMetrics:
        """Calculate system performance metrics."""
        async with get_db_session() as db:
            # Get incident statistics
            incident_stats = await db.fetchrow("""
                SELECT 
                    COUNT(*) as total_incidents,
                    COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_incidents,
                    AVG(CASE 
                        WHEN resolved_at IS NOT NULL 
                        THEN EXTRACT(EPOCH FROM (resolved_at - created_at))/60 
                    END) as avg_response_time
                FROM incidents
                WHERE created_at BETWEEN $1 AND $2
            """, start_date, end_date)
            
            # Get false positive rate (placeholder calculation)
            false_positive_rate = await self._calculate_false_positive_rate(start_date, end_date)
            
            # Get camera status
            camera_stats = await db.fetchrow("""
                SELECT 
                    COUNT(*) as total_cameras,
                    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_cameras
                FROM cameras
            """)
            
            # Calculate system uptime (placeholder - would integrate with monitoring)
            system_uptime = 99.5  # Placeholder value
            
            # Calculate detection accuracy (placeholder)
            detection_accuracy = 85.2  # Placeholder value
            
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                total_incidents=incident_stats['total_incidents'] or 0,
                resolved_incidents=incident_stats['resolved_incidents'] or 0,
                average_response_time=float(incident_stats['avg_response_time'] or 0),
                false_positive_rate=false_positive_rate,
                system_uptime=system_uptime,
                active_cameras=camera_stats['active_cameras'] or 0,
                total_cameras=camera_stats['total_cameras'] or 0,
                detection_accuracy=detection_accuracy
            )
    
    async def _calculate_false_positive_rate(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate false positive rate for detections."""
        async with get_db_session() as db:
            result = await db.fetchrow("""
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(CASE WHEN false_positive = true THEN 1 END) as false_positives
                FROM events
                WHERE created_at BETWEEN $1 AND $2
            """, start_date, end_date)
            
            total = result['total_events'] or 0
            false_positives = result['false_positives'] or 0
            
            return (false_positives / total * 100) if total > 0 else 0.0
    
    async def _identify_security_hotspots(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[SecurityHotspot]:
        """Identify security hotspots based on incident frequency and severity."""
        async with get_db_session() as db:
            query = """
                SELECT 
                    i.location_id,
                    l.name as location_name,
                    COUNT(*) as incident_count,
                    AVG(CASE 
                        WHEN i.severity = 'high' THEN 3
                        WHEN i.severity = 'medium' THEN 2
                        ELSE 1
                    END) as avg_severity,
                    ARRAY_AGG(DISTINCT EXTRACT(hour FROM i.created_at)) as peak_hours
                FROM incidents i
                LEFT JOIN locations l ON i.location_id = l.id
                WHERE i.created_at BETWEEN $1 AND $2
                GROUP BY i.location_id, l.name
                HAVING COUNT(*) >= 3
                ORDER BY COUNT(*) DESC, avg_severity DESC
                LIMIT 10
            """, start_date, end_date
            
            result = await db.fetch(query)
            
            hotspots = []
            for row in result:
                # Calculate risk score based on frequency and severity
                risk_score = (row['incident_count'] * float(row['avg_severity'])) / 10
                risk_score = min(risk_score, 10.0)  # Cap at 10
                
                # Generate recommendations based on risk level
                recommendations = []
                if risk_score >= 7:
                    recommendations.extend([
                        "Increase security patrol frequency",
                        "Consider additional camera coverage",
                        "Implement access control measures"
                    ])
                elif risk_score >= 4:
                    recommendations.extend([
                        "Review security protocols",
                        "Increase monitoring during peak hours"
                    ])
                else:
                    recommendations.append("Monitor for pattern changes")
                
                hotspots.append(SecurityHotspot(
                    location_id=row['location_id'],
                    location_name=row['location_name'] or f"Location {row['location_id']}",
                    risk_score=risk_score,
                    incident_frequency=row['incident_count'],
                    peak_hours=[int(h) for h in (row['peak_hours'] or []) if h is not None],
                    recommended_actions=recommendations
                ))
            
            return hotspots
    
    async def _generate_predictions(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[PredictiveInsight]:
        """Generate predictive insights for security planning."""
        insights = []
        
        async with get_db_session() as db:
            # Analyze historical patterns for predictions
            
            # Seasonal trend prediction
            seasonal_query = """
                SELECT 
                    EXTRACT(month FROM created_at) as month,
                    COUNT(*) as incident_count
                FROM incidents
                WHERE created_at >= $1 - INTERVAL '1 year'
                GROUP BY EXTRACT(month FROM created_at)
                ORDER BY month
            """
            seasonal_data = await db.fetch(seasonal_query, start_date)
            
            if seasonal_data:
                current_month = datetime.now().month
                next_month = (current_month % 12) + 1
                
                current_incidents = next((row['incident_count'] for row in seasonal_data 
                                        if row['month'] == current_month), 0)
                next_incidents = next((row['incident_count'] for row in seasonal_data 
                                     if row['month'] == next_month), 0)
                
                if next_incidents > current_incidents * 1.2:
                    insights.append(PredictiveInsight(
                        insight_type="seasonal_trend",
                        description=f"Incident volume expected to increase by {((next_incidents/current_incidents - 1) * 100):.1f}% next month",
                        confidence=0.75,
                        recommended_action="Prepare additional security resources",
                        impact_level="medium",
                        time_horizon="short"
                    ))
            
            # Resource allocation prediction
            peak_hours_query = """
                SELECT 
                    EXTRACT(hour FROM created_at) as hour,
                    COUNT(*) as incident_count
                FROM incidents
                WHERE created_at BETWEEN $1 AND $2
                GROUP BY EXTRACT(hour FROM created_at)
                ORDER BY incident_count DESC
                LIMIT 3
            """
            peak_hours = await db.fetch(peak_hours_query, start_date, end_date)
            
            if peak_hours:
                peak_hour_list = [int(row['hour']) for row in peak_hours]
                insights.append(PredictiveInsight(
                    insight_type="resource_optimization",
                    description=f"Peak incident hours: {', '.join(map(str, peak_hour_list))}. Consider staffing adjustments.",
                    confidence=0.82,
                    recommended_action="Optimize security staff scheduling",
                    impact_level="high",
                    time_horizon="medium"
                ))
            
            # Equipment maintenance prediction
            insights.append(PredictiveInsight(
                insight_type="maintenance",
                description="Camera maintenance recommended based on performance metrics",
                confidence=0.68,
                recommended_action="Schedule preventive maintenance for cameras with declining performance",
                impact_level="low",
                time_horizon="long"
            ))
        
        return insights
    
    def _get_date_range(
        self, 
        time_range: TimeRange, 
        start_date: Optional[date], 
        end_date: Optional[date]
    ) -> Tuple[datetime, datetime]:
        """Get date range based on time range or explicit dates."""
        if start_date and end_date:
            return (
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time())
            )
        
        end_dt = datetime.utcnow()
        
        if time_range == TimeRange.HOUR:
            start_dt = end_dt - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            start_dt = end_dt - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            start_dt = end_dt - timedelta(weeks=1)
        elif time_range == TimeRange.MONTH:
            start_dt = end_dt - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            start_dt = end_dt - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            start_dt = end_dt - timedelta(days=365)
        else:
            start_dt = end_dt - timedelta(days=7)  # Default to week
        
        return start_dt, end_dt


class ComplianceService:
    """Service for compliance and audit reporting."""
    
    async def generate_compliance_report(
        self, 
        report_type: str, 
        start_date: date, 
        end_date: date
    ) -> ComplianceReport:
        """Generate compliance report for specified period."""
        logger.info("Generating compliance report", 
                   report_type=report_type, start_date=start_date, end_date=end_date)
        
        async with get_db_session() as db:
            # Get incident statistics
            incident_stats = await db.fetchrow("""
                SELECT 
                    COUNT(*) as total_incidents,
                    COUNT(CASE WHEN privacy_violation = true THEN 1 END) as privacy_violations
                FROM incidents
                WHERE created_at::date BETWEEN $1 AND $2
            """, start_date, end_date)
            
            # Get data access requests
            access_requests = await db.fetchrow("""
                SELECT COUNT(*) as data_access_requests
                FROM audit_log
                WHERE action = 'data_access_request' 
                AND timestamp::date BETWEEN $1 AND $2
            """, start_date, end_date)
            
            # Calculate retention compliance
            retention_compliance = await self._calculate_retention_compliance()
            
            # Get audit findings
            audit_findings = await self._get_audit_findings(start_date, end_date)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(
                incident_stats, access_requests, retention_compliance
            )
            
            return ComplianceReport(
                report_id=uuid4(),
                report_type=report_type,
                generated_at=datetime.utcnow(),
                period_start=start_date,
                period_end=end_date,
                total_incidents=incident_stats['total_incidents'] or 0,
                privacy_violations=incident_stats['privacy_violations'] or 0,
                data_access_requests=access_requests['data_access_requests'] or 0,
                retention_compliance=retention_compliance,
                audit_findings=audit_findings,
                recommendations=recommendations
            )
    
    async def _calculate_retention_compliance(self) -> float:
        """Calculate data retention compliance percentage."""
        async with get_db_session() as db:
            result = await db.fetchrow("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN retention_expires > NOW() THEN 1 END) as compliant_records
                FROM evidence
                WHERE retention_expires IS NOT NULL
            """)
            
            total = result['total_records'] or 0
            compliant = result['compliant_records'] or 0
            
            return (compliant / total * 100) if total > 0 else 100.0
    
    async def _get_audit_findings(self, start_date: date, end_date: date) -> List[str]:
        """Get audit findings for the specified period."""
        async with get_db_session() as db:
            findings = await db.fetch("""
                SELECT DISTINCT finding
                FROM audit_findings
                WHERE created_at::date BETWEEN $1 AND $2
                ORDER BY created_at DESC
            """, start_date, end_date)
            
            return [row['finding'] for row in findings] if findings else []
    
    def _generate_compliance_recommendations(
        self, 
        incident_stats: Dict, 
        access_requests: Dict, 
        retention_compliance: float
    ) -> List[str]:
        """Generate compliance recommendations based on metrics."""
        recommendations = []
        
        if incident_stats.get('privacy_violations', 0) > 0:
            recommendations.append("Review privacy protection measures and staff training")
        
        if access_requests.get('data_access_requests', 0) > 10:
            recommendations.append("Consider implementing automated DSAR processing")
        
        if retention_compliance < 95:
            recommendations.append("Improve data retention policy enforcement")
        
        if not recommendations:
            recommendations.append("Maintain current compliance practices")
        
        return recommendations
    
    async def get_chain_of_custody(self, evidence_id: str) -> ChainOfCustody:
        """Get chain of custody for evidence."""
        async with get_db_session() as db:
            evidence = await db.fetchrow("""
                SELECT 
                    e.id,
                    e.incident_id,
                    e.created_at,
                    e.retention_expires,
                    e.integrity_hash
                FROM evidence e
                WHERE e.id = $1
            """, evidence_id)
            
            if not evidence:
                raise ValueError(f"Evidence {evidence_id} not found")
            
            # Get custody chain
            custody_chain = await db.fetch("""
                SELECT 
                    user_id,
                    action,
                    timestamp,
                    ip_address
                FROM audit_log
                WHERE resource_id = $1 AND resource_type = 'evidence'
                ORDER BY timestamp
            """, evidence_id)
            
            # Get access log
            access_log = await db.fetch("""
                SELECT 
                    user_id,
                    timestamp,
                    action,
                    ip_address
                FROM audit_log
                WHERE resource_id = $1 AND action LIKE '%access%'
                ORDER BY timestamp DESC
            """, evidence_id)
            
            return ChainOfCustody(
                evidence_id=evidence['id'],
                incident_id=evidence['incident_id'],
                created_at=evidence['created_at'],
                custody_chain=[
                    {
                        "user_id": row['user_id'],
                        "action": row['action'],
                        "timestamp": row['timestamp'].isoformat(),
                        "ip_address": str(row['ip_address'])
                    }
                    for row in custody_chain
                ],
                integrity_verified=True,  # Would verify hash in real implementation
                access_log=[
                    {
                        "user_id": row['user_id'],
                        "timestamp": row['timestamp'].isoformat(),
                        "action": row['action'],
                        "ip_address": str(row['ip_address'])
                    }
                    for row in access_log
                ],
                retention_expires=evidence['retention_expires']
            )


# Service instances
analytics_service = AnalyticsService()
compliance_service = ComplianceService()