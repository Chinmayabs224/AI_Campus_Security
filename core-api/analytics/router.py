"""
Analytics and reporting router.
"""
from datetime import date
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
import structlog

from core.database import get_db_session
from core.redis import redis_manager
try:
    from auth.dependencies import get_current_user, require_permissions
except ImportError:
    # Mock dependencies for testing
    def get_current_user():
        return {"user_id": "test_user", "roles": ["admin"]}
    
    def require_permissions(permissions):
        def dependency():
            return True
        return dependency
from .models import (
    AnalyticsRequest, AnalyticsResponse, TimeRange, MetricType,
    ComplianceReport, ChainOfCustody
)
from .service import analytics_service, compliance_service

logger = structlog.get_logger()

router = APIRouter()


@router.get("/health")
async def analytics_health():
    """Analytics service health check."""
    return {"service": "analytics", "status": "healthy"}


@router.post("/dashboard", response_model=AnalyticsResponse)
async def get_dashboard_analytics(
    request: AnalyticsRequest,
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["analytics:read"]))
):
    """Get comprehensive dashboard analytics data."""
    try:
        logger.info("Dashboard analytics requested", 
                   user_id=current_user.get("user_id"), 
                   request=request.dict())
        
        analytics_data = await analytics_service.generate_analytics(request)
        return analytics_data
        
    except Exception as e:
        logger.error("Failed to generate dashboard analytics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate analytics data"
        )


@router.get("/patterns")
async def get_incident_patterns(
    time_range: TimeRange = Query(TimeRange.WEEK),
    location_ids: Optional[List[str]] = Query(None),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["analytics:read"]))
):
    """Get incident pattern analysis."""
    try:
        request = AnalyticsRequest(
            time_range=time_range,
            location_ids=location_ids
        )
        analytics_data = await analytics_service.generate_analytics(request)
        return {"patterns": analytics_data.patterns}
        
    except Exception as e:
        logger.error("Failed to get incident patterns", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve incident patterns"
        )


@router.get("/heatmap")
async def get_security_heatmap(
    time_range: TimeRange = Query(TimeRange.WEEK),
    location_ids: Optional[List[str]] = Query(None),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["analytics:read"]))
):
    """Get security incident heat map data."""
    try:
        request = AnalyticsRequest(
            time_range=time_range,
            location_ids=location_ids
        )
        analytics_data = await analytics_service.generate_analytics(request)
        return {"heat_map": analytics_data.heat_map}
        
    except Exception as e:
        logger.error("Failed to generate heat map", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate heat map data"
        )


@router.get("/trends")
async def get_security_trends(
    time_range: TimeRange = Query(TimeRange.MONTH),
    metrics: Optional[List[MetricType]] = Query(None),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["analytics:read"]))
):
    """Get security trend analysis."""
    try:
        request = AnalyticsRequest(
            time_range=time_range,
            metrics=metrics
        )
        analytics_data = await analytics_service.generate_analytics(request)
        return {"trends": analytics_data.trends}
        
    except Exception as e:
        logger.error("Failed to get security trends", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security trends"
        )


@router.get("/performance")
async def get_performance_metrics(
    time_range: TimeRange = Query(TimeRange.DAY),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["analytics:read"]))
):
    """Get system performance metrics."""
    try:
        request = AnalyticsRequest(time_range=time_range)
        analytics_data = await analytics_service.generate_analytics(request)
        return {"performance_metrics": analytics_data.performance_metrics}
        
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get("/hotspots")
async def get_security_hotspots(
    time_range: TimeRange = Query(TimeRange.MONTH),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["analytics:read"]))
):
    """Get security hotspot analysis."""
    try:
        request = AnalyticsRequest(time_range=time_range)
        analytics_data = await analytics_service.generate_analytics(request)
        return {"hotspots": analytics_data.hotspots}
        
    except Exception as e:
        logger.error("Failed to get security hotspots", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security hotspots"
        )


@router.get("/predictions")
async def get_predictive_insights(
    time_range: TimeRange = Query(TimeRange.QUARTER),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["analytics:read"]))
):
    """Get predictive analytics insights."""
    try:
        request = AnalyticsRequest(time_range=time_range)
        analytics_data = await analytics_service.generate_analytics(request)
        return {"predictions": analytics_data.predictions}
        
    except Exception as e:
        logger.error("Failed to get predictive insights", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve predictive insights"
        )


@router.post("/reports/compliance", response_model=ComplianceReport)
async def generate_compliance_report(
    report_type: str,
    start_date: date,
    end_date: date,
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["compliance:read"]))
):
    """Generate compliance report for specified period."""
    try:
        logger.info("Compliance report requested", 
                   user_id=current_user.get("user_id"),
                   report_type=report_type,
                   start_date=start_date,
                   end_date=end_date)
        
        report = await compliance_service.generate_compliance_report(
            report_type, start_date, end_date
        )
        return report
        
    except Exception as e:
        logger.error("Failed to generate compliance report", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate compliance report"
        )


@router.get("/reports/chain-of-custody/{evidence_id}", response_model=ChainOfCustody)
async def get_chain_of_custody(
    evidence_id: str,
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["evidence:read", "compliance:read"]))
):
    """Get chain of custody for evidence."""
    try:
        logger.info("Chain of custody requested", 
                   user_id=current_user.get("user_id"),
                   evidence_id=evidence_id)
        
        custody = await compliance_service.get_chain_of_custody(evidence_id)
        return custody
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to get chain of custody", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chain of custody"
        )


@router.get("/reports/system-performance")
async def get_system_performance_report(
    start_date: date = Query(...),
    end_date: date = Query(...),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions(["analytics:read"]))
):
    """Get system performance and uptime report."""
    try:
        # Generate performance report for the specified period
        request = AnalyticsRequest(
            time_range=TimeRange.DAY,
            start_date=start_date,
            end_date=end_date
        )
        
        analytics_data = await analytics_service.generate_analytics(request)
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "performance_metrics": analytics_data.performance_metrics,
            "trends": analytics_data.trends,
            "generated_at": analytics_data.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to generate system performance report", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate system performance report"
        )