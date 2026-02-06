"""
Data Sources API Router

Endpoints for managing external seismic data sources
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.usgs_service import usgs_service
from app.services.event_monitor import event_monitor

router = APIRouter()


class USGSQueryRequest(BaseModel):
    min_magnitude: float = 4.0
    hours_back: int = 24
    max_results: int = 100


class MonitorConfigRequest(BaseModel):
    check_interval_seconds: int = 300  # 5 minutes
    min_magnitude: float = 4.0


@router.get("/usgs/recent")
async def get_recent_earthquakes(
    min_magnitude: float = 4.0,
    hours_back: int = 24,
    max_results: int = 100
):
    """
    Fetch recent earthquakes from USGS API
    
    Query Parameters:
    - min_magnitude: Minimum earthquake magnitude (default: 4.0)
    - hours_back: How many hours back to search (default: 24)
    - max_results: Maximum number of results (default: 100)
    """
    try:
        events = usgs_service.fetch_recent_earthquakes(
            min_magnitude=min_magnitude,
            hours_back=hours_back,
            max_results=max_results
        )
        
        return {
            "status": "success",
            "count": len(events),
            "events": events
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/usgs/event/{event_id}")
async def get_event_details(event_id: str):
    """Get detailed information about a specific USGS event"""
    try:
        event = usgs_service.get_event_details(event_id)
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        return {
            "status": "success",
            "event": event
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor/start")
async def start_monitoring(config: MonitorConfigRequest):
    """
    Start background earthquake monitoring
    
    Body:
    - check_interval_seconds: How often to check for new events (default: 300)
    - min_magnitude: Minimum magnitude to report (default: 4.0)
    """
    try:
        await event_monitor.start(
            check_interval_seconds=config.check_interval_seconds,
            min_magnitude=config.min_magnitude
        )
        
        return {
            "status": "success",
            "message": "Event monitoring started",
            "config": {
                "check_interval_seconds": config.check_interval_seconds,
                "min_magnitude": config.min_magnitude
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor/stop")
async def stop_monitoring():
    """Stop background earthquake monitoring"""
    try:
        await event_monitor.stop()
        
        return {
            "status": "success",
            "message": "Event monitoring stopped"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/status")
async def get_monitor_status():
    """Get current status of earthquake monitoring"""
    try:
        status = event_monitor.get_status()
        
        return {
            "status": "success",
            "monitor": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor/check-now")
async def trigger_manual_check():
    """Manually trigger an immediate check for new earthquakes"""
    try:
        await event_monitor.check_now()
        
        return {
            "status": "success",
            "message": "Manual check completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
