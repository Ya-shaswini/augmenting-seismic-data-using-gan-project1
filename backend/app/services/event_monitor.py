"""
Background Event Monitor

Continuously monitors earthquake data sources and broadcasts
new events to connected WebSocket clients.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from app.services.usgs_service import usgs_service
from app.websockets.connection_manager import manager

logger = logging.getLogger(__name__)


class EventMonitor:
    """Background service for monitoring earthquake events"""
    
    def __init__(self):
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        self.check_interval_seconds = 300  # 5 minutes default
        self.min_magnitude = 4.0
        
    async def start(
        self,
        check_interval_seconds: int = 300,
        min_magnitude: float = 4.0
    ):
        """
        Start the background monitoring task
        
        Args:
            check_interval_seconds: How often to check for new events (default: 5 minutes)
            min_magnitude: Minimum earthquake magnitude to report (default: 4.0)
        """
        if self.is_running:
            logger.warning("Event monitor is already running")
            return
        
        self.check_interval_seconds = check_interval_seconds
        self.min_magnitude = min_magnitude
        self.is_running = True
        
        # Start background task
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Event monitor started (interval: {check_interval_seconds}s, "
            f"min magnitude: {min_magnitude})"
        )
    
    async def stop(self):
        """Stop the background monitoring task"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop - runs continuously in background"""
        logger.info("Event monitoring loop started")
        
        # Initial check immediately
        await self._check_for_events()
        
        while self.is_running:
            try:
                # Wait for next check interval
                await asyncio.sleep(self.check_interval_seconds)
                
                # Check for new events
                await self._check_for_events()
                
            except asyncio.CancelledError:
                logger.info("Event monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event monitor loop: {e}", exc_info=True)
                # Continue running even if there's an error
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _check_for_events(self):
        """Check for new earthquake events and broadcast them"""
        try:
            logger.info("Checking for new earthquake events...")
            
            # Fetch new events from USGS
            new_events = usgs_service.fetch_new_earthquakes(
                min_magnitude=self.min_magnitude,
                check_interval_minutes=self.check_interval_seconds // 60
            )
            
            if new_events:
                logger.info(f"Found {len(new_events)} new earthquake(s)")
                
                # Broadcast each event to all connected clients
                for event in new_events:
                    await self._broadcast_event(event)
            else:
                logger.debug("No new earthquakes detected")
                
        except Exception as e:
            logger.error(f"Error checking for events: {e}", exc_info=True)
    
    async def _broadcast_event(self, event: dict):
        """Broadcast earthquake event to all WebSocket clients"""
        try:
            # Format event for WebSocket transmission
            ws_message = {
                "type": "earthquake_event",
                "source": "USGS",
                "event_id": event.get('id'),
                "magnitude": event.get('magnitude'),
                "location": event.get('location'),
                "time": event.get('time_formatted'),
                "latitude": event.get('latitude'),
                "longitude": event.get('longitude'),
                "depth_km": event.get('depth_km'),
                "alert_level": event.get('alert_level'),
                "tsunami": event.get('tsunami'),
                "url": event.get('url'),
                "summary": usgs_service.get_earthquake_summary(event),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to all connected clients
            await manager.broadcast(ws_message)
            
            logger.info(
                f"Broadcasted event: M{event.get('magnitude')} - {event.get('location')}"
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting event: {e}", exc_info=True)
    
    async def check_now(self):
        """Manually trigger an immediate check (useful for testing)"""
        logger.info("Manual event check triggered")
        await self._check_for_events()
    
    def get_status(self) -> dict:
        """Get current monitor status"""
        return {
            "is_running": self.is_running,
            "check_interval_seconds": self.check_interval_seconds,
            "min_magnitude": self.min_magnitude,
            "last_check": usgs_service.last_check_time.isoformat() if usgs_service.last_check_time else None,
            "processed_events_count": len(usgs_service.processed_event_ids)
        }


# Global instance
event_monitor = EventMonitor()
