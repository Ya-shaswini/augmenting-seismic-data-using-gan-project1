"""
USGS Earthquake Data Service

Fetches real-time earthquake data from USGS API and provides
it to the application for monitoring and analysis.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class USGSService:
    """Service for fetching earthquake data from USGS API"""
    
    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    def __init__(self):
        self.last_check_time = None
        self.processed_event_ids = set()  # Track processed events to avoid duplicates
        
    def fetch_recent_earthquakes(
        self,
        min_magnitude: float = 4.0,
        hours_back: int = 24,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Fetch recent earthquakes from USGS API
        
        Args:
            min_magnitude: Minimum earthquake magnitude (Richter scale)
            hours_back: How many hours back to search
            max_results: Maximum number of results to return
            
        Returns:
            List of earthquake event dictionaries
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            params = {
                "format": "geojson",
                "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "minmagnitude": min_magnitude,
                "orderby": "time",
                "limit": max_results
            }
            
            logger.info(f"Fetching earthquakes from USGS API (mag >= {min_magnitude}, last {hours_back}h)")
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            events = self._parse_geojson_response(data)
            
            logger.info(f"Found {len(events)} earthquakes")
            return events
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching USGS data: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in fetch_recent_earthquakes: {e}")
            return []
    
    def fetch_new_earthquakes(
        self,
        min_magnitude: float = 4.0,
        check_interval_minutes: int = 15
    ) -> List[Dict]:
        """
        Fetch only NEW earthquakes since last check
        
        Args:
            min_magnitude: Minimum earthquake magnitude
            check_interval_minutes: How far back to check (in minutes)
            
        Returns:
            List of NEW earthquake events (not previously processed)
        """
        # Determine time range
        if self.last_check_time:
            # Check since last time, plus small buffer
            hours_back = (check_interval_minutes + 5) / 60.0
        else:
            # First run - check last hour
            hours_back = 1.0
        
        all_events = self.fetch_recent_earthquakes(
            min_magnitude=min_magnitude,
            hours_back=int(hours_back) + 1
        )
        
        # Filter out already processed events
        new_events = [
            event for event in all_events 
            if event['id'] not in self.processed_event_ids
        ]
        
        # Mark as processed
        for event in new_events:
            self.processed_event_ids.add(event['id'])
        
        # Update last check time
        self.last_check_time = datetime.utcnow()
        
        # Limit size of processed set (keep last 1000)
        if len(self.processed_event_ids) > 1000:
            self.processed_event_ids = set(list(self.processed_event_ids)[-1000:])
        
        return new_events
    
    def _parse_geojson_response(self, data: Dict) -> List[Dict]:
        """Parse USGS GeoJSON response into simplified event dictionaries"""
        events = []
        
        for feature in data.get('features', []):
            try:
                props = feature.get('properties', {})
                coords = feature.get('geometry', {}).get('coordinates', [0, 0, 0])
                
                event = {
                    'id': feature.get('id'),
                    'magnitude': props.get('mag'),
                    'location': props.get('place', 'Unknown'),
                    'time': props.get('time'),  # Unix timestamp in milliseconds
                    'time_formatted': self._format_timestamp(props.get('time')),
                    'latitude': coords[1] if len(coords) > 1 else 0,
                    'longitude': coords[0] if len(coords) > 0 else 0,
                    'depth_km': coords[2] if len(coords) > 2 else 0,
                    'event_type': props.get('type', 'earthquake'),
                    'status': props.get('status', 'automatic'),
                    'tsunami': props.get('tsunami', 0),
                    'significance': props.get('sig', 0),
                    'url': props.get('url', ''),
                    'detail_url': props.get('detail', ''),
                    'felt_reports': props.get('felt', 0),
                    'cdi': props.get('cdi'),  # Community Decimal Intensity
                    'mmi': props.get('mmi'),  # Modified Mercalli Intensity
                    'alert_level': props.get('alert'),  # green, yellow, orange, red
                    'source': 'USGS'
                }
                
                events.append(event)
                
            except Exception as e:
                logger.warning(f"Error parsing event: {e}")
                continue
        
        return events
    
    def _format_timestamp(self, timestamp_ms: Optional[int]) -> str:
        """Convert USGS timestamp (milliseconds) to readable format"""
        if not timestamp_ms:
            return "Unknown"
        
        try:
            dt = datetime.utcfromtimestamp(timestamp_ms / 1000.0)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except:
            return "Invalid timestamp"
    
    def get_event_details(self, event_id: str) -> Optional[Dict]:
        """
        Fetch detailed information about a specific event
        
        Args:
            event_id: USGS event ID
            
        Returns:
            Detailed event dictionary or None if not found
        """
        try:
            url = f"https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {
                "eventid": event_id,
                "format": "geojson"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            events = self._parse_geojson_response(data)
            
            return events[0] if events else None
            
        except Exception as e:
            logger.error(f"Error fetching event details for {event_id}: {e}")
            return None
    
    def get_earthquake_summary(self, event: Dict) -> str:
        """Generate human-readable summary of earthquake event"""
        mag = event.get('magnitude', 'Unknown')
        location = event.get('location', 'Unknown location')
        time = event.get('time_formatted', 'Unknown time')
        depth = event.get('depth_km', 0)
        
        summary = f"M{mag} - {location}"
        if depth:
            summary += f" (depth: {depth:.1f}km)"
        
        return summary


# Global instance
usgs_service = USGSService()
