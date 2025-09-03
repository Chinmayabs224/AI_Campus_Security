"""
Notification preference management
"""

import json
import redis
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
from models import UserPreferences, NotificationPriority, IncidentType, NotificationChannel

logger = logging.getLogger(__name__)

class NotificationPreferenceManager:
    """Manages user notification preferences with Redis backend"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.preference_ttl = timedelta(days=90).total_seconds()  # Cache preferences for 90 days
    
    async def get_user_preferences(self, user_id: str) -> Dict:
        """Get user notification preferences"""
        try:
            # Try to get from cache first
            cached_prefs = self.redis.get(f"user_prefs:{user_id}")
            
            if cached_prefs:
                return json.loads(cached_prefs)
            
            # Return default preferences if not found
            default_prefs = self._get_default_preferences()
            
            # Cache the default preferences
            await self.update_user_preferences(user_id, default_prefs)
            
            return default_prefs
            
        except Exception as e:
            logger.error(f"Error getting user preferences for {user_id}: {str(e)}")
            return self._get_default_preferences()
    
    async def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Update user notification preferences"""
        try:
            # Validate preferences
            validated_prefs = self._validate_preferences(preferences)
            
            # Store in Redis with TTL
            self.redis.setex(
                f"user_prefs:{user_id}",
                int(self.preference_ttl),
                json.dumps(validated_prefs)
            )
            
            # Log preference update
            self._log_preference_update(user_id, validated_prefs)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating user preferences for {user_id}: {str(e)}")
            return False
    
    async def get_users_by_preferences(self, channel: NotificationChannel, 
                                     priority: NotificationPriority) -> List[str]:
        """Get list of users who should receive notifications for given channel and priority"""
        try:
            # This would typically query a database, but for now we'll use Redis scan
            user_ids = []
            
            for key in self.redis.scan_iter(match="user_prefs:*"):
                user_id = key.split(":", 1)[1]
                prefs = await self.get_user_preferences(user_id)
                
                if self._should_notify_user(prefs, channel, priority):
                    user_ids.append(user_id)
            
            return user_ids
            
        except Exception as e:
            logger.error(f"Error getting users by preferences: {str(e)}")
            return []
    
    async def is_in_quiet_hours(self, user_id: str) -> bool:
        """Check if user is currently in quiet hours"""
        try:
            prefs = await self.get_user_preferences(user_id)
            
            quiet_start = prefs.get('quiet_hours_start')
            quiet_end = prefs.get('quiet_hours_end')
            
            if not quiet_start or not quiet_end:
                return False
            
            # Get current time in user's timezone
            # For simplicity, using UTC for now
            current_time = datetime.utcnow().time()
            
            # Parse quiet hours
            start_time = datetime.strptime(quiet_start, "%H:%M").time()
            end_time = datetime.strptime(quiet_end, "%H:%M").time()
            
            # Handle overnight quiet hours (e.g., 22:00 to 08:00)
            if start_time > end_time:
                return current_time >= start_time or current_time <= end_time
            else:
                return start_time <= current_time <= end_time
                
        except Exception as e:
            logger.error(f"Error checking quiet hours for {user_id}: {str(e)}")
            return False
    
    def _get_default_preferences(self) -> Dict:
        """Get default notification preferences"""
        return {
            'push': {
                'enabled': True,
                'min_priority': 'LOW'
            },
            'sms': {
                'enabled': True,
                'min_priority': 'MEDIUM'
            },
            'whatsapp': {
                'enabled': False,
                'min_priority': 'HIGH'
            },
            'email': {
                'enabled': True,
                'min_priority': 'LOW'
            },
            'quiet_hours_start': None,
            'quiet_hours_end': None,
            'timezone': 'UTC',
            'incident_types': [incident.value for incident in IncidentType]
        }
    
    def _validate_preferences(self, preferences: Dict) -> Dict:
        """Validate and sanitize user preferences"""
        validated = self._get_default_preferences()
        
        # Validate channel preferences
        for channel in ['push', 'sms', 'whatsapp', 'email']:
            if channel in preferences:
                channel_prefs = preferences[channel]
                if isinstance(channel_prefs, dict):
                    validated[channel]['enabled'] = bool(channel_prefs.get('enabled', True))
                    
                    # Validate priority
                    min_priority = channel_prefs.get('min_priority', 'LOW')
                    if min_priority in [p.value for p in NotificationPriority]:
                        validated[channel]['min_priority'] = min_priority
        
        # Validate quiet hours
        if 'quiet_hours_start' in preferences:
            try:
                if preferences['quiet_hours_start']:
                    datetime.strptime(preferences['quiet_hours_start'], "%H:%M")
                    validated['quiet_hours_start'] = preferences['quiet_hours_start']
            except ValueError:
                pass  # Keep default
        
        if 'quiet_hours_end' in preferences:
            try:
                if preferences['quiet_hours_end']:
                    datetime.strptime(preferences['quiet_hours_end'], "%H:%M")
                    validated['quiet_hours_end'] = preferences['quiet_hours_end']
            except ValueError:
                pass  # Keep default
        
        # Validate timezone
        if 'timezone' in preferences and isinstance(preferences['timezone'], str):
            validated['timezone'] = preferences['timezone']
        
        # Validate incident types
        if 'incident_types' in preferences and isinstance(preferences['incident_types'], list):
            valid_types = [incident.value for incident in IncidentType]
            validated['incident_types'] = [
                t for t in preferences['incident_types'] if t in valid_types
            ]
        
        return validated
    
    def _should_notify_user(self, preferences: Dict, channel: NotificationChannel, 
                           priority: NotificationPriority) -> bool:
        """Check if user should receive notification based on preferences"""
        channel_prefs = preferences.get(channel.value, {})
        
        # Check if channel is enabled
        if not channel_prefs.get('enabled', True):
            return False
        
        # Check priority threshold
        min_priority = channel_prefs.get('min_priority', 'LOW')
        priority_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        
        return priority_levels.get(priority.value, 1) >= priority_levels.get(min_priority, 1)
    
    def _log_preference_update(self, user_id: str, preferences: Dict):
        """Log preference updates for audit trail"""
        log_entry = {
            'user_id': user_id,
            'action': 'preference_update',
            'timestamp': datetime.utcnow().isoformat(),
            'preferences': preferences
        }
        
        # Store in Redis with TTL for audit
        self.redis.setex(
            f"pref_audit:{user_id}:{datetime.utcnow().timestamp()}",
            timedelta(days=365).total_seconds(),  # Keep audit logs for 1 year
            json.dumps(log_entry)
        )