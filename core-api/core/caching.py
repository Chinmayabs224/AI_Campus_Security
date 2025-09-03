"""
Caching strategies for campus security system.
"""
import json
import pickle
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
import asyncio
import hashlib
import functools

import redis.asyncio as redis
import structlog

logger = structlog.get_logger()

class CacheManager:
    """Advanced caching manager with multiple strategies."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with fallback to default."""
        try:
            # Try local cache first
            if key in self.local_cache:
                self.cache_stats['hits'] += 1
                return self.local_cache[key]['value']
            
            # Try Redis cache
            value = await self.redis.get(key)
            if value is not None:
                self.cache_stats['hits'] += 1
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return pickle.loads(value)
            
            self.cache_stats['misses'] += 1
            return default
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats['misses'] += 1
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600,
        local_cache: bool = False
    ) -> bool:
        """Set value in cache with TTL."""
        try:
            # Store in Redis
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                serialized = pickle.dumps(value)
            
            await self.redis.setex(key, ttl, serialized)
            
            # Store in local cache if requested
            if local_cache and ttl <= 300:  # Only cache locally for short TTL
                self.local_cache[key] = {
                    'value': value,
                    'expires': datetime.now() + timedelta(seconds=ttl)
                }
            
            self.cache_stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            await self.redis.delete(key)
            if key in self.local_cache:
                del self.local_cache[key]
            
            self.cache_stats['deletes'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
            
            # Clear matching local cache entries
            local_keys_to_delete = [
                k for k in self.local_cache.keys() 
                if pattern.replace('*', '') in k
            ]
            for key in local_keys_to_delete:
                del self.local_cache[key]
            
            return len(keys) + len(local_keys_to_delete)
            
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0
    
    def cleanup_local_cache(self):
        """Remove expired entries from local cache."""
        now = datetime.now()
        expired_keys = [
            key for key, data in self.local_cache.items()
            if data['expires'] < now
        ]
        for key in expired_keys:
            del self.local_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate_percent': round(hit_rate, 2),
            'local_cache_size': len(self.local_cache)
        }


class SecurityDataCache:
    """Specialized caching for security data."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    async def cache_incident(self, incident_id: str, incident_data: Dict[str, Any], ttl: int = 1800):
        """Cache incident data."""
        key = f"incident:{incident_id}"
        await self.cache.set(key, incident_data, ttl)
    
    async def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get cached incident data."""
        key = f"incident:{incident_id}"
        return await self.cache.get(key)
    
    async def cache_camera_status(self, camera_id: str, status_data: Dict[str, Any], ttl: int = 60):
        """Cache camera status with short TTL."""
        key = f"camera_status:{camera_id}"
        await self.cache.set(key, status_data, ttl, local_cache=True)
    
    async def get_camera_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get cached camera status."""
        key = f"camera_status:{camera_id}"
        return await self.cache.get(key)
    
    async def cache_user_permissions(self, user_id: str, permissions: List[str], ttl: int = 3600):
        """Cache user permissions."""
        key = f"user_permissions:{user_id}"
        await self.cache.set(key, permissions, ttl)
    
    async def get_user_permissions(self, user_id: str) -> Optional[List[str]]:
        """Get cached user permissions."""
        key = f"user_permissions:{user_id}"
        return await self.cache.get(key)
    
    async def cache_analytics_data(self, query_hash: str, data: Dict[str, Any], ttl: int = 7200):
        """Cache analytics query results."""
        key = f"analytics:{query_hash}"
        await self.cache.set(key, data, ttl)
    
    async def get_analytics_data(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached analytics data."""
        key = f"analytics:{query_hash}"
        return await self.cache.get(key)
    
    async def invalidate_incident_cache(self, incident_id: str):
        """Invalidate incident-related cache entries."""
        await self.cache.delete(f"incident:{incident_id}")
        await self.cache.clear_pattern(f"analytics:*incident*{incident_id}*")
    
    async def invalidate_user_cache(self, user_id: str):
        """Invalidate user-related cache entries."""
        await self.cache.clear_pattern(f"user_*:{user_id}")


def cached(
    ttl: int = 3600,
    key_prefix: str = "",
    local_cache: bool = False,
    cache_manager: Optional[CacheManager] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if cache_manager is None:
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = cache_manager._generate_key(
                key_prefix or func.__name__, 
                *args, 
                **kwargs
            )
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl, local_cache)
            
            return result
        return wrapper
    return decorator


class CacheWarmer:
    """Proactive cache warming for frequently accessed data."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.warming_tasks = []
    
    async def warm_incident_data(self):
        """Warm cache with recent incident data."""
        try:
            # This would typically fetch from database
            # For now, we'll simulate the warming process
            logger.info("Warming incident data cache")
            
            # Simulate fetching recent incidents
            recent_incidents = []  # Would fetch from DB
            
            for incident in recent_incidents:
                await self.cache.set(
                    f"incident:{incident['id']}", 
                    incident, 
                    ttl=1800
                )
            
            logger.info(f"Warmed cache with {len(recent_incidents)} incidents")
            
        except Exception as e:
            logger.error(f"Failed to warm incident cache: {e}")
    
    async def warm_camera_status(self):
        """Warm cache with camera status data."""
        try:
            logger.info("Warming camera status cache")
            
            # Simulate fetching camera statuses
            camera_statuses = []  # Would fetch from monitoring system
            
            for status in camera_statuses:
                await self.cache.set(
                    f"camera_status:{status['camera_id']}", 
                    status, 
                    ttl=60,
                    local_cache=True
                )
            
            logger.info(f"Warmed cache with {len(camera_statuses)} camera statuses")
            
        except Exception as e:
            logger.error(f"Failed to warm camera status cache: {e}")
    
    async def warm_user_permissions(self):
        """Warm cache with user permission data."""
        try:
            logger.info("Warming user permissions cache")
            
            # Simulate fetching active users and their permissions
            active_users = []  # Would fetch from DB
            
            for user in active_users:
                permissions = []  # Would fetch user permissions
                await self.cache.set(
                    f"user_permissions:{user['id']}", 
                    permissions, 
                    ttl=3600
                )
            
            logger.info(f"Warmed cache with {len(active_users)} user permissions")
            
        except Exception as e:
            logger.error(f"Failed to warm user permissions cache: {e}")
    
    async def start_warming_schedule(self):
        """Start scheduled cache warming."""
        warming_tasks = [
            self._schedule_warming(self.warm_incident_data, 300),  # Every 5 minutes
            self._schedule_warming(self.warm_camera_status, 60),   # Every minute
            self._schedule_warming(self.warm_user_permissions, 1800)  # Every 30 minutes
        ]
        
        self.warming_tasks = [asyncio.create_task(task) for task in warming_tasks]
        logger.info("Started cache warming schedule")
    
    async def _schedule_warming(self, warming_func: Callable, interval: int):
        """Schedule a warming function to run at intervals."""
        while True:
            try:
                await warming_func()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduled warming: {e}")
                await asyncio.sleep(interval)
    
    async def stop_warming_schedule(self):
        """Stop scheduled cache warming."""
        for task in self.warming_tasks:
            task.cancel()
        
        await asyncio.gather(*self.warming_tasks, return_exceptions=True)
        logger.info("Stopped cache warming schedule")