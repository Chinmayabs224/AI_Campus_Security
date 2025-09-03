"""
Redis connection management and utilities.
"""
import asyncio
import json
from typing import Optional, Any, Union
from contextlib import asynccontextmanager

import redis.asyncio as redis
import structlog

from .config import settings

logger = structlog.get_logger()


class RedisManager:
    """Redis connection manager."""
    
    def __init__(self):
        self.pool: Optional[redis.ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
    
    async def connect(self) -> None:
        """Initialize Redis connection."""
        try:
            # Create connection pool
            self.pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=settings.REDIS_POOL_SIZE,
                socket_timeout=settings.REDIS_TIMEOUT,
                socket_connect_timeout=settings.REDIS_TIMEOUT,
                decode_responses=True
            )
            
            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Redis connection", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connections."""
        try:
            if self.client:
                await self.client.close()
                logger.info("Redis client closed")
            
            if self.pool:
                await self.pool.disconnect()
                logger.info("Redis pool disconnected")
                
        except Exception as e:
            logger.error("Error closing Redis connections", error=str(e))
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self.client:
            raise RuntimeError("Redis not initialized")
        
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error("Redis GET failed", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: Union[str, int, float],
        expire: Optional[int] = None
    ) -> bool:
        """Set value in Redis with optional expiration."""
        if not self.client:
            raise RuntimeError("Redis not initialized")
        
        try:
            return await self.client.set(key, value, ex=expire)
        except Exception as e:
            logger.error("Redis SET failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.client:
            raise RuntimeError("Redis not initialized")
        
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Redis DELETE failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self.client:
            raise RuntimeError("Redis not initialized")
        
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error("Redis EXISTS failed", key=key, error=str(e))
            return False
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment value in Redis."""
        if not self.client:
            raise RuntimeError("Redis not initialized")
        
        try:
            return await self.client.incr(key, amount)
        except Exception as e:
            logger.error("Redis INCR failed", key=key, error=str(e))
            return None
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        if not self.client:
            raise RuntimeError("Redis not initialized")
        
        try:
            return await self.client.expire(key, seconds)
        except Exception as e:
            logger.error("Redis EXPIRE failed", key=key, error=str(e))
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from Redis."""
        value = await self.get(key)
        if value is None:
            return None
        
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from Redis", key=key, error=str(e))
            return None
    
    async def set_json(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set JSON value in Redis."""
        try:
            json_value = json.dumps(value)
            return await self.set(key, json_value, expire)
        except (TypeError, ValueError) as e:
            logger.error("Failed to encode JSON for Redis", key=key, error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            if not self.client:
                return False
            
            await self.client.ping()
            return True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False
    
    @asynccontextmanager
    async def pipeline(self):
        """Get Redis pipeline context manager."""
        if not self.client:
            raise RuntimeError("Redis not initialized")
        
        pipe = self.client.pipeline()
        try:
            yield pipe
            await pipe.execute()
        except Exception:
            await pipe.reset()
            raise


# Global Redis manager instance
redis_manager = RedisManager()


# Cache utilities
class CacheManager:
    """High-level cache management utilities."""
    
    @staticmethod
    async def get_or_set(
        key: str,
        factory_func,
        expire: int = 300
    ) -> Any:
        """Get value from cache or set it using factory function."""
        # Try to get from cache
        cached_value = await redis_manager.get_json(key)
        if cached_value is not None:
            return cached_value
        
        # Generate new value
        new_value = await factory_func() if asyncio.iscoroutinefunction(factory_func) else factory_func()
        
        # Cache the new value
        await redis_manager.set_json(key, new_value, expire)
        
        return new_value
    
    @staticmethod
    async def invalidate_pattern(pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        if not redis_manager.client:
            return 0
        
        try:
            keys = await redis_manager.client.keys(pattern)
            if keys:
                return await redis_manager.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error("Failed to invalidate cache pattern", pattern=pattern, error=str(e))
            return 0


# Global cache manager instance
cache_manager = CacheManager()