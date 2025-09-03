"""
Authentication utilities for edge devices.
"""
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional

import structlog
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.config import settings
from core.redis import redis_manager

logger = structlog.get_logger()
security = HTTPBearer()


class EdgeDeviceAuth:
    """Edge device authentication manager."""
    
    @staticmethod
    async def authenticate_device(
        device_id: str,
        device_secret: str,
        timestamp: str,
        signature: str
    ) -> bool:
        """Authenticate edge device using HMAC signature."""
        try:
            # Check if device exists and is active
            device_key = f"edge_device:{device_id}"
            device_data = await redis_manager.get_json(device_key)
            
            if not device_data:
                logger.warning("Unknown edge device attempted authentication", device_id=device_id)
                return False
            
            if not device_data.get("active", True):
                logger.warning("Inactive edge device attempted authentication", device_id=device_id)
                return False
            
            # Verify timestamp (prevent replay attacks)
            try:
                request_time = datetime.fromisoformat(timestamp)
                current_time = datetime.utcnow()
                time_diff = abs((current_time - request_time).total_seconds())
                
                if time_diff > 300:  # 5 minutes tolerance
                    logger.warning("Edge device authentication with stale timestamp", 
                                 device_id=device_id, time_diff=time_diff)
                    return False
            except ValueError:
                logger.warning("Edge device authentication with invalid timestamp", 
                             device_id=device_id, timestamp=timestamp)
                return False
            
            # Verify HMAC signature
            stored_secret = device_data.get("secret")
            if not stored_secret:
                logger.error("Edge device missing secret", device_id=device_id)
                return False
            
            # Create expected signature
            message = f"{device_id}:{timestamp}:{device_secret}"
            expected_signature = hmac.new(
                stored_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Edge device authentication failed - invalid signature", 
                             device_id=device_id)
                return False
            
            # Update last seen
            await redis_manager.set_json(
                f"edge_device_last_seen:{device_id}",
                {"timestamp": datetime.utcnow().isoformat()},
                expire=86400  # 24 hours
            )
            
            logger.info("Edge device authenticated successfully", device_id=device_id)
            return True
            
        except Exception as e:
            logger.error("Edge device authentication error", error=str(e), device_id=device_id)
            return False
    
    @staticmethod
    async def register_device(
        device_id: str,
        device_secret: str,
        location: Optional[str] = None,
        capabilities: Optional[list] = None
    ) -> str:
        """Register a new edge device."""
        try:
            # Generate device secret
            device_data = {
                "device_id": device_id,
                "secret": device_secret,
                "location": location,
                "capabilities": capabilities or [],
                "active": True,
                "registered_at": datetime.utcnow().isoformat()
            }
            
            device_key = f"edge_device:{device_id}"
            await redis_manager.set_json(device_key, device_data)
            
            logger.info("Edge device registered", device_id=device_id, location=location)
            return device_secret
            
        except Exception as e:
            logger.error("Failed to register edge device", error=str(e), device_id=device_id)
            raise
    
    @staticmethod
    async def revoke_device(device_id: str) -> bool:
        """Revoke access for an edge device."""
        try:
            device_key = f"edge_device:{device_id}"
            device_data = await redis_manager.get_json(device_key)
            
            if device_data:
                device_data["active"] = False
                device_data["revoked_at"] = datetime.utcnow().isoformat()
                await redis_manager.set_json(device_key, device_data)
                
                logger.info("Edge device access revoked", device_id=device_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to revoke edge device", error=str(e), device_id=device_id)
            return False


async def get_authenticated_device(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """FastAPI dependency for edge device authentication."""
    try:
        # Extract authentication data from token
        # Format: "device_id:timestamp:signature"
        auth_parts = credentials.credentials.split(":")
        
        if len(auth_parts) != 3:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication format"
            )
        
        device_id, timestamp, signature = auth_parts
        
        # Get device secret from request headers or use a default approach
        # For now, we'll use a simplified approach where the secret is part of the signature
        device_secret = "temp_secret"  # This should be properly implemented
        
        # Authenticate device
        is_authenticated = await EdgeDeviceAuth.authenticate_device(
            device_id, device_secret, timestamp, signature
        )
        
        if not is_authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Device authentication failed"
            )
        
        return device_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authentication dependency error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error"
        )


class RateLimiter:
    """Rate limiter for edge devices."""
    
    @staticmethod
    async def check_rate_limit(device_id: str, limit: int = 100, window: int = 60) -> bool:
        """Check if device is within rate limits."""
        try:
            rate_key = f"rate_limit:device:{device_id}"
            current_requests = await redis_manager.incr(rate_key)
            
            if current_requests == 1:
                await redis_manager.expire(rate_key, window)
            
            if current_requests > limit:
                logger.warning("Rate limit exceeded for device", 
                             device_id=device_id, requests=current_requests, limit=limit)
                return False
            
            return True
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e), device_id=device_id)
            return True  # Allow request if check fails


async def check_device_rate_limit(device_id: str = Depends(get_authenticated_device)):
    """FastAPI dependency for rate limiting."""
    is_allowed = await RateLimiter.check_rate_limit(device_id)
    
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )