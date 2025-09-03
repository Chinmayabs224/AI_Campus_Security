"""
JWT token service for authentication and session management.
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import uuid

import structlog
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from core.config import settings
from .models import User, UserSession, APIKey

logger = structlog.get_logger()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class JWTService:
    """JWT token management service."""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
        self.edge_token_expire_hours = settings.EDGE_DEVICE_TOKEN_EXPIRE_HOURS
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        role: str,
        session_id: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "session_id": session_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(
        self,
        user_id: str,
        session_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": str(user_id),
            "session_id": session_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_edge_device_token(
        self,
        device_id: str,
        scopes: list,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT token for edge devices."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.edge_token_expire_hours)
        
        to_encode = {
            "sub": device_id,
            "scopes": scopes,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "edge_device"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                return None
            
            return payload
            
        except JWTError as e:
            logger.debug("JWT verification failed", error=str(e))
            return None
    
    async def create_user_session(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new user session with tokens."""
        try:
            # Get user information
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            
            if not user or not user.is_active:
                raise ValueError("User not found or inactive")
            
            # Create session
            session_id = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
            
            # Generate tokens
            access_token = self.create_access_token(
                user_id=str(user.id),
                username=user.username,
                role=user.role,
                session_id=session_id
            )
            
            refresh_token = self.create_refresh_token(
                user_id=str(user.id),
                session_id=session_id
            )
            
            # Store session in database
            session = UserSession(
                id=uuid.UUID(session_id),
                user_id=user.id,
                session_token=hashlib.sha256(access_token.encode()).hexdigest(),
                refresh_token=hashlib.sha256(refresh_token.encode()).hexdigest(),
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            db.add(session)
            
            # Update user last login
            await db.execute(
                update(User)
                .where(User.id == user.id)
                .values(last_login=datetime.utcnow())
            )
            
            await db.commit()
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60,
                "user": user
            }
            
        except Exception as e:
            await db.rollback()
            logger.error("Failed to create user session", error=str(e))
            raise
    
    async def refresh_access_token(
        self,
        db: AsyncSession,
        refresh_token: str
    ) -> Optional[Dict[str, Any]]:
        """Refresh access token using refresh token."""
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token)
            if not payload or payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("sub")
            session_id = payload.get("session_id")
            
            if not user_id or not session_id:
                return None
            
            # Find active session
            refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            
            result = await db.execute(
                select(UserSession, User)
                .join(User)
                .where(
                    UserSession.refresh_token == refresh_token_hash,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow(),
                    User.is_active == True
                )
            )
            
            session_user = result.first()
            if not session_user:
                return None
            
            session, user = session_user
            
            # Create new access token
            access_token = self.create_access_token(
                user_id=str(user.id),
                username=user.username,
                role=user.role,
                session_id=str(session.id)
            )
            
            # Update session
            await db.execute(
                update(UserSession)
                .where(UserSession.id == session.id)
                .values(
                    session_token=hashlib.sha256(access_token.encode()).hexdigest(),
                    last_accessed=datetime.utcnow()
                )
            )
            
            await db.commit()
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }
            
        except Exception as e:
            await db.rollback()
            logger.error("Failed to refresh access token", error=str(e))
            return None
    
    async def revoke_session(
        self,
        db: AsyncSession,
        session_id: str
    ) -> bool:
        """Revoke a user session."""
        try:
            result = await db.execute(
                update(UserSession)
                .where(UserSession.id == uuid.UUID(session_id))
                .values(is_active=False)
            )
            
            await db.commit()
            return result.rowcount > 0
            
        except Exception as e:
            await db.rollback()
            logger.error("Failed to revoke session", error=str(e))
            return False
    
    async def revoke_all_user_sessions(
        self,
        db: AsyncSession,
        user_id: uuid.UUID
    ) -> int:
        """Revoke all sessions for a user."""
        try:
            result = await db.execute(
                update(UserSession)
                .where(UserSession.user_id == user_id)
                .values(is_active=False)
            )
            
            await db.commit()
            return result.rowcount
            
        except Exception as e:
            await db.rollback()
            logger.error("Failed to revoke user sessions", error=str(e))
            return 0
    
    async def cleanup_expired_sessions(self, db: AsyncSession) -> int:
        """Clean up expired sessions."""
        try:
            result = await db.execute(
                delete(UserSession)
                .where(UserSession.expires_at < datetime.utcnow())
            )
            
            await db.commit()
            return result.rowcount
            
        except Exception as e:
            await db.rollback()
            logger.error("Failed to cleanup expired sessions", error=str(e))
            return 0
    
    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and return (key, hash)."""
        # Generate random key
        key = secrets.token_urlsafe(32)
        
        # Create hash for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        return key, key_hash
    
    def verify_api_key_hash(self, key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return secrets.compare_digest(key_hash, stored_hash)


# Global JWT service instance
jwt_service = JWTService()