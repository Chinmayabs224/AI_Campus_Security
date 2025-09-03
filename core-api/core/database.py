"""
Database connection management with asyncpg.
"""
import asyncio
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

import asyncpg
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData

from .config import settings

logger = structlog.get_logger()

# SQLAlchemy base and metadata
Base = declarative_base()
metadata = MetaData()


class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self) -> None:
        """Initialize database connections."""
        try:
            # Create SQLAlchemy async engine
            self.engine = create_async_engine(
                settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                pool_timeout=settings.DATABASE_POOL_TIMEOUT,
                echo=settings.ENVIRONMENT == "development"
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create asyncpg connection pool for raw queries
            self.pool = await asyncpg.create_pool(
                settings.DATABASE_URL,
                min_size=5,
                max_size=settings.DATABASE_POOL_SIZE,
                command_timeout=settings.DATABASE_POOL_TIMEOUT
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database connections", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Close database connections."""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("AsyncPG pool closed")
            
            if self.engine:
                await self.engine.dispose()
                logger.info("SQLAlchemy engine disposed")
                
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get raw asyncpg connection context manager."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute a raw SQL command and return status."""
        async with self.get_connection() as conn:
            return await conn.execute(command, *args)
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False


# Global database manager instance
database_manager = DatabaseManager()


# Dependency for FastAPI
async def get_db_session():
    """FastAPI dependency for database sessions."""
    async with database_manager.get_session() as session:
        yield session


async def get_db_connection():
    """FastAPI dependency for raw database connections."""
    async with database_manager.get_connection() as connection:
        yield connection