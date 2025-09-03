"""
Pytest configuration and fixtures for integration testing.
"""
import asyncio
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any
import uuid
from datetime import datetime, timedelta
import json
import tempfile
import shutil

from fastapi.testclient import TestClient
from httpx import AsyncClient
import redis.asyncio as redis
import asyncpg
from minio import Minio
import cv2
import numpy as np

# Import application components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../core-api'))

from main import create_app
from core.config import settings
from core.database import database_manager
from core.redis import redis_manager
from auth.models import User, UserRole
from events.models import SecurityEvent, EventType
from incidents.models import Incident, IncidentStatus, SeverityLevel


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_app():
    """Create test FastAPI application."""
    app = create_app()
    return app


@pytest.fixture(scope="session")
async def test_client(test_app):
    """Create test client for FastAPI application."""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="session")
async def test_db():
    """Create test database connection."""
    # Use test database
    test_db_url = settings.DATABASE_URL.replace("/campus_security", "/campus_security_test")
    
    # Create test database if it doesn't exist
    conn = await asyncpg.connect(settings.DATABASE_URL.replace("/campus_security", "/postgres"))
    try:
        await conn.execute("CREATE DATABASE campus_security_test")
    except asyncpg.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()
    
    # Connect to test database
    await database_manager.connect(test_db_url)
    
    # Run migrations
    os.system("cd core-api && alembic upgrade head")
    
    yield database_manager
    
    # Cleanup
    await database_manager.disconnect()


@pytest.fixture(scope="session")
async def test_redis():
    """Create test Redis connection."""
    test_redis_url = settings.REDIS_URL + "_test"
    await redis_manager.connect(test_redis_url)
    
    yield redis_manager
    
    # Cleanup
    await redis_manager.disconnect()


@pytest.fixture(scope="session")
async def test_storage():
    """Create test MinIO storage."""
    client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False
    )
    
    # Create test bucket
    test_bucket = "test-evidence"
    if not client.bucket_exists(test_bucket):
        client.make_bucket(test_bucket)
    
    yield client, test_bucket
    
    # Cleanup test bucket
    objects = client.list_objects(test_bucket, recursive=True)
    for obj in objects:
        client.remove_object(test_bucket, obj.object_name)
    client.remove_bucket(test_bucket)


@pytest.fixture
async def test_user(test_db):
    """Create test user."""
    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        username="test_user",
        email="test@example.com",
        role=UserRole.SECURITY_GUARD,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    # Insert user into database
    query = """
        INSERT INTO users (id, username, email, role, is_active, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
    """
    await database_manager.execute(
        query, user.id, user.username, user.email, 
        user.role.value, user.is_active, user.created_at
    )
    
    yield user
    
    # Cleanup
    await database_manager.execute("DELETE FROM users WHERE id = $1", user.id)


@pytest.fixture
async def test_admin_user(test_db):
    """Create test admin user."""
    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        username="admin_user",
        email="admin@example.com",
        role=UserRole.ADMIN,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    # Insert user into database
    query = """
        INSERT INTO users (id, username, email, role, is_active, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
    """
    await database_manager.execute(
        query, user.id, user.username, user.email, 
        user.role.value, user.is_active, user.created_at
    )
    
    yield user
    
    # Cleanup
    await database_manager.execute("DELETE FROM users WHERE id = $1", user.id)


@pytest.fixture
def mock_video_frame():
    """Create mock video frame for testing."""
    # Create a simple test frame (640x480, 3 channels)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def mock_video_clip():
    """Create mock video clip for testing."""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_clip.mp4")
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    for i in range(90):  # 3 seconds at 30fps
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_security_event():
    """Create mock security event."""
    return SecurityEvent(
        id=uuid.uuid4(),
        camera_id="test_camera_01",
        timestamp=datetime.utcnow(),
        event_type=EventType.INTRUSION,
        confidence_score=0.85,
        bounding_boxes=[
            {
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 300,
                "class": "person",
                "confidence": 0.85
            }
        ],
        metadata={
            "location": "Building A - Entrance",
            "zone": "restricted",
            "camera_model": "Test Camera v1.0"
        }
    )


@pytest.fixture
def mock_incident(mock_security_event):
    """Create mock incident."""
    return Incident(
        id=uuid.uuid4(),
        event_ids=[mock_security_event.id],
        severity=SeverityLevel.HIGH,
        status=IncidentStatus.OPEN,
        created_at=datetime.utcnow(),
        location="Building A - Entrance",
        description="Unauthorized intrusion detected"
    )


@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers for test requests."""
    # Mock JWT token (in real implementation, this would be properly signed)
    token = f"Bearer test_token_{test_user.id}"
    return {"Authorization": token}


@pytest.fixture
def admin_auth_headers(test_admin_user):
    """Create admin authentication headers for test requests."""
    token = f"Bearer admin_token_{test_admin_user.id}"
    return {"Authorization": token}


@pytest.fixture
async def cleanup_database(test_db):
    """Cleanup database after each test."""
    yield
    
    # Clean up test data
    tables = [
        "audit_logs", "evidence", "incidents", "events", 
        "users", "compliance_reports", "dsar_requests"
    ]
    
    for table in tables:
        await database_manager.execute(f"DELETE FROM {table} WHERE 1=1")


@pytest.fixture
async def cleanup_redis(test_redis):
    """Cleanup Redis after each test."""
    yield
    
    # Clean up test data
    await redis_manager.flushdb()


class MockCamera:
    """Mock camera for testing RTSP streams."""
    
    def __init__(self, camera_id: str, stream_url: str):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.is_streaming = False
        self.frame_count = 0
    
    def start_stream(self):
        """Start mock camera stream."""
        self.is_streaming = True
        self.frame_count = 0
    
    def stop_stream(self):
        """Stop mock camera stream."""
        self.is_streaming = False
    
    def get_frame(self):
        """Get next frame from mock camera."""
        if not self.is_streaming:
            return None
        
        self.frame_count += 1
        # Generate a simple test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add frame number as text for debugging
        cv2.putText(frame, f"Frame {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame


@pytest.fixture
def mock_cameras():
    """Create mock cameras for testing."""
    cameras = [
        MockCamera("test_camera_01", "rtsp://test:554/stream1"),
        MockCamera("test_camera_02", "rtsp://test:554/stream2"),
        MockCamera("test_camera_03", "rtsp://test:554/stream3"),
    ]
    return cameras


@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for testing."""
    return {
        "alert_latency_max": 5.0,  # seconds
        "detection_accuracy_min": 0.7,  # 70%
        "false_positive_rate_max": 0.3,  # 30%
        "api_response_time_max": 1.0,  # seconds
        "concurrent_streams_min": 10,  # streams
        "storage_write_time_max": 2.0,  # seconds
    }


@pytest.fixture
def compliance_requirements():
    """Define compliance requirements for testing."""
    return {
        "gdpr": {
            "data_retention_days": 365,
            "dsar_response_days": 30,
            "audit_log_retention_years": 7,
            "encryption_required": True,
            "anonymization_required": True
        },
        "ferpa": {
            "access_logging_required": True,
            "consent_tracking_required": True,
            "data_minimization_required": True,
            "secure_transmission_required": True
        },
        "coppa": {
            "parental_consent_required": True,
            "data_collection_limited": True,
            "deletion_on_request": True
        }
    }