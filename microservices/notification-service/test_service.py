"""
Test script for the notification service
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app import app, notification_service
from models import NotificationRequest, NotificationChannel, NotificationPriority

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data
    assert data['service'] == 'notification-service'

def test_send_notification_missing_fields(client):
    """Test send notification with missing required fields"""
    # Missing user_id
    response = client.post('/send', 
                          json={'title': 'Test', 'message': 'Test message', 'channels': ['push']})
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'Missing required field: user_id' in data['error']

def test_send_notification_invalid_channel(client):
    """Test send notification with invalid channel"""
    response = client.post('/send', json={
        'user_id': 'test_user',
        'title': 'Test Alert',
        'message': 'Test message',
        'channels': ['invalid_channel'],
        'priority': 'MEDIUM'
    })
    assert response.status_code == 400

@patch('app.notification_service.send_notification')
def test_send_notification_success(mock_send, client):
    """Test successful notification sending"""
    # Mock the async response
    mock_response = Mock()
    mock_response.notification_id = 'test_123'
    mock_response.success = True
    mock_response.results = {'push': {'success': True, 'message_id': 'fcm_123'}}
    mock_response.timestamp = '2024-01-15T10:30:00Z'
    
    # Create a mock coroutine
    async def mock_coroutine(*args, **kwargs):
        return mock_response
    
    mock_send.return_value = mock_coroutine()
    
    response = client.post('/send', json={
        'user_id': 'test_user',
        'title': 'Test Alert',
        'message': 'Test message',
        'channels': ['push'],
        'priority': 'MEDIUM'
    })
    
    assert response.status_code == 200

def test_get_user_preferences_default(client):
    """Test getting default user preferences"""
    with patch('app.preference_manager.get_user_preferences') as mock_get_prefs:
        # Mock async function
        async def mock_coroutine(*args, **kwargs):
            return {
                'push': {'enabled': True, 'min_priority': 'LOW'},
                'sms': {'enabled': True, 'min_priority': 'MEDIUM'},
                'email': {'enabled': True, 'min_priority': 'LOW'}
            }
        
        mock_get_prefs.return_value = mock_coroutine()
        
        response = client.get('/preferences/test_user')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'push' in data
        assert 'sms' in data
        assert 'email' in data

def test_update_user_preferences(client):
    """Test updating user preferences"""
    with patch('app.preference_manager.update_user_preferences') as mock_update_prefs:
        # Mock async function
        async def mock_coroutine(*args, **kwargs):
            return True
        
        mock_update_prefs.return_value = mock_coroutine()
        
        response = client.put('/preferences/test_user', json={
            'push': {'enabled': False, 'min_priority': 'HIGH'},
            'sms': {'enabled': True, 'min_priority': 'CRITICAL'}
        })
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['message'] == 'Preferences updated successfully'

def test_bulk_send_notifications(client):
    """Test bulk notification sending"""
    with patch('app.notification_service.send_notification') as mock_send:
        # Mock the async response
        mock_response = Mock()
        mock_response.notification_id = 'test_123'
        mock_response.success = True
        mock_response.results = {'push': {'success': True}}
        mock_response.timestamp = '2024-01-15T10:30:00Z'
        
        # Create a mock coroutine
        async def mock_coroutine(*args, **kwargs):
            return mock_response
        
        mock_send.return_value = mock_coroutine()
        
        response = client.post('/bulk-send', json={
            'notifications': [
                {
                    'user_id': 'user1',
                    'title': 'Test Alert 1',
                    'message': 'Test message 1',
                    'channels': ['push'],
                    'priority': 'MEDIUM'
                },
                {
                    'user_id': 'user2',
                    'title': 'Test Alert 2',
                    'message': 'Test message 2',
                    'channels': ['email'],
                    'priority': 'HIGH'
                }
            ]
        })
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'results' in data
        assert len(data['results']) == 2

class TestNotificationModels:
    """Test notification data models"""
    
    def test_notification_request_creation(self):
        """Test creating a notification request"""
        request = NotificationRequest(
            notification_id='test_123',
            user_id='user_456',
            title='Test Alert',
            message='Test message',
            channels=[NotificationChannel.PUSH, NotificationChannel.SMS],
            priority=NotificationPriority.HIGH,
            incident_id='incident_789'
        )
        
        assert request.notification_id == 'test_123'
        assert request.user_id == 'user_456'
        assert request.title == 'Test Alert'
        assert request.message == 'Test message'
        assert len(request.channels) == 2
        assert NotificationChannel.PUSH in request.channels
        assert NotificationChannel.SMS in request.channels
        assert request.priority == NotificationPriority.HIGH
        assert request.incident_id == 'incident_789'

    def test_notification_channel_enum(self):
        """Test notification channel enum values"""
        assert NotificationChannel.PUSH.value == 'push'
        assert NotificationChannel.SMS.value == 'sms'
        assert NotificationChannel.WHATSAPP.value == 'whatsapp'
        assert NotificationChannel.EMAIL.value == 'email'

    def test_notification_priority_enum(self):
        """Test notification priority enum values"""
        assert NotificationPriority.LOW.value == 'LOW'
        assert NotificationPriority.MEDIUM.value == 'MEDIUM'
        assert NotificationPriority.HIGH.value == 'HIGH'
        assert NotificationPriority.CRITICAL.value == 'CRITICAL'

if __name__ == '__main__':
    # Run basic tests
    print("Running notification service tests...")
    
    # Test model creation
    test_models = TestNotificationModels()
    test_models.test_notification_request_creation()
    test_models.test_notification_channel_enum()
    test_models.test_notification_priority_enum()
    
    print("✅ Model tests passed")
    
    # Test app creation
    with app.test_client() as client:
        # Test health check
        response = client.get('/health')
        assert response.status_code == 200
        print("✅ Health check test passed")
        
        # Test missing fields validation
        response = client.post('/send', json={'title': 'Test'})
        assert response.status_code == 400
        print("✅ Validation test passed")
    
    print("🎉 All basic tests passed!")
    print("\nTo run full test suite with pytest:")
    print("pip install pytest")
    print("pytest test_service.py -v")