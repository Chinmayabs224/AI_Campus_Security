# Notification Service

Multi-channel notification Flask microservice for the AI Campus Security system. Supports Firebase Cloud Messaging (FCM) for push notifications, Twilio for SMS and WhatsApp, and SMTP for email notifications.

## Features

- **Multi-channel notifications**: Push, SMS, WhatsApp, and Email
- **Real-time WebSocket alerts**: Live dashboard updates and mobile notifications
- **Alert prioritization and routing**: Intelligent routing based on incident type, location, and priority
- **Escalation management**: Automatic escalation for unacknowledged critical alerts
- **User preference management**: Configurable notification preferences per user
- **Priority-based filtering**: Different notification thresholds based on priority levels
- **Quiet hours support**: Respect user-defined quiet hours
- **Bulk notifications**: Send notifications to multiple users efficiently
- **Audit logging**: Complete audit trail for compliance
- **Rate limiting**: Prevent notification spam
- **Retry mechanism**: Automatic retry for failed deliveries

## API Endpoints

### Real-Time Alert Distribution

#### Distribute Alert
```
POST /alert/distribute
```

Distribute a security alert via WebSocket and traditional notifications with intelligent routing.

**Request Body:**
```json
{
  "title": "Security Alert: Intrusion Detected",
  "message": "Unauthorized person detected in restricted area",
  "priority": "HIGH",
  "incident_type": "intrusion",
  "location": "Building A - Floor 2",
  "incident_id": "incident_123",
  "camera_id": "CAM_001",
  "confidence": 0.95,
  "timestamp": "2024-01-15T10:30:00Z",
  "channels": ["push", "sms"]
}
```

**Response:**
```json
{
  "alert_id": "alert_1234567890",
  "routing_results": [
    {
      "rule_id": "high_priority",
      "rule_name": "High Priority Security Events",
      "target_users": ["security_guard_1", "security_guard_2"],
      "channels": ["push", "sms"]
    }
  ],
  "websocket_result": {
    "sent_count": 3,
    "failed_count": 0,
    "target_connections": 3
  }
}
```

#### Acknowledge Alert
```
POST /alert/acknowledge
```

Acknowledge an alert and cancel any pending escalations.

**Request Body:**
```json
{
  "alert_id": "alert_1234567890",
  "user_id": "security_guard_1"
}
```

#### Get Active Alerts
```
GET /alerts/active?user_id=security_guard_1
```

Get all active alerts for a specific user.

**Response:**
```json
{
  "user_id": "security_guard_1",
  "active_alerts": [
    {
      "alert_id": "alert_123",
      "title": "Security Alert",
      "message": "Intrusion detected",
      "priority": "HIGH",
      "location": "Building A",
      "timestamp": "2024-01-15T10:30:00Z",
      "acknowledged": false
    }
  ],
  "count": 1
}
```

### Send Notification
```
POST /send
```

Send a notification to a user through specified channels.

**Request Body:**
```json
{
  "user_id": "user123",
  "title": "Security Alert",
  "message": "Intrusion detected in Building A",
  "channels": ["push", "sms", "email"],
  "priority": "HIGH",
  "incident_id": "incident_456",
  "metadata": {
    "location": "Building A - Floor 2",
    "camera_id": "CAM_001",
    "confidence": 0.95,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Response:**
```json
{
  "notification_id": "notif_1234567890",
  "success": true,
  "results": {
    "push": {"success": true, "message_id": "fcm_msg_123"},
    "sms": {"success": true, "message_id": "twilio_msg_456"},
    "email": {"success": true, "message_id": "email_msg_789"}
  },
  "timestamp": "2024-01-15T10:30:05Z"
}
```

### Bulk Send Notifications
```
POST /bulk-send
```

Send notifications to multiple users.

**Request Body:**
```json
{
  "notifications": [
    {
      "user_id": "user123",
      "title": "Security Alert",
      "message": "Intrusion detected",
      "channels": ["push", "sms"],
      "priority": "HIGH"
    },
    {
      "user_id": "user456",
      "title": "Security Alert", 
      "message": "Intrusion detected",
      "channels": ["email"],
      "priority": "HIGH"
    }
  ]
}
```

### Get User Preferences
```
GET /preferences/{user_id}
```

Get notification preferences for a user.

**Response:**
```json
{
  "push": {
    "enabled": true,
    "min_priority": "LOW"
  },
  "sms": {
    "enabled": true,
    "min_priority": "MEDIUM"
  },
  "whatsapp": {
    "enabled": false,
    "min_priority": "HIGH"
  },
  "email": {
    "enabled": true,
    "min_priority": "LOW"
  },
  "quiet_hours_start": "22:00",
  "quiet_hours_end": "08:00",
  "timezone": "UTC",
  "incident_types": ["intrusion", "loitering", "crowding"]
}
```

### Update User Preferences
```
PUT /preferences/{user_id}
```

Update notification preferences for a user.

**Request Body:**
```json
{
  "push": {
    "enabled": true,
    "min_priority": "LOW"
  },
  "sms": {
    "enabled": false,
    "min_priority": "HIGH"
  },
  "quiet_hours_start": "23:00",
  "quiet_hours_end": "07:00"
}
```

### Health Check
```
GET /health
```

Check service health status.

## Configuration

Copy `.env.example` to `.env` and configure the following:

### Firebase Cloud Messaging (FCM)
- `FCM_CREDENTIALS_PATH`: Path to Firebase service account credentials JSON file
- `FCM_PROJECT_ID`: Firebase project ID

### Twilio (SMS & WhatsApp)
- `TWILIO_ACCOUNT_SID`: Twilio account SID
- `TWILIO_AUTH_TOKEN`: Twilio auth token
- `TWILIO_PHONE_NUMBER`: Twilio phone number for SMS
- `TWILIO_WHATSAPP_NUMBER`: Twilio WhatsApp number

### Email (SMTP)
- `SMTP_HOST`: SMTP server hostname
- `SMTP_PORT`: SMTP server port
- `SMTP_USERNAME`: SMTP username
- `SMTP_PASSWORD`: SMTP password
- `EMAIL_FROM`: From email address

### Redis
- `REDIS_HOST`: Redis server hostname
- `REDIS_PORT`: Redis server port
- `REDIS_DB`: Redis database number

## WebSocket API

The service provides a WebSocket server for real-time alert distribution.

### Connection
Connect to `ws://localhost:8765` and send authentication message:

```json
{
  "type": "auth",
  "user_id": "security_guard_1",
  "token": "valid_auth_token",
  "connection_type": "dashboard",
  "subscriptions": {
    "locations": ["Building A", "Building B"],
    "incident_types": ["intrusion", "loitering", "crowding"]
  }
}
```

### Message Types

#### Security Alert
```json
{
  "type": "security_alert",
  "alert": {
    "alert_id": "alert_123",
    "title": "Security Alert",
    "message": "Intrusion detected",
    "priority": "HIGH",
    "location": "Building A",
    "incident_type": "intrusion"
  }
}
```

#### Alert Acknowledgment
```json
{
  "type": "acknowledge_alert",
  "alert_id": "alert_123"
}
```

#### Alert Escalation
```json
{
  "type": "alert_escalation",
  "alert_id": "alert_123",
  "escalation_level": 2
}
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Set up Firebase credentials:
```bash
# Download service account key from Firebase Console
# Save as firebase-credentials.json in the service directory
```

4. Start Redis server:
```bash
redis-server
```

5. Run the complete service (Flask + WebSocket):
```bash
python run_service.py
```

Or run components separately:

**Flask HTTP API only:**
```bash
python app.py
```

**WebSocket server only:**
```bash
python websocket_server.py
```

## Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t notification-service .

# Run container
docker run -d \
  --name notification-service \
  -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/firebase-credentials.json:/app/firebase-credentials.json \
  notification-service
```

## Testing

The service includes placeholder implementations for user data retrieval (phone numbers, email addresses, FCM tokens). In a production environment, these should be replaced with actual integrations to your user management system.

### Test Real-Time Alert Distribution
```bash
# Distribute a security alert
curl -X POST http://localhost:5000/alert/distribute \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Security Alert: Intrusion Detected",
    "message": "Unauthorized person detected in Building A",
    "priority": "HIGH",
    "incident_type": "intrusion",
    "location": "Building A - Floor 2",
    "camera_id": "CAM_001",
    "confidence": 0.95
  }'
```

### Test WebSocket Connection
```bash
# Run WebSocket client examples
python websocket_client_example.py dashboard  # Dashboard client
python websocket_client_example.py mobile    # Mobile client
python websocket_client_example.py admin     # Admin client
```

### Test Traditional Notification
```bash
curl -X POST http://localhost:5000/send \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "title": "Test Alert",
    "message": "This is a test notification",
    "channels": ["push", "sms", "email"],
    "priority": "MEDIUM"
  }'
```

### Run Unit Tests
```bash
python test_service.py
# or
pytest test_service.py -v
```

## Integration with Core API

The notification service is designed to be called by the core API when security incidents are detected. Example integration:

```python
import httpx

async def send_security_alert(incident_data):
    notification_payload = {
        "user_id": incident_data["assigned_user"],
        "title": f"Security Alert: {incident_data['type']}",
        "message": f"Incident detected at {incident_data['location']}",
        "channels": ["push", "sms"],
        "priority": incident_data["priority"],
        "incident_id": incident_data["id"],
        "metadata": {
            "location": incident_data["location"],
            "camera_id": incident_data["camera_id"],
            "confidence": incident_data["confidence"],
            "timestamp": incident_data["timestamp"]
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://notification-service:5000/send",
            json=notification_payload
        )
        return response.json()
```

## Monitoring and Logging

The service logs all notification attempts and maintains audit trails in Redis. Monitor the following metrics:

- Notification delivery success rates by channel
- Response times for each provider
- Failed delivery reasons
- User preference update frequency

## Security Considerations

- API endpoints should be secured with authentication in production
- Sensitive configuration (API keys, passwords) should be stored securely
- Rate limiting is implemented to prevent abuse
- All notification attempts are logged for audit purposes
- User data retrieval should be secured and comply with privacy regulations