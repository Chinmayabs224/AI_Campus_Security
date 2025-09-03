"""
AI Campus Security - Notification Service
Multi-channel notification Flask microservice with FCM, Twilio, and email support
"""

from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import os
import logging
import redis
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import threading

# Import notification providers
from providers.fcm_provider import FCMProvider
from providers.twilio_provider import TwilioProvider
from providers.email_provider import EmailProvider
from models import NotificationRequest, NotificationResponse, NotificationChannel, NotificationPriority
from config import Config
from preferences import NotificationPreferenceManager
from websocket_manager import WebSocketManager, AlertMessage
from alert_router import AlertRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Redis for caching and rate limiting
redis_client = redis.Redis(
    host=app.config['REDIS_HOST'],
    port=app.config['REDIS_PORT'],
    db=app.config['REDIS_DB'],
    decode_responses=True
)

# Initialize notification providers
fcm_provider = FCMProvider(app.config)
twilio_provider = TwilioProvider(app.config)
email_provider = EmailProvider(app.config)
preference_manager = NotificationPreferenceManager(redis_client)

# Initialize WebSocket manager and alert router
websocket_manager = WebSocketManager(redis_client)
alert_router = AlertRouter(redis_client, websocket_manager)

class NotificationService:
    """Core notification service handling multi-channel delivery"""
    
    def __init__(self):
        self.providers = {
            NotificationChannel.PUSH: fcm_provider,
            NotificationChannel.SMS: twilio_provider,
            NotificationChannel.WHATSAPP: twilio_provider,
            NotificationChannel.EMAIL: email_provider
        }
    
    async def send_notification(self, notification: NotificationRequest) -> NotificationResponse:
        """Send notification through specified channels with user preferences"""
        try:
            # Get user notification preferences
            user_preferences = await preference_manager.get_user_preferences(notification.user_id)
            
            # Filter channels based on user preferences and notification priority
            enabled_channels = self._filter_channels(notification.channels, user_preferences, notification.priority)
            
            results = {}
            for channel in enabled_channels:
                try:
                    provider = self.providers[channel]
                    result = await provider.send(notification, channel)
                    results[channel.value] = result
                    logger.info(f"Notification sent via {channel.value} to user {notification.user_id}")
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel.value}: {str(e)}")
                    results[channel.value] = {"success": False, "error": str(e)}
            
            # Log notification for audit trail
            await self._log_notification(notification, results)
            
            return NotificationResponse(
                notification_id=notification.notification_id,
                success=any(result.get("success", False) for result in results.values()),
                results=results,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Notification service error: {str(e)}")
            return NotificationResponse(
                notification_id=notification.notification_id,
                success=False,
                results={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    def _filter_channels(self, requested_channels: List[NotificationChannel], 
                        user_preferences: Dict, priority: NotificationPriority) -> List[NotificationChannel]:
        """Filter channels based on user preferences and priority"""
        enabled_channels = []
        
        for channel in requested_channels:
            # Check if user has enabled this channel
            if user_preferences.get(channel.value, {}).get('enabled', True):
                # Check priority-based filtering
                min_priority = user_preferences.get(channel.value, {}).get('min_priority', 'LOW')
                if self._priority_meets_threshold(priority, min_priority):
                    enabled_channels.append(channel)
        
        return enabled_channels
    
    def _priority_meets_threshold(self, notification_priority: NotificationPriority, min_priority: str) -> bool:
        """Check if notification priority meets user's minimum threshold"""
        priority_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        return priority_levels.get(notification_priority.value, 1) >= priority_levels.get(min_priority, 1)
    
    async def _log_notification(self, notification: NotificationRequest, results: Dict):
        """Log notification for audit and analytics"""
        log_entry = {
            'notification_id': notification.notification_id,
            'user_id': notification.user_id,
            'incident_id': notification.incident_id,
            'channels': [ch.value for ch in notification.channels],
            'priority': notification.priority.value,
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in Redis with TTL for analytics
        redis_client.setex(
            f"notification_log:{notification.notification_id}",
            timedelta(days=30).total_seconds(),
            json.dumps(log_entry)
        )

# Initialize notification service
notification_service = NotificationService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'notification-service'
    })

@app.route('/send', methods=['POST'])
def send_notification():
    """Send notification endpoint"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_id', 'title', 'message', 'channels']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create notification request
        notification = NotificationRequest(
            notification_id=data.get('notification_id', f"notif_{datetime.utcnow().timestamp()}"),
            user_id=data['user_id'],
            incident_id=data.get('incident_id'),
            title=data['title'],
            message=data['message'],
            channels=[NotificationChannel(ch) for ch in data['channels']],
            priority=NotificationPriority(data.get('priority', 'MEDIUM')),
            metadata=data.get('metadata', {})
        )
        
        # Send notification asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(notification_service.send_notification(notification))
        loop.close()
        
        return jsonify(asdict(response))
        
    except ValueError as e:
        return jsonify({'error': f'Invalid data: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Send notification error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/preferences/<user_id>', methods=['GET'])
def get_user_preferences(user_id: str):
    """Get user notification preferences"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        preferences = loop.run_until_complete(preference_manager.get_user_preferences(user_id))
        loop.close()
        
        return jsonify(preferences)
    except Exception as e:
        logger.error(f"Get preferences error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/preferences/<user_id>', methods=['PUT'])
def update_user_preferences(user_id: str):
    """Update user notification preferences"""
    try:
        data = request.get_json()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(preference_manager.update_user_preferences(user_id, data))
        loop.close()
        
        if success:
            return jsonify({'message': 'Preferences updated successfully'})
        else:
            return jsonify({'error': 'Failed to update preferences'}), 500
            
    except Exception as e:
        logger.error(f"Update preferences error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/bulk-send', methods=['POST'])
def bulk_send_notifications():
    """Send notifications to multiple users"""
    try:
        data = request.get_json()
        notifications = data.get('notifications', [])
        
        if not notifications:
            return jsonify({'error': 'No notifications provided'}), 400
        
        results = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        for notif_data in notifications:
            try:
                notification = NotificationRequest(
                    notification_id=notif_data.get('notification_id', f"notif_{datetime.utcnow().timestamp()}"),
                    user_id=notif_data['user_id'],
                    incident_id=notif_data.get('incident_id'),
                    title=notif_data['title'],
                    message=notif_data['message'],
                    channels=[NotificationChannel(ch) for ch in notif_data['channels']],
                    priority=NotificationPriority(notif_data.get('priority', 'MEDIUM')),
                    metadata=notif_data.get('metadata', {})
                )
                
                response = loop.run_until_complete(notification_service.send_notification(notification))
                results.append(asdict(response))
                
            except Exception as e:
                results.append({
                    'notification_id': notif_data.get('notification_id', 'unknown'),
                    'success': False,
                    'error': str(e)
                })
        
        loop.close()
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Bulk send error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/alert/distribute', methods=['POST'])
def distribute_alert():
    """Distribute real-time alert via WebSocket and notifications"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['title', 'message', 'priority', 'incident_type', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create notification request
        notification = NotificationRequest(
            notification_id=data.get('notification_id', f"alert_{datetime.utcnow().timestamp()}"),
            user_id=data.get('user_id', 'system'),
            incident_id=data.get('incident_id'),
            title=data['title'],
            message=data['message'],
            channels=[NotificationChannel(ch) for ch in data.get('channels', ['push'])],
            priority=NotificationPriority(data['priority']),
            metadata={
                'incident_type': data['incident_type'],
                'location': data['location'],
                'camera_id': data.get('camera_id'),
                'confidence': data.get('confidence'),
                'timestamp': data.get('timestamp', datetime.utcnow().isoformat())
            }
        )
        
        # Route alert through alert router
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        routing_result = loop.run_until_complete(alert_router.route_alert(notification))
        loop.close()
        
        return jsonify(routing_result)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid data: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Distribute alert error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/alert/acknowledge', methods=['POST'])
def acknowledge_alert():
    """Acknowledge an alert"""
    try:
        data = request.get_json()
        
        alert_id = data.get('alert_id')
        user_id = data.get('user_id')
        
        if not alert_id or not user_id:
            return jsonify({'error': 'Missing alert_id or user_id'}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(alert_router.acknowledge_alert_escalation(alert_id, user_id))
        loop.close()
        
        return jsonify({
            'success': success,
            'alert_id': alert_id,
            'acknowledged_by': user_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Acknowledge alert error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/alerts/active', methods=['GET'])
def get_active_alerts():
    """Get active alerts for a user"""
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Missing user_id parameter'}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        alerts = loop.run_until_complete(websocket_manager.get_active_alerts(user_id))
        loop.close()
        
        return jsonify({
            'user_id': user_id,
            'active_alerts': alerts,
            'count': len(alerts),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get active alerts error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/websocket/stats', methods=['GET'])
def get_websocket_stats():
    """Get WebSocket connection statistics"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ws_stats = loop.run_until_complete(websocket_manager.get_connection_stats())
        escalation_stats = loop.run_until_complete(alert_router.get_escalation_stats())
        loop.close()
        
        return jsonify({
            'websocket_stats': ws_stats,
            'escalation_stats': escalation_stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get WebSocket stats error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/routing/rules', methods=['GET'])
def get_routing_rules():
    """Get all routing rules"""
    try:
        rules = {}
        for rule_id, rule in alert_router.routing_rules.items():
            rules[rule_id] = {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'conditions': rule.conditions,
                'target_users': rule.target_users,
                'target_roles': rule.target_roles,
                'channels': rule.channels,
                'escalation_delay_minutes': rule.escalation_delay_minutes,
                'enabled': rule.enabled
            }
        
        return jsonify({
            'routing_rules': rules,
            'count': len(rules),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get routing rules error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)