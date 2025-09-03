"""
Firebase Cloud Messaging (FCM) provider for push notifications
"""

import firebase_admin
from firebase_admin import credentials, messaging
import logging
from typing import Dict, Any, Optional
import json
import os
from models import NotificationRequest, NotificationChannel, DeliveryResult

logger = logging.getLogger(__name__)

class FCMProvider:
    """Firebase Cloud Messaging provider for push notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase app is already initialized
            if not firebase_admin._apps:
                credentials_path = self.config.get('FCM_CREDENTIALS_PATH')
                
                if credentials_path and os.path.exists(credentials_path):
                    # Initialize with service account credentials
                    cred = credentials.Certificate(credentials_path)
                    self.app = firebase_admin.initialize_app(cred)
                else:
                    # Initialize with default credentials (for cloud environments)
                    self.app = firebase_admin.initialize_app()
                
                logger.info("Firebase Admin SDK initialized successfully")
            else:
                self.app = firebase_admin.get_app()
                
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK: {str(e)}")
            raise
    
    async def send(self, notification: NotificationRequest, channel: NotificationChannel) -> Dict[str, Any]:
        """Send push notification via FCM"""
        try:
            if channel != NotificationChannel.PUSH:
                raise ValueError(f"FCM provider only supports PUSH channel, got {channel}")
            
            # Get user's FCM token (this would typically come from a user service)
            fcm_token = await self._get_user_fcm_token(notification.user_id)
            
            if not fcm_token:
                return {
                    "success": False,
                    "error": "No FCM token found for user"
                }
            
            # Build FCM message
            message = self._build_fcm_message(notification, fcm_token)
            
            # Send message
            response = messaging.send(message)
            
            logger.info(f"FCM notification sent successfully: {response}")
            
            return {
                "success": True,
                "message_id": response,
                "provider": "fcm"
            }
            
        except Exception as e:
            logger.error(f"FCM send error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "provider": "fcm"
            }
    
    def _build_fcm_message(self, notification: NotificationRequest, fcm_token: str) -> messaging.Message:
        """Build FCM message from notification request"""
        
        # Build notification payload
        fcm_notification = messaging.Notification(
            title=notification.title,
            body=notification.message
        )
        
        # Build data payload
        data = {
            "notification_id": notification.notification_id,
            "priority": notification.priority.value,
            "timestamp": str(notification.metadata.get('timestamp', '')),
        }
        
        # Add incident-specific data
        if notification.incident_id:
            data["incident_id"] = notification.incident_id
        
        # Add metadata
        if notification.metadata:
            for key, value in notification.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    data[f"meta_{key}"] = str(value)
        
        # Build Android-specific config
        android_config = messaging.AndroidConfig(
            priority='high' if notification.priority.value in ['HIGH', 'CRITICAL'] else 'normal',
            notification=messaging.AndroidNotification(
                icon='security_icon',
                color='#FF0000' if notification.priority.value == 'CRITICAL' else '#FFA500',
                sound='default',
                channel_id='security_alerts'
            )
        )
        
        # Build iOS-specific config
        apns_config = messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(
                    alert=messaging.ApsAlert(
                        title=notification.title,
                        body=notification.message
                    ),
                    badge=1,
                    sound='default',
                    category='SECURITY_ALERT'
                )
            )
        )
        
        # Create and return message
        return messaging.Message(
            notification=fcm_notification,
            data=data,
            token=fcm_token,
            android=android_config,
            apns=apns_config
        )
    
    async def _get_user_fcm_token(self, user_id: str) -> Optional[str]:
        """Get FCM token for user (placeholder - would integrate with user service)"""
        # This is a placeholder implementation
        # In a real system, this would query a user service or database
        # to get the user's current FCM registration token
        
        # For testing purposes, return a mock token
        # In production, this should be replaced with actual token retrieval
        return f"mock_fcm_token_for_user_{user_id}"
    
    async def send_to_topic(self, topic: str, notification: NotificationRequest) -> Dict[str, Any]:
        """Send notification to FCM topic"""
        try:
            # Build FCM message for topic
            message = messaging.Message(
                notification=messaging.Notification(
                    title=notification.title,
                    body=notification.message
                ),
                data={
                    "notification_id": notification.notification_id,
                    "priority": notification.priority.value,
                    "incident_id": notification.incident_id or "",
                },
                topic=topic
            )
            
            # Send message
            response = messaging.send(message)
            
            logger.info(f"FCM topic notification sent successfully: {response}")
            
            return {
                "success": True,
                "message_id": response,
                "provider": "fcm",
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"FCM topic send error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "provider": "fcm",
                "topic": topic
            }
    
    async def subscribe_to_topic(self, fcm_tokens: list, topic: str) -> Dict[str, Any]:
        """Subscribe FCM tokens to a topic"""
        try:
            response = messaging.subscribe_to_topic(fcm_tokens, topic)
            
            logger.info(f"Subscribed {len(fcm_tokens)} tokens to topic {topic}")
            
            return {
                "success": True,
                "success_count": response.success_count,
                "failure_count": response.failure_count,
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"FCM topic subscription error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }