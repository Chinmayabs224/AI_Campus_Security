"""
Twilio provider for SMS and WhatsApp notifications
"""

from twilio.rest import Client
from twilio.base.exceptions import TwilioException
import logging
from typing import Dict, Any, Optional
from models import NotificationRequest, NotificationChannel

logger = logging.getLogger(__name__)

class TwilioProvider:
    """Twilio provider for SMS and WhatsApp notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self._initialize_twilio()
    
    def _initialize_twilio(self):
        """Initialize Twilio client"""
        try:
            account_sid = self.config.get('TWILIO_ACCOUNT_SID')
            auth_token = self.config.get('TWILIO_AUTH_TOKEN')
            
            if not account_sid or not auth_token:
                raise ValueError("Twilio credentials not configured")
            
            self.client = Client(account_sid, auth_token)
            logger.info("Twilio client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {str(e)}")
            raise
    
    async def send(self, notification: NotificationRequest, channel: NotificationChannel) -> Dict[str, Any]:
        """Send notification via Twilio (SMS or WhatsApp)"""
        try:
            if channel == NotificationChannel.SMS:
                return await self._send_sms(notification)
            elif channel == NotificationChannel.WHATSAPP:
                return await self._send_whatsapp(notification)
            else:
                raise ValueError(f"Twilio provider only supports SMS and WhatsApp channels, got {channel}")
                
        except Exception as e:
            logger.error(f"Twilio send error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "provider": "twilio"
            }
    
    async def _send_sms(self, notification: NotificationRequest) -> Dict[str, Any]:
        """Send SMS notification"""
        try:
            # Get user's phone number
            phone_number = await self._get_user_phone_number(notification.user_id)
            
            if not phone_number:
                return {
                    "success": False,
                    "error": "No phone number found for user"
                }
            
            # Format message for SMS
            message_body = self._format_sms_message(notification)
            
            # Send SMS
            message = self.client.messages.create(
                body=message_body,
                from_=self.config.get('TWILIO_PHONE_NUMBER'),
                to=phone_number
            )
            
            logger.info(f"SMS sent successfully: {message.sid}")
            
            return {
                "success": True,
                "message_id": message.sid,
                "provider": "twilio",
                "channel": "sms",
                "to": phone_number
            }
            
        except TwilioException as e:
            logger.error(f"Twilio SMS error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "provider": "twilio",
                "channel": "sms"
            }
    
    async def _send_whatsapp(self, notification: NotificationRequest) -> Dict[str, Any]:
        """Send WhatsApp notification"""
        try:
            # Get user's WhatsApp number
            whatsapp_number = await self._get_user_whatsapp_number(notification.user_id)
            
            if not whatsapp_number:
                return {
                    "success": False,
                    "error": "No WhatsApp number found for user"
                }
            
            # Format message for WhatsApp
            message_body = self._format_whatsapp_message(notification)
            
            # Send WhatsApp message
            message = self.client.messages.create(
                body=message_body,
                from_=self.config.get('TWILIO_WHATSAPP_NUMBER'),
                to=f"whatsapp:{whatsapp_number}"
            )
            
            logger.info(f"WhatsApp message sent successfully: {message.sid}")
            
            return {
                "success": True,
                "message_id": message.sid,
                "provider": "twilio",
                "channel": "whatsapp",
                "to": whatsapp_number
            }
            
        except TwilioException as e:
            logger.error(f"Twilio WhatsApp error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "provider": "twilio",
                "channel": "whatsapp"
            }
    
    def _format_sms_message(self, notification: NotificationRequest) -> str:
        """Format message for SMS (160 character limit consideration)"""
        # Create concise message for SMS
        priority_emoji = {
            'LOW': '🔵',
            'MEDIUM': '🟡',
            'HIGH': '🟠',
            'CRITICAL': '🔴'
        }
        
        emoji = priority_emoji.get(notification.priority.value, '🔵')
        
        # Build message with character limit in mind
        message = f"{emoji} {notification.title}\n{notification.message}"
        
        # Add incident ID if available
        if notification.incident_id:
            message += f"\nID: {notification.incident_id}"
        
        # Add location if available in metadata
        if notification.metadata and 'location' in notification.metadata:
            message += f"\nLocation: {notification.metadata['location']}"
        
        # Truncate if too long (SMS limit is 160 chars for single message)
        if len(message) > 150:  # Leave some buffer
            message = message[:147] + "..."
        
        return message
    
    def _format_whatsapp_message(self, notification: NotificationRequest) -> str:
        """Format message for WhatsApp (richer formatting allowed)"""
        priority_emoji = {
            'LOW': '🔵',
            'MEDIUM': '🟡',
            'HIGH': '🟠',
            'CRITICAL': '🔴'
        }
        
        emoji = priority_emoji.get(notification.priority.value, '🔵')
        
        # Build rich WhatsApp message
        message = f"{emoji} *{notification.title}*\n\n{notification.message}"
        
        # Add incident details if available
        if notification.incident_id:
            message += f"\n\n📋 *Incident ID:* {notification.incident_id}"
        
        # Add metadata if available
        if notification.metadata:
            if 'location' in notification.metadata:
                message += f"\n📍 *Location:* {notification.metadata['location']}"
            
            if 'camera_id' in notification.metadata:
                message += f"\n📹 *Camera:* {notification.metadata['camera_id']}"
            
            if 'confidence' in notification.metadata:
                confidence = float(notification.metadata['confidence']) * 100
                message += f"\n🎯 *Confidence:* {confidence:.1f}%"
            
            if 'timestamp' in notification.metadata:
                message += f"\n⏰ *Time:* {notification.metadata['timestamp']}"
        
        # Add footer
        message += f"\n\n🚨 *Priority:* {notification.priority.value}"
        
        return message
    
    async def _get_user_phone_number(self, user_id: str) -> Optional[str]:
        """Get user's phone number (placeholder - would integrate with user service)"""
        # This is a placeholder implementation
        # In a real system, this would query a user service or database
        # to get the user's phone number
        
        # For testing purposes, return a mock number
        # In production, this should be replaced with actual phone number retrieval
        return f"+1555000{user_id[-4:].zfill(4)}"  # Mock phone number
    
    async def _get_user_whatsapp_number(self, user_id: str) -> Optional[str]:
        """Get user's WhatsApp number (placeholder - would integrate with user service)"""
        # This is a placeholder implementation
        # In many cases, WhatsApp number might be the same as phone number
        
        # For testing purposes, return a mock number
        # In production, this should be replaced with actual WhatsApp number retrieval
        return f"+1555000{user_id[-4:].zfill(4)}"  # Mock WhatsApp number
    
    async def send_bulk_sms(self, phone_numbers: list, message: str) -> Dict[str, Any]:
        """Send bulk SMS messages"""
        try:
            results = []
            
            for phone_number in phone_numbers:
                try:
                    message_obj = self.client.messages.create(
                        body=message,
                        from_=self.config.get('TWILIO_PHONE_NUMBER'),
                        to=phone_number
                    )
                    
                    results.append({
                        "phone_number": phone_number,
                        "success": True,
                        "message_id": message_obj.sid
                    })
                    
                except TwilioException as e:
                    results.append({
                        "phone_number": phone_number,
                        "success": False,
                        "error": str(e)
                    })
            
            successful = sum(1 for r in results if r["success"])
            
            return {
                "success": True,
                "total_sent": len(phone_numbers),
                "successful": successful,
                "failed": len(phone_numbers) - successful,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Bulk SMS error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }