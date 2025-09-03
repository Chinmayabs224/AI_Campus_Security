"""
Email provider for email notifications
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from models import NotificationRequest, NotificationChannel

logger = logging.getLogger(__name__)

class EmailProvider:
    """Email provider for email notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = None
        self._validate_config()
    
    def _validate_config(self):
        """Validate email configuration"""
        required_configs = ['SMTP_HOST', 'SMTP_PORT', 'SMTP_USERNAME', 'SMTP_PASSWORD', 'EMAIL_FROM']
        
        for config_key in required_configs:
            if not self.config.get(config_key):
                raise ValueError(f"Email configuration missing: {config_key}")
        
        logger.info("Email provider configuration validated")
    
    async def send(self, notification: NotificationRequest, channel: NotificationChannel) -> Dict[str, Any]:
        """Send email notification"""
        try:
            if channel != NotificationChannel.EMAIL:
                raise ValueError(f"Email provider only supports EMAIL channel, got {channel}")
            
            # Get user's email address
            email_address = await self._get_user_email(notification.user_id)
            
            if not email_address:
                return {
                    "success": False,
                    "error": "No email address found for user"
                }
            
            # Create email message
            message = self._create_email_message(notification, email_address)
            
            # Send email
            result = await self._send_email(message, email_address)
            
            return result
            
        except Exception as e:
            logger.error(f"Email send error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "provider": "email"
            }
    
    def _create_email_message(self, notification: NotificationRequest, to_email: str) -> MIMEMultipart:
        """Create email message from notification"""
        
        # Create message container
        message = MIMEMultipart("alternative")
        
        # Set headers
        message["Subject"] = self._format_email_subject(notification)
        message["From"] = self.config.get('EMAIL_FROM')
        message["To"] = to_email
        message["Date"] = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
        
        # Add custom headers for tracking
        message["X-Notification-ID"] = notification.notification_id
        message["X-Priority"] = notification.priority.value
        if notification.incident_id:
            message["X-Incident-ID"] = notification.incident_id
        
        # Create plain text version
        text_content = self._create_text_content(notification)
        text_part = MIMEText(text_content, "plain")
        
        # Create HTML version
        html_content = self._create_html_content(notification)
        html_part = MIMEText(html_content, "html")
        
        # Attach parts
        message.attach(text_part)
        message.attach(html_part)
        
        return message
    
    def _format_email_subject(self, notification: NotificationRequest) -> str:
        """Format email subject line"""
        priority_prefix = {
            'LOW': '[INFO]',
            'MEDIUM': '[ALERT]',
            'HIGH': '[URGENT]',
            'CRITICAL': '[CRITICAL]'
        }
        
        prefix = priority_prefix.get(notification.priority.value, '[ALERT]')
        
        # Include incident ID if available
        if notification.incident_id:
            return f"{prefix} {notification.title} - Incident #{notification.incident_id}"
        else:
            return f"{prefix} {notification.title}"
    
    def _create_text_content(self, notification: NotificationRequest) -> str:
        """Create plain text email content"""
        content = f"""
Campus Security Alert

{notification.title}

{notification.message}

Priority: {notification.priority.value}
"""
        
        # Add incident details
        if notification.incident_id:
            content += f"Incident ID: {notification.incident_id}\n"
        
        # Add metadata
        if notification.metadata:
            content += "\nAdditional Details:\n"
            
            if 'location' in notification.metadata:
                content += f"Location: {notification.metadata['location']}\n"
            
            if 'camera_id' in notification.metadata:
                content += f"Camera: {notification.metadata['camera_id']}\n"
            
            if 'confidence' in notification.metadata:
                confidence = float(notification.metadata['confidence']) * 100
                content += f"Detection Confidence: {confidence:.1f}%\n"
            
            if 'timestamp' in notification.metadata:
                content += f"Detection Time: {notification.metadata['timestamp']}\n"
        
        content += f"""
Notification ID: {notification.notification_id}
Sent: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

---
Campus Security System
Do not reply to this automated message.
"""
        
        return content
    
    def _create_html_content(self, notification: NotificationRequest) -> str:
        """Create HTML email content"""
        
        # Priority styling
        priority_colors = {
            'LOW': '#17a2b8',      # Info blue
            'MEDIUM': '#ffc107',   # Warning yellow
            'HIGH': '#fd7e14',     # Orange
            'CRITICAL': '#dc3545'  # Danger red
        }
        
        priority_color = priority_colors.get(notification.priority.value, '#ffc107')
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campus Security Alert</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: {priority_color}; color: white; padding: 20px; text-align: center; }}
        .content {{ background-color: #f8f9fa; padding: 20px; }}
        .priority-badge {{ 
            display: inline-block; 
            background-color: {priority_color}; 
            color: white; 
            padding: 5px 10px; 
            border-radius: 5px; 
            font-weight: bold; 
        }}
        .details {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid {priority_color}; }}
        .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
        .metadata {{ background-color: #e9ecef; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 Campus Security Alert</h1>
        </div>
        
        <div class="content">
            <h2>{notification.title}</h2>
            
            <div class="priority-badge">
                Priority: {notification.priority.value}
            </div>
            
            <div class="details">
                <p>{notification.message}</p>
            </div>
"""
        
        # Add incident details if available
        if notification.incident_id:
            html_content += f"""
            <div class="metadata">
                <strong>Incident ID:</strong> {notification.incident_id}
            </div>
"""
        
        # Add metadata
        if notification.metadata:
            html_content += '<div class="metadata"><strong>Additional Details:</strong><br>'
            
            if 'location' in notification.metadata:
                html_content += f"📍 <strong>Location:</strong> {notification.metadata['location']}<br>"
            
            if 'camera_id' in notification.metadata:
                html_content += f"📹 <strong>Camera:</strong> {notification.metadata['camera_id']}<br>"
            
            if 'confidence' in notification.metadata:
                confidence = float(notification.metadata['confidence']) * 100
                html_content += f"🎯 <strong>Detection Confidence:</strong> {confidence:.1f}%<br>"
            
            if 'timestamp' in notification.metadata:
                html_content += f"⏰ <strong>Detection Time:</strong> {notification.metadata['timestamp']}<br>"
            
            html_content += '</div>'
        
        html_content += f"""
        </div>
        
        <div class="footer">
            <p>Notification ID: {notification.notification_id}</p>
            <p>Sent: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <hr>
            <p>Campus Security System - Do not reply to this automated message.</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    async def _send_email(self, message: MIMEMultipart, to_email: str) -> Dict[str, Any]:
        """Send email via SMTP"""
        try:
            # Create SMTP connection
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.config.get('SMTP_HOST'), self.config.get('SMTP_PORT')) as server:
                if self.config.get('SMTP_USE_TLS', True):
                    server.starttls(context=context)
                
                server.login(self.config.get('SMTP_USERNAME'), self.config.get('SMTP_PASSWORD'))
                
                # Send email
                text = message.as_string()
                server.sendmail(self.config.get('EMAIL_FROM'), to_email, text)
            
            logger.info(f"Email sent successfully to {to_email}")
            
            return {
                "success": True,
                "provider": "email",
                "to": to_email,
                "message_id": message.get("Message-ID", "unknown")
            }
            
        except Exception as e:
            logger.error(f"SMTP send error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "provider": "email",
                "to": to_email
            }
    
    async def _get_user_email(self, user_id: str) -> Optional[str]:
        """Get user's email address (placeholder - would integrate with user service)"""
        # This is a placeholder implementation
        # In a real system, this would query a user service or database
        # to get the user's email address
        
        # For testing purposes, return a mock email
        # In production, this should be replaced with actual email retrieval
        return f"user_{user_id}@campus.edu"
    
    async def send_bulk_email(self, email_addresses: List[str], notification: NotificationRequest) -> Dict[str, Any]:
        """Send bulk email notifications"""
        try:
            results = []
            
            for email_address in email_addresses:
                try:
                    message = self._create_email_message(notification, email_address)
                    result = await self._send_email(message, email_address)
                    results.append({
                        "email": email_address,
                        "success": result["success"],
                        "message_id": result.get("message_id"),
                        "error": result.get("error")
                    })
                    
                except Exception as e:
                    results.append({
                        "email": email_address,
                        "success": False,
                        "error": str(e)
                    })
            
            successful = sum(1 for r in results if r["success"])
            
            return {
                "success": True,
                "total_sent": len(email_addresses),
                "successful": successful,
                "failed": len(email_addresses) - successful,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Bulk email error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }