"""
Configuration settings for the notification service
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for notification service"""
    
    # Redis Configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # Firebase Cloud Messaging Configuration
    FCM_CREDENTIALS_PATH = os.getenv('FCM_CREDENTIALS_PATH', 'firebase-credentials.json')
    FCM_PROJECT_ID = os.getenv('FCM_PROJECT_ID')
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
    
    # Email Configuration
    SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
    EMAIL_FROM = os.getenv('EMAIL_FROM', 'security@campus.edu')
    
    # Rate Limiting Configuration
    RATE_LIMIT_PER_USER_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_USER_PER_MINUTE', 10))
    RATE_LIMIT_PER_USER_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_USER_PER_HOUR', 100))
    
    # Notification Configuration
    MAX_RETRY_ATTEMPTS = int(os.getenv('MAX_RETRY_ATTEMPTS', 3))
    RETRY_DELAY_SECONDS = int(os.getenv('RETRY_DELAY_SECONDS', 5))
    
    # Security Configuration
    API_KEY = os.getenv('NOTIFICATION_API_KEY')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        required_configs = []
        
        # Check FCM configuration
        if not cls.FCM_PROJECT_ID:
            required_configs.append('FCM_PROJECT_ID')
        
        # Check Twilio configuration
        if not cls.TWILIO_ACCOUNT_SID:
            required_configs.append('TWILIO_ACCOUNT_SID')
        if not cls.TWILIO_AUTH_TOKEN:
            required_configs.append('TWILIO_AUTH_TOKEN')
        if not cls.TWILIO_PHONE_NUMBER:
            required_configs.append('TWILIO_PHONE_NUMBER')
        
        # Check email configuration
        if not cls.SMTP_USERNAME:
            required_configs.append('SMTP_USERNAME')
        if not cls.SMTP_PASSWORD:
            required_configs.append('SMTP_PASSWORD')
        
        if required_configs:
            raise ValueError(f"Missing required configuration: {', '.join(required_configs)}")
        
        return True