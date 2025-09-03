"""
Example WebSocket client for testing real-time alert distribution
"""

import asyncio
import json
import websockets
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertWebSocketClient:
    """Example WebSocket client for receiving real-time alerts"""
    
    def __init__(self, uri: str, user_id: str, auth_token: str):
        self.uri = uri
        self.user_id = user_id
        self.auth_token = auth_token
        self.websocket = None
        self.running = False
    
    async def connect(self, connection_type: str = "dashboard", subscriptions: dict = None):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            
            # Send authentication message
            auth_message = {
                "type": "auth",
                "user_id": self.user_id,
                "token": self.auth_token,
                "connection_type": connection_type,
                "subscriptions": subscriptions or {
                    "locations": ["Building A", "Building B"],
                    "incident_types": ["intrusion", "loitering", "crowding"]
                }
            }
            
            await self.websocket.send(json.dumps(auth_message))
            logger.info(f"Connected to WebSocket server as {self.user_id}")
            
            self.running = True
            
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            raise
    
    async def listen_for_alerts(self):
        """Listen for incoming alerts"""
        try:
            async for message in self.websocket:
                await self.handle_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.running = False
            
        except Exception as e:
            logger.error(f"Error listening for alerts: {str(e)}")
            self.running = False
    
    async def handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'connection_established':
                logger.info(f"Connection established: {data.get('connection_id')}")
                
            elif message_type == 'security_alert':
                await self.handle_security_alert(data)
                
            elif message_type == 'alert_acknowledged':
                logger.info(f"Alert {data.get('alert_id')} acknowledged by {data.get('acknowledged_by')}")
                
            elif message_type == 'alert_escalation':
                await self.handle_alert_escalation(data)
                
            elif message_type == 'active_alerts':
                logger.info(f"Received {len(data.get('alerts', []))} active alerts")
                
            elif message_type == 'pong':
                logger.debug("Received pong")
                
            else:
                logger.info(f"Received message: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
    
    async def handle_security_alert(self, data: dict):
        """Handle security alert"""
        alert = data.get('alert', {})
        
        logger.warning(f"""
🚨 SECURITY ALERT 🚨
Alert ID: {alert.get('alert_id')}
Incident: {alert.get('incident_id')}
Title: {alert.get('title')}
Message: {alert.get('message')}
Priority: {alert.get('priority')}
Location: {alert.get('location')}
Type: {alert.get('incident_type')}
Time: {alert.get('timestamp')}
""")
        
        # Simulate acknowledgment after 10 seconds for testing
        if alert.get('priority') in ['HIGH', 'CRITICAL']:
            asyncio.create_task(self.auto_acknowledge(alert.get('alert_id'), 10))
    
    async def handle_alert_escalation(self, data: dict):
        """Handle alert escalation"""
        logger.critical(f"""
🔥 ALERT ESCALATION 🔥
Alert ID: {data.get('alert_id')}
Escalation Level: {data.get('escalation_level')}
Time: {data.get('timestamp')}
""")
    
    async def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        try:
            message = {
                "type": "acknowledge_alert",
                "alert_id": alert_id
            }
            
            await self.websocket.send(json.dumps(message))
            logger.info(f"Acknowledged alert: {alert_id}")
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {str(e)}")
    
    async def auto_acknowledge(self, alert_id: str, delay: int):
        """Auto-acknowledge alert after delay (for testing)"""
        await asyncio.sleep(delay)
        await self.acknowledge_alert(alert_id)
    
    async def send_ping(self):
        """Send ping to keep connection alive"""
        try:
            message = {
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending ping: {str(e)}")
    
    async def get_active_alerts(self):
        """Request active alerts"""
        try:
            message = {
                "type": "get_active_alerts"
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error requesting active alerts: {str(e)}")
    
    async def update_subscriptions(self, subscriptions: dict):
        """Update alert subscriptions"""
        try:
            message = {
                "type": "update_subscriptions",
                "subscriptions": subscriptions
            }
            
            await self.websocket.send(json.dumps(message))
            logger.info("Updated subscriptions")
            
        except Exception as e:
            logger.error(f"Error updating subscriptions: {str(e)}")
    
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.running = False
            logger.info("Disconnected from WebSocket server")

async def run_dashboard_client():
    """Run dashboard client example"""
    client = AlertWebSocketClient(
        uri="ws://localhost:8765",
        user_id="dashboard_user",
        auth_token="valid_dashboard_token"
    )
    
    try:
        await client.connect("dashboard", {
            "locations": ["Building A", "Building B", "Parking Lot"],
            "incident_types": ["intrusion", "loitering", "crowding", "abandoned_object"]
        })
        
        # Start ping task
        async def ping_task():
            while client.running:
                await asyncio.sleep(30)
                if client.running:
                    await client.send_ping()
        
        ping_coroutine = asyncio.create_task(ping_task())
        
        # Request active alerts
        await client.get_active_alerts()
        
        # Listen for alerts
        await client.listen_for_alerts()
        
    except KeyboardInterrupt:
        logger.info("Shutting down client...")
        
    finally:
        await client.disconnect()

async def run_mobile_client():
    """Run mobile client example"""
    client = AlertWebSocketClient(
        uri="ws://localhost:8765",
        user_id="security_guard_1",
        auth_token="valid_mobile_token"
    )
    
    try:
        await client.connect("mobile", {
            "locations": ["Building A"],
            "incident_types": ["intrusion", "violence", "fire", "medical_emergency"]
        })
        
        # Listen for alerts
        await client.listen_for_alerts()
        
    except KeyboardInterrupt:
        logger.info("Shutting down mobile client...")
        
    finally:
        await client.disconnect()

async def run_admin_client():
    """Run admin client example"""
    client = AlertWebSocketClient(
        uri="ws://localhost:8765",
        user_id="security_supervisor",
        auth_token="valid_admin_token"
    )
    
    try:
        await client.connect("admin", {
            "locations": [],  # Empty means all locations
            "incident_types": []  # Empty means all incident types
        })
        
        # Send test alert after 5 seconds
        async def send_test_alert():
            await asyncio.sleep(5)
            
            test_alert = {
                "type": "send_test_alert",
                "title": "Test Security Alert",
                "message": "This is a test alert from admin client",
                "priority": "HIGH",
                "incident_type": "INTRUSION",
                "location": "Building A - Floor 2",
                "target_user": "security_guard_1",
                "metadata": {
                    "camera_id": "CAM_001",
                    "confidence": 0.95
                }
            }
            
            await client.websocket.send(json.dumps(test_alert))
            logger.info("Sent test alert")
        
        asyncio.create_task(send_test_alert())
        
        # Listen for alerts and escalations
        await client.listen_for_alerts()
        
    except KeyboardInterrupt:
        logger.info("Shutting down admin client...")
        
    finally:
        await client.disconnect()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        client_type = sys.argv[1]
        
        if client_type == "dashboard":
            asyncio.run(run_dashboard_client())
        elif client_type == "mobile":
            asyncio.run(run_mobile_client())
        elif client_type == "admin":
            asyncio.run(run_admin_client())
        else:
            print("Usage: python websocket_client_example.py [dashboard|mobile|admin]")
    else:
        print("Running dashboard client by default...")
        asyncio.run(run_dashboard_client())