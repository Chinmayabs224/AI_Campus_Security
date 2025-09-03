"""
WebSocket server for real-time alert distribution
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import websockets
import redis
from websockets.exceptions import ConnectionClosed, WebSocketException

from websocket_manager import WebSocketManager, ConnectionType, AlertMessage
from alert_router import AlertRouter
from models import NotificationPriority, IncidentType

logger = logging.getLogger(__name__)

class WebSocketServer:
    """WebSocket server for real-time notifications"""
    
    def __init__(self, redis_client: redis.Redis, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.redis = redis_client
        self.websocket_manager = WebSocketManager(redis_client)
        self.alert_router = AlertRouter(redis_client, self.websocket_manager)
        self.server = None
        self.running = False
    
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.running = True
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Start cleanup task
            asyncio.create_task(self.periodic_cleanup())
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.running = False
            logger.info("WebSocket server stopped")
    
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        connection_id = None
        
        try:
            # Wait for authentication message
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            auth_data = json.loads(auth_message)
            
            # Validate authentication
            user_id = await self.authenticate_connection(auth_data)
            if not user_id:
                await websocket.send(json.dumps({
                    'type': 'auth_error',
                    'message': 'Authentication failed'
                }))
                return
            
            # Determine connection type
            connection_type = ConnectionType(auth_data.get('connection_type', 'dashboard'))
            
            # Get subscriptions
            subscriptions = auth_data.get('subscriptions', {})
            
            # Register connection
            connection_id = await self.websocket_manager.connect(
                websocket, user_id, connection_type, subscriptions
            )
            
            logger.info(f"WebSocket connection established: {connection_id} for user {user_id}")
            
            # Handle messages
            async for message in websocket:
                await self.handle_message(connection_id, message)
                
        except asyncio.TimeoutError:
            logger.warning("WebSocket authentication timeout")
            await websocket.send(json.dumps({
                'type': 'auth_timeout',
                'message': 'Authentication timeout'
            }))
            
        except ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
            
        except WebSocketException as e:
            logger.error(f"WebSocket error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket connection: {str(e)}")
            
        finally:
            if connection_id:
                await self.websocket_manager.disconnect(connection_id)
    
    async def authenticate_connection(self, auth_data: Dict[str, Any]) -> Optional[str]:
        """Authenticate WebSocket connection"""
        try:
            # Extract authentication info
            token = auth_data.get('token')
            user_id = auth_data.get('user_id')
            
            if not token or not user_id:
                return None
            
            # Validate token (placeholder - would integrate with auth service)
            if await self.validate_auth_token(token, user_id):
                return user_id
            
            return None
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return None
    
    async def validate_auth_token(self, token: str, user_id: str) -> bool:
        """Validate authentication token (placeholder)"""
        # This is a placeholder implementation
        # In a real system, this would validate JWT tokens or session tokens
        # against the authentication service
        
        # For testing, accept any token that starts with "valid_"
        return token.startswith("valid_")
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'ping':
                await self.handle_ping(connection_id)
                
            elif message_type == 'acknowledge_alert':
                await self.handle_alert_acknowledgment(connection_id, data)
                
            elif message_type == 'update_subscriptions':
                await self.handle_subscription_update(connection_id, data)
                
            elif message_type == 'get_active_alerts':
                await self.handle_get_active_alerts(connection_id)
                
            elif message_type == 'send_test_alert':
                await self.handle_test_alert(connection_id, data)
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from connection {connection_id}")
            
        except Exception as e:
            logger.error(f"Error handling message from connection {connection_id}: {str(e)}")
    
    async def handle_ping(self, connection_id: str):
        """Handle ping message"""
        response = {
            'type': 'pong',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.websocket_manager._send_to_connection(connection_id, response)
    
    async def handle_alert_acknowledgment(self, connection_id: str, data: Dict[str, Any]):
        """Handle alert acknowledgment"""
        try:
            alert_id = data.get('alert_id')
            
            if not alert_id:
                return
            
            # Get user ID from connection
            if connection_id not in self.websocket_manager.connections:
                return
            
            connection = self.websocket_manager.connections[connection_id]
            user_id = connection.user_id
            
            # Acknowledge alert
            success = await self.alert_router.acknowledge_alert_escalation(alert_id, user_id)
            
            # Send acknowledgment response
            response = {
                'type': 'alert_acknowledged_response',
                'alert_id': alert_id,
                'success': success,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.websocket_manager._send_to_connection(connection_id, response)
            
        except Exception as e:
            logger.error(f"Error handling alert acknowledgment: {str(e)}")
    
    async def handle_subscription_update(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription update"""
        try:
            subscriptions = data.get('subscriptions', {})
            
            success = await self.websocket_manager.update_connection_subscriptions(
                connection_id, subscriptions
            )
            
            response = {
                'type': 'subscription_updated',
                'success': success,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.websocket_manager._send_to_connection(connection_id, response)
            
        except Exception as e:
            logger.error(f"Error handling subscription update: {str(e)}")
    
    async def handle_get_active_alerts(self, connection_id: str):
        """Handle request for active alerts"""
        try:
            if connection_id not in self.websocket_manager.connections:
                return
            
            connection = self.websocket_manager.connections[connection_id]
            user_id = connection.user_id
            
            active_alerts = await self.websocket_manager.get_active_alerts(user_id)
            
            response = {
                'type': 'active_alerts',
                'alerts': active_alerts,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.websocket_manager._send_to_connection(connection_id, response)
            
        except Exception as e:
            logger.error(f"Error handling get active alerts: {str(e)}")
    
    async def handle_test_alert(self, connection_id: str, data: Dict[str, Any]):
        """Handle test alert creation (for testing purposes)"""
        try:
            if connection_id not in self.websocket_manager.connections:
                return
            
            connection = self.websocket_manager.connections[connection_id]
            
            # Only allow admin connections to send test alerts
            if connection.connection_type != ConnectionType.ADMIN:
                return
            
            # Create test alert
            test_alert = AlertMessage(
                alert_id=f"test_{datetime.utcnow().timestamp()}",
                incident_id=data.get('incident_id', 'test_incident'),
                user_id=data.get('target_user', connection.user_id),
                title=data.get('title', 'Test Alert'),
                message=data.get('message', 'This is a test alert'),
                priority=NotificationPriority(data.get('priority', 'MEDIUM')),
                incident_type=IncidentType(data.get('incident_type', 'SYSTEM_ALERT')),
                location=data.get('location', 'Test Location'),
                timestamp=datetime.utcnow(),
                metadata=data.get('metadata', {})
            )
            
            # Distribute test alert
            result = await self.websocket_manager.distribute_alert(test_alert)
            
            response = {
                'type': 'test_alert_sent',
                'alert_id': test_alert.alert_id,
                'result': result,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.websocket_manager._send_to_connection(connection_id, response)
            
        except Exception as e:
            logger.error(f"Error handling test alert: {str(e)}")
    
    async def broadcast_system_message(self, message: str, message_type: str = "system"):
        """Broadcast system message to all connections"""
        try:
            system_message = {
                'type': message_type,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            for connection_id in self.websocket_manager.connections:
                await self.websocket_manager._send_to_connection(connection_id, system_message)
                
        except Exception as e:
            logger.error(f"Error broadcasting system message: {str(e)}")
    
    async def periodic_cleanup(self):
        """Periodic cleanup of expired alerts and connections"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up expired alerts
                await self.websocket_manager.cleanup_expired_alerts()
                
                # Clean up dead connections (connections that haven't pinged recently)
                current_time = datetime.utcnow()
                dead_connections = []
                
                for connection_id, connection in self.websocket_manager.connections.items():
                    if (current_time - connection.last_ping).total_seconds() > 300:  # 5 minutes
                        dead_connections.append(connection_id)
                
                for connection_id in dead_connections:
                    await self.websocket_manager.disconnect(connection_id)
                    logger.info(f"Cleaned up dead connection: {connection_id}")
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {str(e)}")
    
    async def get_server_stats(self) -> Dict[str, Any]:
        """Get WebSocket server statistics"""
        try:
            ws_stats = await self.websocket_manager.get_connection_stats()
            escalation_stats = await self.alert_router.get_escalation_stats()
            
            return {
                'server_running': self.running,
                'websocket_stats': ws_stats,
                'escalation_stats': escalation_stats,
                'server_info': {
                    'host': self.host,
                    'port': self.port
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting server stats: {str(e)}")
            return {}

# Standalone WebSocket server runner
async def run_websocket_server():
    """Run the WebSocket server as a standalone service"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize Redis
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True
    )
    
    # Create and start server
    server = WebSocketServer(
        redis_client,
        host=os.getenv('WEBSOCKET_HOST', 'localhost'),
        port=int(os.getenv('WEBSOCKET_PORT', 8765))
    )
    
    try:
        await server.start_server()
        
        # Keep server running
        while server.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down WebSocket server...")
        await server.stop_server()
    except Exception as e:
        logger.error(f"WebSocket server error: {str(e)}")
        await server.stop_server()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    asyncio.run(run_websocket_server())