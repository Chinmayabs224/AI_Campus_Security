"""
Startup script for the notification service
Runs both Flask HTTP API and WebSocket server
"""

import asyncio
import threading
import logging
import os
import signal
import sys
from dotenv import load_dotenv

# Import Flask app
from app import app

# Import WebSocket server
from websocket_server import WebSocketServer
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NotificationServiceRunner:
    """Runs both Flask HTTP API and WebSocket server"""
    
    def __init__(self):
        load_dotenv()
        
        # Configuration
        self.flask_host = os.getenv('FLASK_HOST', '0.0.0.0')
        self.flask_port = int(os.getenv('FLASK_PORT', 5000))
        self.websocket_host = os.getenv('WEBSOCKET_HOST', 'localhost')
        self.websocket_port = int(os.getenv('WEBSOCKET_PORT', 8765))
        
        # Initialize Redis
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )
        
        # Initialize WebSocket server
        self.websocket_server = WebSocketServer(
            self.redis_client,
            host=self.websocket_host,
            port=self.websocket_port
        )
        
        self.flask_thread = None
        self.websocket_loop = None
        self.running = False
    
    def run_flask_app(self):
        """Run Flask app in a separate thread"""
        try:
            logger.info(f"Starting Flask HTTP API on {self.flask_host}:{self.flask_port}")
            app.run(
                host=self.flask_host,
                port=self.flask_port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Flask app error: {str(e)}")
    
    async def run_websocket_server(self):
        """Run WebSocket server"""
        try:
            logger.info(f"Starting WebSocket server on ws://{self.websocket_host}:{self.websocket_port}")
            await self.websocket_server.start_server()
            
            # Keep server running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"WebSocket server error: {str(e)}")
        finally:
            await self.websocket_server.stop_server()
    
    def start(self):
        """Start both services"""
        try:
            self.running = True
            
            # Test Redis connection
            try:
                self.redis_client.ping()
                logger.info("Redis connection successful")
            except Exception as e:
                logger.error(f"Redis connection failed: {str(e)}")
                return
            
            # Start Flask app in a separate thread
            self.flask_thread = threading.Thread(target=self.run_flask_app, daemon=True)
            self.flask_thread.start()
            
            # Start WebSocket server in the main thread
            self.websocket_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.websocket_loop)
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Run WebSocket server
            self.websocket_loop.run_until_complete(self.run_websocket_server())
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Service startup error: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop both services"""
        logger.info("Shutting down notification service...")
        
        self.running = False
        
        # Stop WebSocket server
        if self.websocket_loop and not self.websocket_loop.is_closed():
            try:
                # Schedule stop_server coroutine
                future = asyncio.run_coroutine_threadsafe(
                    self.websocket_server.stop_server(),
                    self.websocket_loop
                )
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping WebSocket server: {str(e)}")
            
            try:
                self.websocket_loop.close()
            except Exception as e:
                logger.error(f"Error closing event loop: {str(e)}")
        
        # Flask app will stop when the main process exits
        logger.info("Notification service stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def health_check(self):
        """Perform health check"""
        try:
            # Check Redis
            self.redis_client.ping()
            
            # Check WebSocket server
            ws_running = self.websocket_server.running if self.websocket_server else False
            
            # Check Flask thread
            flask_running = self.flask_thread.is_alive() if self.flask_thread else False
            
            return {
                'redis': True,
                'websocket_server': ws_running,
                'flask_app': flask_running,
                'overall': ws_running and flask_running
            }
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                'redis': False,
                'websocket_server': False,
                'flask_app': False,
                'overall': False,
                'error': str(e)
            }

def main():
    """Main entry point"""
    runner = NotificationServiceRunner()
    
    try:
        logger.info("Starting AI Campus Security Notification Service")
        logger.info("=" * 50)
        
        # Perform initial health check
        health = runner.health_check()
        if not health.get('redis', False):
            logger.error("Redis connection failed. Please ensure Redis is running.")
            return 1
        
        # Start services
        runner.start()
        
    except Exception as e:
        logger.error(f"Failed to start notification service: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())