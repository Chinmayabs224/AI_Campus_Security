#!/bin/bash

# Local Deployment Script for Campus Security System
# This script sets up the complete system for local development and testing

set -e

PROJECT_NAME="campus-security"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"

echo "🏠 Starting local deployment of Campus Security System"

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    echo "📝 Creating environment file from template..."
    cp .env.production $ENV_FILE
    
    # Generate secure passwords and keys
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    MINIO_ACCESS_KEY=$(openssl rand -base64 16)
    MINIO_SECRET_KEY=$(openssl rand -base64 32)
    JWT_SECRET_KEY=$(openssl rand -base64 32)
    ENCRYPTION_KEY=$(openssl rand -base64 32)
    GRAFANA_PASSWORD=$(openssl rand -base64 16)
    
    # Update environment file with generated values
    sed -i.bak "s/your-secure-password/$POSTGRES_PASSWORD/g" $ENV_FILE
    sed -i.bak "s/your-minio-access-key/$MINIO_ACCESS_KEY/g" $ENV_FILE
    sed -i.bak "s/your-minio-secret-key/$MINIO_SECRET_KEY/g" $ENV_FILE
    sed -i.bak "s/your-jwt-secret-key-minimum-32-characters/$JWT_SECRET_KEY/g" $ENV_FILE
    sed -i.bak "s/your-encryption-key-32-characters/$ENCRYPTION_KEY/g" $ENV_FILE
    sed -i.bak "s/your-grafana-password/$GRAFANA_PASSWORD/g" $ENV_FILE
    
    # Update CORS origins for local development
    sed -i.bak 's/\["https:\/\/your-domain.com", "https:\/\/www.your-domain.com"\]/\["http:\/\/localhost:3000", "http:\/\/localhost:80", "http:\/\/localhost"\]/g' $ENV_FILE
    sed -i.bak 's/\["your-domain.com", "www.your-domain.com"\]/\["localhost", "127.0.0.1"\]/g' $ENV_FILE
    
    rm -f $ENV_FILE.bak
    
    echo "✅ Environment file created with secure generated values"
    echo "📝 Please review and update $ENV_FILE with your specific configuration"
fi

# Generate SSL certificates for local development
echo "🔐 Generating SSL certificates for local development..."
if [ ! -f "deployment/nginx/ssl/cert.pem" ]; then
    mkdir -p deployment/nginx/ssl
    
    # Generate self-signed certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout deployment/nginx/ssl/key.pem \
        -out deployment/nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    
    chmod 600 deployment/nginx/ssl/key.pem
    chmod 644 deployment/nginx/ssl/cert.pem
    
    echo "✅ SSL certificates generated"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs/nginx
mkdir -p models
mkdir -p data/uploads
mkdir -p data/evidence

# Download YOLO model if not present
if [ ! -f "models/yolov8n.pt" ]; then
    echo "📥 Downloading YOLO model..."
    if command -v wget &> /dev/null; then
        wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    elif command -v curl &> /dev/null; then
        curl -L -o models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    else
        echo "⚠️  Please download yolov8n.pt manually to the models/ directory"
    fi
fi

# Build Docker images
echo "🐳 Building Docker images..."
./deployment/docker-build.sh

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down --remove-orphans || true

# Start the services
echo "🚀 Starting services..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
services=("core-api" "privacy-service" "notification-service" "compliance-service" "frontend" "postgres" "redis" "minio")

for service in "${services[@]}"; do
    if docker-compose -f $COMPOSE_FILE ps $service | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service is not running"
        docker-compose -f $COMPOSE_FILE logs $service
    fi
done

# Initialize database
echo "🗄️ Initializing database..."
docker-compose -f $COMPOSE_FILE exec -T core-api python -c "
from core.database import database_manager
import asyncio

async def init_db():
    await database_manager.connect()
    # Run any database initialization here
    await database_manager.disconnect()

asyncio.run(init_db())
" || echo "⚠️  Database initialization may need manual setup"

# Create MinIO buckets
echo "🪣 Setting up MinIO buckets..."
docker-compose -f $COMPOSE_FILE exec -T minio mc alias set local http://localhost:9000 $MINIO_ACCESS_KEY $MINIO_SECRET_KEY || true
docker-compose -f $COMPOSE_FILE exec -T minio mc mb local/evidence || true
docker-compose -f $COMPOSE_FILE exec -T minio mc mb local/models || true

# Display access information
echo ""
echo "🎉 Campus Security System deployed successfully!"
echo ""
echo "📊 Service Access URLs:"
echo "  🌐 Frontend:              http://localhost (or https://localhost)"
echo "  🔧 Core API:              http://localhost:8000"
echo "  🔒 Privacy Service:       http://localhost:5001"
echo "  📢 Notification Service:  http://localhost:5002"
echo "  📋 Compliance Service:    http://localhost:5003"
echo "  🎯 Edge Service:          http://localhost:8080"
echo ""
echo "🛠️  Management Tools:"
echo "  📈 Grafana:               http://localhost:3000 (admin/$(grep GRAFANA_PASSWORD $ENV_FILE | cut -d'=' -f2))"
echo "  📊 Prometheus:            http://localhost:9090"
echo "  🗄️  MinIO Console:         http://localhost:9001 ($(grep MINIO_ACCESS_KEY $ENV_FILE | cut -d'=' -f2)/$(grep MINIO_SECRET_KEY $ENV_FILE | cut -d'=' -f2))"
echo ""
echo "🔍 Health Checks:"
echo "  Core API:    curl http://localhost:8000/health"
echo "  Privacy:     curl http://localhost:5001/health"
echo "  Notification: curl http://localhost:5002/health"
echo "  Compliance:  curl http://localhost:5003/health"
echo ""
echo "📝 Logs:"
echo "  View all logs:     docker-compose -f $COMPOSE_FILE logs -f"
echo "  View service logs: docker-compose -f $COMPOSE_FILE logs -f [service-name]"
echo ""
echo "🛑 To stop the system:"
echo "  docker-compose -f $COMPOSE_FILE down"
echo ""
echo "⚠️  Important Notes:"
echo "  - This is a local development setup with self-signed certificates"
echo "  - Update $ENV_FILE with your actual service credentials"
echo "  - For production deployment, use the AWS deployment scripts"
echo "  - Configure your cameras to point to the edge service endpoints"
echo ""

# Save deployment info
cat > deployment-info.txt << EOF
Campus Security System - Local Deployment
Deployed: $(date)
Compose File: $COMPOSE_FILE
Environment File: $ENV_FILE

Service URLs:
- Frontend: http://localhost
- Core API: http://localhost:8000
- Privacy Service: http://localhost:5001
- Notification Service: http://localhost:5002
- Compliance Service: http://localhost:5003
- Edge Service: http://localhost:8080

Management:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- MinIO: http://localhost:9001

Commands:
- View logs: docker-compose -f $COMPOSE_FILE logs -f
- Stop system: docker-compose -f $COMPOSE_FILE down
- Restart: docker-compose -f $COMPOSE_FILE restart
- Update: ./deployment/docker-build.sh && docker-compose -f $COMPOSE_FILE up -d
EOF

echo "📄 Deployment information saved to deployment-info.txt"
echo "✅ Local deployment completed successfully!"