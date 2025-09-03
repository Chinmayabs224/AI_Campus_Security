#!/bin/bash

# Docker Build Script for Campus Security System
# This script builds all Docker images with proper tagging and optimization

set -e

PROJECT_NAME="campus-security"
REGISTRY=${DOCKER_REGISTRY:-""}
VERSION=${VERSION:-"latest"}

echo "🐳 Building Docker images for Campus Security System"

# Function to build and tag images
build_image() {
    local service=$1
    local context=$2
    local dockerfile=${3:-"Dockerfile"}
    
    echo "Building $service..."
    
    if [ -n "$REGISTRY" ]; then
        IMAGE_TAG="$REGISTRY/$PROJECT_NAME-$service:$VERSION"
        LATEST_TAG="$REGISTRY/$PROJECT_NAME-$service:latest"
    else
        IMAGE_TAG="$PROJECT_NAME-$service:$VERSION"
        LATEST_TAG="$PROJECT_NAME-$service:latest"
    fi
    
    docker build \
        --build-arg VERSION=$VERSION \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown") \
        -t $IMAGE_TAG \
        -t $LATEST_TAG \
        -f $context/$dockerfile \
        $context
    
    echo "✅ Built $service: $IMAGE_TAG"
}

# Build core API
build_image "core-api" "./core-api"

# Build microservices
build_image "privacy-service" "./microservices/privacy-service"
build_image "notification-service" "./microservices/notification-service"
build_image "compliance-service" "./microservices/compliance-service"

# Build frontend
build_image "frontend" "./frontend"

# Build edge services
build_image "edge-service" "./edge-services"

# Build monitoring images if they exist
if [ -d "./monitoring/custom-exporter" ]; then
    build_image "monitoring-exporter" "./monitoring/custom-exporter"
fi

echo "📊 Docker images built successfully!"

# List all built images
echo "Built images:"
docker images | grep $PROJECT_NAME

# Optional: Push to registry
if [ -n "$REGISTRY" ] && [ "$PUSH_IMAGES" = "true" ]; then
    echo "🚀 Pushing images to registry..."
    
    docker push $REGISTRY/$PROJECT_NAME-core-api:$VERSION
    docker push $REGISTRY/$PROJECT_NAME-core-api:latest
    
    docker push $REGISTRY/$PROJECT_NAME-privacy-service:$VERSION
    docker push $REGISTRY/$PROJECT_NAME-privacy-service:latest
    
    docker push $REGISTRY/$PROJECT_NAME-notification-service:$VERSION
    docker push $REGISTRY/$PROJECT_NAME-notification-service:latest
    
    docker push $REGISTRY/$PROJECT_NAME-compliance-service:$VERSION
    docker push $REGISTRY/$PROJECT_NAME-compliance-service:latest
    
    docker push $REGISTRY/$PROJECT_NAME-frontend:$VERSION
    docker push $REGISTRY/$PROJECT_NAME-frontend:latest
    
    docker push $REGISTRY/$PROJECT_NAME-edge-service:$VERSION
    docker push $REGISTRY/$PROJECT_NAME-edge-service:latest
    
    echo "✅ Images pushed to registry successfully!"
fi

# Create image manifest
cat > image-manifest.json << EOF
{
    "project": "$PROJECT_NAME",
    "version": "$VERSION",
    "build_date": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "git_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")",
    "images": {
        "core-api": "$PROJECT_NAME-core-api:$VERSION",
        "privacy-service": "$PROJECT_NAME-privacy-service:$VERSION",
        "notification-service": "$PROJECT_NAME-notification-service:$VERSION",
        "compliance-service": "$PROJECT_NAME-compliance-service:$VERSION",
        "frontend": "$PROJECT_NAME-frontend:$VERSION",
        "edge-service": "$PROJECT_NAME-edge-service:$VERSION"
    }
}
EOF

echo "📄 Image manifest saved to image-manifest.json"
echo "🎉 Docker build process completed successfully!"