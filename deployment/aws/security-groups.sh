#!/bin/bash

# Security Groups Setup Script for Campus Security System
# This script creates comprehensive security groups for different service tiers

set -e

REGION=${AWS_REGION:-us-east-1}
VPC_ID=${VPC_ID:-""}
PROJECT_NAME="campus-security"

if [ -z "$VPC_ID" ]; then
    echo "❌ VPC_ID environment variable is required"
    exit 1
fi

echo "🔒 Creating security groups for Campus Security System"

# Web/Application Tier Security Group
echo "Creating web tier security group..."
WEB_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-web-sg \
    --description "Security group for web and application services" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow HTTP and HTTPS from anywhere
aws ec2 authorize-security-group-ingress \
    --group-id $WEB_SG_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $REGION

aws ec2 authorize-security-group-ingress \
    --group-id $WEB_SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Allow SSH from management networks only
aws ec2 authorize-security-group-ingress \
    --group-id $WEB_SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 10.0.0.0/8 \
    --region $REGION

# Allow internal communication between web services
aws ec2 authorize-security-group-ingress \
    --group-id $WEB_SG_ID \
    --protocol tcp \
    --port 8000 \
    --source-group $WEB_SG_ID \
    --region $REGION

aws ec2 authorize-security-group-ingress \
    --group-id $WEB_SG_ID \
    --protocol tcp \
    --port 5000-5003 \
    --source-group $WEB_SG_ID \
    --region $REGION

# Database Tier Security Group
echo "Creating database tier security group..."
DB_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-db-sg \
    --description "Security group for database services" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow PostgreSQL from web tier only
aws ec2 authorize-security-group-ingress \
    --group-id $DB_SG_ID \
    --protocol tcp \
    --port 5432 \
    --source-group $WEB_SG_ID \
    --region $REGION

# Allow Redis from web tier only
aws ec2 authorize-security-group-ingress \
    --group-id $DB_SG_ID \
    --protocol tcp \
    --port 6379 \
    --source-group $WEB_SG_ID \
    --region $REGION

# Edge Services Security Group
echo "Creating edge services security group..."
EDGE_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-edge-sg \
    --description "Security group for edge inference services" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow edge service communication from web tier
aws ec2 authorize-security-group-ingress \
    --group-id $EDGE_SG_ID \
    --protocol tcp \
    --port 8080 \
    --source-group $WEB_SG_ID \
    --region $REGION

# Allow SSH for management
aws ec2 authorize-security-group-ingress \
    --group-id $EDGE_SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 10.0.0.0/8 \
    --region $REGION

# Allow RTSP streams from camera networks
aws ec2 authorize-security-group-ingress \
    --group-id $EDGE_SG_ID \
    --protocol tcp \
    --port 554 \
    --cidr 192.168.0.0/16 \
    --region $REGION

# Load Balancer Security Group
echo "Creating load balancer security group..."
LB_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-lb-sg \
    --description "Security group for load balancer" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow HTTP and HTTPS from anywhere
aws ec2 authorize-security-group-ingress \
    --group-id $LB_SG_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $REGION

aws ec2 authorize-security-group-ingress \
    --group-id $LB_SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Monitoring Security Group
echo "Creating monitoring security group..."
MONITOR_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-monitor-sg \
    --description "Security group for monitoring services" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow Prometheus from web tier
aws ec2 authorize-security-group-ingress \
    --group-id $MONITOR_SG_ID \
    --protocol tcp \
    --port 9090 \
    --source-group $WEB_SG_ID \
    --region $REGION

# Allow Grafana from management networks
aws ec2 authorize-security-group-ingress \
    --group-id $MONITOR_SG_ID \
    --protocol tcp \
    --port 3000 \
    --cidr 10.0.0.0/8 \
    --region $REGION

# Allow node exporter from monitoring
aws ec2 authorize-security-group-ingress \
    --group-id $MONITOR_SG_ID \
    --protocol tcp \
    --port 9100 \
    --source-group $MONITOR_SG_ID \
    --region $REGION

# Management Security Group
echo "Creating management security group..."
MGMT_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-mgmt-sg \
    --description "Security group for management and bastion hosts" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow SSH from specific IP ranges (update with your management IPs)
aws ec2 authorize-security-group-ingress \
    --group-id $MGMT_SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Allow VPN access
aws ec2 authorize-security-group-ingress \
    --group-id $MGMT_SG_ID \
    --protocol udp \
    --port 1194 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Update web security group to allow SSH from management
aws ec2 authorize-security-group-ingress \
    --group-id $WEB_SG_ID \
    --protocol tcp \
    --port 22 \
    --source-group $MGMT_SG_ID \
    --region $REGION

# Update edge security group to allow SSH from management
aws ec2 authorize-security-group-ingress \
    --group-id $EDGE_SG_ID \
    --protocol tcp \
    --port 22 \
    --source-group $MGMT_SG_ID \
    --region $REGION

# Tag all security groups
aws ec2 create-tags \
    --resources $WEB_SG_ID $DB_SG_ID $EDGE_SG_ID $LB_SG_ID $MONITOR_SG_ID $MGMT_SG_ID \
    --tags Key=Project,Value=$PROJECT_NAME Key=Environment,Value=production \
    --region $REGION

# Create security group configuration file
cat > security-groups-config.json << EOF
{
    "project": "$PROJECT_NAME",
    "vpc_id": "$VPC_ID",
    "region": "$REGION",
    "security_groups": {
        "web": {
            "id": "$WEB_SG_ID",
            "name": "${PROJECT_NAME}-web-sg",
            "description": "Web and application services",
            "ports": [80, 443, 22, 8000, "5000-5003"]
        },
        "database": {
            "id": "$DB_SG_ID",
            "name": "${PROJECT_NAME}-db-sg",
            "description": "Database services",
            "ports": [5432, 6379]
        },
        "edge": {
            "id": "$EDGE_SG_ID",
            "name": "${PROJECT_NAME}-edge-sg",
            "description": "Edge inference services",
            "ports": [8080, 22, 554]
        },
        "load_balancer": {
            "id": "$LB_SG_ID",
            "name": "${PROJECT_NAME}-lb-sg",
            "description": "Load balancer",
            "ports": [80, 443]
        },
        "monitoring": {
            "id": "$MONITOR_SG_ID",
            "name": "${PROJECT_NAME}-monitor-sg",
            "description": "Monitoring services",
            "ports": [9090, 3000, 9100]
        },
        "management": {
            "id": "$MGMT_SG_ID",
            "name": "${PROJECT_NAME}-mgmt-sg",
            "description": "Management and bastion hosts",
            "ports": [22, 1194]
        }
    }
}
EOF

# Output summary
cat > security-groups-summary.txt << EOF
🔒 Security Groups Created Successfully!

Security Group IDs:
- Web Tier: $WEB_SG_ID
- Database Tier: $DB_SG_ID
- Edge Services: $EDGE_SG_ID
- Load Balancer: $LB_SG_ID
- Monitoring: $MONITOR_SG_ID
- Management: $MGMT_SG_ID

Access Rules Summary:
Web Tier ($WEB_SG_ID):
- HTTP (80) from anywhere
- HTTPS (443) from anywhere
- SSH (22) from management SG
- Internal communication (8000, 5000-5003)

Database Tier ($DB_SG_ID):
- PostgreSQL (5432) from web tier only
- Redis (6379) from web tier only

Edge Services ($EDGE_SG_ID):
- Edge API (8080) from web tier
- SSH (22) from management SG
- RTSP (554) from camera networks

Load Balancer ($LB_SG_ID):
- HTTP (80) from anywhere
- HTTPS (443) from anywhere

Monitoring ($MONITOR_SG_ID):
- Prometheus (9090) from web tier
- Grafana (3000) from management networks
- Node Exporter (9100) internal

Management ($MGMT_SG_ID):
- SSH (22) from anywhere (restrict as needed)
- VPN (1194) from anywhere

Environment Variables:
export WEB_SG_ID=$WEB_SG_ID
export DB_SG_ID=$DB_SG_ID
export EDGE_SG_ID=$EDGE_SG_ID
export LB_SG_ID=$LB_SG_ID
export MONITOR_SG_ID=$MONITOR_SG_ID
export MGMT_SG_ID=$MGMT_SG_ID
EOF

echo "📄 Security groups configuration saved to security-groups-config.json"
echo "📄 Security groups summary saved to security-groups-summary.txt"
cat security-groups-summary.txt

echo "✅ Security groups setup completed successfully!"
echo "⚠️  Remember to:"
echo "   - Restrict SSH access to specific IP ranges in production"
echo "   - Review and adjust camera network CIDR blocks"
echo "   - Set up VPC Flow Logs for security monitoring"
echo "   - Configure AWS WAF for additional web application protection"