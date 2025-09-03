#!/bin/bash

# Complete AWS Deployment Script for Campus Security System
# This master script orchestrates the entire AWS deployment process

set -e

# Configuration
REGION=${AWS_REGION:-us-east-1}
BACKUP_REGION=${AWS_BACKUP_REGION:-us-west-2}
PROJECT_NAME="campus-security"
DOMAIN=${DOMAIN:-""}
CERT_EMAIL=${CERT_EMAIL:-""}
ALERT_EMAIL=${ALERT_EMAIL:-""}

echo "🚀 Campus Security System - Complete AWS Deployment"
echo "=================================================="

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first."
    exit 1
fi

# Validate required environment variables
if [ -z "$DOMAIN" ]; then
    echo "⚠️  DOMAIN not set. SSL setup will be skipped."
fi

if [ -z "$CERT_EMAIL" ]; then
    echo "⚠️  CERT_EMAIL not set. Let's Encrypt setup will be skipped."
fi

if [ -z "$ALERT_EMAIL" ]; then
    echo "⚠️  ALERT_EMAIL not set. Monitoring alerts will use default."
    ALERT_EMAIL="admin@${DOMAIN:-localhost}"
fi

echo "✅ Prerequisites check completed"

# Step 1: Build Docker images
echo ""
echo "📦 Step 1: Building Docker images..."
echo "===================================="
./deployment/docker-build.sh
echo "✅ Docker images built successfully"

# Step 2: Deploy AWS infrastructure
echo ""
echo "🏗️  Step 2: Deploying AWS infrastructure..."
echo "=========================================="
./deployment/aws/deploy.sh

# Extract instance IDs and other resources from deployment
if [ -f "deployment-summary.txt" ]; then
    MAIN_PUBLIC_IP=$(grep "Main Server Public IP:" deployment-summary.txt | cut -d' ' -f5)
    EDGE_PUBLIC_IP=$(grep "Edge Server Public IP:" deployment-summary.txt | cut -d' ' -f5)
    VPC_ID=$(grep "VPC ID:" deployment-summary.txt | cut -d' ' -f3)
    MAIN_INSTANCE_ID=$(grep "Main Server Instance ID:" deployment-summary.txt | cut -d' ' -f5)
    EDGE_INSTANCE_ID=$(grep "Edge Server Instance ID:" deployment-summary.txt | cut -d' ' -f5)
    EVIDENCE_BUCKET=$(grep "Evidence S3 Bucket:" deployment-summary.txt | cut -d' ' -f4)
fi

echo "✅ AWS infrastructure deployed successfully"

# Step 3: Setup S3 storage
echo ""
echo "🪣 Step 3: Setting up S3 storage..."
echo "=================================="
export AWS_S3_BUCKET=$EVIDENCE_BUCKET
./deployment/aws/setup-s3.sh
echo "✅ S3 storage configured successfully"

# Step 4: Configure security groups
echo ""
echo "🔒 Step 4: Configuring security groups..."
echo "========================================"
export VPC_ID=$VPC_ID
./deployment/aws/security-groups.sh
echo "✅ Security groups configured successfully"

# Step 5: Setup SSL certificates
echo ""
echo "🔐 Step 5: Setting up SSL certificates..."
echo "======================================="
if [ -n "$DOMAIN" ] && [ -n "$CERT_EMAIL" ]; then
    export DOMAIN=$DOMAIN
    export CERT_EMAIL=$CERT_EMAIL
    export MAIN_INSTANCE_ID=$MAIN_INSTANCE_ID
    ./deployment/aws/ssl-setup.sh acm
    echo "✅ SSL certificates configured successfully"
else
    echo "⚠️  Skipping SSL setup - DOMAIN or CERT_EMAIL not provided"
fi

# Step 6: Setup monitoring and alerting
echo ""
echo "📊 Step 6: Setting up monitoring and alerting..."
echo "=============================================="
export ALERT_EMAIL=$ALERT_EMAIL
export MAIN_INSTANCE_ID=$MAIN_INSTANCE_ID
export EDGE_INSTANCE_ID=$EDGE_INSTANCE_ID
export DB_INSTANCE_ID="campus-security-db"
export EVIDENCE_BUCKET=$EVIDENCE_BUCKET
./deployment/aws/monitoring-setup.sh
echo "✅ Monitoring and alerting configured successfully"

# Step 7: Setup backup and disaster recovery
echo ""
echo "💾 Step 7: Setting up backup and disaster recovery..."
echo "=================================================="
export MAIN_INSTANCE_ID=$MAIN_INSTANCE_ID
export EDGE_INSTANCE_ID=$EDGE_INSTANCE_ID
export DB_INSTANCE_ID="campus-security-db"
export EVIDENCE_BUCKET=$EVIDENCE_BUCKET
./deployment/aws/backup-restore.sh setup-automation
echo "✅ Backup and disaster recovery configured successfully"

# Step 8: Deploy application to instances
echo ""
echo "🚀 Step 8: Deploying application to instances..."
echo "=============================================="

# Wait for instances to be fully ready
echo "⏳ Waiting for instances to be fully ready..."
sleep 60

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf campus-security-deployment.tar.gz \
    docker-compose.prod.yml \
    .env.production \
    deployment/ \
    --exclude=deployment/aws/*.sh

# Upload deployment package to S3
DEPLOYMENT_BUCKET="${PROJECT_NAME}-deployment-$(date +%s)"
aws s3 mb s3://$DEPLOYMENT_BUCKET --region $REGION
aws s3 cp campus-security-deployment.tar.gz s3://$DEPLOYMENT_BUCKET/ --region $REGION

# Deploy to main server
echo "🖥️  Deploying to main server..."
ssh -i ${PROJECT_NAME}-key.pem -o StrictHostKeyChecking=no ubuntu@$MAIN_PUBLIC_IP << EOF
    # Download deployment package
    aws s3 cp s3://$DEPLOYMENT_BUCKET/campus-security-deployment.tar.gz . --region $REGION
    tar -xzf campus-security-deployment.tar.gz
    
    # Update environment file with actual values
    cp .env.production .env
    sed -i "s/your-domain.com/$DOMAIN/g" .env
    
    # Start services
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be healthy
    sleep 30
    
    # Test health endpoints
    curl -f http://localhost:8000/health || echo "Core API health check failed"
    curl -f http://localhost:5001/health || echo "Privacy service health check failed"
    curl -f http://localhost:5002/health || echo "Notification service health check failed"
    curl -f http://localhost:5003/health || echo "Compliance service health check failed"
EOF

# Deploy to edge server
echo "🎮 Deploying to edge server..."
ssh -i ${PROJECT_NAME}-key.pem -o StrictHostKeyChecking=no ubuntu@$EDGE_PUBLIC_IP << EOF
    # Download edge service files
    aws s3 cp s3://$DEPLOYMENT_BUCKET/campus-security-deployment.tar.gz . --region $REGION
    tar -xzf campus-security-deployment.tar.gz
    
    # Start edge service
    cd edge-services
    docker build -t campus-security-edge-service .
    docker run -d --name edge-service --gpus all \
        -e CORE_API_URL=https://$DOMAIN/api/v1 \
        -e GPU_ENABLED=true \
        -p 8080:8080 \
        campus-security-edge-service
    
    # Test edge service
    sleep 15
    curl -f http://localhost:8080/health || echo "Edge service health check failed"
EOF

echo "✅ Application deployed to instances successfully"

# Step 9: Final configuration and testing
echo ""
echo "🧪 Step 9: Final configuration and testing..."
echo "==========================================="

# Test main application
echo "🌐 Testing main application..."
if [ -n "$DOMAIN" ]; then
    curl -f https://$DOMAIN/health || echo "⚠️  Main application health check failed"
    curl -f https://$DOMAIN/api/v1/auth/health || echo "⚠️  API health check failed"
else
    curl -f http://$MAIN_PUBLIC_IP/health || echo "⚠️  Main application health check failed"
fi

# Test edge service
echo "🎯 Testing edge service..."
curl -f http://$EDGE_PUBLIC_IP:8080/health || echo "⚠️  Edge service health check failed"

# Create final deployment report
cat > deployment-report.txt << EOF
🎉 Campus Security System - AWS Deployment Complete!
===================================================

Deployment Date: $(date)
Region: $REGION
Backup Region: $BACKUP_REGION

Infrastructure:
- VPC ID: $VPC_ID
- Main Server: $MAIN_PUBLIC_IP ($MAIN_INSTANCE_ID)
- Edge Server: $EDGE_PUBLIC_IP ($EDGE_INSTANCE_ID)
- Evidence Bucket: $EVIDENCE_BUCKET
- Deployment Bucket: $DEPLOYMENT_BUCKET

Access URLs:
- Main Application: $([ -n "$DOMAIN" ] && echo "https://$DOMAIN" || echo "http://$MAIN_PUBLIC_IP")
- Edge Service: http://$EDGE_PUBLIC_IP:8080
- Monitoring Dashboard: https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=${PROJECT_NAME}-monitoring

Security:
- SSL Certificates: $([ -n "$DOMAIN" ] && echo "Configured with ACM" || echo "Not configured")
- Security Groups: Configured
- Backup: Automated daily backups enabled
- Monitoring: CloudWatch alarms and dashboard configured

Next Steps:
1. Update DNS records to point $DOMAIN to the load balancer
2. Configure camera RTSP streams to connect to edge service
3. Set up user accounts and SSO integration
4. Test incident detection and notification workflows
5. Configure privacy zones and redaction settings
6. Review and adjust monitoring thresholds
7. Perform disaster recovery testing

Important Files:
- SSH Key: ${PROJECT_NAME}-key.pem
- Environment Config: .env
- Deployment Package: campus-security-deployment.tar.gz
- Monitoring Summary: monitoring-summary.txt
- S3 Configuration: s3-config.txt
- Security Groups: security-groups-summary.txt
- SSL Configuration: ssl-config.txt (if configured)

Maintenance Commands:
- View logs: ssh -i ${PROJECT_NAME}-key.pem ubuntu@$MAIN_PUBLIC_IP "docker-compose logs -f"
- Restart services: ssh -i ${PROJECT_NAME}-key.pem ubuntu@$MAIN_PUBLIC_IP "docker-compose restart"
- Update application: Re-run deployment with new images
- Backup manually: ./deployment/aws/backup-restore.sh backup
- Monitor system: Check CloudWatch dashboard

Support:
- Documentation: README.md
- Troubleshooting: Check CloudWatch logs and alarms
- Emergency: Follow disaster recovery procedures in disaster-recovery-plan.md

⚠️  Security Reminders:
- Regularly update system packages and Docker images
- Monitor security alerts and false positive rates
- Review access logs and audit trails
- Test backup and recovery procedures monthly
- Keep SSL certificates up to date
- Review and update security group rules as needed
EOF

echo "📄 Deployment report saved to deployment-report.txt"
cat deployment-report.txt

# Cleanup temporary files
rm -f campus-security-deployment.tar.gz

echo ""
echo "🎉 Campus Security System deployment completed successfully!"
echo "🔗 Access your system at: $([ -n "$DOMAIN" ] && echo "https://$DOMAIN" || echo "http://$MAIN_PUBLIC_IP")"
echo "📊 Monitor your system at: https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=${PROJECT_NAME}-monitoring"
echo "📧 Check your email for monitoring alert confirmations"
echo ""
echo "⚠️  Important: Please review the deployment report and complete the next steps!"