# Campus Security System - Deployment Guide

This directory contains all the deployment scripts and configurations for the AI-Powered Campus Security System.

## Overview

The system supports multiple deployment scenarios:
- **Local Development**: Docker Compose setup for development and testing
- **AWS Production**: Complete AWS infrastructure with auto-scaling, monitoring, and disaster recovery
- **Hybrid Edge-Cloud**: Edge inference nodes with centralized cloud management

## Quick Start

### Local Development Deployment

```bash
# Clone the repository
git clone <repository-url>
cd campus-security

# Deploy locally with Docker Compose
./deployment/deploy-local.sh

# Access the system
open http://localhost
```

### AWS Production Deployment

```bash
# Set required environment variables
export DOMAIN="security.yourschool.edu"
export CERT_EMAIL="admin@yourschool.edu"
export ALERT_EMAIL="security@yourschool.edu"
export AWS_REGION="us-east-1"

# Run complete AWS deployment
./deployment/aws/deploy-complete.sh
```

## Deployment Components

### Docker Images

All services are containerized with multi-stage builds for optimization:

- **core-api**: FastAPI backend with PostgreSQL and Redis
- **privacy-service**: Flask microservice for face detection and redaction
- **notification-service**: Flask microservice for multi-channel notifications
- **compliance-service**: Flask microservice for GDPR/FERPA compliance
- **frontend**: React.js dashboard with Nginx
- **edge-service**: GPU-enabled inference service with YOLO

### AWS Infrastructure

The AWS deployment creates:

- **Compute**: EC2 instances with GPU support for edge inference
- **Storage**: S3 buckets with lifecycle policies for evidence storage
- **Database**: RDS PostgreSQL with automated backups
- **Caching**: ElastiCache Redis for session management
- **Networking**: VPC with security groups and load balancers
- **SSL/TLS**: Certificate Manager or Let's Encrypt certificates
- **Monitoring**: CloudWatch dashboards, alarms, and log aggregation
- **Backup**: Automated snapshots and cross-region replication

## File Structure

```
deployment/
├── README.md                     # This file
├── docker-compose.prod.yml       # Production Docker Compose
├── docker-compose.override.yml   # Development overrides
├── docker-build.sh              # Build all Docker images
├── deploy-local.sh               # Local deployment script
├── nginx/
│   └── nginx.conf               # Nginx reverse proxy configuration
├── ssl/
│   └── generate-certs.sh        # SSL certificate generation
└── aws/
    ├── deploy.sh                # Basic AWS infrastructure
    ├── deploy-complete.sh       # Complete AWS deployment
    ├── setup-s3.sh             # S3 storage configuration
    ├── security-groups.sh      # Security group setup
    ├── ssl-setup.sh            # SSL/TLS configuration
    ├── monitoring-setup.sh     # CloudWatch monitoring
    └── backup-restore.sh       # Backup and disaster recovery
```

## Configuration

### Environment Variables

Copy `.env.production` to `.env` and update with your values:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
POSTGRES_PASSWORD=secure-password

# Storage
AWS_S3_BUCKET=your-evidence-bucket
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key

# Security
JWT_SECRET_KEY=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key

# Notifications
FCM_SERVER_KEY=your-firebase-key
TWILIO_ACCOUNT_SID=your-twilio-sid
SENDGRID_API_KEY=your-sendgrid-key

# Domain and SSL
DOMAIN=security.yourschool.edu
CORS_ORIGINS=["https://security.yourschool.edu"]
```

### Camera Configuration

Configure your IP cameras to stream to the edge services:

```bash
# Edge service environment
CAMERA_RTSP_URLS=rtsp://camera1.local/stream,rtsp://camera2.local/stream
CORE_API_URL=https://your-domain.com/api/v1
GPU_ENABLED=true
```

## Deployment Scenarios

### 1. Development Setup

For local development and testing:

```bash
# Start development environment
docker-compose -f docker-compose.prod.yml -f docker-compose.override.yml up -d

# Access development tools
open http://localhost:8025  # MailHog (email testing)
open http://localhost:8081  # Redis Commander
open http://localhost:5050  # pgAdmin
```

### 2. Single Server Production

For small deployments on a single server:

```bash
# Deploy to single server
./deployment/deploy-local.sh

# Configure SSL with Let's Encrypt
./deployment/ssl/generate-certs.sh your-domain.com
```

### 3. AWS Multi-Server Production

For scalable production deployments:

```bash
# Complete AWS deployment
export DOMAIN="security.yourschool.edu"
export CERT_EMAIL="admin@yourschool.edu"
./deployment/aws/deploy-complete.sh
```

### 4. Hybrid Edge-Cloud

For distributed edge inference with cloud management:

```bash
# Deploy cloud infrastructure
./deployment/aws/deploy.sh

# Deploy edge nodes separately
docker run -d --gpus all \
  -e CORE_API_URL=https://your-domain.com/api/v1 \
  -e CAMERA_RTSP_URLS=rtsp://camera1.local/stream \
  campus-security-edge-service
```

## Monitoring and Maintenance

### Health Checks

All services provide health check endpoints:

```bash
curl https://your-domain.com/health              # Main application
curl https://your-domain.com/api/v1/auth/health  # Authentication
curl http://edge-server:8080/health              # Edge service
```

### Monitoring

- **CloudWatch Dashboard**: System metrics and performance
- **Grafana**: Custom dashboards and visualizations
- **Prometheus**: Metrics collection and alerting
- **Log Aggregation**: Centralized logging with CloudWatch Logs

### Backup and Recovery

```bash
# Manual backup
./deployment/aws/backup-restore.sh backup

# Restore from backup
./deployment/aws/backup-restore.sh restore

# Test disaster recovery
./deployment/aws/backup-restore.sh test-recovery
```

## Security Considerations

### Network Security

- VPC with private subnets for databases
- Security groups with minimal required access
- WAF protection for web applications
- VPN access for management

### Data Protection

- Encryption at rest for all data stores
- TLS 1.2+ for all communications
- Automatic face redaction and privacy zones
- GDPR/FERPA compliance features

### Access Control

- SSO/SAML integration
- Role-based access control (RBAC)
- API key authentication for edge devices
- Audit logging for all access

## Troubleshooting

### Common Issues

1. **Docker build failures**: Check Docker daemon and available disk space
2. **SSL certificate issues**: Verify domain DNS and firewall settings
3. **Database connection errors**: Check security groups and credentials
4. **High false positive rates**: Adjust model confidence thresholds

### Log Analysis

```bash
# View application logs
docker-compose logs -f core-api

# View system logs
journalctl -u docker -f

# View AWS CloudWatch logs
aws logs tail /aws/ec2/campus-security/core-api --follow
```

### Performance Tuning

- Adjust worker processes based on CPU cores
- Configure Redis memory limits
- Optimize database queries and indexes
- Scale edge services based on camera load

## Support and Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Architecture Diagrams**: See `design.md` in project root
- **Security Policies**: See `compliance/` directory
- **Incident Response**: See `incident-response.md`

## Contributing

1. Test changes in development environment
2. Update documentation for any configuration changes
3. Ensure all health checks pass
4. Follow security best practices
5. Update deployment scripts if needed

## License

This deployment configuration is part of the Campus Security System project. See the main project LICENSE file for details.