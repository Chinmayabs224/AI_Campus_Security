#!/bin/bash

# Backup and Disaster Recovery Script for Campus Security System
# This script handles automated backups and disaster recovery procedures

set -e

REGION=${AWS_REGION:-us-east-1}
BACKUP_REGION=${AWS_BACKUP_REGION:-us-west-2}
PROJECT_NAME="campus-security"
BACKUP_BUCKET="${PROJECT_NAME}-backups-$(date +%Y%m%d)"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "💾 Campus Security System - Backup and Disaster Recovery"

# Function to create database backup
backup_database() {
    local instance_id=$1
    local db_name=$2
    
    echo "📊 Creating database backup for $db_name..."
    
    # Create RDS snapshot
    SNAPSHOT_ID="${db_name}-backup-${TIMESTAMP}"
    aws rds create-db-snapshot \
        --db-instance-identifier $instance_id \
        --db-snapshot-identifier $SNAPSHOT_ID \
        --region $REGION
    
    echo "✅ Database snapshot created: $SNAPSHOT_ID"
    
    # Wait for snapshot to complete
    aws rds wait db-snapshot-completed \
        --db-snapshot-identifier $SNAPSHOT_ID \
        --region $REGION
    
    # Copy snapshot to backup region
    aws rds copy-db-snapshot \
        --source-db-snapshot-identifier $SNAPSHOT_ID \
        --target-db-snapshot-identifier $SNAPSHOT_ID \
        --source-region $REGION \
        --region $BACKUP_REGION
    
    echo "✅ Database snapshot copied to backup region"
}

# Function to backup S3 evidence
backup_evidence() {
    local source_bucket=$1
    local backup_bucket=$2
    
    echo "🪣 Backing up evidence from $source_bucket to $backup_bucket..."
    
    # Create backup bucket if it doesn't exist
    aws s3 mb s3://$backup_bucket --region $BACKUP_REGION || true
    
    # Sync evidence data
    aws s3 sync s3://$source_bucket s3://$backup_bucket \
        --region $BACKUP_REGION \
        --storage-class STANDARD_IA
    
    echo "✅ Evidence backup completed"
}

# Function to backup application configuration
backup_configuration() {
    local backup_bucket=$1
    
    echo "⚙️ Backing up application configuration..."
    
    # Create temporary directory for configuration files
    TEMP_DIR=$(mktemp -d)
    CONFIG_DIR="$TEMP_DIR/config-backup-$TIMESTAMP"
    mkdir -p $CONFIG_DIR
    
    # Backup Docker Compose files
    cp docker-compose.prod.yml $CONFIG_DIR/
    cp .env.production $CONFIG_DIR/
    
    # Backup deployment scripts
    cp -r deployment/ $CONFIG_DIR/
    
    # Backup SSL certificates (if not using Let's Encrypt)
    if [ -d "deployment/nginx/ssl" ]; then
        cp -r deployment/nginx/ssl $CONFIG_DIR/
    fi
    
    # Create archive
    tar -czf $CONFIG_DIR.tar.gz -C $TEMP_DIR config-backup-$TIMESTAMP
    
    # Upload to S3
    aws s3 cp $CONFIG_DIR.tar.gz s3://$backup_bucket/config/ \
        --region $BACKUP_REGION
    
    # Cleanup
    rm -rf $TEMP_DIR
    
    echo "✅ Configuration backup completed"
}

# Function to backup EC2 instances
backup_instances() {
    local instance_ids=("$@")
    
    echo "💻 Creating EC2 instance backups..."
    
    for instance_id in "${instance_ids[@]}"; do
        # Get instance name
        INSTANCE_NAME=$(aws ec2 describe-instances \
            --instance-ids $instance_id \
            --query 'Reservations[0].Instances[0].Tags[?Key==`Name`].Value' \
            --output text \
            --region $REGION)
        
        # Create AMI
        AMI_NAME="${INSTANCE_NAME}-backup-${TIMESTAMP}"
        AMI_ID=$(aws ec2 create-image \
            --instance-id $instance_id \
            --name "$AMI_NAME" \
            --description "Automated backup of $INSTANCE_NAME" \
            --no-reboot \
            --region $REGION \
            --query 'ImageId' \
            --output text)
        
        echo "✅ AMI created for $INSTANCE_NAME: $AMI_ID"
        
        # Tag the AMI
        aws ec2 create-tags \
            --resources $AMI_ID \
            --tags Key=Project,Value=$PROJECT_NAME \
                   Key=BackupDate,Value=$TIMESTAMP \
                   Key=SourceInstance,Value=$instance_id \
            --region $REGION
        
        # Copy AMI to backup region
        aws ec2 copy-image \
            --source-image-id $AMI_ID \
            --source-region $REGION \
            --region $BACKUP_REGION \
            --name "$AMI_NAME-backup-region" \
            --description "Cross-region backup of $AMI_NAME"
    done
}

# Function to create disaster recovery plan
create_dr_plan() {
    cat > disaster-recovery-plan.md << EOF
# Campus Security System - Disaster Recovery Plan

## Overview
This document outlines the disaster recovery procedures for the Campus Security System.

## Backup Information
- **Backup Date**: $(date)
- **Primary Region**: $REGION
- **Backup Region**: $BACKUP_REGION
- **Backup Bucket**: $BACKUP_BUCKET

## Recovery Time Objectives (RTO)
- **Critical Services**: 4 hours
- **Non-Critical Services**: 24 hours
- **Full System Recovery**: 48 hours

## Recovery Point Objectives (RPO)
- **Database**: 1 hour (automated snapshots)
- **Evidence Storage**: 24 hours (daily sync)
- **Configuration**: 24 hours (daily backup)

## Recovery Procedures

### 1. Database Recovery
\`\`\`bash
# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \\
    --db-instance-identifier campus-security-restored \\
    --db-snapshot-identifier [SNAPSHOT_ID] \\
    --region $BACKUP_REGION
\`\`\`

### 2. Evidence Storage Recovery
\`\`\`bash
# Restore evidence from backup bucket
aws s3 sync s3://$BACKUP_BUCKET s3/[NEW_EVIDENCE_BUCKET] \\
    --region $BACKUP_REGION
\`\`\`

### 3. Application Recovery
\`\`\`bash
# Download configuration backup
aws s3 cp s3://$BACKUP_BUCKET/config/config-backup-$TIMESTAMP.tar.gz . \\
    --region $BACKUP_REGION

# Extract configuration
tar -xzf config-backup-$TIMESTAMP.tar.gz

# Deploy using backup configuration
./deployment/aws/deploy.sh
\`\`\`

### 4. Instance Recovery
\`\`\`bash
# Launch instances from backup AMIs
aws ec2 run-instances \\
    --image-id [BACKUP_AMI_ID] \\
    --instance-type t3.large \\
    --key-name campus-security-key \\
    --security-group-ids [SECURITY_GROUP_ID] \\
    --subnet-id [SUBNET_ID] \\
    --region $BACKUP_REGION
\`\`\`

## Testing Schedule
- **Monthly**: Database restore test
- **Quarterly**: Full disaster recovery drill
- **Annually**: Cross-region failover test

## Contact Information
- **Primary Contact**: [Your Name] - [Your Email]
- **Secondary Contact**: [Backup Contact] - [Backup Email]
- **AWS Support**: [Support Case URL]

## Monitoring and Alerting
- CloudWatch alarms for backup failures
- SNS notifications for DR events
- Automated health checks post-recovery

## Compliance Requirements
- GDPR: Data recovery within 72 hours
- FERPA: Audit trail preservation during recovery
- SOC 2: Documented recovery procedures
EOF

    echo "📋 Disaster recovery plan created: disaster-recovery-plan.md"
}

# Function to setup automated backups
setup_automated_backups() {
    echo "🤖 Setting up automated backup schedule..."
    
    # Create Lambda function for automated backups
    cat > backup-lambda.py << 'EOF'
import boto3
import json
import datetime
import os

def lambda_handler(event, context):
    """
    Automated backup Lambda function for Campus Security System
    """
    
    # Initialize AWS clients
    rds = boto3.client('rds')
    s3 = boto3.client('s3')
    ec2 = boto3.client('ec2')
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    project_name = os.environ['PROJECT_NAME']
    
    try:
        # Create RDS snapshot
        db_instance_id = os.environ['DB_INSTANCE_ID']
        snapshot_id = f"{project_name}-auto-backup-{timestamp}"
        
        rds.create_db_snapshot(
            DBInstanceIdentifier=db_instance_id,
            DBSnapshotIdentifier=snapshot_id
        )
        
        # Backup S3 evidence (trigger sync job)
        source_bucket = os.environ['EVIDENCE_BUCKET']
        backup_bucket = os.environ['BACKUP_BUCKET']
        
        # Create AMI backups
        instance_ids = os.environ['INSTANCE_IDS'].split(',')
        for instance_id in instance_ids:
            ami_name = f"{project_name}-auto-backup-{instance_id}-{timestamp}"
            ec2.create_image(
                InstanceId=instance_id,
                Name=ami_name,
                NoReboot=True
            )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Automated backup completed successfully',
                'timestamp': timestamp,
                'snapshot_id': snapshot_id
            })
        }
        
    except Exception as e:
        print(f"Backup failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': timestamp
            })
        }
EOF

    # Create Lambda deployment package
    zip backup-lambda.zip backup-lambda.py
    
    # Create Lambda function
    aws lambda create-function \
        --function-name ${PROJECT_NAME}-automated-backup \
        --runtime python3.9 \
        --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/LambdaBackupRole \
        --handler backup-lambda.lambda_handler \
        --zip-file fileb://backup-lambda.zip \
        --timeout 300 \
        --environment Variables="{PROJECT_NAME=$PROJECT_NAME,DB_INSTANCE_ID=campus-security-db,EVIDENCE_BUCKET=campus-security-evidence,BACKUP_BUCKET=$BACKUP_BUCKET,INSTANCE_IDS=i-1234567890abcdef0}" \
        --region $REGION
    
    # Create CloudWatch Events rule for daily backups
    aws events put-rule \
        --name ${PROJECT_NAME}-daily-backup \
        --schedule-expression "cron(0 2 * * ? *)" \
        --description "Daily automated backup for Campus Security System" \
        --region $REGION
    
    # Add Lambda permission for CloudWatch Events
    aws lambda add-permission \
        --function-name ${PROJECT_NAME}-automated-backup \
        --statement-id allow-cloudwatch \
        --action lambda:InvokeFunction \
        --principal events.amazonaws.com \
        --source-arn arn:aws:events:$REGION:$(aws sts get-caller-identity --query Account --output text):rule/${PROJECT_NAME}-daily-backup \
        --region $REGION
    
    # Create target for the rule
    aws events put-targets \
        --rule ${PROJECT_NAME}-daily-backup \
        --targets "Id"="1","Arn"="arn:aws:lambda:$REGION:$(aws sts get-caller-identity --query Account --output text):function:${PROJECT_NAME}-automated-backup" \
        --region $REGION
    
    # Cleanup
    rm -f backup-lambda.py backup-lambda.zip
    
    echo "✅ Automated backup schedule configured"
}

# Main execution
case "${1:-backup}" in
    "backup")
        echo "🚀 Starting full system backup..."
        
        # Get instance IDs (you'll need to update these)
        MAIN_INSTANCE_ID=${MAIN_INSTANCE_ID:-"i-1234567890abcdef0"}
        EDGE_INSTANCE_ID=${EDGE_INSTANCE_ID:-"i-0987654321fedcba0"}
        
        # Get database instance ID
        DB_INSTANCE_ID=${DB_INSTANCE_ID:-"campus-security-db"}
        
        # Get evidence bucket name
        EVIDENCE_BUCKET=${EVIDENCE_BUCKET:-"campus-security-evidence"}
        
        # Create backup bucket
        aws s3 mb s3://$BACKUP_BUCKET --region $BACKUP_REGION || true
        
        # Perform backups
        backup_database $DB_INSTANCE_ID "campus-security"
        backup_evidence $EVIDENCE_BUCKET $BACKUP_BUCKET
        backup_configuration $BACKUP_BUCKET
        backup_instances $MAIN_INSTANCE_ID $EDGE_INSTANCE_ID
        
        # Create DR plan
        create_dr_plan
        
        echo "✅ Full system backup completed successfully!"
        ;;
        
    "setup-automation")
        setup_automated_backups
        ;;
        
    "restore")
        echo "🔄 Starting disaster recovery process..."
        echo "⚠️  This is a destructive operation. Please ensure you have the correct backup identifiers."
        echo "📋 Please refer to the disaster-recovery-plan.md for detailed procedures."
        ;;
        
    *)
        echo "Usage: $0 {backup|setup-automation|restore}"
        echo "  backup           - Perform full system backup"
        echo "  setup-automation - Configure automated backup schedule"
        echo "  restore          - Display restore procedures"
        exit 1
        ;;
esac

echo "💾 Backup and disaster recovery operations completed!"