#!/bin/bash

# S3 Setup Script for Campus Security Evidence Storage
# This script creates and configures S3 buckets for evidence storage with proper lifecycle policies

set -e

REGION=${AWS_REGION:-us-east-1}
PROJECT_NAME="campus-security"
BUCKET_NAME="${PROJECT_NAME}-evidence-$(date +%s)"

echo "🪣 Setting up S3 infrastructure for evidence storage"

# Create main evidence bucket
echo "Creating evidence storage bucket: $BUCKET_NAME"
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Enable versioning for evidence integrity
echo "Enabling versioning for evidence integrity..."
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled \
    --region $REGION

# Set up server-side encryption
echo "Configuring server-side encryption..."
cat > encryption-config.json << EOF
{
    "Rules": [
        {
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            },
            "BucketKeyEnabled": true
        }
    ]
}
EOF

aws s3api put-bucket-encryption \
    --bucket $BUCKET_NAME \
    --server-side-encryption-configuration file://encryption-config.json \
    --region $REGION

# Set up lifecycle policy for cost optimization and compliance
echo "Setting up lifecycle policy..."
cat > lifecycle-policy.json << EOF
{
    "Rules": [
        {
            "ID": "EvidenceRetentionPolicy",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "evidence/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                },
                {
                    "Days": 365,
                    "StorageClass": "DEEP_ARCHIVE"
                }
            ],
            "Expiration": {
                "Days": 2555
            }
        },
        {
            "ID": "IncompleteMultipartUploads",
            "Status": "Enabled",
            "Filter": {},
            "AbortIncompleteMultipartUpload": {
                "DaysAfterInitiation": 7
            }
        },
        {
            "ID": "DeleteOldVersions",
            "Status": "Enabled",
            "Filter": {},
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 90
            }
        }
    ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket $BUCKET_NAME \
    --lifecycle-configuration file://lifecycle-policy.json \
    --region $REGION

# Set up bucket policy for secure access
echo "Configuring bucket access policy..."
cat > bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyInsecureConnections",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ],
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        },
        {
            "Sid": "AllowApplicationAccess",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/CampusSecurityRole"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        }
    ]
}
EOF

aws s3api put-bucket-policy \
    --bucket $BUCKET_NAME \
    --policy file://bucket-policy.json \
    --region $REGION

# Enable access logging
LOGGING_BUCKET="${PROJECT_NAME}-access-logs-$(date +%s)"
echo "Creating access logging bucket: $LOGGING_BUCKET"
aws s3 mb s3://$LOGGING_BUCKET --region $REGION

cat > logging-config.json << EOF
{
    "LoggingEnabled": {
        "TargetBucket": "$LOGGING_BUCKET",
        "TargetPrefix": "evidence-access-logs/"
    }
}
EOF

aws s3api put-bucket-logging \
    --bucket $BUCKET_NAME \
    --bucket-logging-status file://logging-config.json \
    --region $REGION

# Set up bucket notification for compliance monitoring
echo "Setting up bucket notifications..."
cat > notification-config.json << EOF
{
    "CloudWatchConfigurations": [
        {
            "Id": "EvidenceAccessMonitoring",
            "CloudWatchConfiguration": {
                "LogGroupName": "/aws/s3/$BUCKET_NAME"
            },
            "Events": [
                "s3:ObjectCreated:*",
                "s3:ObjectRemoved:*"
            ]
        }
    ]
}
EOF

# Create CloudWatch log group
aws logs create-log-group \
    --log-group-name "/aws/s3/$BUCKET_NAME" \
    --region $REGION

# Enable bucket notifications (requires SNS topic or Lambda function)
# aws s3api put-bucket-notification-configuration \
#     --bucket $BUCKET_NAME \
#     --notification-configuration file://notification-config.json \
#     --region $REGION

# Create backup bucket for disaster recovery
BACKUP_BUCKET="${PROJECT_NAME}-backup-$(date +%s)"
BACKUP_REGION=${AWS_BACKUP_REGION:-us-west-2}

echo "Creating backup bucket in different region: $BACKUP_BUCKET"
aws s3 mb s3://$BACKUP_BUCKET --region $BACKUP_REGION

# Set up cross-region replication
cat > replication-config.json << EOF
{
    "Role": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/S3ReplicationRole",
    "Rules": [
        {
            "ID": "EvidenceBackupReplication",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "evidence/"
            },
            "Destination": {
                "Bucket": "arn:aws:s3:::$BACKUP_BUCKET",
                "StorageClass": "STANDARD_IA"
            }
        }
    ]
}
EOF

# Note: You need to create the S3ReplicationRole first
# aws s3api put-bucket-replication \
#     --bucket $BUCKET_NAME \
#     --replication-configuration file://replication-config.json \
#     --region $REGION

# Create IAM role for application access
echo "Creating IAM role for application access..."
cat > trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

aws iam create-role \
    --role-name CampusSecurityRole \
    --assume-role-policy-document file://trust-policy.json

cat > s3-access-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetObjectVersion",
                "s3:DeleteObjectVersion"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        }
    ]
}
EOF

aws iam put-role-policy \
    --role-name CampusSecurityRole \
    --policy-name S3EvidenceAccess \
    --policy-document file://s3-access-policy.json

# Create instance profile
aws iam create-instance-profile --instance-profile-name CampusSecurityProfile
aws iam add-role-to-instance-profile \
    --instance-profile-name CampusSecurityProfile \
    --role-name CampusSecurityRole

# Output configuration
cat > s3-config.txt << EOF
S3 Infrastructure Setup Complete!

Evidence Storage:
- Primary Bucket: $BUCKET_NAME (Region: $REGION)
- Backup Bucket: $BACKUP_BUCKET (Region: $BACKUP_REGION)
- Access Logs: $LOGGING_BUCKET

Configuration:
- Versioning: Enabled
- Encryption: AES256
- Lifecycle Policy: 30d -> IA, 90d -> Glacier, 365d -> Deep Archive, 7y -> Delete
- Access Policy: Secure transport required

IAM:
- Role: CampusSecurityRole
- Instance Profile: CampusSecurityProfile

Environment Variables for Application:
AWS_S3_BUCKET=$BUCKET_NAME
AWS_S3_REGION=$REGION
AWS_S3_BACKUP_BUCKET=$BACKUP_BUCKET

Next Steps:
1. Update your application configuration with the bucket name
2. Attach the instance profile to your EC2 instances
3. Test bucket access from your application
4. Set up monitoring and alerting for bucket access
EOF

echo "📄 S3 configuration saved to s3-config.txt"
cat s3-config.txt

# Cleanup temporary files
rm -f encryption-config.json lifecycle-policy.json bucket-policy.json
rm -f logging-config.json notification-config.json replication-config.json
rm -f trust-policy.json s3-access-policy.json

echo "✅ S3 setup completed successfully!"