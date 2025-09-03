#!/bin/bash

# AWS Monitoring and Alerting Setup for Campus Security System
# This script configures CloudWatch monitoring, alarms, and SNS notifications

set -e

REGION=${AWS_REGION:-us-east-1}
PROJECT_NAME="campus-security"
ALERT_EMAIL=${ALERT_EMAIL:-"admin@campus-security.local"}

echo "📊 Setting up AWS monitoring and alerting for Campus Security System"

# Create SNS topic for alerts
echo "📢 Creating SNS topic for alerts..."
TOPIC_ARN=$(aws sns create-topic \
    --name ${PROJECT_NAME}-alerts \
    --region $REGION \
    --query 'TopicArn' \
    --output text)

# Subscribe email to SNS topic
aws sns subscribe \
    --topic-arn $TOPIC_ARN \
    --protocol email \
    --notification-endpoint $ALERT_EMAIL \
    --region $REGION

echo "✅ SNS topic created: $TOPIC_ARN"
echo "📧 Please check your email and confirm the subscription"

# Create CloudWatch Log Groups
echo "📝 Creating CloudWatch Log Groups..."
LOG_GROUPS=(
    "/aws/ec2/${PROJECT_NAME}/core-api"
    "/aws/ec2/${PROJECT_NAME}/privacy-service"
    "/aws/ec2/${PROJECT_NAME}/notification-service"
    "/aws/ec2/${PROJECT_NAME}/compliance-service"
    "/aws/ec2/${PROJECT_NAME}/edge-service"
    "/aws/ec2/${PROJECT_NAME}/nginx"
    "/aws/s3/${PROJECT_NAME}-evidence"
)

for log_group in "${LOG_GROUPS[@]}"; do
    aws logs create-log-group \
        --log-group-name $log_group \
        --region $REGION || true
    
    # Set retention policy (30 days)
    aws logs put-retention-policy \
        --log-group-name $log_group \
        --retention-in-days 30 \
        --region $REGION
done

# Create custom CloudWatch metrics
echo "📈 Setting up custom CloudWatch metrics..."

# Create Lambda function for custom metrics
cat > custom-metrics-lambda.py << 'EOF'
import boto3
import json
import requests
import datetime

def lambda_handler(event, context):
    """
    Custom metrics collection for Campus Security System
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    try:
        # Collect metrics from core API
        core_api_url = os.environ.get('CORE_API_URL', 'http://localhost:8000')
        
        # Get health status
        health_response = requests.get(f"{core_api_url}/health", timeout=10)
        health_status = 1 if health_response.status_code == 200 else 0
        
        # Get metrics endpoint
        metrics_response = requests.get(f"{core_api_url}/metrics", timeout=10)
        
        # Parse Prometheus metrics (simplified)
        metrics_data = metrics_response.text
        
        # Extract key metrics
        active_connections = 0
        security_incidents = 0
        false_positive_rate = 0
        
        for line in metrics_data.split('\n'):
            if line.startswith('active_connections'):
                active_connections = float(line.split()[-1])
            elif line.startswith('security_incidents_total'):
                security_incidents = float(line.split()[-1])
            elif line.startswith('security_false_positive_rate'):
                false_positive_rate = float(line.split()[-1])
        
        # Send custom metrics to CloudWatch
        cloudwatch.put_metric_data(
            Namespace='CampusSecurity/Application',
            MetricData=[
                {
                    'MetricName': 'HealthStatus',
                    'Value': health_status,
                    'Unit': 'Count',
                    'Timestamp': datetime.datetime.utcnow()
                },
                {
                    'MetricName': 'ActiveConnections',
                    'Value': active_connections,
                    'Unit': 'Count',
                    'Timestamp': datetime.datetime.utcnow()
                },
                {
                    'MetricName': 'SecurityIncidents',
                    'Value': security_incidents,
                    'Unit': 'Count',
                    'Timestamp': datetime.datetime.utcnow()
                },
                {
                    'MetricName': 'FalsePositiveRate',
                    'Value': false_positive_rate,
                    'Unit': 'Percent',
                    'Timestamp': datetime.datetime.utcnow()
                }
            ]
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps('Metrics collected successfully')
        }
        
    except Exception as e:
        print(f"Error collecting metrics: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
EOF

# Create deployment package
echo "requests" > requirements.txt
pip install -r requirements.txt -t .
zip -r custom-metrics-lambda.zip custom-metrics-lambda.py requests*

# Create Lambda function
aws lambda create-function \
    --function-name ${PROJECT_NAME}-custom-metrics \
    --runtime python3.9 \
    --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/LambdaMetricsRole \
    --handler custom-metrics-lambda.lambda_handler \
    --zip-file fileb://custom-metrics-lambda.zip \
    --timeout 60 \
    --environment Variables="{CORE_API_URL=https://your-domain.com}" \
    --region $REGION

# Schedule metrics collection every 5 minutes
aws events put-rule \
    --name ${PROJECT_NAME}-metrics-collection \
    --schedule-expression "rate(5 minutes)" \
    --description "Collect custom metrics every 5 minutes" \
    --region $REGION

aws lambda add-permission \
    --function-name ${PROJECT_NAME}-custom-metrics \
    --statement-id allow-cloudwatch-metrics \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn arn:aws:events:$REGION:$(aws sts get-caller-identity --query Account --output text):rule/${PROJECT_NAME}-metrics-collection \
    --region $REGION

aws events put-targets \
    --rule ${PROJECT_NAME}-metrics-collection \
    --targets "Id"="1","Arn"="arn:aws:lambda:$REGION:$(aws sts get-caller-identity --query Account --output text):function:${PROJECT_NAME}-custom-metrics" \
    --region $REGION

# Create CloudWatch Alarms
echo "🚨 Creating CloudWatch alarms..."

# High CPU utilization alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "${PROJECT_NAME}-high-cpu" \
    --alarm-description "High CPU utilization on main server" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $TOPIC_ARN \
    --dimensions Name=InstanceId,Value=${MAIN_INSTANCE_ID:-"i-1234567890abcdef0"} \
    --region $REGION

# High memory utilization alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "${PROJECT_NAME}-high-memory" \
    --alarm-description "High memory utilization" \
    --metric-name MemoryUtilization \
    --namespace CWAgent \
    --statistic Average \
    --period 300 \
    --threshold 85 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $TOPIC_ARN \
    --region $REGION

# Database connection alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "${PROJECT_NAME}-db-connections" \
    --alarm-description "High database connections" \
    --metric-name DatabaseConnections \
    --namespace AWS/RDS \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $TOPIC_ARN \
    --dimensions Name=DBInstanceIdentifier,Value=${DB_INSTANCE_ID:-"campus-security-db"} \
    --region $REGION

# Application health alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "${PROJECT_NAME}-app-health" \
    --alarm-description "Application health check failure" \
    --metric-name HealthStatus \
    --namespace CampusSecurity/Application \
    --statistic Average \
    --period 300 \
    --threshold 1 \
    --comparison-operator LessThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $TOPIC_ARN \
    --treat-missing-data breaching \
    --region $REGION

# High false positive rate alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "${PROJECT_NAME}-high-false-positives" \
    --alarm-description "High false positive rate in security detection" \
    --metric-name FalsePositiveRate \
    --namespace CampusSecurity/Application \
    --statistic Average \
    --period 900 \
    --threshold 30 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $TOPIC_ARN \
    --region $REGION

# S3 bucket size alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "${PROJECT_NAME}-evidence-storage-size" \
    --alarm-description "Evidence storage approaching limit" \
    --metric-name BucketSizeBytes \
    --namespace AWS/S3 \
    --statistic Average \
    --period 86400 \
    --threshold 1000000000000 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions $TOPIC_ARN \
    --dimensions Name=BucketName,Value=${EVIDENCE_BUCKET:-"campus-security-evidence"} Name=StorageType,Value=StandardStorage \
    --region $REGION

# Create CloudWatch Dashboard
echo "📊 Creating CloudWatch dashboard..."
cat > dashboard-config.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/EC2", "CPUUtilization", "InstanceId", "${MAIN_INSTANCE_ID:-i-1234567890abcdef0}" ],
                    [ "CWAgent", "MemoryUtilization", "InstanceId", "${MAIN_INSTANCE_ID:-i-1234567890abcdef0}" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "$REGION",
                "title": "Server Performance"
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "CampusSecurity/Application", "ActiveConnections" ],
                    [ ".", "SecurityIncidents" ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "$REGION",
                "title": "Application Metrics"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", "${DB_INSTANCE_ID:-campus-security-db}" ],
                    [ ".", "DatabaseConnections", ".", "." ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "$REGION",
                "title": "Database Performance"
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "CampusSecurity/Application", "FalsePositiveRate" ],
                    [ ".", "HealthStatus" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "$REGION",
                "title": "Security Metrics"
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 12,
            "width": 24,
            "height": 6,
            "properties": {
                "query": "SOURCE '/aws/ec2/${PROJECT_NAME}/core-api'\n| fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 100",
                "region": "$REGION",
                "title": "Recent Errors",
                "view": "table"
            }
        }
    ]
}
EOF

aws cloudwatch put-dashboard \
    --dashboard-name "${PROJECT_NAME}-monitoring" \
    --dashboard-body file://dashboard-config.json \
    --region $REGION

# Create monitoring summary
cat > monitoring-summary.txt << EOF
🔍 AWS Monitoring Setup Complete!

SNS Topic: $TOPIC_ARN
Email Subscription: $ALERT_EMAIL (please confirm subscription)

CloudWatch Alarms Created:
- ${PROJECT_NAME}-high-cpu: CPU > 80% for 10 minutes
- ${PROJECT_NAME}-high-memory: Memory > 85% for 10 minutes
- ${PROJECT_NAME}-db-connections: DB connections > 80 for 10 minutes
- ${PROJECT_NAME}-app-health: Application health check failures
- ${PROJECT_NAME}-high-false-positives: False positive rate > 30%
- ${PROJECT_NAME}-evidence-storage-size: Evidence storage > 1TB

CloudWatch Dashboard: ${PROJECT_NAME}-monitoring
Custom Metrics Collection: Every 5 minutes via Lambda

Log Groups Created:
$(printf '%s\n' "${LOG_GROUPS[@]}")

Next Steps:
1. Confirm email subscription for alerts
2. Install CloudWatch Agent on EC2 instances for detailed metrics
3. Configure application to send custom metrics
4. Set up log forwarding from Docker containers
5. Review and adjust alarm thresholds based on baseline performance

CloudWatch Agent Installation:
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
sudo rpm -U ./amazon-cloudwatch-agent.rpm

Dashboard URL:
https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=${PROJECT_NAME}-monitoring
EOF

echo "📄 Monitoring summary saved to monitoring-summary.txt"
cat monitoring-summary.txt

# Cleanup temporary files
rm -f custom-metrics-lambda.py custom-metrics-lambda.zip requirements.txt
rm -f dashboard-config.json
rm -rf requests* urllib3* certifi* charset_normalizer* idna*

echo "✅ AWS monitoring and alerting setup completed successfully!"
echo "📧 Please check your email and confirm the SNS subscription"