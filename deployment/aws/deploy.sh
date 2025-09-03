#!/bin/bash

# AWS Campus Security System Deployment Script
# This script deploys the containerized campus security system to AWS EC2

set -e

# Configuration
REGION=${AWS_REGION:-us-east-1}
KEY_NAME=${AWS_KEY_NAME:-campus-security-key}
INSTANCE_TYPE=${AWS_INSTANCE_TYPE:-t3.large}
GPU_INSTANCE_TYPE=${AWS_GPU_INSTANCE_TYPE:-g4dn.xlarge}
VPC_CIDR="10.0.0.0/16"
SUBNET_CIDR="10.0.1.0/24"
PROJECT_NAME="campus-security"

echo "🚀 Starting AWS deployment for Campus Security System"

# Check AWS CLI installation
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first."
    exit 1
fi

# Create VPC and networking
echo "📡 Creating VPC and networking infrastructure..."
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block $VPC_CIDR \
    --region $REGION \
    --query 'Vpc.VpcId' \
    --output text)

aws ec2 create-tags \
    --resources $VPC_ID \
    --tags Key=Name,Value=${PROJECT_NAME}-vpc \
    --region $REGION

# Enable DNS hostnames
aws ec2 modify-vpc-attribute \
    --vpc-id $VPC_ID \
    --enable-dns-hostnames \
    --region $REGION

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
    --region $REGION \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)

aws ec2 attach-internet-gateway \
    --internet-gateway-id $IGW_ID \
    --vpc-id $VPC_ID \
    --region $REGION

aws ec2 create-tags \
    --resources $IGW_ID \
    --tags Key=Name,Value=${PROJECT_NAME}-igw \
    --region $REGION

# Create subnet
SUBNET_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block $SUBNET_CIDR \
    --region $REGION \
    --query 'Subnet.SubnetId' \
    --output text)

aws ec2 create-tags \
    --resources $SUBNET_ID \
    --tags Key=Name,Value=${PROJECT_NAME}-subnet \
    --region $REGION

# Create route table
ROUTE_TABLE_ID=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'RouteTable.RouteTableId' \
    --output text)

aws ec2 create-route \
    --route-table-id $ROUTE_TABLE_ID \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID \
    --region $REGION

aws ec2 associate-route-table \
    --subnet-id $SUBNET_ID \
    --route-table-id $ROUTE_TABLE_ID \
    --region $REGION

# Create security groups
echo "🔒 Creating security groups..."

# Web security group
WEB_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-web-sg \
    --description "Security group for web services" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow HTTP, HTTPS, and SSH
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

aws ec2 authorize-security-group-ingress \
    --group-id $WEB_SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Database security group
DB_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-db-sg \
    --description "Security group for database services" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow PostgreSQL from web security group
aws ec2 authorize-security-group-ingress \
    --group-id $DB_SG_ID \
    --protocol tcp \
    --port 5432 \
    --source-group $WEB_SG_ID \
    --region $REGION

# Edge security group
EDGE_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT_NAME}-edge-sg \
    --description "Security group for edge services" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Allow edge communication
aws ec2 authorize-security-group-ingress \
    --group-id $EDGE_SG_ID \
    --protocol tcp \
    --port 8080 \
    --source-group $WEB_SG_ID \
    --region $REGION

aws ec2 authorize-security-group-ingress \
    --group-id $EDGE_SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Get latest Ubuntu AMI
echo "🔍 Finding latest Ubuntu AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --region $REGION \
    --output text)

# Create key pair if it doesn't exist
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &> /dev/null; then
    echo "🔑 Creating key pair..."
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --region $REGION \
        --query 'KeyMaterial' \
        --output text > ${KEY_NAME}.pem
    chmod 400 ${KEY_NAME}.pem
    echo "✅ Key pair created and saved as ${KEY_NAME}.pem"
fi

# Create user data script for main server
cat > user-data-main.sh << 'EOF'
#!/bin/bash
apt-get update
apt-get install -y docker.io docker-compose-plugin awscli

# Start Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /opt/campus-security
cd /opt/campus-security

# Download deployment files from S3 (you'll need to upload them first)
# aws s3 cp s3://your-deployment-bucket/docker-compose.prod.yml .
# aws s3 cp s3://your-deployment-bucket/deployment/ ./deployment/ --recursive

# Set up SSL certificates (Let's Encrypt)
apt-get install -y certbot
# certbot certonly --standalone -d your-domain.com

# Create environment file
cat > .env << 'ENVEOF'
POSTGRES_PASSWORD=your-secure-password
MINIO_ACCESS_KEY=your-minio-access-key
MINIO_SECRET_KEY=your-minio-secret-key
JWT_SECRET_KEY=your-jwt-secret
FCM_SERVER_KEY=your-fcm-key
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
SENDGRID_API_KEY=your-sendgrid-key
GRAFANA_PASSWORD=your-grafana-password
ENVEOF

# Start services
# docker-compose -f docker-compose.prod.yml up -d
EOF

# Create user data script for edge server
cat > user-data-edge.sh << 'EOF'
#!/bin/bash
apt-get update
apt-get install -y docker.io awscli

# Install NVIDIA Docker for GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# Start Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Create edge service directory
mkdir -p /opt/edge-service
cd /opt/edge-service

# Download edge service files
# aws s3 cp s3://your-deployment-bucket/edge-services/ . --recursive

# Create environment file
cat > .env << 'ENVEOF'
CORE_API_URL=https://your-main-server.com/api/v1
REDIS_URL=redis://your-main-server.com:6379/4
GPU_ENABLED=true
CAMERA_RTSP_URLS=rtsp://camera1.local/stream,rtsp://camera2.local/stream
ENVEOF

# Start edge service
# docker run -d --name edge-service --gpus all --env-file .env your-edge-image
EOF

# Launch main server instance
echo "🖥️ Launching main server instance..."
MAIN_INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $WEB_SG_ID \
    --subnet-id $SUBNET_ID \
    --associate-public-ip-address \
    --user-data file://user-data-main.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PROJECT_NAME}-main},{Key=Project,Value=${PROJECT_NAME}}]" \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

# Launch edge server instance with GPU
echo "🎮 Launching edge server instance with GPU..."
EDGE_INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $GPU_INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $EDGE_SG_ID \
    --subnet-id $SUBNET_ID \
    --associate-public-ip-address \
    --user-data file://user-data-edge.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PROJECT_NAME}-edge},{Key=Project,Value=${PROJECT_NAME}}]" \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

# Wait for instances to be running
echo "⏳ Waiting for instances to be running..."
aws ec2 wait instance-running --instance-ids $MAIN_INSTANCE_ID --region $REGION
aws ec2 wait instance-running --instance-ids $EDGE_INSTANCE_ID --region $REGION

# Get public IP addresses
MAIN_PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $MAIN_INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

EDGE_PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $EDGE_INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

# Create S3 bucket for evidence storage
echo "🪣 Creating S3 bucket for evidence storage..."
BUCKET_NAME="${PROJECT_NAME}-evidence-$(date +%s)"
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Set up bucket lifecycle policy
cat > lifecycle-policy.json << EOF
{
    "Rules": [
        {
            "ID": "EvidenceRetention",
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
                }
            ],
            "Expiration": {
                "Days": 2555
            }
        }
    ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket $BUCKET_NAME \
    --lifecycle-configuration file://lifecycle-policy.json \
    --region $REGION

# Enable bucket versioning
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled \
    --region $REGION

# Create deployment summary
cat > deployment-summary.txt << EOF
🎉 Campus Security System Deployment Complete!

Infrastructure Details:
- VPC ID: $VPC_ID
- Subnet ID: $SUBNET_ID
- Main Server Instance ID: $MAIN_INSTANCE_ID
- Edge Server Instance ID: $EDGE_INSTANCE_ID
- Evidence S3 Bucket: $BUCKET_NAME

Access Information:
- Main Server Public IP: $MAIN_PUBLIC_IP
- Edge Server Public IP: $EDGE_PUBLIC_IP
- SSH Key: ${KEY_NAME}.pem

Next Steps:
1. Wait 5-10 minutes for instances to fully initialize
2. SSH into main server: ssh -i ${KEY_NAME}.pem ubuntu@$MAIN_PUBLIC_IP
3. Upload your application files to the instances
4. Configure SSL certificates
5. Start the services with Docker Compose
6. Configure edge devices to connect to: https://$MAIN_PUBLIC_IP

Security Groups:
- Web SG: $WEB_SG_ID (HTTP, HTTPS, SSH)
- Database SG: $DB_SG_ID (PostgreSQL from web)
- Edge SG: $EDGE_SG_ID (Edge communication, SSH)

Remember to:
- Update DNS records to point to $MAIN_PUBLIC_IP
- Configure SSL certificates
- Set up monitoring and alerting
- Configure backup procedures
- Update security group rules as needed
EOF

echo "📄 Deployment summary saved to deployment-summary.txt"
cat deployment-summary.txt

# Cleanup temporary files
rm -f user-data-main.sh user-data-edge.sh lifecycle-policy.json

echo "✅ Deployment script completed successfully!"
echo "🔗 Main server will be available at: https://$MAIN_PUBLIC_IP"
echo "🔗 Edge server available at: $EDGE_PUBLIC_IP:8080"