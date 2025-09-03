#!/bin/bash

# SSL/TLS Certificate Setup for Campus Security System
# This script configures SSL certificates using Let's Encrypt or AWS Certificate Manager

set -e

REGION=${AWS_REGION:-us-east-1}
DOMAIN=${DOMAIN:-"campus-security.example.com"}
EMAIL=${CERT_EMAIL:-"admin@example.com"}
PROJECT_NAME="campus-security"

echo "🔐 Setting up SSL/TLS certificates for Campus Security System"

# Function to setup Let's Encrypt certificates
setup_letsencrypt() {
    local domain=$1
    local email=$2
    
    echo "🌐 Setting up Let's Encrypt certificates for $domain"
    
    # Install Certbot
    sudo apt-get update
    sudo apt-get install -y certbot python3-certbot-nginx
    
    # Stop nginx temporarily
    sudo systemctl stop nginx || true
    
    # Obtain certificate
    sudo certbot certonly \
        --standalone \
        --email $email \
        --agree-tos \
        --no-eff-email \
        --domains $domain
    
    # Copy certificates to nginx directory
    sudo mkdir -p /etc/nginx/ssl
    sudo cp /etc/letsencrypt/live/$domain/fullchain.pem /etc/nginx/ssl/cert.pem
    sudo cp /etc/letsencrypt/live/$domain/privkey.pem /etc/nginx/ssl/key.pem
    
    # Set proper permissions
    sudo chmod 644 /etc/nginx/ssl/cert.pem
    sudo chmod 600 /etc/nginx/ssl/key.pem
    
    # Setup auto-renewal
    echo "0 12 * * * /usr/bin/certbot renew --quiet --post-hook 'systemctl reload nginx'" | sudo crontab -
    
    echo "✅ Let's Encrypt certificates configured for $domain"
}

# Function to setup AWS Certificate Manager
setup_acm() {
    local domain=$1
    
    echo "☁️ Setting up AWS Certificate Manager for $domain"
    
    # Request certificate
    CERT_ARN=$(aws acm request-certificate \
        --domain-name $domain \
        --subject-alternative-names "www.$domain" \
        --validation-method DNS \
        --region $REGION \
        --query 'CertificateArn' \
        --output text)
    
    echo "📜 Certificate requested: $CERT_ARN"
    
    # Get DNS validation records
    echo "⏳ Waiting for DNS validation records..."
    sleep 10
    
    aws acm describe-certificate \
        --certificate-arn $CERT_ARN \
        --region $REGION \
        --query 'Certificate.DomainValidationOptions[*].[DomainName,ResourceRecord.Name,ResourceRecord.Value]' \
        --output table
    
    echo "📋 Please create the DNS records shown above to validate the certificate"
    echo "⏳ Waiting for certificate validation (this may take several minutes)..."
    
    # Wait for certificate to be issued
    aws acm wait certificate-validated \
        --certificate-arn $CERT_ARN \
        --region $REGION
    
    echo "✅ Certificate validated and issued: $CERT_ARN"
    
    # Create Application Load Balancer
    echo "⚖️ Creating Application Load Balancer..."
    
    # Get VPC and subnet information
    VPC_ID=$(aws ec2 describe-vpcs \
        --filters "Name=tag:Name,Values=${PROJECT_NAME}-vpc" \
        --query 'Vpcs[0].VpcId' \
        --output text \
        --region $REGION)
    
    SUBNET_IDS=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[*].SubnetId' \
        --output text \
        --region $REGION)
    
    # Create security group for ALB
    ALB_SG_ID=$(aws ec2 create-security-group \
        --group-name ${PROJECT_NAME}-alb-sg \
        --description "Security group for Application Load Balancer" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' \
        --output text)
    
    # Allow HTTP and HTTPS
    aws ec2 authorize-security-group-ingress \
        --group-id $ALB_SG_ID \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    aws ec2 authorize-security-group-ingress \
        --group-id $ALB_SG_ID \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    # Create ALB
    ALB_ARN=$(aws elbv2 create-load-balancer \
        --name ${PROJECT_NAME}-alb \
        --subnets $SUBNET_IDS \
        --security-groups $ALB_SG_ID \
        --scheme internet-facing \
        --type application \
        --ip-address-type ipv4 \
        --region $REGION \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text)
    
    # Get ALB DNS name
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --load-balancer-arns $ALB_ARN \
        --region $REGION \
        --query 'LoadBalancers[0].DNSName' \
        --output text)
    
    # Create target group
    TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
        --name ${PROJECT_NAME}-targets \
        --protocol HTTP \
        --port 80 \
        --vpc-id $VPC_ID \
        --health-check-path /health \
        --health-check-interval-seconds 30 \
        --health-check-timeout-seconds 5 \
        --healthy-threshold-count 2 \
        --unhealthy-threshold-count 3 \
        --region $REGION \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text)
    
    # Register targets (you'll need to update with actual instance IDs)
    if [ -n "$MAIN_INSTANCE_ID" ]; then
        aws elbv2 register-targets \
            --target-group-arn $TARGET_GROUP_ARN \
            --targets Id=$MAIN_INSTANCE_ID,Port=80 \
            --region $REGION
    fi
    
    # Create HTTPS listener
    aws elbv2 create-listener \
        --load-balancer-arn $ALB_ARN \
        --protocol HTTPS \
        --port 443 \
        --certificates CertificateArn=$CERT_ARN \
        --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN \
        --region $REGION
    
    # Create HTTP listener (redirect to HTTPS)
    aws elbv2 create-listener \
        --load-balancer-arn $ALB_ARN \
        --protocol HTTP \
        --port 80 \
        --default-actions Type=redirect,RedirectConfig="{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}" \
        --region $REGION
    
    echo "✅ Application Load Balancer created: $ALB_DNS"
    echo "📋 Please update your DNS to point $domain to $ALB_DNS"
    
    # Save configuration
    cat > ssl-config.txt << EOF
SSL Configuration Complete!

Certificate ARN: $CERT_ARN
Load Balancer ARN: $ALB_ARN
Load Balancer DNS: $ALB_DNS
Target Group ARN: $TARGET_GROUP_ARN
Security Group: $ALB_SG_ID

DNS Configuration:
Create a CNAME record pointing $domain to $ALB_DNS

Environment Variables:
export CERT_ARN=$CERT_ARN
export ALB_ARN=$ALB_ARN
export TARGET_GROUP_ARN=$TARGET_GROUP_ARN
EOF
}

# Function to setup self-signed certificates (development only)
setup_selfsigned() {
    local domain=$1
    
    echo "🔧 Setting up self-signed certificates for $domain (development only)"
    
    mkdir -p ssl
    
    # Generate private key
    openssl genrsa -out ssl/key.pem 2048
    
    # Generate certificate signing request
    openssl req -new -key ssl/key.pem -out ssl/cert.csr -subj "/C=US/ST=State/L=City/O=Organization/CN=$domain"
    
    # Generate self-signed certificate
    openssl x509 -req -in ssl/cert.csr -signkey ssl/key.pem -out ssl/cert.pem -days 365
    
    # Set proper permissions
    chmod 600 ssl/key.pem
    chmod 644 ssl/cert.pem
    
    # Clean up CSR
    rm ssl/cert.csr
    
    echo "✅ Self-signed certificates generated for $domain"
    echo "⚠️  Warning: Self-signed certificates should only be used for development"
}

# Function to setup SSL security headers
setup_security_headers() {
    cat > ssl-security-headers.conf << 'EOF'
# SSL Security Headers Configuration
# Include this in your nginx server block

# HSTS (HTTP Strict Transport Security)
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

# Prevent MIME type sniffing
add_header X-Content-Type-Options "nosniff" always;

# XSS Protection
add_header X-XSS-Protection "1; mode=block" always;

# Frame Options
add_header X-Frame-Options "SAMEORIGIN" always;

# Referrer Policy
add_header Referrer-Policy "strict-origin-when-cross-origin" always;

# Content Security Policy
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' wss: https:; media-src 'self'; object-src 'none'; child-src 'none'; worker-src 'none'; frame-ancestors 'self'; form-action 'self'; base-uri 'self';" always;

# Feature Policy
add_header Permissions-Policy "camera=(), microphone=(), geolocation=(), interest-cohort=()" always;

# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_session_tickets off;
ssl_stapling on;
ssl_stapling_verify on;

# Perfect Forward Secrecy
ssl_dhparam /etc/nginx/ssl/dhparam.pem;
EOF

    # Generate DH parameters
    openssl dhparam -out dhparam.pem 2048
    
    echo "✅ SSL security headers configuration created"
}

# Main execution
case "${1:-letsencrypt}" in
    "letsencrypt")
        if [ -z "$DOMAIN" ] || [ "$DOMAIN" = "campus-security.example.com" ]; then
            echo "❌ Please set DOMAIN environment variable to your actual domain"
            echo "   Example: export DOMAIN=security.yourschool.edu"
            exit 1
        fi
        
        if [ -z "$EMAIL" ] || [ "$EMAIL" = "admin@example.com" ]; then
            echo "❌ Please set CERT_EMAIL environment variable to your email"
            echo "   Example: export CERT_EMAIL=admin@yourschool.edu"
            exit 1
        fi
        
        setup_letsencrypt $DOMAIN $EMAIL
        setup_security_headers
        ;;
        
    "acm")
        if [ -z "$DOMAIN" ] || [ "$DOMAIN" = "campus-security.example.com" ]; then
            echo "❌ Please set DOMAIN environment variable to your actual domain"
            echo "   Example: export DOMAIN=security.yourschool.edu"
            exit 1
        fi
        
        setup_acm $DOMAIN
        setup_security_headers
        ;;
        
    "selfsigned")
        setup_selfsigned $DOMAIN
        setup_security_headers
        ;;
        
    *)
        echo "Usage: $0 {letsencrypt|acm|selfsigned}"
        echo "  letsencrypt  - Use Let's Encrypt for free SSL certificates"
        echo "  acm          - Use AWS Certificate Manager with Application Load Balancer"
        echo "  selfsigned   - Generate self-signed certificates (development only)"
        echo ""
        echo "Environment Variables:"
        echo "  DOMAIN       - Your domain name (required)"
        echo "  CERT_EMAIL   - Email for Let's Encrypt (required for letsencrypt)"
        echo "  AWS_REGION   - AWS region (default: us-east-1)"
        exit 1
        ;;
esac

echo "🔐 SSL/TLS setup completed successfully!"
echo "📋 Don't forget to:"
echo "   - Update your DNS records"
echo "   - Test SSL configuration with SSL Labs"
echo "   - Configure automatic certificate renewal"
echo "   - Update your application configuration to use HTTPS"