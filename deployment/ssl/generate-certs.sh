#!/bin/bash

# SSL Certificate Generation Script for Campus Security System
# This script generates self-signed certificates for development/testing
# For production, use Let's Encrypt or a proper CA

set -e

DOMAIN=${1:-campus-security.local}
CERT_DIR="./deployment/nginx/ssl"

echo "🔐 Generating SSL certificates for domain: $DOMAIN"

# Create SSL directory if it doesn't exist
mkdir -p $CERT_DIR

# Generate private key
openssl genrsa -out $CERT_DIR/key.pem 2048

# Generate certificate signing request
openssl req -new -key $CERT_DIR/key.pem -out $CERT_DIR/cert.csr -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"

# Generate self-signed certificate
openssl x509 -req -in $CERT_DIR/cert.csr -signkey $CERT_DIR/key.pem -out $CERT_DIR/cert.pem -days 365

# Set proper permissions
chmod 600 $CERT_DIR/key.pem
chmod 644 $CERT_DIR/cert.pem

# Clean up CSR
rm $CERT_DIR/cert.csr

echo "✅ SSL certificates generated successfully!"
echo "📁 Certificate files:"
echo "   - Private key: $CERT_DIR/key.pem"
echo "   - Certificate: $CERT_DIR/cert.pem"
echo ""
echo "⚠️  Note: These are self-signed certificates for development/testing only."
echo "   For production, use Let's Encrypt or a proper Certificate Authority."