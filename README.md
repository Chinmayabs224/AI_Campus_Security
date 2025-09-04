# AI-Powered Campus Security System

A comprehensive security monitoring system leveraging edge computing and cloud analytics for intelligent threat detection, incident management, and evidence handling with strict privacy compliance.

## 🚀 Features

### Core Capabilities
- **Real-time AI Threat Detection** using YOLO models trained on UCF Crime Dataset
- **Edge Computing** for low-latency processing at camera locations
- **Comprehensive Evidence Management** with encryption and chain of custody
- **Incident Management Workflow** with automated escalation and notifications
- **Privacy-First Design** with GDPR compliance and data protection
- **Role-Based Access Control** with SSO/SAML integration
- **Real-time Dashboard** with live monitoring and analytics

### Security Features
- **End-to-End Encryption** for all evidence and communications
- **Audit Logging** for complete compliance and forensic capabilities
- **Multi-Factor Authentication** with JWT and refresh token management
- **Rate Limiting** and DDoS protection
- **Vulnerability Scanning** and security hardening
- **Data Retention Policies** with automated cleanup

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Edge Devices  │    │   Core API      │    │   Frontend      │
│                 │    │                 │    │                 │
│ • YOLO Models   │◄──►│ • FastAPI       │◄──►│ • React.js      │
│ • RTSP Streams  │    │ • PostgreSQL    │    │ • Real-time UI  │
│ • Local Buffer  │    │ • Redis Cache   │    │ • Mobile App    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Microservices  │
                    │                 │
                    │ • Privacy       │
                    │ • Notifications │
                    │ • Compliance    │
                    └─────────────────┘
```

## 📁 Project Structure

```
├── core-api/              # FastAPI central services
│   ├── auth/              # Authentication & authorization
│   ├── events/            # Event ingestion & processing
│   ├── incidents/         # Incident management
│   ├── evidence/          # Evidence storage & retrieval
│   ├── analytics/         # Security analytics
│   └── security/          # Security & compliance
├── edge-services/         # Edge computing services
│   ├── inference-engine/  # AI model inference
│   ├── stream-processor/  # RTSP stream handling
│   └── event-generator/   # Security event generation
├── frontend/              # React.js dashboard
├── mobile-app/            # React Native mobile app
├── microservices/         # Specialized services
│   ├── privacy-service/   # Privacy & GDPR compliance
│   └── notification-service/ # Alert notifications
├── ml-training/           # Model training pipeline
│   ├── data-processing/   # UCF Crime dataset processing
│   ├── model-training/    # YOLO model training
│   ├── model-evaluation/  # Validation & benchmarking
│   └── model-retraining/  # Automated retraining
├── deployment/            # Deployment configurations
├── monitoring/            # Prometheus & Grafana
└── testing/               # Test suites & validation
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Chinmayabs224/AI_Campus_Security.git
cd AI_Campus_Security
```

### 2. Environment Setup
```bash
# Copy environment files
cp core-api/.env.example core-api/.env
cp microservices/notification-service/.env.example microservices/notification-service/.env

# Edit configuration files with your settings
```

### 3. Start Services
```bash
# Start all services with Docker Compose
docker-compose up -d

# Or start individual services
cd core-api && python -m uvicorn main:app --reload
cd frontend && npm start
```

### 4. Access Applications
- **Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001
- **Grafana**: http://localhost:3001

## 🤖 ML Training Pipeline

### Dataset Setup
```bash
cd ml-training
python setup_ml_env.py
python data-processing/download_dataset.py
```

### Model Training
```bash
# Train YOLO security model
python model-training/train_yolo_security.py

# Evaluate model performance
python model-evaluation/security_model_validator.py path/to/model.pt

# Compare model variants
python model-evaluation/yolo_variant_comparator.py --models nano:yolov8n.pt small:yolov8s.pt
```

### Security Classes
- **Critical**: Violence, Emergency situations
- **High**: Theft, Suspicious behavior
- **Medium**: Trespassing, Vandalism
- **Low**: Crowd detection, Loitering, Abandoned objects

## 🔧 API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/logout` - User logout
- `GET /api/v1/auth/me` - Current user info

### Events
- `POST /api/v1/events/ingest` - Ingest security events
- `GET /api/v1/events/` - List events with filtering
- `POST /api/v1/events/batch` - Batch event ingestion

### Incidents
- `POST /api/v1/incidents/` - Create incident
- `GET /api/v1/incidents/` - List incidents
- `PUT /api/v1/incidents/{id}` - Update incident
- `POST /api/v1/incidents/{id}/assign` - Assign incident

### Evidence
- `POST /api/v1/evidence/upload` - Get upload URL
- `GET /api/v1/evidence/{id}` - Get evidence
- `POST /api/v1/evidence/{id}/download-url` - Get download URL

## 🔒 Security & Compliance

### Data Protection
- **Encryption at Rest**: AES-256 encryption for all stored data
- **Encryption in Transit**: TLS 1.3 for all communications
- **Key Management**: Secure key rotation and management
- **Access Controls**: Role-based permissions with audit trails

### Privacy Compliance
- **GDPR Compliance**: Data subject rights and consent management
- **Data Retention**: Configurable retention policies
- **Right to Erasure**: Secure data deletion capabilities
- **Privacy by Design**: Built-in privacy protection

### Audit & Monitoring
- **Complete Audit Trails**: All actions logged with user attribution
- **Real-time Monitoring**: Prometheus metrics and Grafana dashboards
- **Security Scanning**: Automated vulnerability assessments
- **Compliance Reporting**: Automated compliance status reports

## 📊 Performance Metrics

### Model Performance
- **mAP@0.5**: ≥70% for production deployment
- **Real-time Processing**: ≥15 FPS minimum, ≥30 FPS target
- **False Positive Rate**: ≤30% acceptable, ≤10% excellent
- **Critical Event Detection**: ≥90% recall for violence/emergency

### System Performance
- **API Response Time**: <100ms for 95th percentile
- **Event Processing**: <500ms end-to-end latency
- **Concurrent Users**: Support for 1000+ simultaneous users
- **Uptime**: 99.9% availability target

## 🧪 Testing

### Test Suites
```bash
# Run API tests
cd core-api && python -m pytest

# Run frontend tests
cd frontend && npm test

# Run integration tests
cd testing && python run_e2e_tests.py

# Run security tests
cd testing/security && python test_security_compliance.py
```

### TestSprite Integration
The project includes comprehensive TestSprite test cases covering:
- User registration and authentication
- Role-based access control
- Security event processing
- Evidence management workflows
- Compliance and audit trails

## 🚀 Deployment

### Docker Deployment
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# With monitoring
docker-compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Monitor deployment
kubectl get pods -n campus-security
```

### AWS Deployment
```bash
# Deploy to AWS
cd deployment/aws
./deploy-complete.sh
```

## 📈 Monitoring & Analytics

### Metrics Dashboard
- **System Health**: CPU, memory, disk usage
- **Security Events**: Event rates, threat levels, response times
- **User Activity**: Login patterns, access patterns
- **Model Performance**: Inference times, accuracy metrics

### Alerting
- **Security Alerts**: Real-time threat notifications
- **System Alerts**: Performance and availability monitoring
- **Compliance Alerts**: Policy violations and audit failures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/Chinmayabs224/AI_Campus_Security/wiki)
- **Issues**: [GitHub Issues](https://github.com/Chinmayabs224/AI_Campus_Security/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Chinmayabs224/AI_Campus_Security/discussions)

## 🙏 Acknowledgments

- **UCF Crime Dataset** for training data
- **YOLO** for object detection framework
- **FastAPI** for high-performance API framework
- **React** for modern frontend development
- **MinIO** for object storage solution

---

**Built with ❤️ for campus safety and security**