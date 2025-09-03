# Campus Security System - Security Module

This module provides comprehensive security controls and vulnerability management for the campus security system.

## Overview

The security module implements multiple layers of security controls including:

- **Network Security Policies**: Firewall rules, network segmentation, and access controls
- **Vulnerability Management**: Container scanning and vulnerability assessment
- **Secrets Management**: HashiCorp Vault integration for secure credential storage
- **Incident Response**: Automated security incident detection and response procedures
- **Security Hardening**: System and application hardening configurations
- **Policy Enforcement**: Comprehensive security policy engine

## Components

### 1. Network Security (`network_policies.py`)

Manages network-level security controls:

```python
from security import network_policies

# Add trusted network
network_policies.add_trusted_network("10.0.0.0/8")

# Block malicious network
network_policies.add_blocked_network("192.168.100.0/24")

# Check if IP is allowed
allowed = network_policies.is_ip_allowed("10.0.1.100")

# Generate iptables rules
rules = network_policies.generate_iptables_rules()
```

**Features:**
- Firewall rule management
- Network zone definitions (Edge, Internal, DMZ, Management)
- Rate limiting per network
- IP allowlist/blocklist
- iptables rule generation

### 2. Vulnerability Scanner (`vulnerability_scanner.py`)

Provides container vulnerability scanning:

```python
from security import vulnerability_scanner

# Scan container image
result = await vulnerability_scanner.scan_container_image("myapp", "latest")

# Generate vulnerability report
report = await vulnerability_scanner.generate_vulnerability_report("myapp", "latest")

# Export scan results
await vulnerability_scanner.export_scan_results("scan_results.json")
```

**Features:**
- Multi-tool scanning (Trivy, Grype)
- Vulnerability severity classification
- Policy violation detection
- Automated reporting
- Risk score calculation

### 3. Secrets Management (`vault_integration.py`)

HashiCorp Vault integration for secure secrets:

```python
from security import SecretsManager, VaultConfig

# Initialize secrets manager
vault_config = VaultConfig(url="http://vault:8200", token="vault-token")
secrets_manager = SecretsManager(vault_config)
await secrets_manager.initialize()

# Store secret
await secrets_manager.set_secret("database/credentials", {
    "username": "dbuser",
    "password": "secure_password"
})

# Retrieve secret
credentials = await secrets_manager.get_secret("database/credentials")
```

**Features:**
- AppRole authentication
- Encrypted secret storage
- Dynamic credential generation
- Secret rotation
- Audit logging

### 4. Incident Response (`incident_response.py`)

Automated security incident management:

```python
from security import incident_response, IncidentType, IncidentSeverity

# Create security incident
incident = await incident_response.create_incident(
    title="Unauthorized access detected",
    description="Multiple failed login attempts from suspicious IP",
    incident_type=IncidentType.UNAUTHORIZED_ACCESS,
    severity=IncidentSeverity.HIGH,
    reported_by="security_monitor"
)

# Update incident status
await incident_response.update_incident_status(
    incident.id, 
    IncidentStatus.INVESTIGATING,
    "Investigation started"
)
```

**Features:**
- Automated incident creation
- Response playbooks
- Escalation procedures
- Notification system
- Timeline tracking

### 5. Security Middleware (`middleware.py`)

Enhanced security middleware for API protection:

```python
from security import enhanced_security_middleware

# Applied automatically in main.py
app.add_middleware(enhanced_security_middleware)
```

**Features:**
- Request validation
- Malicious pattern detection
- Rate limiting
- Security headers
- Suspicious activity tracking

### 6. Security Hardening (`hardening.py`)

System and application hardening utilities:

```python
from security import security_hardening

# Apply comprehensive hardening
result = await security_hardening.apply_system_hardening()

# Generate security checklist
checklist = security_hardening.generate_security_checklist()

# Export hardening report
security_hardening.export_hardening_report("hardening_report.json")
```

**Features:**
- Network security hardening
- File permission hardening
- Service configuration
- Container security
- Database security

### 7. Security Policies (`security_policies.py`)

Comprehensive security policy engine:

```python
from security import security_policy_engine

# Validate password
valid, violations = security_policy_engine.validate_password("MySecurePass123!")

# Validate API request
valid, violations = security_policy_engine.validate_api_request(request_data)

# Check compliance requirements
requirements = security_policy_engine.check_compliance_requirements(
    "personal_data", 
    "analytics"
)
```

**Features:**
- Password policy enforcement
- API security validation
- Data classification policies
- Compliance checking (GDPR, FERPA)
- Policy violation tracking

## CLI Usage

The security module includes a comprehensive CLI tool:

```bash
# Network security management
python -m security.cli network add-trusted --network 10.0.0.0/8
python -m security.cli network add-blocked --network 192.168.100.0/24
python -m security.cli network check-ip --ip 10.0.1.100

# Vulnerability scanning
python -m security.cli scan container --image myapp --tag latest
python -m security.cli scan report --image myapp --output report.json

# Secrets management
python -m security.cli secrets get --path database/credentials
python -m security.cli secrets set --path api/keys --data '{"key":"value"}'

# Incident management
python -m security.cli incidents list
python -m security.cli incidents show --incident-id abc123
python -m security.cli incidents update-status --incident-id abc123 --status investigating

# Security status and hardening
python -m security.cli status
python -m security.cli harden
```

## Configuration

Security settings are managed through environment variables:

```bash
# Vault configuration
SECURITY_VAULT_URL=http://localhost:8200
SECURITY_VAULT_TOKEN=vault-token
SECURITY_VAULT_ROLE_ID=role-id
SECURITY_VAULT_SECRET_ID=secret-id

# Vulnerability scanning
SECURITY_ENABLE_VULNERABILITY_SCANNING=true
SECURITY_MAX_CRITICAL_VULNERABILITIES=0
SECURITY_MAX_HIGH_VULNERABILITIES=5

# Network security
SECURITY_TRUSTED_NETWORKS=["10.0.0.0/16", "192.168.0.0/16"]
SECURITY_BLOCKED_NETWORKS=[]

# Incident response
SECURITY_ENABLE_AUTO_INCIDENT_RESPONSE=true
SECURITY_INCIDENT_NOTIFICATION_WEBHOOK=https://webhook.example.com
SECURITY_SECURITY_TEAM_EMAIL=security@campus.edu

# Compliance
SECURITY_ENABLE_GDPR_COMPLIANCE=true
SECURITY_ENABLE_FERPA_COMPLIANCE=true
SECURITY_DATA_RETENTION_DAYS=2555
```

## Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Module                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Network Security│  │ Vulnerability   │  │ Secrets      │ │
│  │ - Firewall      │  │ Management      │  │ Management   │ │
│  │ - Rate Limiting │  │ - Container Scan│  │ - Vault      │ │
│  │ - IP Filtering  │  │ - Risk Analysis │  │ - Encryption │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Incident        │  │ Security        │  │ Policy       │ │
│  │ Response        │  │ Hardening       │  │ Engine       │ │
│  │ - Auto Response │  │ - System Config │  │ - Validation │ │
│  │ - Playbooks     │  │ - Best Practices│  │ - Compliance │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                Security Middleware                          │
│  - Request Validation                                       │
│  - Threat Detection                                         │
│  - Security Headers                                         │
└─────────────────────────────────────────────────────────────┘
```

## Security Best Practices

### 1. Network Security
- Implement network segmentation
- Use firewall rules to restrict access
- Enable DDoS protection
- Monitor network traffic

### 2. Application Security
- Validate all inputs
- Use HTTPS/TLS encryption
- Implement proper authentication
- Apply security headers

### 3. Data Security
- Encrypt data at rest and in transit
- Implement access controls
- Use data classification
- Regular backups

### 4. Monitoring & Response
- Enable comprehensive logging
- Set up real-time monitoring
- Implement incident response procedures
- Regular security assessments

## Compliance

The security module supports compliance with:

- **GDPR**: Data protection and privacy rights
- **FERPA**: Educational record privacy
- **NIST Cybersecurity Framework**: Security controls
- **ISO 27001**: Information security management
- **SOC 2**: Security and availability controls

## Integration

The security module integrates with:

- **FastAPI**: Security middleware and authentication
- **PostgreSQL**: Secure database connections
- **Redis**: Secure caching and session management
- **HashiCorp Vault**: Secrets management
- **Prometheus**: Security metrics
- **Structlog**: Security event logging

## Monitoring

Security events are logged and monitored through:

- **Audit Logs**: All security events
- **Metrics**: Security-related metrics
- **Alerts**: Real-time security alerts
- **Reports**: Regular security reports
- **Dashboards**: Security status visualization

## Troubleshooting

### Common Issues

1. **Vault Connection Failed**
   - Check Vault URL and credentials
   - Verify network connectivity
   - Check Vault server status

2. **Vulnerability Scan Failed**
   - Ensure Trivy/Grype are installed
   - Check container image availability
   - Verify scan permissions

3. **Network Policy Violations**
   - Review firewall rules
   - Check IP allowlists/blocklists
   - Verify network configuration

4. **Incident Response Not Triggered**
   - Check incident response configuration
   - Verify notification handlers
   - Review escalation rules

For additional support, check the logs or contact the security team.