"""
Security module for campus security system.
"""

from .config import security_settings, SecuritySettings
from .network_policies import network_policies, NetworkSecurityPolicies, FirewallRule
from .vulnerability_scanner import vulnerability_scanner, VulnerabilityScanner, SeverityLevel
from .vault_integration import SecretsManager, VaultConfig, VaultClient
from .incident_response import incident_response, IncidentResponseProcedures, SecurityIncident
from .middleware import enhanced_security_middleware, EnhancedSecurityMiddleware
from .init import security_manager, SecurityManager
from .hardening import security_hardening, SecurityHardening
from .security_policies import security_policy_engine, SecurityPolicyEngine, PolicyViolation

__all__ = [
    # Configuration
    "security_settings",
    "SecuritySettings",
    
    # Network Security
    "network_policies", 
    "NetworkSecurityPolicies",
    "FirewallRule",
    
    # Vulnerability Management
    "vulnerability_scanner",
    "VulnerabilityScanner", 
    "SeverityLevel",
    
    # Secrets Management
    "SecretsManager",
    "VaultConfig",
    "VaultClient",
    
    # Incident Response
    "incident_response",
    "IncidentResponseProcedures",
    "SecurityIncident",
    
    # Middleware
    "enhanced_security_middleware",
    "EnhancedSecurityMiddleware",
    
    # Security Manager
    "security_manager",
    "SecurityManager",
    
    # Security Hardening
    "security_hardening",
    "SecurityHardening",
    
    # Security Policies
    "security_policy_engine",
    "SecurityPolicyEngine",
    "PolicyViolation"
]