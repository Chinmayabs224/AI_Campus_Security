"""
Security initialization and setup for campus security system.
"""
import asyncio
import os
import json
from typing import Optional
from datetime import datetime
import structlog
import aiofiles

from .config import security_settings
from .network_policies import network_policies
from .vulnerability_scanner import vulnerability_scanner
from .vault_integration import SecretsManager, VaultConfig
from .incident_response import incident_response
from .docker_security import docker_security_scanner
from .secrets_scanner import secrets_scanner
from .compliance_monitor import compliance_monitor

logger = structlog.get_logger()


class SecurityManager:
    """Central security manager for initializing and coordinating security components."""
    
    def __init__(self):
        self.secrets_manager: Optional[SecretsManager] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all security components."""
        if self.initialized:
            logger.warning("Security manager already initialized")
            return
        
        logger.info("Initializing security manager")
        
        try:
            # Initialize network policies
            await self._initialize_network_policies()
            
            # Initialize secrets management
            await self._initialize_secrets_management()
            
            # Initialize vulnerability scanning
            await self._initialize_vulnerability_scanning()
            
            # Initialize incident response
            await self._initialize_incident_response()
            
            # Initialize Docker security scanning
            await self._initialize_docker_security()
            
            # Initialize secrets scanning
            await self._initialize_secrets_scanning()
            
            # Initialize compliance monitoring
            await self._initialize_compliance_monitoring()
            
            # Set up security monitoring
            await self._setup_security_monitoring()
            
            self.initialized = True
            logger.info("Security manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize security manager", error=str(e))
            raise
    
    async def _initialize_network_policies(self):
        """Initialize network security policies."""
        logger.info("Initializing network security policies")
        
        # Initialize default firewall rules
        network_policies.initialize_default_rules()
        
        # Add trusted networks from configuration
        for network in security_settings.TRUSTED_NETWORKS:
            network_policies.add_trusted_network(network)
        
        # Add blocked networks from configuration
        for network in security_settings.BLOCKED_NETWORKS:
            network_policies.add_blocked_network(network)
        
        # Set up rate limiting rules
        network_policies.add_rate_limit_rule("0.0.0.0/0", 100, 200)  # Global default
        network_policies.add_rate_limit_rule("10.0.0.0/16", 500, 1000)  # Internal networks
        
        logger.info("Network security policies initialized")
    
    async def _initialize_secrets_management(self):
        """Initialize secrets management with Vault."""
        if not security_settings.VAULT_URL:
            logger.warning("Vault URL not configured, skipping secrets management initialization")
            return
        
        logger.info("Initializing secrets management")
        
        vault_config = VaultConfig(
            url=security_settings.VAULT_URL,
            token=security_settings.VAULT_TOKEN,
            role_id=security_settings.VAULT_ROLE_ID,
            secret_id=security_settings.VAULT_SECRET_ID,
            mount_point=security_settings.VAULT_MOUNT_POINT,
            namespace=security_settings.VAULT_NAMESPACE
        )
        
        self.secrets_manager = SecretsManager(vault_config)
        
        try:
            await self.secrets_manager.initialize()
            
            # Set up default application secrets if they don't exist
            await self.secrets_manager.setup_application_secrets()
            
            logger.info("Secrets management initialized")
            
        except Exception as e:
            logger.error("Failed to initialize secrets management", error=str(e))
            # Continue without Vault if it's not available
            self.secrets_manager = None
    
    async def _initialize_vulnerability_scanning(self):
        """Initialize vulnerability scanning."""
        if not security_settings.ENABLE_VULNERABILITY_SCANNING:
            logger.info("Vulnerability scanning disabled")
            return
        
        logger.info("Initializing vulnerability scanning")
        
        # Set up scanning policies
        vulnerability_scanner.scan_policies.update({
            "max_critical_vulnerabilities": security_settings.MAX_CRITICAL_VULNERABILITIES,
            "max_high_vulnerabilities": security_settings.MAX_HIGH_VULNERABILITIES,
            "scan_frequency_hours": 24,
            "auto_fix_enabled": True
        })
        
        # Schedule initial scans for key images
        key_images = [
            ("campus-security-api", "latest"),
            ("campus-security-edge", "latest"),
            ("campus-security-frontend", "latest")
        ]
        
        for image_name, tag in key_images:
            try:
                await vulnerability_scanner.scan_container_image(image_name, tag)
                logger.info("Initial vulnerability scan completed", image=f"{image_name}:{tag}")
            except Exception as e:
                logger.warning("Initial vulnerability scan failed", 
                             image=f"{image_name}:{tag}", error=str(e))
        
        logger.info("Vulnerability scanning initialized")
    
    async def _initialize_incident_response(self):
        """Initialize incident response procedures."""
        logger.info("Initializing incident response")
        
        # Add notification handlers
        if security_settings.INCIDENT_NOTIFICATION_WEBHOOK:
            incident_response.add_notification_handler(self._webhook_notification_handler)
        
        # Add email notification handler
        incident_response.add_notification_handler(self._email_notification_handler)
        
        logger.info("Incident response initialized")
    
    async def _initialize_docker_security(self):
        """Initialize Docker security scanning."""
        logger.info("Initializing Docker security scanning")
        
        # Set up exclusions for common paths
        docker_security_scanner.excluded_paths.update({
            ".git", "__pycache__", "node_modules", ".pytest_cache",
            "venv", ".venv", "build", "dist"
        })
        
        # Scan existing Dockerfiles
        dockerfile_paths = [
            "core-api/Dockerfile",
            "edge-services/Dockerfile",
            "frontend/Dockerfile"
        ]
        
        for dockerfile_path in dockerfile_paths:
            try:
                if os.path.exists(dockerfile_path):
                    await docker_security_scanner.scan_dockerfile(dockerfile_path)
                    logger.info("Dockerfile security scan completed", path=dockerfile_path)
            except Exception as e:
                logger.warning("Dockerfile security scan failed", 
                             path=dockerfile_path, error=str(e))
        
        logger.info("Docker security scanning initialized")
    
    async def _initialize_secrets_scanning(self):
        """Initialize secrets scanning."""
        logger.info("Initializing secrets scanning")
        
        # Add common exclusions
        secrets_scanner.add_excluded_path(".git")
        secrets_scanner.add_excluded_path("__pycache__")
        secrets_scanner.add_excluded_path("node_modules")
        secrets_scanner.add_excluded_path(".pytest_cache")
        secrets_scanner.add_excluded_path("venv")
        secrets_scanner.add_excluded_path(".venv")
        
        # Perform initial scan of critical directories
        scan_directories = ["core-api", "microservices", "scripts"]
        
        for directory in scan_directories:
            try:
                if os.path.exists(directory):
                    results = await secrets_scanner.scan_directory(directory, recursive=True)
                    if results:
                        logger.warning("Secrets detected during initial scan", 
                                     directory=directory, 
                                     secrets_count=sum(len(matches) for matches in results.values()))
                    else:
                        logger.info("No secrets detected in directory", directory=directory)
            except Exception as e:
                logger.warning("Secrets scan failed", directory=directory, error=str(e))
        
        logger.info("Secrets scanning initialized")
    
    async def _initialize_compliance_monitoring(self):
        """Initialize compliance monitoring."""
        logger.info("Initializing compliance monitoring")
        
        # Run initial compliance check
        try:
            results = await compliance_monitor.run_compliance_check()
            logger.info("Initial compliance check completed",
                       rules_checked=results["rules_checked"],
                       violations_found=results["violations_found"])
            
            if results["violations_found"] > 0:
                logger.warning("Compliance violations detected",
                             violations=results["violations_found"])
        
        except Exception as e:
            logger.error("Initial compliance check failed", error=str(e))
        
        logger.info("Compliance monitoring initialized")
    
    async def _setup_security_monitoring(self):
        """Set up continuous security monitoring."""
        logger.info("Setting up security monitoring")
        
        # Schedule periodic vulnerability scans
        if security_settings.ENABLE_VULNERABILITY_SCANNING:
            asyncio.create_task(self._periodic_vulnerability_scan())
        
        # Schedule security health checks
        asyncio.create_task(self._periodic_security_health_check())
        
        # Schedule periodic secrets scanning
        asyncio.create_task(self._periodic_secrets_scan())
        
        # Schedule periodic compliance checks
        asyncio.create_task(self._periodic_compliance_check())
        
        logger.info("Security monitoring set up")
    
    async def _periodic_vulnerability_scan(self):
        """Periodic vulnerability scanning task."""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                
                logger.info("Starting periodic vulnerability scan")
                
                # Scan all tracked images
                for image_key in vulnerability_scanner.scan_results.keys():
                    image_name, image_tag = image_key.split(":", 1)
                    try:
                        await vulnerability_scanner.scan_container_image(image_name, image_tag)
                    except Exception as e:
                        logger.error("Periodic vulnerability scan failed",
                                   image=image_key, error=str(e))
                
                logger.info("Periodic vulnerability scan completed")
                
            except Exception as e:
                logger.error("Periodic vulnerability scan task failed", error=str(e))
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _periodic_security_health_check(self):
        """Periodic security health check task."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Check for high-risk vulnerabilities
                await self._check_vulnerability_alerts()
                
                # Check for security policy violations
                await self._check_policy_violations()
                
            except Exception as e:
                logger.error("Security health check failed", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _check_vulnerability_alerts(self):
        """Check for vulnerability alerts that need attention."""
        for image_key, scan_result in vulnerability_scanner.scan_results.items():
            if scan_result.critical_count > security_settings.MAX_CRITICAL_VULNERABILITIES:
                # Create incident for critical vulnerabilities
                await incident_response.create_incident(
                    title=f"Critical vulnerabilities detected in {image_key}",
                    description=f"Container image {image_key} has {scan_result.critical_count} critical vulnerabilities",
                    incident_type=incident_response.IncidentType.VULNERABILITY_EXPLOIT,
                    severity=incident_response.IncidentSeverity.CRITICAL,
                    reported_by="security_monitor",
                    affected_systems=[image_key],
                    indicators_of_compromise=[f"Critical vulnerabilities: {scan_result.critical_count}"]
                )
    
    async def _check_policy_violations(self):
        """Check for security policy violations."""
        # This would integrate with other monitoring systems
        # For now, just log that the check is running
        logger.debug("Security policy violation check completed")
    
    async def _periodic_secrets_scan(self):
        """Periodic secrets scanning task."""
        while True:
            try:
                await asyncio.sleep(7 * 24 * 3600)  # Run weekly
                
                logger.info("Starting periodic secrets scan")
                
                # Clear previous results
                secrets_scanner.clear_results()
                
                # Scan critical directories
                scan_directories = ["core-api", "microservices", "scripts"]
                total_secrets = 0
                
                for directory in scan_directories:
                    if os.path.exists(directory):
                        results = await secrets_scanner.scan_directory(directory, recursive=True)
                        secrets_count = sum(len(matches) for matches in results.values())
                        total_secrets += secrets_count
                        
                        if secrets_count > 0:
                            logger.warning("Secrets detected in periodic scan",
                                         directory=directory,
                                         secrets_count=secrets_count)
                
                # Generate report if secrets found
                if total_secrets > 0:
                    report_file = f"security_reports/secrets_scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                    os.makedirs("security_reports", exist_ok=True)
                    await secrets_scanner.generate_report(report_file)
                    
                    # Create security incident for high-risk secrets
                    high_risk_secrets = sum(
                        1 for matches in secrets_scanner.scan_results.values()
                        for match in matches
                        if match.severity in ["critical", "high"]
                    )
                    
                    if high_risk_secrets > 0:
                        await incident_response.create_incident(
                            title=f"High-risk secrets detected in codebase",
                            description=f"Periodic scan found {high_risk_secrets} high-risk secrets in codebase",
                            incident_type=incident_response.IncidentType.POLICY_VIOLATION,
                            severity=incident_response.IncidentSeverity.HIGH,
                            reported_by="security_monitor",
                            indicators_of_compromise=[f"High-risk secrets: {high_risk_secrets}"]
                        )
                
                logger.info("Periodic secrets scan completed", total_secrets=total_secrets)
                
            except Exception as e:
                logger.error("Periodic secrets scan failed", error=str(e))
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _periodic_compliance_check(self):
        """Periodic compliance checking task."""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                
                logger.info("Starting periodic compliance check")
                
                # Run compliance checks for all frameworks
                results = await compliance_monitor.run_compliance_check()
                
                # Generate daily compliance report
                report_file = f"security_reports/compliance_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
                os.makedirs("security_reports", exist_ok=True)
                
                report = await compliance_monitor.generate_compliance_report()
                async with aiofiles.open(report_file, 'w') as f:
                    await f.write(json.dumps(report, indent=2, default=str))
                
                # Create incidents for critical compliance violations
                critical_violations = [
                    v for v in compliance_monitor.violations.values()
                    if v.severity == "critical" and v.status == "open"
                ]
                
                for violation in critical_violations:
                    await incident_response.create_incident(
                        title=f"Critical compliance violation: {violation.title}",
                        description=violation.description,
                        incident_type=incident_response.IncidentType.POLICY_VIOLATION,
                        severity=incident_response.IncidentSeverity.CRITICAL,
                        reported_by="compliance_monitor",
                        indicators_of_compromise=[f"Compliance rule: {violation.rule_id}"]
                    )
                
                logger.info("Periodic compliance check completed",
                           violations_found=results["violations_found"])
                
            except Exception as e:
                logger.error("Periodic compliance check failed", error=str(e))
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _webhook_notification_handler(self, recipient: str, notification_data: dict):
        """Send incident notification via webhook."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "recipient": recipient,
                    "notification": notification_data,
                    "timestamp": notification_data.get("detected_at"),
                    "source": "campus_security_system"
                }
                
                async with session.post(
                    security_settings.INCIDENT_NOTIFICATION_WEBHOOK,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("Webhook notification sent", recipient=recipient)
                    else:
                        logger.error("Webhook notification failed",
                                   recipient=recipient, status=response.status)
                        
        except Exception as e:
            logger.error("Webhook notification error", recipient=recipient, error=str(e))
    
    async def _email_notification_handler(self, recipient: str, notification_data: dict):
        """Send incident notification via email."""
        # This would integrate with an email service
        # For now, just log the notification
        logger.info("Email notification would be sent",
                   recipient=recipient,
                   incident_id=notification_data.get("incident_id"),
                   severity=notification_data.get("severity"))
    
    async def shutdown(self):
        """Shutdown security manager and cleanup resources."""
        if not self.initialized:
            return
        
        logger.info("Shutting down security manager")
        
        if self.secrets_manager:
            await self.secrets_manager.close()
        
        self.initialized = False
        logger.info("Security manager shutdown complete")
    
    def get_security_status(self) -> dict:
        """Get overall security status."""
        return {
            "initialized": self.initialized,
            "secrets_manager_available": self.secrets_manager is not None,
            "vulnerability_scanning_enabled": security_settings.ENABLE_VULNERABILITY_SCANNING,
            "incident_response_enabled": security_settings.ENABLE_AUTO_INCIDENT_RESPONSE,
            "network_policies_active": len(network_policies.firewall_rules) > 0,
            "total_vulnerabilities": sum(
                scan.total_vulnerabilities 
                for scan in vulnerability_scanner.scan_results.values()
            ),
            "active_incidents": len([
                incident for incident in incident_response.incidents.values()
                if incident.status not in [
                    incident_response.IncidentStatus.RESOLVED,
                    incident_response.IncidentStatus.CLOSED
                ]
            ]),
            "docker_security_scans": len(docker_security_scanner.scan_results),
            "secrets_detected": sum(
                len(matches) for matches in secrets_scanner.scan_results.values()
            ),
            "compliance_violations": len([
                v for v in compliance_monitor.violations.values() if v.status == "open"
            ]),
            "compliance_status": {
                framework.value: status.value 
                for framework, status in compliance_monitor.compliance_status.items()
            }
        }


# Global security manager instance
security_manager = SecurityManager()