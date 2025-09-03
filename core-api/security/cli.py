#!/usr/bin/env python3
"""
Security management CLI tool for campus security system.
"""
import asyncio
import click
import json
import sys
from typing import Optional
from datetime import datetime
import structlog

from .config import security_settings
from .network_policies import network_policies
from .vulnerability_scanner import vulnerability_scanner, SeverityLevel
from .vault_integration import SecretsManager, VaultConfig
from .incident_response import incident_response, IncidentSeverity, IncidentType, IncidentStatus

logger = structlog.get_logger()


@click.group()
def cli():
    """Campus Security System - Security Management CLI"""
    pass


@cli.group()
def network():
    """Network security management commands"""
    pass


@network.command()
@click.option('--network', required=True, help='Network CIDR (e.g., 192.168.1.0/24)')
def add_trusted(network: str):
    """Add a trusted network"""
    if network_policies.add_trusted_network(network):
        click.echo(f"✅ Added trusted network: {network}")
    else:
        click.echo(f"❌ Failed to add trusted network: {network}")


@network.command()
@click.option('--network', required=True, help='Network CIDR (e.g., 192.168.1.0/24)')
def add_blocked(network: str):
    """Add a blocked network"""
    if network_policies.add_blocked_network(network):
        click.echo(f"✅ Added blocked network: {network}")
    else:
        click.echo(f"❌ Failed to add blocked network: {network}")


@network.command()
@click.option('--ip', required=True, help='IP address to check')
def check_ip(ip: str):
    """Check if IP address is allowed"""
    allowed = network_policies.is_ip_allowed(ip)
    status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
    click.echo(f"IP {ip}: {status}")


@network.command()
def generate_iptables():
    """Generate iptables rules from network policies"""
    rules = network_policies.generate_iptables_rules()
    click.echo("# Generated iptables rules")
    for rule in rules:
        click.echo(rule)


@network.command()
def export_config():
    """Export network security configuration"""
    config = network_policies.export_config()
    click.echo(json.dumps(config, indent=2))


@cli.group()
def scan():
    """Vulnerability scanning commands"""
    pass


@scan.command()
@click.option('--image', required=True, help='Container image name')
@click.option('--tag', default='latest', help='Container image tag')
def container(image: str, tag: str):
    """Scan container image for vulnerabilities"""
    async def run_scan():
        try:
            click.echo(f"🔍 Scanning container {image}:{tag}...")
            result = await vulnerability_scanner.scan_container_image(image, tag)
            
            click.echo(f"\n📊 Scan Results for {image}:{tag}")
            click.echo(f"Total vulnerabilities: {result.total_vulnerabilities}")
            click.echo(f"Critical: {result.critical_count}")
            click.echo(f"High: {result.high_count}")
            click.echo(f"Medium: {result.medium_count}")
            click.echo(f"Low: {result.low_count}")
            click.echo(f"Risk Score: {result.get_risk_score():.1f}")
            
            if result.critical_count > 0:
                click.echo("\n⚠️  CRITICAL VULNERABILITIES FOUND!")
                for vuln in result.vulnerabilities:
                    if vuln.severity == SeverityLevel.CRITICAL:
                        click.echo(f"  - {vuln.cve_id}: {vuln.title}")
            
        except Exception as e:
            click.echo(f"❌ Scan failed: {str(e)}")
    
    asyncio.run(run_scan())


@scan.command()
@click.option('--image', required=True, help='Container image name')
@click.option('--tag', default='latest', help='Container image tag')
@click.option('--output', help='Output file for report')
def report(image: str, tag: str, output: Optional[str]):
    """Generate vulnerability report"""
    async def generate_report():
        try:
            report_data = await vulnerability_scanner.generate_vulnerability_report(image, tag)
            
            if output:
                with open(output, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                click.echo(f"📄 Report saved to {output}")
            else:
                click.echo(json.dumps(report_data, indent=2, default=str))
                
        except Exception as e:
            click.echo(f"❌ Report generation failed: {str(e)}")
    
    asyncio.run(generate_report())


@cli.group()
def secrets():
    """Secrets management commands"""
    pass


@secrets.command()
@click.option('--path', required=True, help='Secret path in Vault')
def get(path: str):
    """Get secret from Vault"""
    async def get_secret():
        try:
            vault_config = VaultConfig(
                url=security_settings.VAULT_URL,
                token=security_settings.VAULT_TOKEN,
                role_id=security_settings.VAULT_ROLE_ID,
                secret_id=security_settings.VAULT_SECRET_ID
            )
            
            secrets_manager = SecretsManager(vault_config)
            await secrets_manager.initialize()
            
            secret = await secrets_manager.get_secret(path)
            if secret:
                click.echo(json.dumps(secret, indent=2))
            else:
                click.echo(f"❌ Secret not found: {path}")
                
            await secrets_manager.close()
            
        except Exception as e:
            click.echo(f"❌ Failed to get secret: {str(e)}")
    
    asyncio.run(get_secret())


@secrets.command()
@click.option('--path', required=True, help='Secret path in Vault')
@click.option('--data', required=True, help='Secret data as JSON string')
def set(path: str, data: str):
    """Set secret in Vault"""
    async def set_secret():
        try:
            secret_data = json.loads(data)
            
            vault_config = VaultConfig(
                url=security_settings.VAULT_URL,
                token=security_settings.VAULT_TOKEN,
                role_id=security_settings.VAULT_ROLE_ID,
                secret_id=security_settings.VAULT_SECRET_ID
            )
            
            secrets_manager = SecretsManager(vault_config)
            await secrets_manager.initialize()
            
            success = await secrets_manager.set_secret(path, secret_data)
            if success:
                click.echo(f"✅ Secret set: {path}")
            else:
                click.echo(f"❌ Failed to set secret: {path}")
                
            await secrets_manager.close()
            
        except json.JSONDecodeError:
            click.echo("❌ Invalid JSON data")
        except Exception as e:
            click.echo(f"❌ Failed to set secret: {str(e)}")
    
    asyncio.run(set_secret())


@cli.group()
def incidents():
    """Security incident management commands"""
    pass


@incidents.command()
def list():
    """List security incidents"""
    click.echo("📋 Security Incidents:")
    click.echo("-" * 80)
    
    for incident in incident_response.incidents.values():
        status_icon = {
            IncidentStatus.DETECTED: "🔍",
            IncidentStatus.INVESTIGATING: "🔎",
            IncidentStatus.CONTAINED: "🔒",
            IncidentStatus.RESOLVED: "✅",
            IncidentStatus.CLOSED: "📁"
        }.get(incident.status, "❓")
        
        severity_icon = {
            IncidentSeverity.CRITICAL: "🚨",
            IncidentSeverity.HIGH: "⚠️",
            IncidentSeverity.MEDIUM: "⚡",
            IncidentSeverity.LOW: "ℹ️"
        }.get(incident.severity, "❓")
        
        click.echo(f"{status_icon} {severity_icon} [{incident.id[:8]}] {incident.title}")
        click.echo(f"    Type: {incident.incident_type.value}")
        click.echo(f"    Status: {incident.status.value}")
        click.echo(f"    Detected: {incident.detected_at.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo()


@incidents.command()
@click.option('--incident-id', required=True, help='Incident ID')
def show(incident_id: str):
    """Show incident details"""
    incident = incident_response.incidents.get(incident_id)
    if not incident:
        click.echo(f"❌ Incident not found: {incident_id}")
        return
    
    click.echo(f"🔍 Incident Details: {incident.id}")
    click.echo("-" * 50)
    click.echo(f"Title: {incident.title}")
    click.echo(f"Type: {incident.incident_type.value}")
    click.echo(f"Severity: {incident.severity.value}")
    click.echo(f"Status: {incident.status.value}")
    click.echo(f"Detected: {incident.detected_at}")
    click.echo(f"Reported by: {incident.reported_by}")
    
    if incident.assigned_to:
        click.echo(f"Assigned to: {incident.assigned_to}")
    
    if incident.affected_systems:
        click.echo(f"Affected systems: {', '.join(incident.affected_systems)}")
    
    if incident.indicators_of_compromise:
        click.echo("Indicators of Compromise:")
        for ioc in incident.indicators_of_compromise:
            click.echo(f"  - {ioc}")
    
    click.echo(f"\nDescription: {incident.description}")


@incidents.command()
@click.option('--incident-id', required=True, help='Incident ID')
@click.option('--status', required=True, type=click.Choice(['investigating', 'contained', 'resolved', 'closed']))
@click.option('--notes', help='Status update notes')
def update_status(incident_id: str, status: str, notes: Optional[str]):
    """Update incident status"""
    async def update():
        try:
            status_enum = IncidentStatus(status)
            await incident_response.update_incident_status(incident_id, status_enum, notes)
            click.echo(f"✅ Incident {incident_id} status updated to {status}")
        except ValueError as e:
            click.echo(f"❌ {str(e)}")
        except Exception as e:
            click.echo(f"❌ Failed to update status: {str(e)}")
    
    asyncio.run(update())


@incidents.command()
@click.option('--incident-id', required=True, help='Incident ID')
@click.option('--output', help='Output file for report')
def generate_report(incident_id: str, output: Optional[str]):
    """Generate incident report"""
    async def generate():
        try:
            report = await incident_response.generate_incident_report(incident_id)
            
            if output:
                with open(output, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                click.echo(f"📄 Report saved to {output}")
            else:
                click.echo(json.dumps(report, indent=2, default=str))
                
        except Exception as e:
            click.echo(f"❌ Report generation failed: {str(e)}")
    
    asyncio.run(generate())


@cli.command()
def status():
    """Show overall security status"""
    click.echo("🔒 Campus Security System Status")
    click.echo("=" * 40)
    
    # Network policies status
    click.echo(f"📡 Network Policies:")
    click.echo(f"  Trusted networks: {len(network_policies.trusted_networks)}")
    click.echo(f"  Blocked networks: {len(network_policies.blocked_networks)}")
    click.echo(f"  Firewall rules: {len(network_policies.firewall_rules)}")
    
    # Vulnerability scanning status
    click.echo(f"\n🔍 Vulnerability Scanning:")
    click.echo(f"  Scanned images: {len(vulnerability_scanner.scan_results)}")
    total_vulns = sum(scan.total_vulnerabilities for scan in vulnerability_scanner.scan_results.values())
    click.echo(f"  Total vulnerabilities: {total_vulns}")
    
    # Incident response status
    click.echo(f"\n🚨 Incident Response:")
    active_incidents = len([i for i in incident_response.incidents.values() 
                           if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]])
    click.echo(f"  Active incidents: {active_incidents}")
    click.echo(f"  Total incidents: {len(incident_response.incidents)}")
    
    # Configuration status
    click.echo(f"\n⚙️  Configuration:")
    click.echo(f"  Vault URL: {security_settings.VAULT_URL}")
    click.echo(f"  Vulnerability scanning: {'✅' if security_settings.ENABLE_VULNERABILITY_SCANNING else '❌'}")
    click.echo(f"  Auto incident response: {'✅' if security_settings.ENABLE_AUTO_INCIDENT_RESPONSE else '❌'}")


@cli.command()
def harden():
    """Apply security hardening configurations"""
    click.echo("🔒 Applying security hardening...")
    
    # Initialize default network policies
    network_policies.initialize_default_rules()
    click.echo("✅ Network security policies initialized")
    
    # Add common blocked networks (known malicious ranges)
    malicious_networks = [
        "0.0.0.0/8",      # Invalid source
        "127.0.0.0/8",    # Loopback (external)
        "169.254.0.0/16", # Link-local
        "224.0.0.0/4",    # Multicast
        "240.0.0.0/4"     # Reserved
    ]
    
    for network in malicious_networks:
        network_policies.add_blocked_network(network)
    
    click.echo("✅ Malicious networks blocked")
    
    # Set up rate limiting for different network zones
    network_policies.add_rate_limit_rule("0.0.0.0/0", 60, 120)  # Global: 60 req/min
    network_policies.add_rate_limit_rule("10.0.0.0/8", 300, 600)  # Internal: 300 req/min
    network_policies.add_rate_limit_rule("192.168.0.0/16", 300, 600)  # Private: 300 req/min
    
    click.echo("✅ Rate limiting rules configured")
    
    click.echo("\n🔒 Security hardening completed!")
    click.echo("Run 'python -m security.cli status' to verify configuration")


if __name__ == '__main__':
    cli()