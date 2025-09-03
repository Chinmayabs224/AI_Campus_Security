"""
Network security policies and firewall rules for campus security system.
"""
import ipaddress
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class NetworkZone(Enum):
    """Network security zones."""
    EDGE = "edge"
    INTERNAL = "internal"
    DMZ = "dmz"
    MANAGEMENT = "management"
    EXTERNAL = "external"


class Protocol(Enum):
    """Network protocols."""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"


@dataclass
class FirewallRule:
    """Firewall rule definition."""
    name: str
    source_zone: NetworkZone
    destination_zone: NetworkZone
    source_networks: List[str]
    destination_networks: List[str]
    ports: List[int]
    protocol: Protocol
    action: str  # ALLOW, DENY, LOG
    priority: int = 100
    enabled: bool = True


class NetworkSecurityPolicies:
    """Network security policies manager."""
    
    def __init__(self):
        self.firewall_rules: List[FirewallRule] = []
        self.trusted_networks: Set[str] = set()
        self.blocked_networks: Set[str] = set()
        self.rate_limit_rules: Dict[str, Dict] = {}
        
    def initialize_default_rules(self):
        """Initialize default security rules."""
        default_rules = [
            # Edge device communication
            FirewallRule(
                name="edge_to_api",
                source_zone=NetworkZone.EDGE,
                destination_zone=NetworkZone.INTERNAL,
                source_networks=["10.0.1.0/24"],  # Edge network
                destination_networks=["10.0.2.0/24"],  # API network
                ports=[8000, 443],
                protocol=Protocol.TCP,
                action="ALLOW",
                priority=10
            ),
            
            # Web dashboard access
            FirewallRule(
                name="web_dashboard_access",
                source_zone=NetworkZone.EXTERNAL,
                destination_zone=NetworkZone.DMZ,
                source_networks=["0.0.0.0/0"],
                destination_networks=["10.0.3.0/24"],  # DMZ network
                ports=[80, 443],
                protocol=Protocol.TCP,
                action="ALLOW",
                priority=20
            ),
            
            # Database access (internal only)
            FirewallRule(
                name="database_access",
                source_zone=NetworkZone.INTERNAL,
                destination_zone=NetworkZone.INTERNAL,
                source_networks=["10.0.2.0/24"],  # API network
                destination_networks=["10.0.4.0/24"],  # Database network
                ports=[5432],
                protocol=Protocol.TCP,
                action="ALLOW",
                priority=30
            ),
            
            # Redis access (internal only)
            FirewallRule(
                name="redis_access",
                source_zone=NetworkZone.INTERNAL,
                destination_zone=NetworkZone.INTERNAL,
                source_networks=["10.0.2.0/24"],  # API network
                destination_networks=["10.0.5.0/24"],  # Cache network
                ports=[6379],
                protocol=Protocol.TCP,
                action="ALLOW",
                priority=40
            ),
            
            # Management access
            FirewallRule(
                name="management_ssh",
                source_zone=NetworkZone.MANAGEMENT,
                destination_zone=NetworkZone.INTERNAL,
                source_networks=["10.0.100.0/24"],  # Management network
                destination_networks=["10.0.0.0/16"],  # All internal
                ports=[22],
                protocol=Protocol.TCP,
                action="ALLOW",
                priority=50
            ),
            
            # Block all other traffic
            FirewallRule(
                name="default_deny",
                source_zone=NetworkZone.EXTERNAL,
                destination_zone=NetworkZone.INTERNAL,
                source_networks=["0.0.0.0/0"],
                destination_networks=["10.0.0.0/16"],
                ports=[],
                protocol=Protocol.TCP,
                action="DENY",
                priority=1000
            )
        ]
        
        self.firewall_rules.extend(default_rules)
        logger.info("Default firewall rules initialized", rules_count=len(default_rules))
    
    def add_trusted_network(self, network: str) -> bool:
        """Add a trusted network."""
        try:
            # Validate network format
            ipaddress.ip_network(network, strict=False)
            self.trusted_networks.add(network)
            logger.info("Trusted network added", network=network)
            return True
        except ValueError as e:
            logger.error("Invalid network format", network=network, error=str(e))
            return False
    
    def add_blocked_network(self, network: str) -> bool:
        """Add a blocked network."""
        try:
            # Validate network format
            ipaddress.ip_network(network, strict=False)
            self.blocked_networks.add(network)
            logger.info("Blocked network added", network=network)
            return True
        except ValueError as e:
            logger.error("Invalid network format", network=network, error=str(e))
            return False
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check blocked networks first
            for blocked_network in self.blocked_networks:
                if ip in ipaddress.ip_network(blocked_network, strict=False):
                    logger.warning("IP blocked by network policy", ip=ip_address, network=blocked_network)
                    return False
            
            # Check trusted networks
            for trusted_network in self.trusted_networks:
                if ip in ipaddress.ip_network(trusted_network, strict=False):
                    return True
            
            # Default policy for unknown IPs
            return True
            
        except ValueError as e:
            logger.error("Invalid IP address format", ip=ip_address, error=str(e))
            return False
    
    def add_rate_limit_rule(self, network: str, requests_per_minute: int, burst_size: int = None):
        """Add rate limiting rule for a network."""
        if burst_size is None:
            burst_size = requests_per_minute * 2
            
        self.rate_limit_rules[network] = {
            "requests_per_minute": requests_per_minute,
            "burst_size": burst_size
        }
        logger.info("Rate limit rule added", network=network, rpm=requests_per_minute)
    
    def get_rate_limit_for_ip(self, ip_address: str) -> Optional[Dict]:
        """Get rate limit configuration for IP address."""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            for network, limits in self.rate_limit_rules.items():
                if ip in ipaddress.ip_network(network, strict=False):
                    return limits
            
            return None
            
        except ValueError:
            return None
    
    def generate_iptables_rules(self) -> List[str]:
        """Generate iptables rules from firewall policies."""
        rules = []
        
        # Clear existing rules
        rules.append("iptables -F")
        rules.append("iptables -X")
        rules.append("iptables -t nat -F")
        rules.append("iptables -t nat -X")
        
        # Default policies
        rules.append("iptables -P INPUT DROP")
        rules.append("iptables -P FORWARD DROP")
        rules.append("iptables -P OUTPUT ACCEPT")
        
        # Allow loopback
        rules.append("iptables -A INPUT -i lo -j ACCEPT")
        rules.append("iptables -A OUTPUT -o lo -j ACCEPT")
        
        # Allow established connections
        rules.append("iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT")
        
        # Generate rules from policies
        for rule in sorted(self.firewall_rules, key=lambda x: x.priority):
            if not rule.enabled:
                continue
                
            for src_net in rule.source_networks:
                for dst_net in rule.destination_networks:
                    if rule.ports:
                        for port in rule.ports:
                            iptables_rule = self._generate_iptables_rule(rule, src_net, dst_net, port)
                            if iptables_rule:
                                rules.append(iptables_rule)
                    else:
                        iptables_rule = self._generate_iptables_rule(rule, src_net, dst_net)
                        if iptables_rule:
                            rules.append(iptables_rule)
        
        return rules
    
    def _generate_iptables_rule(self, rule: FirewallRule, src_net: str, dst_net: str, port: int = None) -> str:
        """Generate individual iptables rule."""
        action_map = {
            "ALLOW": "ACCEPT",
            "DENY": "DROP",
            "LOG": "LOG"
        }
        
        iptables_rule = f"iptables -A INPUT -s {src_net} -d {dst_net}"
        
        if rule.protocol != Protocol.ICMP:
            iptables_rule += f" -p {rule.protocol.value}"
            
        if port:
            iptables_rule += f" --dport {port}"
            
        iptables_rule += f" -j {action_map.get(rule.action, 'DROP')}"
        
        return iptables_rule
    
    def export_config(self) -> Dict:
        """Export network security configuration."""
        return {
            "firewall_rules": [
                {
                    "name": rule.name,
                    "source_zone": rule.source_zone.value,
                    "destination_zone": rule.destination_zone.value,
                    "source_networks": rule.source_networks,
                    "destination_networks": rule.destination_networks,
                    "ports": rule.ports,
                    "protocol": rule.protocol.value,
                    "action": rule.action,
                    "priority": rule.priority,
                    "enabled": rule.enabled
                }
                for rule in self.firewall_rules
            ],
            "trusted_networks": list(self.trusted_networks),
            "blocked_networks": list(self.blocked_networks),
            "rate_limit_rules": self.rate_limit_rules
        }


# Global network security policies instance
network_policies = NetworkSecurityPolicies()