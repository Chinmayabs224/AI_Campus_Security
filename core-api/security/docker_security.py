"""
Docker security scanning and container hardening for campus security system.
"""
import asyncio
import json
import os
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import structlog
import aiofiles
import yaml

logger = structlog.get_logger()


@dataclass
class DockerSecurityPolicy:
    """Docker security policy configuration."""
    name: str
    description: str
    enabled: bool = True
    severity: str = "medium"
    remediation: str = ""


class DockerSecurityScanner:
    """Docker security scanner and policy enforcer."""
    
    def __init__(self):
        self.security_policies: List[DockerSecurityPolicy] = []
        self.scan_results: Dict[str, Any] = {}
        self.initialize_policies()
    
    def initialize_policies(self):
        """Initialize Docker security policies."""
        policies = [
            DockerSecurityPolicy(
                name="no_root_user",
                description="Container should not run as root user",
                severity="high",
                remediation="Add USER directive with non-root user in Dockerfile"
            ),
            DockerSecurityPolicy(
                name="read_only_filesystem",
                description="Container filesystem should be read-only when possible",
                severity="medium",
                remediation="Use --read-only flag or readOnlyRootFilesystem in security context"
            ),
            DockerSecurityPolicy(
                name="no_privileged_mode",
                description="Container should not run in privileged mode",
                severity="critical",
                remediation="Remove --privileged flag from container configuration"
            ),
            DockerSecurityPolicy(
                name="limited_capabilities",
                description="Container should drop unnecessary Linux capabilities",
                severity="high",
                remediation="Use --cap-drop=ALL and only add required capabilities"
            ),
            DockerSecurityPolicy(
                name="no_host_network",
                description="Container should not use host network mode",
                severity="high",
                remediation="Remove --network=host and use bridge or custom networks"
            ),
            DockerSecurityPolicy(
                name="no_host_pid",
                description="Container should not share host PID namespace",
                severity="high",
                remediation="Remove --pid=host from container configuration"
            ),
            DockerSecurityPolicy(
                name="resource_limits",
                description="Container should have CPU and memory limits",
                severity="medium",
                remediation="Set --memory and --cpus limits in container configuration"
            ),
            DockerSecurityPolicy(
                name="health_check",
                description="Container should have health check configured",
                severity="low",
                remediation="Add HEALTHCHECK directive to Dockerfile"
            ),
            DockerSecurityPolicy(
                name="minimal_base_image",
                description="Use minimal base images (alpine, distroless)",
                severity="medium",
                remediation="Switch to alpine or distroless base images"
            ),
            DockerSecurityPolicy(
                name="no_secrets_in_env",
                description="Secrets should not be passed via environment variables",
                severity="high",
                remediation="Use Docker secrets or external secret management"
            )
        ]
        
        self.security_policies = policies
        logger.info("Docker security policies initialized", count=len(policies))
    
    async def scan_dockerfile(self, dockerfile_path: str) -> Dict[str, Any]:
        """Scan Dockerfile for security issues."""
        logger.info("Scanning Dockerfile for security issues", path=dockerfile_path)
        
        if not os.path.exists(dockerfile_path):
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
        
        scan_result = {
            "dockerfile_path": dockerfile_path,
            "scan_date": datetime.utcnow().isoformat(),
            "violations": [],
            "recommendations": [],
            "score": 100
        }
        
        try:
            async with aiofiles.open(dockerfile_path, 'r') as f:
                dockerfile_content = await f.read()
            
            # Analyze Dockerfile content
            violations = await self._analyze_dockerfile_content(dockerfile_content)
            scan_result["violations"] = violations
            
            # Generate recommendations
            recommendations = self._generate_dockerfile_recommendations(violations)
            scan_result["recommendations"] = recommendations
            
            # Calculate security score
            score = self._calculate_security_score(violations)
            scan_result["score"] = score
            
            # Store scan result
            self.scan_results[dockerfile_path] = scan_result
            
            logger.info("Dockerfile security scan completed",
                       path=dockerfile_path,
                       violations=len(violations),
                       score=score)
            
            return scan_result
            
        except Exception as e:
            logger.error("Dockerfile security scan failed", path=dockerfile_path, error=str(e))
            raise
    
    async def _analyze_dockerfile_content(self, content: str) -> List[Dict[str, Any]]:
        """Analyze Dockerfile content for security violations."""
        violations = []
        lines = content.split('\n')
        
        has_user_directive = False
        has_healthcheck = False
        runs_as_root = True
        uses_privileged = False
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for USER directive
            if line.upper().startswith('USER '):
                has_user_directive = True
                user = line.split()[1]
                if user != 'root' and user != '0':
                    runs_as_root = False
            
            # Check for HEALTHCHECK
            if line.upper().startswith('HEALTHCHECK '):
                has_healthcheck = True
            
            # Check for secrets in ENV
            if line.upper().startswith('ENV '):
                env_vars = line[4:].strip()
                if any(keyword in env_vars.lower() for keyword in ['password', 'secret', 'key', 'token']):
                    violations.append({
                        "policy": "no_secrets_in_env",
                        "line": line_num,
                        "content": line,
                        "severity": "high",
                        "message": "Potential secret found in environment variable"
                    })
            
            # Check for privileged operations
            if 'privileged' in line.lower():
                uses_privileged = True
                violations.append({
                    "policy": "no_privileged_mode",
                    "line": line_num,
                    "content": line,
                    "severity": "critical",
                    "message": "Privileged mode detected"
                })
            
            # Check for host network usage
            if '--network=host' in line or '--net=host' in line:
                violations.append({
                    "policy": "no_host_network",
                    "line": line_num,
                    "content": line,
                    "severity": "high",
                    "message": "Host network mode detected"
                })
            
            # Check for host PID usage
            if '--pid=host' in line:
                violations.append({
                    "policy": "no_host_pid",
                    "line": line_num,
                    "content": line,
                    "severity": "high",
                    "message": "Host PID namespace sharing detected"
                })
        
        # Check for missing USER directive
        if not has_user_directive or runs_as_root:
            violations.append({
                "policy": "no_root_user",
                "line": 0,
                "content": "",
                "severity": "high",
                "message": "Container runs as root user"
            })
        
        # Check for missing HEALTHCHECK
        if not has_healthcheck:
            violations.append({
                "policy": "health_check",
                "line": 0,
                "content": "",
                "severity": "low",
                "message": "No health check configured"
            })
        
        return violations
    
    def _generate_dockerfile_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on violations."""
        recommendations = []
        
        violation_policies = {v["policy"] for v in violations}
        
        for policy in self.security_policies:
            if policy.name in violation_policies and policy.remediation:
                recommendations.append(policy.remediation)
        
        # Add general recommendations
        if any(v["severity"] == "critical" for v in violations):
            recommendations.append("Address critical security issues immediately before deployment")
        
        if len(violations) > 5:
            recommendations.append("Consider using a security-hardened base image")
        
        return recommendations
    
    def _calculate_security_score(self, violations: List[Dict[str, Any]]) -> int:
        """Calculate security score based on violations."""
        score = 100
        
        severity_penalties = {
            "critical": 30,
            "high": 15,
            "medium": 8,
            "low": 3
        }
        
        for violation in violations:
            penalty = severity_penalties.get(violation["severity"], 5)
            score -= penalty
        
        return max(0, score)
    
    async def generate_secure_dockerfile(self, base_image: str, app_name: str, 
                                       output_path: str, config: Dict[str, Any] = None) -> str:
        """Generate a security-hardened Dockerfile."""
        config = config or {}
        
        dockerfile_content = f"""# Security-hardened Dockerfile for {app_name}
# Generated by Campus Security System

# Use minimal base image
FROM {base_image}

# Create non-root user
RUN addgroup -g 1001 -S {app_name} && \\
    adduser -u 1001 -S {app_name} -G {app_name}

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown={app_name}:{app_name} . /app/

# Install dependencies (if needed)
{config.get('install_commands', '# No additional dependencies')}

# Set security-focused environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PATH="/app:$PATH"

# Expose port (if specified)
{f'EXPOSE {config.get("port", 8000)}' if config.get("port") else '# No port exposed'}

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD {config.get('healthcheck_cmd', 'curl -f http://localhost:8000/health || exit 1')}

# Switch to non-root user
USER {app_name}

# Set read-only filesystem (can be overridden at runtime)
# Use: docker run --read-only --tmpfs /tmp --tmpfs /var/run

# Start application
CMD {config.get('start_command', '["python", "main.py"]')}
"""
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(dockerfile_content)
        
        logger.info("Secure Dockerfile generated", output_path=output_path)
        return dockerfile_content
    
    async def scan_container_runtime(self, container_id: str) -> Dict[str, Any]:
        """Scan running container for security issues."""
        logger.info("Scanning container runtime security", container_id=container_id)
        
        scan_result = {
            "container_id": container_id,
            "scan_date": datetime.utcnow().isoformat(),
            "violations": [],
            "recommendations": []
        }
        
        try:
            # Get container information
            inspect_cmd = ["docker", "inspect", container_id]
            process = await asyncio.create_subprocess_exec(
                *inspect_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Failed to inspect container: {stderr.decode()}")
            
            container_info = json.loads(stdout.decode())[0]
            
            # Analyze container configuration
            violations = self._analyze_container_config(container_info)
            scan_result["violations"] = violations
            
            # Generate recommendations
            recommendations = self._generate_runtime_recommendations(violations)
            scan_result["recommendations"] = recommendations
            
            logger.info("Container runtime security scan completed",
                       container_id=container_id,
                       violations=len(violations))
            
            return scan_result
            
        except Exception as e:
            logger.error("Container runtime security scan failed",
                        container_id=container_id, error=str(e))
            raise
    
    def _analyze_container_config(self, container_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze container configuration for security violations."""
        violations = []
        
        config = container_info.get("Config", {})
        host_config = container_info.get("HostConfig", {})
        
        # Check if running as root
        if config.get("User") in ["", "root", "0"]:
            violations.append({
                "policy": "no_root_user",
                "severity": "high",
                "message": "Container is running as root user"
            })
        
        # Check privileged mode
        if host_config.get("Privileged", False):
            violations.append({
                "policy": "no_privileged_mode",
                "severity": "critical",
                "message": "Container is running in privileged mode"
            })
        
        # Check network mode
        network_mode = host_config.get("NetworkMode", "")
        if network_mode == "host":
            violations.append({
                "policy": "no_host_network",
                "severity": "high",
                "message": "Container is using host network mode"
            })
        
        # Check PID mode
        pid_mode = host_config.get("PidMode", "")
        if pid_mode == "host":
            violations.append({
                "policy": "no_host_pid",
                "severity": "high",
                "message": "Container is sharing host PID namespace"
            })
        
        # Check resource limits
        memory = host_config.get("Memory", 0)
        cpu_quota = host_config.get("CpuQuota", 0)
        
        if memory == 0:
            violations.append({
                "policy": "resource_limits",
                "severity": "medium",
                "message": "No memory limit set"
            })
        
        if cpu_quota == 0:
            violations.append({
                "policy": "resource_limits",
                "severity": "medium",
                "message": "No CPU limit set"
            })
        
        # Check capabilities
        cap_add = host_config.get("CapAdd", [])
        if cap_add and "ALL" in cap_add:
            violations.append({
                "policy": "limited_capabilities",
                "severity": "high",
                "message": "Container has all capabilities enabled"
            })
        
        # Check read-only filesystem
        read_only = host_config.get("ReadonlyRootfs", False)
        if not read_only:
            violations.append({
                "policy": "read_only_filesystem",
                "severity": "medium",
                "message": "Container filesystem is not read-only"
            })
        
        return violations
    
    def _generate_runtime_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate runtime security recommendations."""
        recommendations = []
        
        violation_policies = {v["policy"] for v in violations}
        
        if "no_root_user" in violation_policies:
            recommendations.append("Recreate container with non-root user")
        
        if "no_privileged_mode" in violation_policies:
            recommendations.append("Remove --privileged flag and use specific capabilities instead")
        
        if "resource_limits" in violation_policies:
            recommendations.append("Set memory and CPU limits: --memory=512m --cpus=1.0")
        
        if "read_only_filesystem" in violation_policies:
            recommendations.append("Use --read-only flag with tmpfs mounts for writable directories")
        
        return recommendations
    
    async def generate_security_report(self, output_file: str):
        """Generate comprehensive Docker security report."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "security_policies": [
                {
                    "name": policy.name,
                    "description": policy.description,
                    "severity": policy.severity,
                    "enabled": policy.enabled
                }
                for policy in self.security_policies
            ],
            "scan_results": self.scan_results,
            "summary": {
                "total_scans": len(self.scan_results),
                "average_score": sum(
                    result.get("score", 0) for result in self.scan_results.values()
                ) / len(self.scan_results) if self.scan_results else 0
            }
        }
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        logger.info("Docker security report generated", output_file=output_file)


# Global Docker security scanner instance
docker_security_scanner = DockerSecurityScanner()