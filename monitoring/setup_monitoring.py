#!/usr/bin/env python3
"""
Setup script for monitoring infrastructure.
"""
import os
import subprocess
import sys
import time
import requests
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e.stderr}")
        return None

def wait_for_service(url, service_name, timeout=60):
    """Wait for a service to become available."""
    print(f"Waiting for {service_name} to be available at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {service_name} is available")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    print(f"✗ {service_name} failed to start within {timeout} seconds")
    return False

def setup_monitoring_stack():
    """Set up the complete monitoring stack."""
    print("Setting up Campus Security Monitoring Stack...")
    
    # Create necessary directories
    directories = [
        "monitoring/prometheus",
        "monitoring/grafana/provisioning/datasources",
        "monitoring/grafana/provisioning/dashboards",
        "monitoring/grafana/dashboards",
        "monitoring/alertmanager"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Start monitoring services
    print("\nStarting monitoring services...")
    result = run_command("docker-compose -f docker-compose.monitoring.yml up -d", cwd="monitoring")
    
    if result is None:
        print("✗ Failed to start monitoring services")
        return False
    
    print("✓ Monitoring services started")
    
    # Wait for services to be available
    services = [
        ("http://localhost:9090", "Prometheus"),
        ("http://localhost:3001", "Grafana"),
        ("http://localhost:9093", "Alertmanager"),
        ("http://localhost:16686", "Jaeger")
    ]
    
    all_services_ready = True
    for url, service_name in services:
        if not wait_for_service(url, service_name):
            all_services_ready = False
    
    if not all_services_ready:
        print("\n✗ Some services failed to start properly")
        return False
    
    print("\n✓ All monitoring services are running successfully!")
    
    # Display access information
    print("\n" + "="*50)
    print("MONITORING SERVICES ACCESS INFORMATION")
    print("="*50)
    print("Prometheus:   http://localhost:9090")
    print("Grafana:      http://localhost:3001 (admin/admin123)")
    print("Alertmanager: http://localhost:9093")
    print("Jaeger:       http://localhost:16686")
    print("="*50)
    
    return True

def install_python_dependencies():
    """Install required Python packages for monitoring."""
    print("Installing Python monitoring dependencies...")
    
    packages = [
        "prometheus-client",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-jaeger-thrift",
        "opentelemetry-instrumentation-fastapi",
        "opentelemetry-instrumentation-asyncpg",
        "opentelemetry-instrumentation-redis"
    ]
    
    for package in packages:
        result = run_command(f"pip install {package}")
        if result is not None:
            print(f"✓ Installed {package}")
        else:
            print(f"✗ Failed to install {package}")
            return False
    
    return True

def create_monitoring_config():
    """Create additional monitoring configuration files."""
    
    # Create systemd service for edge metrics (if on Linux)
    if sys.platform.startswith('linux'):
        systemd_service = """[Unit]
Description=Campus Security Edge Metrics
After=network.target

[Service]
Type=simple
User=security
WorkingDirectory=/opt/campus-security/edge-services
ExecStart=/usr/bin/python3 -m monitoring.metrics
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        try:
            with open("/tmp/campus-security-edge-metrics.service", "w") as f:
                f.write(systemd_service)
            print("✓ Created systemd service template at /tmp/campus-security-edge-metrics.service")
            print("  Copy to /etc/systemd/system/ and enable with: sudo systemctl enable campus-security-edge-metrics")
        except Exception as e:
            print(f"✗ Failed to create systemd service: {e}")
    
    # Create monitoring health check script
    health_check_script = """#!/bin/bash
# Health check script for monitoring services

echo "Checking monitoring services health..."

services=(
    "prometheus:9090"
    "grafana:3001"
    "alertmanager:9093"
    "jaeger:16686"
)

all_healthy=true

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s -f "http://localhost:$port" > /dev/null; then
        echo "✓ $name is healthy"
    else
        echo "✗ $name is not responding"
        all_healthy=false
    fi
done

if $all_healthy; then
    echo "All monitoring services are healthy"
    exit 0
else
    echo "Some monitoring services are unhealthy"
    exit 1
fi
"""
    
    try:
        with open("monitoring/health_check.sh", "w") as f:
            f.write(health_check_script)
        os.chmod("monitoring/health_check.sh", 0o755)
        print("✓ Created monitoring health check script")
    except Exception as e:
        print(f"✗ Failed to create health check script: {e}")

def main():
    """Main setup function."""
    print("Campus Security System - Monitoring Setup")
    print("="*50)
    
    # Check if Docker is available
    if run_command("docker --version") is None:
        print("✗ Docker is not available. Please install Docker first.")
        sys.exit(1)
    
    if run_command("docker-compose --version") is None:
        print("✗ Docker Compose is not available. Please install Docker Compose first.")
        sys.exit(1)
    
    print("✓ Docker and Docker Compose are available")
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("✗ Failed to install Python dependencies")
        sys.exit(1)
    
    # Create additional configuration
    create_monitoring_config()
    
    # Set up monitoring stack
    if not setup_monitoring_stack():
        print("✗ Failed to set up monitoring stack")
        sys.exit(1)
    
    print("\n🎉 Monitoring setup completed successfully!")
    print("\nNext steps:")
    print("1. Access Grafana at http://localhost:3001 (admin/admin123)")
    print("2. Import additional dashboards as needed")
    print("3. Configure alert notification channels in Alertmanager")
    print("4. Run './monitoring/health_check.sh' to verify all services")

if __name__ == "__main__":
    main()