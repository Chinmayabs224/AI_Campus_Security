# Edge Model Deployment System

A comprehensive edge model deployment system for AI Campus Security that provides secure model distribution, versioning, rollback capabilities, performance monitoring, and automated update scheduling.

## Overview

The Edge Model Deployment System consists of three main components:

1. **EdgeModelDeploymentSystem** - Core deployment and device management
2. **ModelDriftDetector** - Performance monitoring and drift detection
3. **AutomatedUpdateScheduler** - Automated update scheduling and rollout strategies

## Features

### Core Deployment Features
- ✅ Secure model distribution to edge devices
- ✅ Model versioning and rollback capabilities
- ✅ Device registration and authentication
- ✅ Encrypted model transfer
- ✅ Concurrent deployment management
- ✅ Health monitoring and validation

### Performance Monitoring
- ✅ Real-time performance metrics collection
- ✅ Model drift detection using statistical methods
- ✅ Automated alert generation
- ✅ Performance degradation tracking
- ✅ Baseline establishment and comparison

### Automated Updates
- ✅ Multiple rollout strategies (Canary, Blue-Green, Rolling)
- ✅ Scheduled update windows
- ✅ Automatic rollback on failure
- ✅ Health checks and validation
- ✅ Maintenance window management

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Edge Model Deployment System                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Deployment    │  │  Drift Detector │  │ Update Scheduler│  │
│  │     System      │  │                 │  │                 │  │
│  │                 │  │                 │  │                 │  │
│  │ • Device Mgmt   │  │ • Performance   │  │ • Rollout       │  │
│  │ • Model Dist    │  │   Monitoring    │  │   Strategies    │  │
│  │ • Versioning    │  │ • Drift         │  │ • Scheduling    │  │
│  │ • Security      │  │   Detection     │  │ • Health Checks │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        Data Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   SQLite DB     │  │   File Storage  │  │   Monitoring    │  │
│  │                 │  │                 │  │     Data        │  │
│  │ • Devices       │  │ • Model Files   │  │                 │  │
│  │ • Versions      │  │ • Configs       │  │ • Metrics       │  │
│  │ • Jobs          │  │ • Logs          │  │ • Alerts        │  │
│  │ • Schedules     │  │ • Backups       │  │ • Reports       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- Required packages: `asyncio`, `aiohttp`, `aiofiles`, `cryptography`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create configuration directories
mkdir -p ../../data/model_deployment
mkdir -p ../../data/drift_detection
mkdir -p ../../data/update_scheduler

# Copy configuration files
cp config/deployment_config.yaml ../../data/model_deployment/
cp config/drift_detection_config.yaml ../../data/drift_detection/
cp config/update_scheduler_config.yaml ../../data/update_scheduler/
```

## Configuration

### Deployment Configuration (`deployment_config.yaml`)
```yaml
deployment:
  base_dir: "../../data/model_deployment"
  max_concurrent_deployments: 5
  deployment_timeout_minutes: 30
  heartbeat_interval_seconds: 60
  performance_monitoring_interval_minutes: 15
  auto_rollback_on_failure: true
  require_device_authentication: true

security:
  use_tls: true
  verify_model_signatures: true
  encrypt_model_transfer: true
  api_key_length: 32

versioning:
  max_versions_per_device: 3
  cleanup_old_versions: true
  version_retention_days: 30

monitoring:
  performance_degradation_threshold: 0.1
  error_rate_threshold: 0.05
  latency_threshold_ms: 1000
  memory_usage_threshold: 0.8

scheduling:
  enable_auto_updates: true
  update_window_start: "02:00"
  update_window_end: "04:00"
  batch_size: 3
  rollout_strategy: "canary"  # canary, blue_green, rolling
```

### Drift Detection Configuration (`drift_detection_config.yaml`)
```yaml
drift_detection:
  output_dir: "../../data/drift_detection"
  monitoring_interval_hours: 6
  baseline_window_days: 7
  detection_window_days: 1
  min_samples_for_detection: 100
  enable_statistical_tests: true
  enable_visualization: true

thresholds:
  performance_drift:
    accuracy_threshold: 0.05
    precision_threshold: 0.05
    recall_threshold: 0.05
    latency_threshold: 0.2
  
  statistical_drift:
    psi_threshold: 0.2
    kl_divergence_threshold: 0.1
    js_divergence_threshold: 0.1
  
  severity_levels:
    low: 0.05
    medium: 0.1
    high: 0.2
    critical: 0.3

alerts:
  enable_notifications: true
  notification_channels: ["email", "webhook"]
  escalation_hours: 24
  auto_trigger_retraining: false
```

## Usage Examples

### Basic Deployment Workflow

```python
import asyncio
from edge_model_deployment import EdgeModelDeploymentSystem

async def main():
    # Initialize deployment system
    deployment_system = EdgeModelDeploymentSystem()
    
    # Register edge devices
    device_info = {
        'device_id': 'edge_001',
        'device_name': 'Campus North Gate',
        'ip_address': '192.168.1.100',
        'port': 8080,
        'hardware_info': {
            'gpu': 'NVIDIA Jetson Xavier NX',
            'memory_gb': 8,
            'storage_gb': 32
        }
    }
    
    device = await deployment_system.register_edge_device(device_info)
    print(f"Registered device: {device.device_id}")
    
    # Register model version
    model_version = await deployment_system.register_model_version(
        model_path="../../models/yolov8n_security.pt",
        performance_metrics={
            'mAP50': 0.85,
            'precision': 0.82,
            'recall': 0.78,
            'inference_time_ms': 45
        }
    )
    print(f"Registered model version: {model_version.version_id}")
    
    # Create deployment job
    deployment_job = await deployment_system.create_deployment_job(
        model_version_id=model_version.version_id,
        target_devices=[device.device_id],
        deployment_type='full'
    )
    
    # Execute deployment
    success = await deployment_system.execute_deployment_job(deployment_job.job_id)
    print(f"Deployment {'successful' if success else 'failed'}")
    
    # Start performance monitoring
    await deployment_system.start_performance_monitoring()
    
    # Cleanup
    await deployment_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Drift Detection

```python
import asyncio
from model_drift_detector import ModelDriftDetector

async def main():
    # Initialize drift detector
    drift_detector = ModelDriftDetector()
    
    # Start drift monitoring
    await drift_detector.start_drift_monitoring()
    
    # Let it run for monitoring
    await asyncio.sleep(3600)  # Monitor for 1 hour
    
    # Get drift summary
    summary = await drift_detector.get_drift_summary("edge_001", days=7)
    print(f"Drift summary: {summary}")
    
    # Stop monitoring
    await drift_detector.stop_drift_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

### Automated Updates

```python
import asyncio
from datetime import datetime, timedelta
from automated_update_scheduler import (
    AutomatedUpdateScheduler, UpdateStrategy, UpdatePriority
)

async def main():
    # Initialize scheduler
    scheduler = AutomatedUpdateScheduler()
    
    # Start scheduler
    await scheduler.start_scheduler()
    
    # Schedule an update
    schedule_id = await scheduler.schedule_update(
        model_version_id="v20241203_120000_abc12345",
        target_devices=["edge_001", "edge_002", "edge_003"],
        strategy=UpdateStrategy.CANARY,
        priority=UpdatePriority.HIGH,
        scheduled_time=datetime.now() + timedelta(minutes=30)
    )
    
    print(f"Scheduled update: {schedule_id}")
    
    # Monitor status
    while True:
        await asyncio.sleep(60)  # Check every minute
        status = await scheduler.get_update_status(schedule_id)
        print(f"Update status: {status['status']}")
        
        if status['status'] in ['completed', 'failed', 'cancelled']:
            break
    
    # Stop scheduler
    await scheduler.stop_scheduler()

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

### EdgeModelDeploymentSystem

#### Device Management
- `register_edge_device(device_info: Dict) -> EdgeDevice`
- `get_device_status(device_id: str) -> Dict`
- `get_device_performance_metrics(device: EdgeDevice) -> Dict`

#### Model Management
- `register_model_version(model_path: str, performance_metrics: Dict) -> ModelVersion`
- `cleanup_old_versions()`

#### Deployment
- `create_deployment_job(model_version_id: str, target_devices: List[str], deployment_type: str) -> DeploymentJob`
- `execute_deployment_job(job_id: str) -> bool`
- `get_deployment_status(job_id: str) -> Dict`

#### Monitoring
- `start_performance_monitoring()`
- `start_automated_update_scheduler()`

### ModelDriftDetector

#### Monitoring
- `start_drift_monitoring()`
- `stop_drift_monitoring()`

#### Analysis
- `get_drift_summary(device_id: str, days: int) -> Dict`
- `acknowledge_alert(alert_id: str) -> bool`

### AutomatedUpdateScheduler

#### Scheduling
- `schedule_update(model_version_id: str, target_devices: List[str], **kwargs) -> str`
- `cancel_update(schedule_id: str) -> bool`
- `get_update_status(schedule_id: str) -> Dict`

#### Control
- `start_scheduler()`
- `stop_scheduler()`

## Rollout Strategies

### Canary Deployment
- Deploys to a small percentage of devices first
- Monitors canary health before full rollout
- Automatic rollback if canary fails

### Blue-Green Deployment
- Deploys to all devices simultaneously
- Keeps previous version available for quick rollback
- Validation period before finalizing

### Rolling Deployment
- Deploys in batches with delays between batches
- Stops and rolls back if any batch fails
- Configurable batch size and delay

## Security Features

### Model Transfer Security
- TLS encryption for all communications
- Model file encryption during transfer
- Device authentication with API keys
- Model signature verification

### Access Control
- Device-specific API keys
- Role-based access control
- Audit logging for all operations
- Secure key management

## Monitoring and Alerting

### Performance Metrics
- Inference time and throughput
- Memory and CPU usage
- Error rates and accuracy
- Model drift indicators

### Alert Types
- Performance degradation
- Model drift detection
- Deployment failures
- Device connectivity issues

### Notification Channels
- Email notifications
- Webhook integrations
- Slack/Teams integration
- Custom notification handlers

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python -m pytest test_edge_deployment.py -v --asyncio-mode=auto

# Run specific test categories
python -m pytest test_edge_deployment.py::TestEdgeModelDeploymentSystem -v
python -m pytest test_edge_deployment.py::TestModelDriftDetector -v
python -m pytest test_edge_deployment.py::TestAutomatedUpdateScheduler -v
python -m pytest test_edge_deployment.py::TestIntegration -v
```

## Troubleshooting

### Common Issues

1. **Device Connection Failures**
   - Check network connectivity
   - Verify API keys
   - Check TLS certificates

2. **Model Transfer Failures**
   - Check available storage space
   - Verify model file integrity
   - Check encryption settings

3. **Drift Detection Issues**
   - Ensure sufficient baseline data
   - Check threshold configurations
   - Verify metric collection

4. **Scheduler Issues**
   - Check maintenance windows
   - Verify device availability
   - Check rollout strategy settings

### Logging

Logs are written to:
- `../../data/model_deployment/deployment.log`
- `../../data/drift_detection/drift_detection.log`
- `../../data/update_scheduler/update_scheduler.log`

### Database Management

SQLite databases are located at:
- `../../data/model_deployment/deployment.db`
- `../../data/drift_detection/drift_detection.db`
- `../../data/update_scheduler/update_scheduler.db`

## Performance Considerations

### Scalability
- Supports up to 1000 edge devices
- Concurrent deployment limit configurable
- Database optimization for high-volume operations

### Resource Usage
- Memory usage scales with number of devices
- Storage requirements depend on model sizes and retention policies
- Network bandwidth optimized with compression and encryption

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure security best practices are followed

## License

This project is part of the AI Campus Security system and follows the same licensing terms.