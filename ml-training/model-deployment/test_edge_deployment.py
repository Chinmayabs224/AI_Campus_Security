"""
Test Suite for Edge Model Deployment System

This module provides comprehensive tests for the edge model deployment system,
including unit tests, integration tests, and end-to-end deployment scenarios.
"""

import asyncio
import pytest
import tempfile
import shutil
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

# Import the modules to test
from edge_model_deployment import (
    EdgeModelDeploymentSystem, ModelVersion, EdgeDevice, DeploymentJob
)
from model_drift_detector import ModelDriftDetector, DriftMetrics
from automated_update_scheduler import (
    AutomatedUpdateScheduler, UpdateSchedule, UpdateStrategy, UpdatePriority
)


class TestEdgeModelDeploymentSystem:
    """Test cases for EdgeModelDeploymentSystem"""
    
    @pytest.fixture
    async def deployment_system(self):
        """Create a test deployment system"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test config
            config = {
                'deployment': {
                    'base_dir': temp_dir,
                    'max_concurrent_deployments': 5,
                    'deployment_timeout_minutes': 30,
                    'heartbeat_interval_seconds': 60,
                    'performance_monitoring_interval_minutes': 15,
                    'auto_rollback_on_failure': True,
                    'require_device_authentication': True
                },
                'security': {
                    'use_tls': True,
                    'verify_model_signatures': True,
                    'encrypt_model_transfer': True,
                    'api_key_length': 32
                },
                'versioning': {
                    'max_versions_per_device': 3,
                    'cleanup_old_versions': True,
                    'version_retention_days': 30
                },
                'monitoring': {
                    'performance_degradation_threshold': 0.1,
                    'error_rate_threshold': 0.05,
                    'latency_threshold_ms': 1000,
                    'memory_usage_threshold': 0.8
                },
                'scheduling': {
                    'enable_auto_updates': True,
                    'update_window_start': "02:00",
                    'update_window_end': "04:00",
                    'batch_size': 3,
                    'rollout_strategy': "canary"
                }
            }
            
            # Create temporary config file
            config_path = Path(temp_dir) / "test_config.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(config, f)
            
            system = EdgeModelDeploymentSystem(str(config_path))
            yield system
            
            # Cleanup
            await system.shutdown()
    
    @pytest.fixture
    def sample_model_file(self):
        """Create a sample model file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            # Write some dummy model data
            f.write(b"dummy_model_data_for_testing")
            model_path = f.name
        
        yield model_path
        
        # Cleanup
        Path(model_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_device_registration(self, deployment_system):
        """Test edge device registration"""
        device_info = {
            'device_id': 'test_device_001',
            'device_name': 'Test Device 1',
            'ip_address': '192.168.1.100',
            'port': 8080,
            'hardware_info': {
                'gpu': 'NVIDIA Jetson Xavier NX',
                'memory_gb': 8,
                'storage_gb': 32
            }
        }
        
        device = await deployment_system.register_edge_device(device_info)
        
        assert device.device_id == 'test_device_001'
        assert device.device_name == 'Test Device 1'
        assert device.ip_address == '192.168.1.100'
        assert device.port == 8080
        assert len(device.api_key) > 0
        assert device.hardware_info['gpu'] == 'NVIDIA Jetson Xavier NX'
        
        # Verify device is stored in database
        assert 'test_device_001' in deployment_system.edge_devices
    
    @pytest.mark.asyncio
    async def test_model_version_registration(self, deployment_system, sample_model_file):
        """Test model version registration"""
        performance_metrics = {
            'mAP50': 0.85,
            'precision': 0.82,
            'recall': 0.78,
            'inference_time_ms': 45
        }
        
        model_version = await deployment_system.register_model_version(
            sample_model_file, performance_metrics
        )
        
        assert model_version.version_id.startswith('v')
        assert model_version.model_size > 0
        assert len(model_version.model_hash) == 64  # SHA-256 hash length
        assert model_version.performance_metrics == performance_metrics
        assert Path(model_version.model_path).exists()
    
    @pytest.mark.asyncio
    async def test_deployment_job_creation(self, deployment_system, sample_model_file):
        """Test deployment job creation"""
        # Register devices
        device1_info = {
            'device_id': 'test_device_001',
            'device_name': 'Test Device 1',
            'ip_address': '192.168.1.100',
            'port': 8080
        }
        device2_info = {
            'device_id': 'test_device_002',
            'device_name': 'Test Device 2',
            'ip_address': '192.168.1.101',
            'port': 8080
        }
        
        await deployment_system.register_edge_device(device1_info)
        await deployment_system.register_edge_device(device2_info)
        
        # Register model version
        model_version = await deployment_system.register_model_version(
            sample_model_file, {'mAP50': 0.85}
        )
        
        # Create deployment job
        job = await deployment_system.create_deployment_job(
            model_version.version_id,
            ['test_device_001', 'test_device_002'],
            'full'
        )
        
        assert job.job_id.startswith('deploy_')
        assert job.model_version_id == model_version.version_id
        assert job.target_devices == ['test_device_001', 'test_device_002']
        assert job.deployment_type == 'full'
        assert job.status == 'pending'
        assert len(job.progress) == 2
    
    @pytest.mark.asyncio
    async def test_model_package_preparation(self, deployment_system, sample_model_file):
        """Test model package preparation for transfer"""
        # Register model and device
        model_version = await deployment_system.register_model_version(
            sample_model_file, {'mAP50': 0.85}
        )
        
        device_info = {
            'device_id': 'test_device_001',
            'device_name': 'Test Device 1',
            'ip_address': '192.168.1.100',
            'port': 8080
        }
        device = await deployment_system.register_edge_device(device_info)
        
        # Prepare model package
        package = await deployment_system._prepare_model_package(model_version, device)
        
        assert package['version_id'] == model_version.version_id
        assert package['model_hash'] == model_version.model_hash
        assert package['model_size'] == model_version.model_size
        assert 'model_data' in package
        assert package['encrypted'] == True  # Based on config
        assert package['performance_metrics'] == model_version.performance_metrics
    
    @pytest.mark.asyncio
    async def test_device_connectivity_check(self, deployment_system):
        """Test device connectivity checking"""
        device_info = {
            'device_id': 'test_device_001',
            'device_name': 'Test Device 1',
            'ip_address': '192.168.1.100',
            'port': 8080
        }
        device = await deployment_system.register_edge_device(device_info)
        
        # Mock the HTTP client to simulate device response
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Test successful connectivity
            mock_response = Mock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            is_connected = await deployment_system._check_device_connectivity(device)
            assert is_connected == True
            
            # Test failed connectivity
            mock_response.status = 500
            is_connected = await deployment_system._check_device_connectivity(device)
            assert is_connected == False
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, deployment_system):
        """Test performance monitoring functionality"""
        device_info = {
            'device_id': 'test_device_001',
            'device_name': 'Test Device 1',
            'ip_address': '192.168.1.100',
            'port': 8080
        }
        device = await deployment_system.register_edge_device(device_info)
        device.status = "online"
        device.current_model_version = "v20241203_120000_abc12345"
        
        # Mock device metrics response
        mock_metrics = {
            'inference_time_ms': 45.5,
            'memory_usage_mb': 512.0,
            'cpu_usage_percent': 35.2,
            'gpu_usage_percent': 78.9,
            'error_rate': 0.02,
            'throughput_fps': 22.1
        }
        
        with patch.object(deployment_system, '_get_device_performance_metrics', 
                         return_value=mock_metrics):
            await deployment_system._monitor_device_performance(device)
            
            # Check that metrics were stored
            assert device.performance_metrics == mock_metrics
            assert device.last_heartbeat is not None
    
    @pytest.mark.asyncio
    async def test_cleanup_old_versions(self, deployment_system, sample_model_file):
        """Test cleanup of old model versions"""
        # Register multiple model versions
        old_time = datetime.now() - timedelta(days=35)  # Older than retention period
        
        model_version = await deployment_system.register_model_version(
            sample_model_file, {'mAP50': 0.85}
        )
        
        # Manually set old creation time in database
        with sqlite3.connect(deployment_system.db_path) as conn:
            conn.execute('''
                UPDATE model_versions 
                SET created_at = ? 
                WHERE version_id = ?
            ''', (old_time.isoformat(), model_version.version_id))
        
        # Run cleanup
        await deployment_system.cleanup_old_versions()
        
        # Verify old version was cleaned up
        model_file = Path(model_version.model_path)
        assert not model_file.exists()


class TestModelDriftDetector:
    """Test cases for ModelDriftDetector"""
    
    @pytest.fixture
    async def drift_detector(self):
        """Create a test drift detector"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'drift_detection': {
                    'output_dir': temp_dir,
                    'monitoring_interval_hours': 6,
                    'baseline_window_days': 7,
                    'detection_window_days': 1,
                    'min_samples_for_detection': 10,  # Lower for testing
                    'enable_statistical_tests': True,
                    'enable_visualization': False  # Disable for testing
                },
                'thresholds': {
                    'performance_drift': {
                        'accuracy_threshold': 0.05,
                        'precision_threshold': 0.05,
                        'recall_threshold': 0.05,
                        'latency_threshold': 0.2
                    },
                    'statistical_drift': {
                        'psi_threshold': 0.2,
                        'kl_divergence_threshold': 0.1,
                        'js_divergence_threshold': 0.1
                    },
                    'severity_levels': {
                        'low': 0.05,
                        'medium': 0.1,
                        'high': 0.2,
                        'critical': 0.3
                    }
                },
                'alerts': {
                    'enable_notifications': False,  # Disable for testing
                    'notification_channels': [],
                    'escalation_hours': 24,
                    'auto_trigger_retraining': False
                }
            }
            
            config_path = Path(temp_dir) / "test_drift_config.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(config, f)
            
            detector = ModelDriftDetector(str(config_path))
            yield detector
            
            await detector.stop_drift_monitoring()
    
    @pytest.mark.asyncio
    async def test_baseline_establishment(self, drift_detector):
        """Test baseline performance establishment"""
        device_id = "test_device_001"
        model_version_id = "v20241203_120000_abc12345"
        
        # Mock historical performance data
        with patch.object(drift_detector, '_get_historical_performance') as mock_historical:
            # Generate mock historical data
            historical_data = []
            for i in range(100):  # 100 samples for reliable baseline
                historical_data.append({
                    'accuracy': 0.85 + np.random.normal(0, 0.02),
                    'precision': 0.82 + np.random.normal(0, 0.02),
                    'recall': 0.78 + np.random.normal(0, 0.02),
                    'inference_time_ms': 45 + np.random.normal(0, 5),
                    'error_rate': 0.02 + np.random.normal(0, 0.005)
                })
            
            mock_historical.return_value = historical_data
            
            await drift_detector._establish_baseline(device_id, model_version_id)
            
            # Verify baseline was established
            baseline_metrics = await drift_detector._get_baseline_performance(device_id, model_version_id)
            
            assert baseline_metrics is not None
            assert 'accuracy' in baseline_metrics
            assert 'precision' in baseline_metrics
            assert 'recall' in baseline_metrics
            assert baseline_metrics['accuracy']['mean'] > 0.8
            assert baseline_metrics['accuracy']['std'] > 0
    
    @pytest.mark.asyncio
    async def test_drift_calculation(self, drift_detector):
        """Test drift metrics calculation"""
        device_id = "test_device_001"
        model_version_id = "v20241203_120000_abc12345"
        
        # Mock baseline metrics
        baseline_metrics = {
            'accuracy': {'mean': 0.85, 'std': 0.02, 'count': 100},
            'precision': {'mean': 0.82, 'std': 0.02, 'count': 100},
            'recall': {'mean': 0.78, 'std': 0.02, 'count': 100},
            'inference_time_ms': {'mean': 45.0, 'std': 5.0, 'count': 100}
        }
        
        # Mock recent data with drift
        recent_data = []
        for i in range(50):
            recent_data.append({
                'accuracy': 0.75 + np.random.normal(0, 0.02),  # Significant drop
                'precision': 0.72 + np.random.normal(0, 0.02),  # Significant drop
                'recall': 0.68 + np.random.normal(0, 0.02),    # Significant drop
                'inference_time_ms': 55 + np.random.normal(0, 5)  # Increase in latency
            })
        
        drift_metrics = await drift_detector._calculate_drift_metrics(
            device_id, model_version_id, baseline_metrics, recent_data
        )
        
        assert drift_metrics.device_id == device_id
        assert drift_metrics.model_version_id == model_version_id
        assert drift_metrics.accuracy_drift > 0.1  # Should detect significant drift
        assert drift_metrics.precision_drift > 0.1
        assert drift_metrics.recall_drift > 0.1
        assert drift_metrics.drift_detected == True
        assert drift_metrics.drift_severity in ['medium', 'high', 'critical']
    
    @pytest.mark.asyncio
    async def test_psi_calculation(self, drift_detector):
        """Test Population Stability Index calculation"""
        baseline_mean = 0.85
        baseline_std = 0.02
        
        # Test with similar distribution (no drift)
        recent_values = [0.85 + np.random.normal(0, 0.02) for _ in range(50)]
        psi = drift_detector._calculate_psi(baseline_mean, baseline_std, recent_values)
        assert psi < 0.1  # Should be low for similar distributions
        
        # Test with different distribution (drift)
        recent_values = [0.75 + np.random.normal(0, 0.05) for _ in range(50)]
        psi = drift_detector._calculate_psi(baseline_mean, baseline_std, recent_values)
        assert psi > 0.2  # Should be high for different distributions
    
    @pytest.mark.asyncio
    async def test_drift_alert_generation(self, drift_detector):
        """Test drift alert generation"""
        drift_metrics = DriftMetrics(
            device_id="test_device_001",
            model_version_id="v20241203_120000_abc12345",
            timestamp=datetime.now(),
            accuracy_drift=0.15,  # High drift
            precision_drift=0.12,
            recall_drift=0.10,
            latency_drift=0.25,
            psi_score=0.3,
            kl_divergence=0.15,
            js_divergence=0.12,
            drift_severity="high",
            drift_detected=True,
            sample_size=50,
            confidence_interval=(0.70, 0.80)
        )
        
        await drift_detector._generate_drift_alert(drift_metrics)
        
        # Verify alert was stored
        with sqlite3.connect(drift_detector.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM drift_alerts')
            alert_count = cursor.fetchone()[0]
            assert alert_count > 0
    
    @pytest.mark.asyncio
    async def test_drift_summary(self, drift_detector):
        """Test drift summary generation"""
        device_id = "test_device_001"
        
        # Create some test drift metrics
        drift_metrics = DriftMetrics(
            device_id=device_id,
            model_version_id="v20241203_120000_abc12345",
            timestamp=datetime.now(),
            accuracy_drift=0.08,
            precision_drift=0.06,
            recall_drift=0.05,
            latency_drift=0.15,
            psi_score=0.15,
            kl_divergence=0.08,
            js_divergence=0.06,
            drift_severity="medium",
            drift_detected=True,
            sample_size=100,
            confidence_interval=(0.75, 0.85)
        )
        
        await drift_detector._store_drift_metrics(drift_metrics)
        
        # Get drift summary
        summary = await drift_detector.get_drift_summary(device_id, days=7)
        
        assert summary['device_id'] == device_id
        assert summary['period_days'] == 7
        assert summary['total_checks'] >= 1
        assert summary['drift_detections'] >= 1
        assert 'severity_breakdown' in summary


class TestAutomatedUpdateScheduler:
    """Test cases for AutomatedUpdateScheduler"""
    
    @pytest.fixture
    async def update_scheduler(self):
        """Create a test update scheduler"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'scheduler': {
                    'output_dir': temp_dir,
                    'check_interval_minutes': 1,  # Fast for testing
                    'max_concurrent_updates': 10,
                    'default_update_window_start': "00:00",  # Always in window for testing
                    'default_update_window_end': "23:59",
                    'enable_maintenance_windows': False  # Disable for testing
                },
                'rollout_strategies': {
                    'canary': {
                        'canary_percentage': 0.1,
                        'canary_duration_minutes': 1,  # Fast for testing
                        'success_threshold': 0.95
                    },
                    'rolling': {
                        'batch_size': 2,
                        'batch_delay_minutes': 1,  # Fast for testing
                        'max_unavailable_percentage': 0.2
                    }
                },
                'health_monitoring': {
                    'health_check_interval_minutes': 1,
                    'health_check_timeout_seconds': 30,
                    'failure_threshold': 3,
                    'recovery_threshold': 2
                },
                'rollback': {
                    'auto_rollback_enabled': True,
                    'rollback_trigger_failure_rate': 0.2,
                    'rollback_timeout_minutes': 1,
                    'preserve_rollback_data_days': 7
                },
                'notifications': {
                    'enable_notifications': False,  # Disable for testing
                    'notification_channels': [],
                    'notify_on_start': False,
                    'notify_on_completion': False,
                    'notify_on_failure': False,
                    'notify_on_rollback': False
                }
            }
            
            config_path = Path(temp_dir) / "test_scheduler_config.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(config, f)
            
            scheduler = AutomatedUpdateScheduler(str(config_path))
            yield scheduler
            
            await scheduler.stop_scheduler()
    
    @pytest.mark.asyncio
    async def test_update_scheduling(self, update_scheduler):
        """Test update scheduling functionality"""
        model_version_id = "v20241203_120000_abc12345"
        target_devices = ["test_device_001", "test_device_002", "test_device_003"]
        
        schedule_id = await update_scheduler.schedule_update(
            model_version_id=model_version_id,
            target_devices=target_devices,
            strategy=UpdateStrategy.ROLLING,
            priority=UpdatePriority.HIGH,
            scheduled_time=datetime.now() + timedelta(seconds=5)
        )
        
        assert schedule_id.startswith("update_")
        
        # Get schedule status
        status = await update_scheduler.get_update_status(schedule_id)
        
        assert status is not None
        assert status['schedule_id'] == schedule_id
        assert status['model_version_id'] == model_version_id
        assert status['target_devices'] == target_devices
        assert status['strategy'] == UpdateStrategy.ROLLING.value
        assert status['priority'] == UpdatePriority.HIGH.value
        assert status['status'] == 'pending'
    
    @pytest.mark.asyncio
    async def test_update_cancellation(self, update_scheduler):
        """Test update cancellation"""
        schedule_id = await update_scheduler.schedule_update(
            model_version_id="v20241203_120000_abc12345",
            target_devices=["test_device_001"],
            scheduled_time=datetime.now() + timedelta(hours=1)  # Future schedule
        )
        
        # Cancel the update
        success = await update_scheduler.cancel_update(schedule_id)
        assert success == True
        
        # Verify cancellation
        status = await update_scheduler.get_update_status(schedule_id)
        assert status['status'] == 'cancelled'
    
    @pytest.mark.asyncio
    async def test_schedule_due_check(self, update_scheduler):
        """Test schedule due checking logic"""
        current_time = datetime.now()
        
        # Create a schedule that is due
        schedule = UpdateSchedule(
            schedule_id="test_schedule_001",
            model_version_id="v20241203_120000_abc12345",
            target_devices=["test_device_001"],
            strategy=UpdateStrategy.IMMEDIATE,
            priority=UpdatePriority.NORMAL,
            scheduled_time=current_time - timedelta(minutes=1),  # Past time
            update_window_start=datetime.strptime("00:00", "%H:%M").time(),
            update_window_end=datetime.strptime("23:59", "%H:%M").time()
        )
        
        is_due = await update_scheduler._is_schedule_due(schedule, current_time)
        assert is_due == True
        
        # Create a schedule that is not due
        schedule.scheduled_time = current_time + timedelta(hours=1)  # Future time
        is_due = await update_scheduler._is_schedule_due(schedule, current_time)
        assert is_due == False
    
    @pytest.mark.asyncio
    async def test_health_check_logic(self, update_scheduler):
        """Test health check logic"""
        from automated_update_scheduler import UpdateJob
        
        # Create a job with good metrics
        job = UpdateJob(
            job_id="test_job_001",
            schedule_id="test_schedule_001",
            device_id="test_device_001",
            model_version_id="v20241203_120000_abc12345",
            pre_update_metrics={
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.78,
                'inference_time_ms': 45.0,
                'error_rate': 0.02
            },
            post_update_metrics={
                'accuracy': 0.86,  # Slight improvement
                'precision': 0.83,
                'recall': 0.79,
                'inference_time_ms': 44.0,  # Slight improvement
                'error_rate': 0.015  # Improvement
            }
        )
        
        health_passed = await update_scheduler._perform_health_check(job)
        assert health_passed == True
        
        # Create a job with degraded metrics
        job.post_update_metrics = {
            'accuracy': 0.70,  # Significant degradation
            'precision': 0.68,
            'recall': 0.65,
            'inference_time_ms': 60.0,  # Significant increase
            'error_rate': 0.08  # High error rate
        }
        
        health_passed = await update_scheduler._perform_health_check(job)
        assert health_passed == False


class TestIntegration:
    """Integration tests for the complete edge deployment system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_deployment_workflow(self):
        """Test complete end-to-end deployment workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test model file
            model_file = Path(temp_dir) / "test_model.pt"
            with open(model_file, 'wb') as f:
                f.write(b"dummy_model_data_for_testing")
            
            # Initialize deployment system
            config = {
                'deployment': {
                    'base_dir': temp_dir,
                    'max_concurrent_deployments': 5,
                    'deployment_timeout_minutes': 30,
                    'heartbeat_interval_seconds': 60,
                    'performance_monitoring_interval_minutes': 15,
                    'auto_rollback_on_failure': True,
                    'require_device_authentication': True
                },
                'security': {
                    'use_tls': False,  # Disable for testing
                    'verify_model_signatures': False,
                    'encrypt_model_transfer': False,
                    'api_key_length': 32
                },
                'versioning': {
                    'max_versions_per_device': 3,
                    'cleanup_old_versions': True,
                    'version_retention_days': 30
                },
                'monitoring': {
                    'performance_degradation_threshold': 0.1,
                    'error_rate_threshold': 0.05,
                    'latency_threshold_ms': 1000,
                    'memory_usage_threshold': 0.8
                },
                'scheduling': {
                    'enable_auto_updates': True,
                    'update_window_start': "02:00",
                    'update_window_end': "04:00",
                    'batch_size': 3,
                    'rollout_strategy': "canary"
                }
            }
            
            config_path = Path(temp_dir) / "test_config.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(config, f)
            
            deployment_system = EdgeModelDeploymentSystem(str(config_path))
            
            try:
                # Step 1: Register edge devices
                device_infos = [
                    {
                        'device_id': f'test_device_{i:03d}',
                        'device_name': f'Test Device {i}',
                        'ip_address': f'192.168.1.{100 + i}',
                        'port': 8080,
                        'hardware_info': {
                            'gpu': 'NVIDIA Jetson Xavier NX',
                            'memory_gb': 8,
                            'storage_gb': 32
                        }
                    }
                    for i in range(1, 4)
                ]
                
                devices = []
                for device_info in device_infos:
                    device = await deployment_system.register_edge_device(device_info)
                    devices.append(device)
                
                assert len(devices) == 3
                
                # Step 2: Register model version
                performance_metrics = {
                    'mAP50': 0.85,
                    'precision': 0.82,
                    'recall': 0.78,
                    'inference_time_ms': 45
                }
                
                model_version = await deployment_system.register_model_version(
                    str(model_file), performance_metrics
                )
                
                assert model_version.version_id is not None
                assert Path(model_version.model_path).exists()
                
                # Step 3: Create deployment job
                target_device_ids = [device.device_id for device in devices]
                
                deployment_job = await deployment_system.create_deployment_job(
                    model_version.version_id,
                    target_device_ids,
                    'full'
                )
                
                assert deployment_job.job_id is not None
                assert deployment_job.target_devices == target_device_ids
                assert deployment_job.status == 'pending'
                
                # Step 4: Mock deployment execution (since we don't have real devices)
                with patch.object(deployment_system, '_check_device_connectivity', return_value=True), \
                     patch.object(deployment_system, '_transfer_model_to_device', return_value=True), \
                     patch.object(deployment_system, '_activate_model_on_device', return_value=True):
                    
                    success = await deployment_system.execute_deployment_job(deployment_job.job_id)
                    assert success == True
                
                # Step 5: Verify deployment status
                final_status = await deployment_system.get_deployment_status(deployment_job.job_id)
                assert final_status['status'] == 'completed'
                
                # Step 6: Verify device status updates
                for device_id in target_device_ids:
                    device_status = await deployment_system.get_device_status(device_id)
                    assert device_status['current_model_version'] == model_version.version_id
                
            finally:
                await deployment_system.shutdown()


# Test runner
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])