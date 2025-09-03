"""
Integration Tests for Automated Model Retraining System
Tests the complete retraining workflow
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import json

from retraining_pipeline import ModelRetrainingPipeline, RetrainingJob
from data_collection_pipeline import DataCollectionPipeline
from automated_labeling import AutomatedLabelingWorkflow
from ab_testing_framework import ABTestingFramework, DeploymentStrategy
from retraining_scheduler import RetrainingScheduler

class TestRetrainingSystem:
    """Integration tests for the retraining system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration"""
        config = {
            'retraining': {
                'output_dir': str(temp_dir / 'retraining'),
                'trigger_conditions': {
                    'min_new_samples': 10,  # Lower threshold for testing
                    'performance_degradation_threshold': 0.05,
                    'false_positive_rate_threshold': 0.15,
                    'days_since_last_training': 1
                },
                'data_collection_window_days': 1,
                'min_samples_per_class': 2,
                'validation_split': 0.2
            },
            'validation': {
                'performance_thresholds': {
                    'min_mAP50_improvement': 0.01,
                    'max_performance_degradation': 0.02,
                    'min_security_score': 0.5,
                    'max_false_positive_rate': 0.2
                }
            },
            'training': {
                'epochs': 5,  # Reduced for testing
                'patience': 3,
                'model_sizes': ['n'],
                'use_transfer_learning': True
            },
            'deployment': {
                'require_manual_approval': False,
                'backup_current_model': True,
                'rollback_on_failure': True
            }
        }
        
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    @pytest.mark.asyncio
    async def test_data_collection_pipeline(self, temp_dir):
        """Test data collection pipeline"""
        # Create test configuration
        config = {
            'data_collection': {
                'output_dir': str(temp_dir / 'collection'),
                'min_confidence_threshold': 0.3,
                'max_confidence_threshold': 0.9,
                'collection_window_days': 1,
                'max_samples_per_class': 100,
                'frame_extraction_interval': 30
            },
            'quality_filters': {
                'min_resolution': [320, 240],
                'max_blur_threshold': 50,
                'min_brightness': 20,
                'max_brightness': 235
            }
        }
        
        config_path = temp_dir / 'collection_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Initialize pipeline
        pipeline = DataCollectionPipeline(str(config_path))
        
        # Test data collection
        incidents = await pipeline.collect_recent_incidents(days_back=1)
        
        assert isinstance(incidents, list)
        assert len(incidents) > 0
        
        # Test metadata saving
        metadata_path = await pipeline.save_collection_metadata(incidents)
        assert Path(metadata_path).exists()
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert 'collection_timestamp' in metadata
        assert 'total_incidents' in metadata
        assert metadata['total_incidents'] == len(incidents)
    
    @pytest.mark.asyncio
    async def test_automated_labeling_workflow(self, temp_dir):
        """Test automated labeling workflow"""
        # Create test configuration
        config = {
            'labeling': {
                'output_dir': str(temp_dir / 'labeling'),
                'confidence_threshold': 0.25,
                'nms_threshold': 0.45,
                'ensemble_agreement_threshold': 0.6,
                'batch_size': 10
            },
            'classes': {
                0: 'person',
                1: 'suspicious_activity',
                2: 'violence'
            },
            'security_priority_classes': [1, 2],
            'models': {
                'primary_model': 'yolov8n.pt',  # Use pretrained model for testing
                'ensemble_models': []
            },
            'validation': {
                'require_human_validation': False,
                'auto_approve_threshold': 0.8,
                'batch_size': 10
            }
        }
        
        config_path = temp_dir / 'labeling_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Initialize workflow
        workflow = AutomatedLabelingWorkflow(str(config_path))
        await workflow.load_models()
        
        # Create test incident data
        test_incidents = [
            {
                'incident_id': 'test_incident_1',
                'video_path': str(temp_dir / 'test_video.mp4')
            }
        ]
        
        # Create dummy video file
        (temp_dir / 'test_video.mp4').touch()
        
        # Test task creation
        tasks = await workflow.create_labeling_tasks(test_incidents)
        assert isinstance(tasks, list)
        
        # Test batch processing
        if tasks:
            results = await workflow.process_labeling_batch(batch_size=5)
            assert 'processed' in results
            assert 'auto_approved' in results
        
        # Test statistics
        stats = await workflow.get_labeling_statistics()
        assert 'overall' in stats
        assert 'daily' in stats
    
    @pytest.mark.asyncio
    async def test_retraining_pipeline(self, test_config, temp_dir):
        """Test complete retraining pipeline"""
        # Initialize pipeline
        pipeline = ModelRetrainingPipeline(test_config)
        
        # Test trigger checking
        job = await pipeline.check_retraining_triggers()
        
        if job:
            assert isinstance(job, RetrainingJob)
            assert job.job_id is not None
            assert job.trigger_reason is not None
            
            # Test job execution (mock mode)
            # In a real test, we would need actual training data and models
            print(f"Would execute retraining job: {job.job_id}")
            print(f"Trigger reason: {job.trigger_reason}")
        else:
            print("No retraining triggers detected (expected in test environment)")
    
    @pytest.mark.asyncio
    async def test_ab_testing_framework(self, temp_dir):
        """Test A/B testing framework"""
        # Create test configuration
        config = {
            'ab_testing': {
                'output_dir': str(temp_dir / 'ab_testing'),
                'default_duration_hours': 1,  # Short duration for testing
                'min_sample_size': 10,
                'confidence_level': 0.95,
                'statistical_power': 0.8
            },
            'deployment': {
                'edge_nodes': ['test_node_1', 'test_node_2'],
                'rollback_timeout_minutes': 5,
                'health_check_interval_seconds': 10
            },
            'success_criteria': {
                'min_detection_rate_improvement': 0.01,
                'max_false_positive_increase': 0.02,
                'max_inference_time_increase_ms': 100,
                'min_confidence_improvement': 0.02
            },
            'rollback_criteria': {
                'max_error_rate': 0.1,
                'max_false_positive_rate': 0.2,
                'max_inference_time_ms': 1000,
                'min_detection_rate': 0.5
            }
        }
        
        config_path = temp_dir / 'ab_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Initialize framework
        framework = ABTestingFramework(str(config_path))
        
        # Create dummy model files
        model_a_path = temp_dir / 'model_a.pt'
        model_b_path = temp_dir / 'model_b.pt'
        model_a_path.touch()
        model_b_path.touch()
        
        # Test A/B test creation
        test_config = await framework.create_ab_test(
            model_a_path=str(model_a_path),
            model_b_path=str(model_b_path),
            strategy=DeploymentStrategy.CANARY,
            traffic_split={"model_a": 0.8, "model_b": 0.2},
            duration_hours=1
        )
        
        assert test_config.test_id is not None
        assert test_config.strategy == DeploymentStrategy.CANARY
        
        # Test starting A/B test
        success = await framework.start_ab_test(test_config.test_id)
        assert success is True
        
        # Test getting active tests
        active_tests = await framework.get_active_tests()
        assert len(active_tests) > 0
        
        # Test stopping test
        await framework.stop_test(test_config.test_id, "test_complete")
        
        # Verify test was stopped
        active_tests = await framework.get_active_tests()
        assert len([t for t in active_tests if t['test_id'] == test_config.test_id]) == 0
    
    @pytest.mark.asyncio
    async def test_retraining_scheduler(self, temp_dir):
        """Test retraining scheduler"""
        # Create scheduler configuration
        config = {
            'scheduler': {
                'check_interval_hours': 1,
                'max_concurrent_jobs': 1,
                'enable_auto_deployment': False,
                'notification_webhook': None,
                'backup_retention_days': 7
            }
        }
        
        config_path = temp_dir / 'scheduler_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Initialize scheduler
        scheduler = RetrainingScheduler(str(config_path))
        
        # Test status
        status = scheduler.get_status()
        assert 'running' in status
        assert 'active_jobs' in status
        assert 'config' in status
        
        # Test manual trigger (would normally start a job)
        job_id = await scheduler.trigger_manual_retraining("test_trigger")
        print(f"Manual retraining triggered: {job_id}")
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, test_config, temp_dir):
        """Test complete end-to-end retraining workflow"""
        print("Testing end-to-end retraining workflow...")
        
        # 1. Data Collection
        collection_pipeline = DataCollectionPipeline()
        incidents = await collection_pipeline.collect_recent_incidents(days_back=1)
        print(f"Collected {len(incidents)} incidents")
        
        # 2. Automated Labeling
        labeling_workflow = AutomatedLabelingWorkflow()
        await labeling_workflow.load_models()
        
        if incidents:
            tasks = await labeling_workflow.create_labeling_tasks(incidents[:2])  # Limit for testing
            print(f"Created {len(tasks)} labeling tasks")
            
            if tasks:
                results = await labeling_workflow.process_labeling_batch(batch_size=5)
                print(f"Processed labeling batch: {results}")
        
        # 3. Retraining Pipeline
        retraining_pipeline = ModelRetrainingPipeline(test_config)
        job = await retraining_pipeline.check_retraining_triggers()
        
        if job:
            print(f"Retraining job created: {job.job_id}")
            print(f"Trigger: {job.trigger_reason}")
            
            # In a full test, we would execute the job
            # success = await retraining_pipeline.execute_retraining_job(job)
            # print(f"Retraining job success: {success}")
        
        # 4. A/B Testing (if retraining was successful)
        ab_framework = ABTestingFramework()
        
        # Create dummy models for testing
        model_a = temp_dir / 'current_model.pt'
        model_b = temp_dir / 'new_model.pt'
        model_a.touch()
        model_b.touch()
        
        test_config = await ab_framework.create_ab_test(
            model_a_path=str(model_a),
            model_b_path=str(model_b),
            strategy=DeploymentStrategy.CANARY
        )
        
        print(f"A/B test created: {test_config.test_id}")
        
        print("End-to-end workflow test completed")

def run_integration_tests():
    """Run all integration tests"""
    import subprocess
    import sys
    
    # Run pytest
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        __file__, 
        '-v', 
        '--asyncio-mode=auto'
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")

if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        print("Running simple integration test...")
        
        # Test data collection
        pipeline = DataCollectionPipeline()
        incidents = await pipeline.collect_recent_incidents(days_back=1)
        print(f"Data collection test: {len(incidents)} incidents collected")
        
        # Test labeling workflow
        workflow = AutomatedLabelingWorkflow()
        await workflow.load_models()
        stats = await workflow.get_labeling_statistics()
        print(f"Labeling workflow test: {stats}")
        
        # Test A/B testing framework
        framework = ABTestingFramework()
        active_tests = await framework.get_active_tests()
        print(f"A/B testing test: {len(active_tests)} active tests")
        
        print("Simple integration test completed")
    
    asyncio.run(simple_test())