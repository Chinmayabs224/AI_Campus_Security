#!/usr/bin/env python3
"""
Test Suite for Enhanced Security Model Validation
Comprehensive tests for the enhanced validation framework.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_security_validator import EnhancedSecurityValidator
from yolo_variant_comparator import YOLOVariantComparator

class TestEnhancedSecurityValidator(unittest.TestCase):
    """Test cases for EnhancedSecurityValidator."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_file = self.test_dir / "test_config.yaml"
        self.model_file = self.test_dir / "test_model.pt"
        
        # Create mock config
        config_content = """
dataset:
  yolo_data_dir: "test_data"
classes:
  violence: 0
  theft: 1
  intrusion: 2
  loitering: 3
  normal: 4
"""
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Create mock model file
        self.model_file.touch()
        
        # Create test data directory structure
        (self.test_dir / "test_data").mkdir()
        (self.test_dir / "enhanced_validation_results").mkdir()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    @patch('enhanced_security_validator.YOLO')
    def test_validator_initialization(self, mock_yolo):
        """Test validator initialization."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        validator = EnhancedSecurityValidator(
            str(self.model_file), 
            str(self.config_file)
        )
        
        self.assertEqual(validator.model_path, self.model_file)
        self.assertEqual(validator.config_path, self.config_file)
        self.assertIsNotNone(validator.config)
        self.assertIn('violence', validator.security_events)
        self.assertIn('initial_target', validator.false_positive_thresholds)
    
    @patch('enhanced_security_validator.YOLO')
    def test_false_positive_analysis(self, mock_yolo):
        """Test false positive rate analysis."""
        # Mock YOLO model and validation results
        mock_model = Mock()
        mock_results = Mock()
        mock_results.box.mp = 0.8  # 80% precision = 20% FP rate
        mock_results.box.mr = 0.7  # 70% recall
        mock_results.box.map50 = 0.75
        
        mock_model.val.return_value = mock_results
        mock_yolo.return_value = mock_model
        
        validator = EnhancedSecurityValidator(
            str(self.model_file), 
            str(self.config_file)
        )
        
        # Test false positive analysis
        fp_analysis = validator._analyze_false_positive_rates()
        
        self.assertIn('threshold_analysis', fp_analysis)
        self.assertIn('optimal_thresholds', fp_analysis)
        
        # Check that thresholds were tested
        threshold_analysis = fp_analysis['threshold_analysis']
        self.assertGreater(len(threshold_analysis), 0)
        
        # Check that false positive rates were calculated
        for threshold_data in threshold_analysis.values():
            if isinstance(threshold_data, dict):
                self.assertIn('false_positive_rate', threshold_data)
                self.assertIn('precision', threshold_data)
    
    @patch('enhanced_security_validator.YOLO')
    def test_environmental_adaptation(self, mock_yolo):
        """Test environmental adaptation testing."""
        mock_model = Mock()
        mock_results = [Mock()]
        mock_results[0].boxes = Mock()
        mock_results[0].boxes.conf = Mock()
        mock_results[0].boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8, 0.7])
        
        mock_model.predict.return_value = mock_results
        mock_yolo.return_value = mock_model
        
        validator = EnhancedSecurityValidator(
            str(self.model_file), 
            str(self.config_file)
        )
        
        # Test environmental adaptation
        env_results = validator._test_environmental_adaptation()
        
        self.assertIn('condition_performance', env_results)
        self.assertIn('adaptation_analysis', env_results)
        
        # Check that different conditions were tested
        condition_performance = env_results['condition_performance']
        expected_conditions = ['daylight', 'low_light', 'artificial_light']
        
        for condition in expected_conditions:
            if condition in condition_performance:
                self.assertIn('0.3', condition_performance[condition])  # Confidence threshold
    
    @patch('enhanced_security_validator.YOLO')
    def test_security_scenario_validation(self, mock_yolo):
        """Test security scenario validation."""
        mock_model = Mock()
        mock_results = Mock()
        mock_results.box.ap_class_index = np.array([0, 1, 2])  # violence, theft, intrusion
        mock_results.box.ap50 = np.array([0.8, 0.7, 0.75])
        mock_results.box.p = np.array([0.85, 0.75, 0.8])
        mock_results.box.r = np.array([0.9, 0.8, 0.85])
        
        mock_model.val.return_value = mock_results
        mock_yolo.return_value = mock_model
        
        validator = EnhancedSecurityValidator(
            str(self.model_file), 
            str(self.config_file)
        )
        
        # Test security scenario validation
        scenario_results = validator._validate_security_scenarios()
        
        self.assertIn('scenario_summary', scenario_results)
        
        # Check that security events were tested
        for event_type in ['violence', 'theft', 'intrusion']:
            detection_key = f'{event_type}_detection'
            if detection_key in scenario_results:
                event_data = scenario_results[detection_key]
                if isinstance(event_data, dict):
                    self.assertIn('priority', event_data)
    
    @patch('enhanced_security_validator.YOLO')
    def test_compliance_assessment(self, mock_yolo):
        """Test compliance assessment for requirements 6.2 and 6.3."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        validator = EnhancedSecurityValidator(
            str(self.model_file), 
            str(self.config_file)
        )
        
        # Mock validation results
        validation_results = {
            'false_positive_analysis': {
                'optimal_thresholds': {
                    'initial_target': {
                        'threshold': 0.5,
                        'false_positive_rate': 0.25,
                        'recall': 0.8
                    },
                    'improved_target': {
                        'threshold': 0.7,
                        'false_positive_rate': 0.08,
                        'recall': 0.7
                    }
                }
            },
            'environmental_adaptation': {
                'adaptation_analysis': {
                    'stability_score': 0.75,
                    'best_condition': 'daylight',
                    'worst_condition': 'low_light'
                },
                'threshold_recommendations': {
                    'daylight': {'recommended_threshold': 0.5},
                    'low_light': {'recommended_threshold': 0.4}
                }
            }
        }
        
        # Test compliance assessment
        compliance = validator._assess_compliance(validation_results)
        
        self.assertIn('requirement_6_2_compliance', compliance)
        self.assertIn('requirement_6_3_compliance', compliance)
        self.assertIn('overall_compliance', compliance)
        
        # Check requirement 6.2 compliance
        req_6_2 = compliance['requirement_6_2_compliance']
        self.assertTrue(req_6_2['initial_target_met'])
        self.assertTrue(req_6_2['improved_target_achievable'])
        self.assertEqual(req_6_2['compliance_status'], 'fully_compliant')
        
        # Check requirement 6.3 compliance
        req_6_3 = compliance['requirement_6_3_compliance']
        self.assertTrue(req_6_3['adaptation_capability'])
        self.assertTrue(req_6_3['threshold_adaptation'])
    
    @patch('enhanced_security_validator.YOLO')
    def test_threshold_optimization(self, mock_yolo):
        """Test detection threshold optimization."""
        mock_model = Mock()
        mock_results = Mock()
        mock_results.box.mp = 0.8
        mock_results.box.mr = 0.7
        mock_results.box.map50 = 0.75
        
        mock_model.val.return_value = mock_results
        mock_yolo.return_value = mock_model
        
        validator = EnhancedSecurityValidator(
            str(self.model_file), 
            str(self.config_file)
        )
        
        # Test threshold optimization
        optimization_results = validator._optimize_detection_thresholds()
        
        self.assertIn('global_optimization', optimization_results)
        self.assertIn('class_specific_optimization', optimization_results)
        self.assertIn('deployment_recommendations', optimization_results)
        
        # Check global optimization
        global_opt = optimization_results['global_optimization']
        if 'threshold_metrics' in global_opt:
            self.assertGreater(len(global_opt['threshold_metrics']), 0)


class TestYOLOVariantComparator(unittest.TestCase):
    """Test cases for YOLOVariantComparator."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_file = self.test_dir / "test_config.yaml"
        
        # Create mock config
        config_content = """
dataset:
  yolo_data_dir: "test_data"
classes:
  violence: 0
  theft: 1
  normal: 2
"""
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Create test data directory
        (self.test_dir / "test_data").mkdir()
        (self.test_dir / "yolo_variant_comparison").mkdir()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_comparator_initialization(self):
        """Test comparator initialization."""
        comparator = YOLOVariantComparator(str(self.config_file))
        
        self.assertEqual(comparator.config_path, self.config_file)
        self.assertIsNotNone(comparator.config)
        self.assertIn('accuracy', comparator.comparison_dimensions)
        self.assertIn('critical_classes', comparator.security_criteria)
    
    @patch('yolo_variant_comparator.YOLO')
    def test_add_yolo_variant(self, mock_yolo):
        """Test adding YOLO variants."""
        mock_model = Mock()
        mock_model.model.parameters.return_value = [Mock(numel=Mock(return_value=1000000))]
        mock_yolo.return_value = mock_model
        
        comparator = YOLOVariantComparator(str(self.config_file))
        
        # Create mock model file
        model_file = self.test_dir / "yolov8n.pt"
        model_file.touch()
        
        # Add variant
        comparator.add_yolo_variant("nano", str(model_file), "YOLOv8 Nano")
        
        self.assertIn("nano", comparator.yolo_variants)
        variant_info = comparator.yolo_variants["nano"]["variant_info"]
        self.assertEqual(variant_info["variant_type"], "yolov8n")
        self.assertEqual(variant_info["size_category"], "nano")
    
    @patch('yolo_variant_comparator.YOLO')
    def test_variant_info_extraction(self, mock_yolo):
        """Test variant information extraction."""
        mock_model = Mock()
        mock_model.model.parameters.return_value = [Mock(numel=Mock(return_value=3000000))]
        mock_yolo.return_value = mock_model
        
        comparator = YOLOVariantComparator(str(self.config_file))
        
        # Create mock model files with different names
        model_files = {
            "yolov8n.pt": ("nano", "yolov8n"),
            "yolov8s.pt": ("small", "yolov8s"),
            "yolov8m.pt": ("medium", "yolov8m")
        }
        
        for filename, (expected_category, expected_type) in model_files.items():
            model_file = self.test_dir / filename
            model_file.touch()
            
            variant_info = comparator._extract_variant_info(filename, str(model_file))
            
            self.assertEqual(variant_info["size_category"], expected_category)
            self.assertEqual(variant_info["variant_type"], expected_type)
    
    @patch('yolo_variant_comparator.YOLO')
    @patch('yolo_variant_comparator.time')
    def test_performance_measurement(self, mock_time, mock_yolo):
        """Test performance measurement methods."""
        mock_model = Mock()
        mock_model.predict.return_value = [Mock()]
        mock_yolo.return_value = mock_model
        
        # Mock time measurements
        mock_time.perf_counter.side_effect = [0.0, 0.05, 0.0, 0.04, 0.0, 0.06]  # 50ms, 40ms, 60ms
        
        comparator = YOLOVariantComparator(str(self.config_file))
        
        # Test inference speed measurement
        speed_metrics = comparator._measure_inference_speed(mock_model, 640)
        
        self.assertIn('mean_time_ms', speed_metrics)
        self.assertIn('fps', speed_metrics)
        self.assertGreater(speed_metrics['fps'], 0)
    
    @patch('yolo_variant_comparator.YOLO')
    def test_security_performance_comparison(self, mock_yolo):
        """Test security-specific performance comparison."""
        mock_model = Mock()
        mock_results = Mock()
        mock_results.box.mp = 0.8
        mock_results.box.mr = 0.75
        mock_results.box.map50 = 0.78
        mock_results.box.ap_class_index = np.array([0, 1])  # violence, theft
        mock_results.box.ap50 = np.array([0.85, 0.75])
        mock_results.box.p = np.array([0.9, 0.8])
        mock_results.box.r = np.array([0.8, 0.7])
        
        mock_model.val.return_value = mock_results
        mock_yolo.return_value = mock_model
        
        comparator = YOLOVariantComparator(str(self.config_file))
        comparator.yolo_variants = {"test_variant": {"model": mock_model}}
        
        # Test security performance comparison
        security_results = comparator._compare_security_performance()
        
        self.assertIn("test_variant", security_results)
        variant_results = security_results["test_variant"]
        
        self.assertIn('security_metrics', variant_results)
        self.assertIn('false_positive_analysis', variant_results)
        self.assertIn('security_score', variant_results)
        
        # Check security score calculation
        security_score = variant_results['security_score']
        self.assertGreater(security_score, 0)
        self.assertLessEqual(security_score, 1)
    
    @patch('yolo_variant_comparator.YOLO')
    def test_deployment_suitability_assessment(self, mock_yolo):
        """Test deployment suitability assessment."""
        mock_yolo.return_value = Mock()
        
        comparator = YOLOVariantComparator(str(self.config_file))
        
        # Mock comparison results
        comparison_results = {
            'accuracy_comparison': {
                'test_variant': {'mAP50': 0.75}
            },
            'performance_comparison': {
                'test_variant': {'summary': {'best_fps': 25, 'memory_increase_mb': 300}}
            },
            'security_comparison': {
                'test_variant': {'security_score': 0.8}
            }
        }
        
        # Mock variant info
        comparator.yolo_variants = {
            'test_variant': {
                'variant_info': {'size_mb': 45}
            }
        }
        
        # Test deployment suitability
        suitability = comparator._assess_deployment_suitability(comparison_results)
        
        self.assertIn('test_variant', suitability)
        variant_suitability = suitability['test_variant']
        
        self.assertIn('edge_deployment', variant_suitability)
        self.assertIn('server_deployment', variant_suitability)
        self.assertIn('mobile_deployment', variant_suitability)
        
        # Check edge deployment suitability
        edge_suitability = variant_suitability['edge_deployment']
        self.assertIn('suitable', edge_suitability)
        self.assertIn('score', edge_suitability)
        self.assertIn('requirements_met', edge_suitability)


class TestIntegration(unittest.TestCase):
    """Integration tests for the validation framework."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_file = self.test_dir / "test_config.yaml"
        
        # Create comprehensive mock config
        config_content = """
dataset:
  yolo_data_dir: "test_data"
classes:
  violence: 0
  emergency: 1
  theft: 2
  suspicious: 3
  intrusion: 4
  loitering: 5
  crowding: 6
  abandoned_object: 7
  normal: 8
"""
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Create test directories
        (self.test_dir / "test_data").mkdir()
        (self.test_dir / "enhanced_validation_results").mkdir()
        (self.test_dir / "yolo_variant_comparison").mkdir()
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.test_dir)
    
    @patch('enhanced_security_validator.YOLO')
    def test_end_to_end_validation(self, mock_yolo):
        """Test end-to-end validation workflow."""
        # Mock YOLO model with comprehensive results
        mock_model = Mock()
        
        # Mock validation results for different thresholds
        def mock_val(**kwargs):
            conf = kwargs.get('conf', 0.25)
            mock_results = Mock()
            
            # Simulate different performance at different confidence levels
            if conf <= 0.3:
                mock_results.box.mp = 0.7  # 70% precision = 30% FP rate
                mock_results.box.mr = 0.85  # 85% recall
            elif conf <= 0.5:
                mock_results.box.mp = 0.8  # 80% precision = 20% FP rate
                mock_results.box.mr = 0.75  # 75% recall
            else:
                mock_results.box.mp = 0.92  # 92% precision = 8% FP rate
                mock_results.box.mr = 0.65  # 65% recall
            
            mock_results.box.map50 = mock_results.box.mp * 0.9  # Approximate mAP
            
            # Mock per-class results
            mock_results.box.ap_class_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            mock_results.box.ap50 = np.array([0.8, 0.85, 0.75, 0.7, 0.82, 0.65, 0.7, 0.6, 0.9])
            mock_results.box.p = np.array([0.85, 0.9, 0.8, 0.75, 0.87, 0.7, 0.75, 0.65, 0.95])
            mock_results.box.r = np.array([0.8, 0.85, 0.7, 0.65, 0.8, 0.6, 0.65, 0.55, 0.85])
            
            return mock_results
        
        mock_model.val.side_effect = mock_val
        
        # Mock prediction results for environmental testing
        mock_predict_results = [Mock()]
        mock_predict_results[0].boxes = Mock()
        mock_predict_results[0].boxes.conf = Mock()
        mock_predict_results[0].boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8, 0.7, 0.6])
        mock_model.predict.return_value = mock_predict_results
        
        mock_yolo.return_value = mock_model
        
        # Create model file
        model_file = self.test_dir / "test_model.pt"
        model_file.touch()
        
        # Initialize validator
        validator = EnhancedSecurityValidator(
            str(model_file), 
            str(self.config_file)
        )
        
        # Run enhanced validation
        results = validator.run_enhanced_validation()
        
        # Verify results structure
        self.assertIn('validation_summary', results)
        self.assertIn('false_positive_analysis', results)
        self.assertIn('environmental_adaptation', results)
        self.assertIn('security_scenario_tests', results)
        self.assertIn('compliance_assessment', results)
        
        # Verify compliance assessment
        compliance = results['compliance_assessment']
        self.assertIn('requirement_6_2_compliance', compliance)
        self.assertIn('requirement_6_3_compliance', compliance)
        
        # Verify validation summary
        summary = results['validation_summary']
        self.assertIn('validation_status', summary)
        self.assertIn('compliance_summary', summary)
        self.assertIn('deployment_readiness', summary)
        
        # Check that the model meets requirements
        req_6_2 = compliance['requirement_6_2_compliance']
        self.assertTrue(req_6_2['initial_target_met'])  # Should meet 30% FP target
        self.assertTrue(req_6_2['improved_target_achievable'])  # Should meet 10% FP target
    
    @patch('yolo_variant_comparator.YOLO')
    @patch('yolo_variant_comparator.time')
    @patch('yolo_variant_comparator.psutil')
    def test_variant_comparison_workflow(self, mock_psutil, mock_time, mock_yolo):
        """Test complete variant comparison workflow."""
        # Mock system info
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1GB
        mock_psutil.Process.return_value = mock_process
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB
        
        # Mock time measurements
        mock_time.perf_counter.side_effect = [
            0.0, 0.03,  # 30ms inference for nano
            0.0, 0.05,  # 50ms inference for small
            0.0, 0.08   # 80ms inference for medium
        ] * 10  # Repeat for multiple measurements
        
        # Mock YOLO models with different characteristics
        def create_mock_model(variant_type):
            mock_model = Mock()
            
            # Different parameter counts for different variants
            param_counts = {'n': 3000000, 's': 11000000, 'm': 25000000}
            param_count = param_counts.get(variant_type[-1], 10000000)
            
            mock_param = Mock()
            mock_param.numel.return_value = param_count
            mock_model.model.parameters.return_value = [mock_param]
            
            # Mock validation results with different accuracy for different variants
            mock_results = Mock()
            accuracies = {'n': 0.65, 's': 0.72, 'm': 0.78}
            accuracy = accuracies.get(variant_type[-1], 0.7)
            
            mock_results.box.mp = accuracy + 0.05  # Precision slightly higher
            mock_results.box.mr = accuracy  # Recall
            mock_results.box.map50 = accuracy
            mock_results.box.map = accuracy - 0.1
            
            # Mock per-class results
            mock_results.box.ap_class_index = np.array([0, 1, 2])
            mock_results.box.ap50 = np.array([accuracy, accuracy + 0.05, accuracy - 0.05])
            mock_results.box.p = np.array([accuracy + 0.1, accuracy + 0.05, accuracy])
            mock_results.box.r = np.array([accuracy, accuracy - 0.05, accuracy + 0.05])
            
            mock_model.val.return_value = mock_results
            
            # Mock prediction for performance testing
            mock_predict_result = [Mock()]
            mock_predict_result[0].boxes = Mock()
            mock_model.predict.return_value = mock_predict_result
            
            return mock_model
        
        # Mock YOLO constructor to return different models
        model_mapping = {}
        
        def mock_yolo_constructor(path):
            if 'yolov8n' in path:
                if path not in model_mapping:
                    model_mapping[path] = create_mock_model('yolov8n')
                return model_mapping[path]
            elif 'yolov8s' in path:
                if path not in model_mapping:
                    model_mapping[path] = create_mock_model('yolov8s')
                return model_mapping[path]
            elif 'yolov8m' in path:
                if path not in model_mapping:
                    model_mapping[path] = create_mock_model('yolov8m')
                return model_mapping[path]
            else:
                return create_mock_model('unknown')
        
        mock_yolo.side_effect = mock_yolo_constructor
        
        # Create mock model files
        model_files = {
            'nano': self.test_dir / 'yolov8n.pt',
            'small': self.test_dir / 'yolov8s.pt',
            'medium': self.test_dir / 'yolov8m.pt'
        }
        
        for model_file in model_files.values():
            model_file.touch()
        
        # Initialize comparator
        comparator = YOLOVariantComparator(str(self.config_file))
        
        # Add variants
        for name, path in model_files.items():
            comparator.add_yolo_variant(name, str(path))
        
        # Run comparison
        results = comparator.run_comprehensive_comparison()
        
        # Verify results structure
        self.assertIn('accuracy_comparison', results)
        self.assertIn('performance_comparison', results)
        self.assertIn('security_comparison', results)
        self.assertIn('deployment_suitability', results)
        self.assertIn('variant_rankings', results)
        self.assertIn('recommendations', results)
        
        # Verify all variants were compared
        for variant_name in ['nano', 'small', 'medium']:
            self.assertIn(variant_name, results['accuracy_comparison'])
            self.assertIn(variant_name, results['performance_comparison'])
            self.assertIn(variant_name, results['deployment_suitability'])
        
        # Verify rankings
        rankings = results['variant_rankings']
        self.assertIn('overall_best', rankings)
        self.assertIn('accuracy_ranking', rankings)
        self.assertIn('performance_ranking', rankings)
        
        # Verify recommendations
        recommendations = results['recommendations']
        self.assertIn('best_for_edge', recommendations)
        self.assertIn('best_for_server', recommendations)
        self.assertIn('variant_analysis', recommendations)


def run_validation_tests():
    """Run all validation tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestEnhancedSecurityValidator))
    test_suite.addTest(unittest.makeSuite(TestYOLOVariantComparator))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)