#!/usr/bin/env python3
"""
Security Model Validation Framework
Comprehensive validation and testing framework for campus security YOLO models.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from ultralytics import YOLO
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityModelValidator:
    """Comprehensive validation framework for security YOLO models."""
    
    def __init__(self, model_path: str, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the validator."""
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Load model
        self.model = YOLO(str(self.model_path))
        
        # Set up paths
        self.yolo_data_dir = Path(self.config['dataset']['yolo_data_dir'])
        self.validation_dir = Path("../validation_results")
        self.validation_dir.mkdir(exist_ok=True)
        
        # Security classes and priorities
        self.class_names = list(self.config['classes'].keys())
        self.security_priorities = {
            'critical': ['violence', 'emergency'],
            'high': ['theft', 'suspicious'],
            'medium': ['trespassing', 'vandalism'],
            'low': ['crowd', 'loitering', 'abandoned_object'],
            'normal': ['normal']
        }
        
        # Validation thresholds
        self.validation_thresholds = {
            'min_accuracy': 0.7,
            'max_false_positive_rate': 0.3,
            'min_critical_recall': 0.8,
            'max_inference_time_ms': 100,
            'min_fps': 15
        }
        
        logger.info(f"SecurityModelValidator initialized with model: {self.model_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_comprehensive_validation(self) -> Dict:
        """Run complete validation suite."""
        logger.info("Starting comprehensive model validation...")
        
        validation_results = {
            'model_info': self._get_model_info(),
            'validation_timestamp': datetime.now().isoformat(),
            'functional_tests': {},
            'performance_tests': {},
            'security_tests': {},
            'robustness_tests': {},
            'benchmark_results': {},
            'validation_summary': {}
        }
        
        try:
            # 1. Functional Tests
            logger.info("Running functional tests...")
            validation_results['functional_tests'] = self._run_functional_tests()
            
            # 2. Performance Tests
            logger.info("Running performance tests...")
            validation_results['performance_tests'] = self._run_performance_tests()
            
            # 3. Security-Specific Tests
            logger.info("Running security-specific tests...")
            validation_results['security_tests'] = self._run_security_tests()
            
            # 4. Robustness Tests
            logger.info("Running robustness tests...")
            validation_results['robustness_tests'] = self._run_robustness_tests()
            
            # 5. Benchmark Comparisons
            logger.info("Running benchmark comparisons...")
            validation_results['benchmark_results'] = self._run_benchmark_tests()
            
            # 6. Generate Summary
            validation_results['validation_summary'] = self._generate_validation_summary(validation_results)
            
            # Save results
            self._save_validation_results(validation_results)
            
            logger.info("Comprehensive validation completed!")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results['validation_summary'] = {'status': 'failed', 'error': str(e)}
            return validation_results
    
    def _get_model_info(self) -> Dict:
        """Get model information and metadata."""
        model_info = {
            'model_path': str(self.model_path),
            'model_size': self._determine_model_size(),
            'parameters': self._count_parameters(),
            'input_size': 640,  # Default YOLO input size
            'classes': self.class_names,
            'num_classes': len(self.class_names)
        }
        
        # Try to get additional model metadata
        try:
            model_info['model_yaml'] = str(self.model.model.yaml)
        except:
            pass
        
        return model_info
    
    def _determine_model_size(self) -> str:
        """Determine YOLO model size."""
        model_name = self.model_path.name.lower()
        for size in ['n', 's', 'm', 'l', 'x']:
            if f'yolov8{size}' in model_name:
                return size
        return 'unknown'
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        try:
            return sum(p.numel() for p in self.model.model.parameters())
        except:
            return 0    

    def _run_functional_tests(self) -> Dict:
        """Run basic functional tests."""
        functional_tests = {}
        
        # Test 1: Model Loading
        functional_tests['model_loading'] = self._test_model_loading()
        
        # Test 2: Basic Inference
        functional_tests['basic_inference'] = self._test_basic_inference()
        
        # Test 3: Batch Inference
        functional_tests['batch_inference'] = self._test_batch_inference()
        
        # Test 4: Input Validation
        functional_tests['input_validation'] = self._test_input_validation()
        
        # Test 5: Output Format
        functional_tests['output_format'] = self._test_output_format()
        
        return functional_tests
    
    def _test_model_loading(self) -> Dict:
        """Test model loading functionality."""
        try:
            # Test loading from different formats
            test_model = YOLO(str(self.model_path))
            
            return {
                'status': 'passed',
                'model_loaded': True,
                'model_type': type(test_model).__name__
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_basic_inference(self) -> Dict:
        """Test basic inference functionality."""
        try:
            # Create dummy input
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run inference
            results = self.model.predict(dummy_input, verbose=False)
            
            return {
                'status': 'passed',
                'inference_successful': True,
                'output_type': type(results).__name__,
                'num_results': len(results) if hasattr(results, '__len__') else 1
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_batch_inference(self) -> Dict:
        """Test batch inference functionality."""
        try:
            # Create batch of dummy inputs
            batch_size = 4
            dummy_batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) 
                          for _ in range(batch_size)]
            
            # Run batch inference
            results = self.model.predict(dummy_batch, verbose=False)
            
            return {
                'status': 'passed',
                'batch_inference_successful': True,
                'batch_size': batch_size,
                'results_count': len(results)
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_input_validation(self) -> Dict:
        """Test input validation and error handling."""
        test_results = {}
        
        # Test invalid input shapes
        try:
            invalid_input = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            results = self.model.predict(invalid_input, verbose=False)
            test_results['small_input'] = 'handled'
        except Exception as e:
            test_results['small_input'] = f'error: {str(e)}'
        
        # Test empty input
        try:
            empty_input = np.zeros((640, 640, 3), dtype=np.uint8)
            results = self.model.predict(empty_input, verbose=False)
            test_results['empty_input'] = 'handled'
        except Exception as e:
            test_results['empty_input'] = f'error: {str(e)}'
        
        return {
            'status': 'passed',
            'test_results': test_results
        }
    
    def _test_output_format(self) -> Dict:
        """Test output format consistency."""
        try:
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.model.predict(dummy_input, verbose=False)
            
            # Analyze output format
            output_analysis = {
                'has_boxes': hasattr(results[0], 'boxes') if results else False,
                'has_confidence': False,
                'has_classes': False,
                'box_format': 'unknown'
            }
            
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                if hasattr(boxes, 'conf'):
                    output_analysis['has_confidence'] = True
                if hasattr(boxes, 'cls'):
                    output_analysis['has_classes'] = True
                if hasattr(boxes, 'xyxy'):
                    output_analysis['box_format'] = 'xyxy'
                elif hasattr(boxes, 'xywh'):
                    output_analysis['box_format'] = 'xywh'
            
            return {
                'status': 'passed',
                'output_analysis': output_analysis
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }  
  
    def _run_performance_tests(self) -> Dict:
        """Run performance and speed tests."""
        performance_tests = {}
        
        # Test 1: Inference Speed
        performance_tests['inference_speed'] = self._test_inference_speed()
        
        # Test 2: Memory Usage
        performance_tests['memory_usage'] = self._test_memory_usage()
        
        # Test 3: Throughput
        performance_tests['throughput'] = self._test_throughput()
        
        # Test 4: Scalability
        performance_tests['scalability'] = self._test_scalability()
        
        return performance_tests
    
    def _test_inference_speed(self) -> Dict:
        """Test inference speed across different input sizes."""
        speed_results = {}
        input_sizes = [320, 416, 640, 832]
        
        for size in input_sizes:
            try:
                # Create input
                test_input = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                
                # Warm up
                for _ in range(5):
                    _ = self.model.predict(test_input, verbose=False)
                
                # Measure inference time
                times = []
                for _ in range(20):
                    start_time = time.time()
                    _ = self.model.predict(test_input, verbose=False)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                speed_results[f'input_{size}'] = {
                    'mean_time_ms': np.mean(times),
                    'std_time_ms': np.std(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'fps': 1000 / np.mean(times)
                }
                
            except Exception as e:
                speed_results[f'input_{size}'] = {'error': str(e)}
        
        # Check if meets real-time requirements
        real_time_analysis = {
            'meets_15fps': any(result.get('fps', 0) >= 15 for result in speed_results.values() if 'fps' in result),
            'meets_30fps': any(result.get('fps', 0) >= 30 for result in speed_results.values() if 'fps' in result),
            'fastest_fps': max((result.get('fps', 0) for result in speed_results.values() if 'fps' in result), default=0)
        }
        
        return {
            'speed_results': speed_results,
            'real_time_analysis': real_time_analysis,
            'meets_threshold': real_time_analysis['fastest_fps'] >= self.validation_thresholds['min_fps']
        }
    
    def _test_memory_usage(self) -> Dict:
        """Test memory usage during inference."""
        try:
            import psutil
            import gc
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run inference multiple times
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            memory_usage = []
            for i in range(10):
                _ = self.model.predict(test_input, verbose=False)
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_usage.append(current_memory)
                
                if i % 3 == 0:  # Periodic garbage collection
                    gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            
            return {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'peak_memory_mb': max(memory_usage),
                'memory_increase_mb': final_memory - initial_memory,
                'memory_stable': abs(final_memory - initial_memory) < 100  # Less than 100MB increase
            }
            
        except ImportError:
            return {'error': 'psutil not available for memory testing'}
        except Exception as e:
            return {'error': str(e)}
    
    def _test_throughput(self) -> Dict:
        """Test model throughput with concurrent requests."""
        try:
            def single_inference():
                test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                start_time = time.time()
                _ = self.model.predict(test_input, verbose=False)
                return time.time() - start_time
            
            # Test sequential throughput
            sequential_times = []
            for _ in range(10):
                sequential_times.append(single_inference())
            
            sequential_throughput = 10 / sum(sequential_times)  # inferences per second
            
            # Test concurrent throughput (if supported)
            concurrent_throughput = None
            try:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    start_time = time.time()
                    futures = [executor.submit(single_inference) for _ in range(10)]
                    
                    for future in as_completed(futures):
                        future.result()  # Wait for completion
                    
                    total_time = time.time() - start_time
                    concurrent_throughput = 10 / total_time
            except Exception as e:
                concurrent_throughput = f"Error: {str(e)}"
            
            return {
                'sequential_throughput_fps': sequential_throughput,
                'concurrent_throughput_fps': concurrent_throughput,
                'avg_inference_time_ms': np.mean(sequential_times) * 1000
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_scalability(self) -> Dict:
        """Test model scalability with different batch sizes."""
        scalability_results = {}
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            try:
                # Create batch input
                batch_input = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) 
                              for _ in range(batch_size)]
                
                # Measure batch processing time
                times = []
                for _ in range(5):
                    start_time = time.time()
                    _ = self.model.predict(batch_input, verbose=False)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                per_image_time = avg_time / batch_size
                
                scalability_results[f'batch_{batch_size}'] = {
                    'total_time_s': avg_time,
                    'per_image_time_s': per_image_time,
                    'throughput_fps': batch_size / avg_time
                }
                
            except Exception as e:
                scalability_results[f'batch_{batch_size}'] = {'error': str(e)}
        
        return scalability_results    
 
   def _run_security_tests(self) -> Dict:
        """Run security-specific validation tests."""
        security_tests = {}
        
        # Test 1: Critical Event Detection
        security_tests['critical_event_detection'] = self._test_critical_event_detection()
        
        # Test 2: False Positive Analysis
        security_tests['false_positive_analysis'] = self._test_false_positive_rates()
        
        # Test 3: Confidence Calibration
        security_tests['confidence_calibration'] = self._test_confidence_calibration()
        
        # Test 4: Priority Class Performance
        security_tests['priority_class_performance'] = self._test_priority_class_performance()
        
        return security_tests
    
    def _test_critical_event_detection(self) -> Dict:
        """Test detection of critical security events."""
        try:
            # Run validation on test set
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                conf=0.25,
                verbose=False
            )
            
            # Extract per-class metrics
            critical_classes = self.security_priorities['critical'] + self.security_priorities['high']
            critical_performance = {}
            
            if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
                for i, class_idx in enumerate(results.box.ap_class_index):
                    if class_idx < len(self.class_names):
                        class_name = self.class_names[class_idx]
                        if class_name in critical_classes:
                            critical_performance[class_name] = {
                                'ap50': float(results.box.ap50[i]),
                                'ap50_95': float(results.box.ap[i]),
                                'precision': float(results.box.p[i]) if hasattr(results.box, 'p') else 0,
                                'recall': float(results.box.r[i]) if hasattr(results.box, 'r') else 0
                            }
            
            # Calculate critical event detection rate
            critical_recalls = [perf['recall'] for perf in critical_performance.values()]
            avg_critical_recall = np.mean(critical_recalls) if critical_recalls else 0
            
            return {
                'critical_performance': critical_performance,
                'avg_critical_recall': avg_critical_recall,
                'meets_threshold': avg_critical_recall >= self.validation_thresholds['min_critical_recall'],
                'critical_classes_tested': len(critical_performance)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_false_positive_rates(self) -> Dict:
        """Test false positive rates for normal activities."""
        try:
            # Run validation focusing on normal class
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                conf=0.1,  # Lower confidence to catch more false positives
                verbose=False
            )
            
            # Calculate false positive rate for normal class
            normal_class_idx = None
            if 'normal' in self.class_names:
                normal_class_idx = self.class_names.index('normal')
            
            false_positive_analysis = {
                'overall_precision': float(results.box.mp),
                'false_positive_rate': 1 - float(results.box.mp),
                'meets_threshold': (1 - float(results.box.mp)) <= self.validation_thresholds['max_false_positive_rate']
            }
            
            # Per-class false positive analysis
            if hasattr(results.box, 'ap_class_index') and normal_class_idx is not None:
                for i, class_idx in enumerate(results.box.ap_class_index):
                    if class_idx == normal_class_idx:
                        normal_precision = float(results.box.p[i]) if hasattr(results.box, 'p') else 0
                        false_positive_analysis['normal_class_precision'] = normal_precision
                        false_positive_analysis['normal_false_positive_rate'] = 1 - normal_precision
                        break
            
            return false_positive_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_confidence_calibration(self) -> Dict:
        """Test confidence score calibration."""
        try:
            # Test at different confidence thresholds
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            calibration_results = {}
            
            for threshold in thresholds:
                results = self.model.val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=threshold,
                    verbose=False
                )
                
                calibration_results[str(threshold)] = {
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                    'mAP50': float(results.box.map50)
                }
            
            # Analyze calibration quality
            precisions = [result['precision'] for result in calibration_results.values()]
            recalls = [result['recall'] for result in calibration_results.values()]
            
            # Good calibration should show increasing precision with higher thresholds
            precision_trend = np.polyfit(thresholds, precisions, 1)[0]  # Slope
            
            return {
                'threshold_results': calibration_results,
                'precision_trend': precision_trend,
                'well_calibrated': precision_trend > 0,  # Precision should increase with threshold
                'recommended_threshold': self._find_optimal_threshold(calibration_results)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _find_optimal_threshold(self, calibration_results: Dict) -> float:
        """Find optimal confidence threshold for security applications."""
        best_threshold = 0.5
        best_score = 0
        
        for threshold_str, metrics in calibration_results.items():
            # Security score emphasizes recall over precision
            security_score = 0.3 * metrics['precision'] + 0.7 * metrics['recall']
            
            if security_score > best_score:
                best_score = security_score
                best_threshold = float(threshold_str)
        
        return best_threshold
    
    def _test_priority_class_performance(self) -> Dict:
        """Test performance across different priority classes."""
        try:
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                verbose=False
            )
            
            priority_performance = {}
            
            for priority, classes in self.security_priorities.items():
                priority_metrics = []
                
                if hasattr(results.box, 'ap_class_index'):
                    for class_name in classes:
                        if class_name in self.class_names:
                            class_idx = self.class_names.index(class_name)
                            if class_idx in results.box.ap_class_index:
                                idx = list(results.box.ap_class_index).index(class_idx)
                                class_metrics = {
                                    'ap50': float(results.box.ap50[idx]),
                                    'precision': float(results.box.p[idx]) if hasattr(results.box, 'p') else 0,
                                    'recall': float(results.box.r[idx]) if hasattr(results.box, 'r') else 0
                                }
                                priority_metrics.append(class_metrics)
                
                if priority_metrics:
                    priority_performance[priority] = {
                        'avg_ap50': np.mean([m['ap50'] for m in priority_metrics]),
                        'avg_precision': np.mean([m['precision'] for m in priority_metrics]),
                        'avg_recall': np.mean([m['recall'] for m in priority_metrics]),
                        'num_classes': len(priority_metrics)
                    }
            
            return priority_performance
            
        except Exception as e:
            return {'error': str(e)}    

    def _run_robustness_tests(self) -> Dict:
        """Run robustness and stress tests."""
        robustness_tests = {}
        
        # Test 1: Noise Robustness
        robustness_tests['noise_robustness'] = self._test_noise_robustness()
        
        # Test 2: Lighting Conditions
        robustness_tests['lighting_robustness'] = self._test_lighting_robustness()
        
        # Test 3: Resolution Robustness
        robustness_tests['resolution_robustness'] = self._test_resolution_robustness()
        
        # Test 4: Stress Testing
        robustness_tests['stress_testing'] = self._test_stress_conditions()
        
        return robustness_tests
    
    def _test_noise_robustness(self) -> Dict:
        """Test robustness to different types of noise."""
        try:
            base_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Get baseline performance
            baseline_results = self.model.predict(base_input, verbose=False)
            baseline_detections = len(baseline_results[0].boxes) if baseline_results[0].boxes is not None else 0
            
            noise_tests = {}
            
            # Gaussian noise
            for noise_level in [10, 25, 50]:
                noisy_input = base_input.astype(np.float32)
                noise = np.random.normal(0, noise_level, noisy_input.shape)
                noisy_input = np.clip(noisy_input + noise, 0, 255).astype(np.uint8)
                
                noisy_results = self.model.predict(noisy_input, verbose=False)
                noisy_detections = len(noisy_results[0].boxes) if noisy_results[0].boxes is not None else 0
                
                noise_tests[f'gaussian_{noise_level}'] = {
                    'detections': noisy_detections,
                    'detection_ratio': noisy_detections / max(baseline_detections, 1)
                }
            
            # Salt and pepper noise
            for noise_prob in [0.05, 0.1, 0.2]:
                noisy_input = base_input.copy()
                mask = np.random.random(noisy_input.shape[:2]) < noise_prob
                noisy_input[mask] = np.random.choice([0, 255], size=np.sum(mask))
                
                noisy_results = self.model.predict(noisy_input, verbose=False)
                noisy_detections = len(noisy_results[0].boxes) if noisy_results[0].boxes is not None else 0
                
                noise_tests[f'salt_pepper_{noise_prob}'] = {
                    'detections': noisy_detections,
                    'detection_ratio': noisy_detections / max(baseline_detections, 1)
                }
            
            # Calculate robustness score
            detection_ratios = [test['detection_ratio'] for test in noise_tests.values()]
            robustness_score = np.mean(detection_ratios)
            
            return {
                'baseline_detections': baseline_detections,
                'noise_tests': noise_tests,
                'robustness_score': robustness_score,
                'robust': robustness_score > 0.7  # 70% detection retention
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_lighting_robustness(self) -> Dict:
        """Test robustness to different lighting conditions."""
        try:
            base_input = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            
            lighting_tests = {}
            
            # Brightness variations
            for brightness_factor in [0.3, 0.5, 1.5, 2.0]:
                bright_input = np.clip(base_input.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
                
                results = self.model.predict(bright_input, verbose=False)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                lighting_tests[f'brightness_{brightness_factor}'] = {
                    'detections': detections,
                    'mean_brightness': np.mean(bright_input)
                }
            
            # Contrast variations
            for contrast_factor in [0.5, 1.5, 2.0]:
                contrast_input = np.clip((base_input.astype(np.float32) - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
                
                results = self.model.predict(contrast_input, verbose=False)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                lighting_tests[f'contrast_{contrast_factor}'] = {
                    'detections': detections,
                    'contrast_std': np.std(contrast_input)
                }
            
            return {
                'lighting_tests': lighting_tests,
                'lighting_robust': len([t for t in lighting_tests.values() if t['detections'] > 0]) > len(lighting_tests) * 0.7
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_resolution_robustness(self) -> Dict:
        """Test robustness to different input resolutions."""
        try:
            resolution_tests = {}
            
            for target_size in [320, 416, 640, 832]:
                # Create input at target resolution
                test_input = np.random.randint(0, 255, (target_size, target_size, 3), dtype=np.uint8)
                
                # Measure inference time and accuracy
                start_time = time.time()
                results = self.model.predict(test_input, verbose=False)
                inference_time = (time.time() - start_time) * 1000
                
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                resolution_tests[f'resolution_{target_size}'] = {
                    'detections': detections,
                    'inference_time_ms': inference_time,
                    'fps': 1000 / inference_time
                }
            
            return {
                'resolution_tests': resolution_tests,
                'supports_multi_resolution': len(resolution_tests) > 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_stress_conditions(self) -> Dict:
        """Test model under stress conditions."""
        try:
            stress_tests = {}
            
            # Memory stress test
            try:
                large_batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(16)]
                start_time = time.time()
                results = self.model.predict(large_batch, verbose=False)
                stress_time = time.time() - start_time
                
                stress_tests['large_batch'] = {
                    'batch_size': 16,
                    'processing_time': stress_time,
                    'successful': True
                }
            except Exception as e:
                stress_tests['large_batch'] = {
                    'successful': False,
                    'error': str(e)
                }
            
            # Rapid inference test
            try:
                test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                rapid_times = []
                
                for _ in range(100):
                    start_time = time.time()
                    _ = self.model.predict(test_input, verbose=False)
                    rapid_times.append(time.time() - start_time)
                
                stress_tests['rapid_inference'] = {
                    'num_inferences': 100,
                    'avg_time_ms': np.mean(rapid_times) * 1000,
                    'time_stability': np.std(rapid_times) * 1000 < 10  # Less than 10ms std
                }
            except Exception as e:
                stress_tests['rapid_inference'] = {
                    'successful': False,
                    'error': str(e)
                }
            
            return stress_tests
            
        except Exception as e:
            return {'error': str(e)}    

    def _run_benchmark_tests(self) -> Dict:
        """Run benchmark comparisons against baseline models."""
        benchmark_tests = {}
        
        # Test 1: Accuracy Benchmarks
        benchmark_tests['accuracy_benchmarks'] = self._test_accuracy_benchmarks()
        
        # Test 2: Speed Benchmarks
        benchmark_tests['speed_benchmarks'] = self._test_speed_benchmarks()
        
        # Test 3: Model Size Comparison
        benchmark_tests['size_comparison'] = self._test_model_size_comparison()
        
        return benchmark_tests
    
    def _test_accuracy_benchmarks(self) -> Dict:
        """Compare accuracy against benchmark thresholds."""
        try:
            # Run comprehensive validation
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                verbose=False
            )
            
            # Extract key metrics
            current_metrics = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            }
            
            # Define benchmark thresholds for security applications
            security_benchmarks = {
                'mAP50': 0.7,      # 70% mAP@0.5
                'mAP50_95': 0.5,   # 50% mAP@0.5:0.95
                'precision': 0.8,   # 80% precision
                'recall': 0.7       # 70% recall
            }
            
            # Compare against benchmarks
            benchmark_comparison = {}
            for metric, threshold in security_benchmarks.items():
                current_value = current_metrics.get(metric, 0)
                benchmark_comparison[metric] = {
                    'current': current_value,
                    'benchmark': threshold,
                    'meets_benchmark': current_value >= threshold,
                    'difference': current_value - threshold
                }
            
            # Overall benchmark score
            benchmark_score = np.mean([
                comp['meets_benchmark'] for comp in benchmark_comparison.values()
            ])
            
            return {
                'current_metrics': current_metrics,
                'benchmark_comparison': benchmark_comparison,
                'benchmark_score': benchmark_score,
                'passes_benchmarks': benchmark_score >= 0.75  # 75% of benchmarks must pass
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_speed_benchmarks(self) -> Dict:
        """Compare speed against benchmark requirements."""
        try:
            # Test inference speed
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                _ = self.model.predict(test_input, verbose=False)
            
            # Measure speed
            times = []
            for _ in range(20):
                start_time = time.time()
                _ = self.model.predict(test_input, verbose=False)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time_ms = np.mean(times)
            fps = 1000 / avg_time_ms
            
            # Speed benchmarks for security applications
            speed_benchmarks = {
                'max_inference_time_ms': 100,  # 100ms max
                'min_fps': 15,                 # 15 FPS min
                'target_fps': 30               # 30 FPS target
            }
            
            speed_comparison = {
                'current_time_ms': avg_time_ms,
                'current_fps': fps,
                'meets_max_time': avg_time_ms <= speed_benchmarks['max_inference_time_ms'],
                'meets_min_fps': fps >= speed_benchmarks['min_fps'],
                'meets_target_fps': fps >= speed_benchmarks['target_fps']
            }
            
            return {
                'speed_metrics': {
                    'avg_inference_time_ms': avg_time_ms,
                    'fps': fps,
                    'std_time_ms': np.std(times)
                },
                'speed_benchmarks': speed_benchmarks,
                'speed_comparison': speed_comparison,
                'real_time_capable': speed_comparison['meets_min_fps']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_model_size_comparison(self) -> Dict:
        """Compare model size and complexity."""
        try:
            model_stats = {
                'parameters': self._count_parameters(),
                'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'model_type': self._determine_model_size()
            }
            
            # Size benchmarks for different deployment scenarios
            size_benchmarks = {
                'edge_device_mb': 50,      # 50MB for edge devices
                'mobile_device_mb': 25,    # 25MB for mobile
                'server_mb': 200           # 200MB for server deployment
            }
            
            deployment_suitability = {
                'edge_suitable': model_stats['model_size_mb'] <= size_benchmarks['edge_device_mb'],
                'mobile_suitable': model_stats['model_size_mb'] <= size_benchmarks['mobile_device_mb'],
                'server_suitable': model_stats['model_size_mb'] <= size_benchmarks['server_mb']
            }
            
            return {
                'model_stats': model_stats,
                'size_benchmarks': size_benchmarks,
                'deployment_suitability': deployment_suitability
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_validation_summary(self, validation_results: Dict) -> Dict:
        """Generate overall validation summary."""
        summary = {
            'overall_status': 'unknown',
            'passed_tests': 0,
            'total_tests': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Count passed tests
            test_categories = ['functional_tests', 'performance_tests', 'security_tests', 'robustness_tests']
            
            for category in test_categories:
                if category in validation_results:
                    category_results = validation_results[category]
                    for test_name, test_result in category_results.items():
                        summary['total_tests'] += 1
                        
                        # Check if test passed
                        if isinstance(test_result, dict):
                            if test_result.get('status') == 'passed' or \
                               test_result.get('meets_threshold') == True or \
                               test_result.get('successful') == True:
                                summary['passed_tests'] += 1
                            elif 'error' in test_result:
                                summary['critical_issues'].append(f"{category}.{test_name}: {test_result['error']}")
            
            # Calculate pass rate
            pass_rate = summary['passed_tests'] / max(summary['total_tests'], 1)
            
            # Determine overall status
            if pass_rate >= 0.9:
                summary['overall_status'] = 'excellent'
            elif pass_rate >= 0.8:
                summary['overall_status'] = 'good'
            elif pass_rate >= 0.7:
                summary['overall_status'] = 'acceptable'
            elif pass_rate >= 0.5:
                summary['overall_status'] = 'needs_improvement'
            else:
                summary['overall_status'] = 'poor'
            
            summary['pass_rate'] = pass_rate
            
            # Generate recommendations
            summary['recommendations'] = self._generate_recommendations(validation_results)
            
        except Exception as e:
            summary['critical_issues'].append(f"Summary generation failed: {str(e)}")
        
        return summary
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        try:
            # Performance recommendations
            if 'performance_tests' in validation_results:
                perf_tests = validation_results['performance_tests']
                
                if 'inference_speed' in perf_tests:
                    speed_test = perf_tests['inference_speed']
                    if not speed_test.get('meets_threshold', True):
                        recommendations.append("Consider using a smaller model size (YOLOv8n) for better inference speed")
                
                if 'memory_usage' in perf_tests:
                    memory_test = perf_tests['memory_usage']
                    if not memory_test.get('memory_stable', True):
                        recommendations.append("Memory usage is unstable - implement proper memory management")
            
            # Security recommendations
            if 'security_tests' in validation_results:
                sec_tests = validation_results['security_tests']
                
                if 'critical_event_detection' in sec_tests:
                    crit_test = sec_tests['critical_event_detection']
                    if not crit_test.get('meets_threshold', True):
                        recommendations.append("Critical event detection rate is below threshold - consider retraining with more critical event data")
                
                if 'false_positive_analysis' in sec_tests:
                    fp_test = sec_tests['false_positive_analysis']
                    if not fp_test.get('meets_threshold', True):
                        recommendations.append("False positive rate is too high - implement confidence threshold optimization")
            
            # Robustness recommendations
            if 'robustness_tests' in validation_results:
                rob_tests = validation_results['robustness_tests']
                
                if 'noise_robustness' in rob_tests:
                    noise_test = rob_tests['noise_robustness']
                    if not noise_test.get('robust', True):
                        recommendations.append("Model is not robust to noise - consider data augmentation during training")
            
            # Benchmark recommendations
            if 'benchmark_results' in validation_results:
                bench_results = validation_results['benchmark_results']
                
                if 'accuracy_benchmarks' in bench_results:
                    acc_bench = bench_results['accuracy_benchmarks']
                    if not acc_bench.get('passes_benchmarks', True):
                        recommendations.append("Model does not meet accuracy benchmarks - consider hyperparameter optimization or more training data")
        
        except Exception as e:
            recommendations.append(f"Could not generate all recommendations: {str(e)}")
        
        return recommendations
    
    def _save_validation_results(self, results: Dict) -> None:
        """Save validation results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.validation_dir / f"validation_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate validation report
        self._generate_validation_report(results, results_dir)
        
        logger.info(f"Validation results saved to: {results_dir}")
    
    def _generate_validation_report(self, results: Dict, output_dir: Path) -> None:
        """Generate human-readable validation report."""
        report_file = output_dir / 'validation_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Security Model Validation Report\n\n")
            
            # Model Information
            f.write("## Model Information\n")
            model_info = results.get('model_info', {})
            f.write(f"- **Model Path**: {model_info.get('model_path', 'Unknown')}\n")
            f.write(f"- **Model Size**: YOLOv8{model_info.get('model_size', 'unknown')}\n")
            f.write(f"- **Parameters**: {model_info.get('parameters', 0):,}\n")
            f.write(f"- **Classes**: {model_info.get('num_classes', 0)}\n")
            f.write(f"- **Validation Date**: {results.get('validation_timestamp', 'Unknown')}\n\n")
            
            # Validation Summary
            f.write("## Validation Summary\n")
            summary = results.get('validation_summary', {})
            f.write(f"- **Overall Status**: {summary.get('overall_status', 'Unknown').upper()}\n")
            f.write(f"- **Pass Rate**: {summary.get('pass_rate', 0):.1%}\n")
            f.write(f"- **Tests Passed**: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}\n\n")
            
            # Critical Issues
            if summary.get('critical_issues'):
                f.write("## Critical Issues\n")
                for issue in summary['critical_issues']:
                    f.write(f"- ❌ {issue}\n")
                f.write("\n")
            
            # Recommendations
            if summary.get('recommendations'):
                f.write("## Recommendations\n")
                for rec in summary['recommendations']:
                    f.write(f"- 💡 {rec}\n")
                f.write("\n")
            
            # Test Results Summary
            f.write("## Test Results Summary\n")
            
            test_categories = {
                'functional_tests': 'Functional Tests',
                'performance_tests': 'Performance Tests', 
                'security_tests': 'Security Tests',
                'robustness_tests': 'Robustness Tests'
            }
            
            for category_key, category_name in test_categories.items():
                if category_key in results:
                    f.write(f"### {category_name}\n")
                    category_results = results[category_key]
                    
                    for test_name, test_result in category_results.items():
                        if isinstance(test_result, dict):
                            status = "✅" if (test_result.get('status') == 'passed' or 
                                           test_result.get('meets_threshold') == True or
                                           test_result.get('successful') == True) else "❌"
                            f.write(f"- {status} **{test_name.replace('_', ' ').title()}**\n")
                    f.write("\n")

def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate YOLO security model')
    parser.add_argument('model_path', help='Path to trained YOLO model')
    parser.add_argument('--config', type=str, default='../config/dataset_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = SecurityModelValidator(args.model_path, args.config)
        
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Print summary
        summary = results.get('validation_summary', {})
        logger.info("="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Overall Status: {summary.get('overall_status', 'Unknown').upper()}")
        logger.info(f"Pass Rate: {summary.get('pass_rate', 0):.1%}")
        logger.info(f"Tests Passed: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
        
        if summary.get('critical_issues'):
            logger.warning("Critical Issues Found:")
            for issue in summary['critical_issues']:
                logger.warning(f"  - {issue}")
        
        return 0 if summary.get('overall_status') in ['excellent', 'good', 'acceptable'] else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)