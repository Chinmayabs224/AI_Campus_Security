#!/usr/bin/env python3
"""
YOLO Variant Comparison Framework
Specialized comparison framework for different YOLO model variants in campus security.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import time
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOVariantComparator:
    """Specialized framework for comparing YOLO model variants."""
    
    def __init__(self, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the YOLO variant comparator."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up paths
        self.yolo_data_dir = Path(self.config['dataset']['yolo_data_dir'])
        self.comparison_dir = Path("../yolo_variant_comparison")
        self.comparison_dir.mkdir(exist_ok=True)
        
        # YOLO variants to compare
        self.yolo_variants = {}
        
        # Comparison dimensions
        self.comparison_dimensions = {
            'accuracy': ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score'],
            'performance': ['inference_time_ms', 'fps', 'memory_usage_mb', 'throughput'],
            'efficiency': ['accuracy_per_param', 'fps_per_mb', 'efficiency_score'],
            'security': ['critical_recall', 'false_positive_rate', 'security_score'],
            'deployment': ['model_size_mb', 'parameters', 'edge_suitable', 'mobile_suitable']
        }
        
        # Security-specific evaluation criteria
        self.security_criteria = {
            'critical_classes': ['violence', 'emergency', 'intrusion'],
            'high_priority_classes': ['theft', 'suspicious'],
            'false_positive_thresholds': {'acceptable': 0.3, 'good': 0.2, 'excellent': 0.1},
            'real_time_requirements': {'edge': 15, 'server': 30, 'mobile': 10}  # FPS
        }
        
        logger.info("YOLOVariantComparator initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def add_yolo_variant(self, variant_name: str, model_path: str, description: str = "") -> None:
        """Add a YOLO variant to the comparison."""
        try:
            model = YOLO(model_path)
            
            # Extract variant information
            variant_info = self._extract_variant_info(variant_name, model_path)
            
            self.yolo_variants[variant_name] = {
                'model': model,
                'path': model_path,
                'description': description,
                'variant_info': variant_info
            }
            
            logger.info(f"Added YOLO variant '{variant_name}' to comparison")
            
        except Exception as e:
            logger.error(f"Failed to add YOLO variant '{variant_name}': {e}")
    
    def _extract_variant_info(self, variant_name: str, model_path: str) -> Dict:
        """Extract detailed information about the YOLO variant."""
        variant_info = {
            'name': variant_name,
            'path': model_path,
            'size_mb': Path(model_path).stat().st_size / (1024 * 1024),
            'variant_type': 'unknown',
            'architecture': 'yolov8'
        }
        
        # Determine variant type from name or path
        model_name = Path(model_path).name.lower()
        for size in ['n', 's', 'm', 'l', 'x']:
            if f'yolov8{size}' in model_name or f'v8{size}' in variant_name.lower():
                variant_info['variant_type'] = f'yolov8{size}'
                variant_info['size_category'] = self._get_size_category(size)
                break
        
        # Load model to get parameter count
        try:
            model = YOLO(model_path)
            variant_info['parameters'] = sum(p.numel() for p in model.model.parameters())
            variant_info['parameters_millions'] = variant_info['parameters'] / 1e6
        except:
            variant_info['parameters'] = 0
            variant_info['parameters_millions'] = 0
        
        return variant_info
    
    def _get_size_category(self, size_suffix: str) -> str:
        """Get size category description."""
        size_categories = {
            'n': 'nano',
            's': 'small', 
            'm': 'medium',
            'l': 'large',
            'x': 'extra_large'
        }
        return size_categories.get(size_suffix, 'unknown')
    
    def run_comprehensive_comparison(self) -> Dict:
        """Run comprehensive comparison of all YOLO variants."""
        logger.info(f"Starting comprehensive comparison of {len(self.yolo_variants)} YOLO variants...")
        
        comparison_results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'variants_compared': list(self.yolo_variants.keys()),
            'variant_info': {name: info['variant_info'] for name, info in self.yolo_variants.items()},
            'accuracy_comparison': {},
            'performance_comparison': {},
            'security_comparison': {},
            'efficiency_comparison': {},
            'deployment_suitability': {},
            'variant_rankings': {},
            'recommendations': {}
        }
        
        try:
            # 1. Accuracy Comparison
            logger.info("Comparing accuracy across variants...")
            comparison_results['accuracy_comparison'] = self._compare_accuracy()
            
            # 2. Performance Comparison
            logger.info("Comparing performance across variants...")
            comparison_results['performance_comparison'] = self._compare_performance()
            
            # 3. Security-Specific Comparison
            logger.info("Comparing security performance...")
            comparison_results['security_comparison'] = self._compare_security_performance()
            
            # 4. Efficiency Analysis
            logger.info("Analyzing efficiency metrics...")
            comparison_results['efficiency_comparison'] = self._analyze_efficiency(comparison_results)
            
            # 5. Deployment Suitability
            logger.info("Assessing deployment suitability...")
            comparison_results['deployment_suitability'] = self._assess_deployment_suitability(comparison_results)
            
            # 6. Generate Rankings
            logger.info("Generating variant rankings...")
            comparison_results['variant_rankings'] = self._generate_variant_rankings(comparison_results)
            
            # 7. Generate Recommendations
            logger.info("Generating recommendations...")
            comparison_results['recommendations'] = self._generate_variant_recommendations(comparison_results)
            
            # Save results
            self._save_comparison_results(comparison_results)
            
            logger.info("YOLO variant comparison completed!")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Variant comparison failed: {e}")
            comparison_results['error'] = str(e)
            return comparison_results
    
    def _compare_accuracy(self) -> Dict:
        """Compare accuracy metrics across YOLO variants."""
        accuracy_results = {}
        
        for variant_name, variant_info in self.yolo_variants.items():
            logger.info(f"Evaluating accuracy for {variant_name}...")
            
            try:
                # Run validation
                results = variant_info['model'].val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    verbose=False
                )
                
                # Extract standard metrics
                mAP50 = float(results.box.map50)
                mAP50_95 = float(results.box.map)
                precision = float(results.box.mp)
                recall = float(results.box.mr)
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                accuracy_results[variant_name] = {
                    'mAP50': mAP50,
                    'mAP50_95': mAP50_95,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }
                
                # Per-class accuracy for security classes
                if hasattr(results.box, 'ap_class_index'):
                    class_metrics = {}
                    class_names = list(self.config['classes'].keys())
                    
                    for i, class_idx in enumerate(results.box.ap_class_index):
                        if class_idx < len(class_names):
                            class_name = class_names[class_idx]
                            class_metrics[class_name] = {
                                'ap50': float(results.box.ap50[i]),
                                'ap50_95': float(results.box.ap[i]),
                                'precision': float(results.box.p[i]) if hasattr(results.box, 'p') else 0,
                                'recall': float(results.box.r[i]) if hasattr(results.box, 'r') else 0
                            }
                    
                    accuracy_results[variant_name]['per_class'] = class_metrics
                
            except Exception as e:
                logger.error(f"Accuracy evaluation failed for {variant_name}: {e}")
                accuracy_results[variant_name] = {'error': str(e)}
        
        return accuracy_results
    
    def _compare_performance(self) -> Dict:
        """Compare performance metrics across YOLO variants."""
        performance_results = {}
        
        # Test configurations
        input_sizes = [320, 640, 832]
        batch_sizes = [1, 4, 8]
        
        for variant_name, variant_info in self.yolo_variants.items():
            logger.info(f"Evaluating performance for {variant_name}...")
            
            variant_performance = {
                'inference_speed': {},
                'memory_usage': {},
                'throughput': {},
                'scalability': {}
            }
            
            try:
                # Inference speed across input sizes
                for input_size in input_sizes:
                    speed_metrics = self._measure_inference_speed(variant_info['model'], input_size)
                    variant_performance['inference_speed'][f'input_{input_size}'] = speed_metrics
                
                # Memory usage
                variant_performance['memory_usage'] = self._measure_memory_usage(variant_info['model'])
                
                # Throughput testing
                variant_performance['throughput'] = self._measure_throughput(variant_info['model'])
                
                # Batch scalability
                for batch_size in batch_sizes:
                    scalability_metrics = self._measure_batch_performance(variant_info['model'], batch_size)
                    variant_performance['scalability'][f'batch_{batch_size}'] = scalability_metrics
                
                # Calculate summary metrics
                variant_performance['summary'] = self._calculate_performance_summary(variant_performance)
                
            except Exception as e:
                logger.error(f"Performance evaluation failed for {variant_name}: {e}")
                variant_performance['error'] = str(e)
            
            performance_results[variant_name] = variant_performance
        
        return performance_results
    
    def _measure_inference_speed(self, model, input_size: int) -> Dict:
        """Measure inference speed for a specific input size."""
        try:
            # Create test input
            test_input = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                _ = model.predict(test_input, verbose=False)
            
            # Measure inference times
            times = []
            for _ in range(20):
                start_time = time.perf_counter()
                _ = model.predict(test_input, verbose=False)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            return {
                'mean_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'fps': 1000 / np.mean(times),
                'input_size': input_size
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _measure_memory_usage(self, model) -> Dict:
        """Measure memory usage during inference."""
        try:
            import psutil
            
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test input
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run inference and measure memory
            memory_samples = []
            for i in range(10):
                _ = model.predict(test_input, verbose=False)
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
            
            peak_memory = max(memory_samples)
            avg_memory = np.mean(memory_samples)
            memory_increase = avg_memory - baseline_memory
            
            return {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'avg_memory_mb': avg_memory,
                'memory_increase_mb': memory_increase,
                'memory_stable': abs(memory_samples[-1] - memory_samples[0]) < 50  # Less than 50MB variation
            }
            
        except ImportError:
            return {'error': 'psutil not available for memory measurement'}
        except Exception as e:
            return {'error': str(e)}
    
    def _measure_throughput(self, model) -> Dict:
        """Measure model throughput."""
        try:
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Sequential throughput
            start_time = time.perf_counter()
            num_inferences = 50
            
            for _ in range(num_inferences):
                _ = model.predict(test_input, verbose=False)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            return {
                'sequential_throughput_fps': num_inferences / total_time,
                'avg_inference_time_ms': (total_time / num_inferences) * 1000,
                'total_time_s': total_time
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _measure_batch_performance(self, model, batch_size: int) -> Dict:
        """Measure batch processing performance."""
        try:
            # Create batch input
            batch_input = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) 
                          for _ in range(batch_size)]
            
            # Warm up
            for _ in range(3):
                _ = model.predict(batch_input, verbose=False)
            
            # Measure batch processing time
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                _ = model.predict(batch_input, verbose=False)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            per_image_time = avg_time / batch_size
            
            return {
                'batch_size': batch_size,
                'total_time_s': avg_time,
                'per_image_time_s': per_image_time,
                'per_image_time_ms': per_image_time * 1000,
                'batch_throughput_fps': batch_size / avg_time
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_performance_summary(self, performance_data: Dict) -> Dict:
        """Calculate summary performance metrics."""
        summary = {}
        
        try:
            # Best FPS across input sizes
            inference_speeds = performance_data.get('inference_speed', {})
            fps_values = [metrics.get('fps', 0) for metrics in inference_speeds.values() 
                         if isinstance(metrics, dict) and 'fps' in metrics]
            
            if fps_values:
                summary['best_fps'] = max(fps_values)
                summary['avg_fps'] = np.mean(fps_values)
                summary['real_time_capable'] = max(fps_values) >= 15
            
            # Memory efficiency
            memory_data = performance_data.get('memory_usage', {})
            if isinstance(memory_data, dict) and 'memory_increase_mb' in memory_data:
                summary['memory_increase_mb'] = memory_data['memory_increase_mb']
                summary['memory_efficient'] = memory_data['memory_increase_mb'] < 100
            
            # Throughput
            throughput_data = performance_data.get('throughput', {})
            if isinstance(throughput_data, dict) and 'sequential_throughput_fps' in throughput_data:
                summary['throughput_fps'] = throughput_data['sequential_throughput_fps']
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _compare_security_performance(self) -> Dict:
        """Compare security-specific performance across variants."""
        security_results = {}
        
        for variant_name, variant_info in self.yolo_variants.items():
            logger.info(f"Evaluating security performance for {variant_name}...")
            
            try:
                # Run validation with security focus
                results = variant_info['model'].val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=0.25,  # Lower confidence to catch more detections
                    verbose=False
                )
                
                # Calculate security-specific metrics
                security_metrics = self._calculate_security_metrics(results)
                
                # False positive analysis
                fp_analysis = self._analyze_false_positives(variant_info['model'])
                
                # Critical class performance
                critical_performance = self._evaluate_critical_classes(results)
                
                security_results[variant_name] = {
                    'security_metrics': security_metrics,
                    'false_positive_analysis': fp_analysis,
                    'critical_performance': critical_performance,
                    'security_score': self._calculate_security_score(security_metrics, critical_performance)
                }
                
            except Exception as e:
                logger.error(f"Security evaluation failed for {variant_name}: {e}")
                security_results[variant_name] = {'error': str(e)}
        
        return security_results
    
    def _calculate_security_metrics(self, results) -> Dict:
        """Calculate security-specific metrics from validation results."""
        return {
            'overall_precision': float(results.box.mp),
            'overall_recall': float(results.box.mr),
            'overall_mAP50': float(results.box.map50),
            'false_positive_rate': 1 - float(results.box.mp)
        }
    
    def _analyze_false_positives(self, model) -> Dict:
        """Analyze false positive rates at different thresholds."""
        fp_analysis = {}
        
        try:
            thresholds = [0.3, 0.5, 0.7]
            
            for threshold in thresholds:
                results = model.val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=threshold,
                    verbose=False
                )
                
                precision = float(results.box.mp)
                fp_rate = 1 - precision
                
                fp_analysis[str(threshold)] = {
                    'precision': precision,
                    'false_positive_rate': fp_rate,
                    'meets_acceptable_threshold': fp_rate <= self.security_criteria['false_positive_thresholds']['acceptable'],
                    'meets_good_threshold': fp_rate <= self.security_criteria['false_positive_thresholds']['good'],
                    'meets_excellent_threshold': fp_rate <= self.security_criteria['false_positive_thresholds']['excellent']
                }
            
        except Exception as e:
            fp_analysis['error'] = str(e)
        
        return fp_analysis
    
    def _evaluate_critical_classes(self, results) -> Dict:
        """Evaluate performance on critical security classes."""
        critical_performance = {}
        
        try:
            class_names = list(self.config['classes'].keys())
            critical_classes = self.security_criteria['critical_classes']
            high_priority_classes = self.security_criteria['high_priority_classes']
            
            if hasattr(results.box, 'ap_class_index'):
                for class_name in critical_classes + high_priority_classes:
                    if class_name in class_names:
                        class_idx = class_names.index(class_name)
                        
                        if class_idx in results.box.ap_class_index:
                            idx = list(results.box.ap_class_index).index(class_idx)
                            
                            critical_performance[class_name] = {
                                'ap50': float(results.box.ap50[idx]),
                                'precision': float(results.box.p[idx]) if hasattr(results.box, 'p') else 0,
                                'recall': float(results.box.r[idx]) if hasattr(results.box, 'r') else 0,
                                'priority': 'critical' if class_name in critical_classes else 'high'
                            }
            
        except Exception as e:
            critical_performance['error'] = str(e)
        
        return critical_performance
    
    def _calculate_security_score(self, security_metrics: Dict, critical_performance: Dict) -> float:
        """Calculate overall security score for the variant."""
        try:
            # Base score from overall metrics
            overall_recall = security_metrics.get('overall_recall', 0)
            fp_rate = security_metrics.get('false_positive_rate', 1)
            
            base_score = overall_recall * 0.6 + (1 - fp_rate) * 0.4
            
            # Bonus for critical class performance
            critical_bonus = 0
            critical_classes = [name for name, perf in critical_performance.items() 
                              if isinstance(perf, dict) and perf.get('priority') == 'critical']
            
            if critical_classes:
                critical_recalls = [critical_performance[name]['recall'] for name in critical_classes]
                avg_critical_recall = np.mean(critical_recalls)
                critical_bonus = avg_critical_recall * 0.2
            
            return min(1.0, base_score + critical_bonus)
            
        except Exception:
            return 0.0
    
    def _analyze_efficiency(self, comparison_results: Dict) -> Dict:
        """Analyze efficiency metrics across variants."""
        efficiency_results = {}
        
        accuracy_data = comparison_results.get('accuracy_comparison', {})
        performance_data = comparison_results.get('performance_comparison', {})
        
        for variant_name in self.yolo_variants.keys():
            try:
                variant_info = self.yolo_variants[variant_name]['variant_info']
                accuracy = accuracy_data.get(variant_name, {})
                performance = performance_data.get(variant_name, {})
                
                # Extract key metrics
                mAP50 = accuracy.get('mAP50', 0)
                model_size_mb = variant_info.get('size_mb', 0)
                parameters = variant_info.get('parameters', 0)
                
                performance_summary = performance.get('summary', {})
                best_fps = performance_summary.get('best_fps', 0)
                
                # Calculate efficiency metrics
                efficiency_metrics = {
                    'accuracy_per_param': mAP50 / (parameters / 1e6) if parameters > 0 else 0,
                    'fps_per_mb': best_fps / model_size_mb if model_size_mb > 0 else 0,
                    'accuracy_per_mb': mAP50 / model_size_mb if model_size_mb > 0 else 0,
                    'efficiency_score': (mAP50 * best_fps) / model_size_mb if model_size_mb > 0 else 0
                }
                
                # Deployment efficiency categories
                efficiency_metrics['deployment_efficiency'] = {
                    'edge_efficiency': self._calculate_edge_efficiency(mAP50, best_fps, model_size_mb),
                    'mobile_efficiency': self._calculate_mobile_efficiency(mAP50, best_fps, model_size_mb),
                    'server_efficiency': self._calculate_server_efficiency(mAP50, best_fps)
                }
                
                efficiency_results[variant_name] = efficiency_metrics
                
            except Exception as e:
                efficiency_results[variant_name] = {'error': str(e)}
        
        return efficiency_results
    
    def _calculate_edge_efficiency(self, accuracy: float, fps: float, size_mb: float) -> float:
        """Calculate efficiency score for edge deployment."""
        # Edge deployment prioritizes: real-time capability, reasonable accuracy, small size
        real_time_score = min(1.0, fps / 15) * 0.4  # 15 FPS minimum
        accuracy_score = accuracy * 0.4
        size_score = max(0, (100 - size_mb) / 100) * 0.2  # Prefer smaller models
        
        return real_time_score + accuracy_score + size_score
    
    def _calculate_mobile_efficiency(self, accuracy: float, fps: float, size_mb: float) -> float:
        """Calculate efficiency score for mobile deployment."""
        # Mobile deployment prioritizes: small size, reasonable performance, acceptable accuracy
        size_score = max(0, (50 - size_mb) / 50) * 0.5  # Strong preference for small models
        performance_score = min(1.0, fps / 10) * 0.3  # 10 FPS minimum for mobile
        accuracy_score = accuracy * 0.2
        
        return size_score + performance_score + accuracy_score
    
    def _calculate_server_efficiency(self, accuracy: float, fps: float) -> float:
        """Calculate efficiency score for server deployment."""
        # Server deployment prioritizes: accuracy, high performance
        accuracy_score = accuracy * 0.6
        performance_score = min(1.0, fps / 30) * 0.4  # 30 FPS target for server
        
        return accuracy_score + performance_score
    
    def _assess_deployment_suitability(self, comparison_results: Dict) -> Dict:
        """Assess deployment suitability for each variant."""
        deployment_suitability = {}
        
        for variant_name in self.yolo_variants.keys():
            try:
                variant_info = self.yolo_variants[variant_name]['variant_info']
                accuracy_data = comparison_results.get('accuracy_comparison', {}).get(variant_name, {})
                performance_data = comparison_results.get('performance_comparison', {}).get(variant_name, {})
                security_data = comparison_results.get('security_comparison', {}).get(variant_name, {})
                
                # Extract key metrics
                mAP50 = accuracy_data.get('mAP50', 0)
                model_size_mb = variant_info.get('size_mb', 0)
                
                performance_summary = performance_data.get('summary', {})
                best_fps = performance_summary.get('best_fps', 0)
                memory_increase = performance_summary.get('memory_increase_mb', 0)
                
                security_score = security_data.get('security_score', 0)
                
                # Assess suitability for different deployment scenarios
                suitability = {
                    'edge_deployment': self._assess_edge_suitability(
                        mAP50, best_fps, model_size_mb, memory_increase, security_score
                    ),
                    'server_deployment': self._assess_server_suitability(
                        mAP50, best_fps, security_score
                    ),
                    'mobile_deployment': self._assess_mobile_suitability(
                        mAP50, best_fps, model_size_mb, memory_increase
                    ),
                    'production_readiness': self._assess_production_readiness(
                        mAP50, best_fps, security_score, security_data
                    )
                }
                
                deployment_suitability[variant_name] = suitability
                
            except Exception as e:
                deployment_suitability[variant_name] = {'error': str(e)}
        
        return deployment_suitability
    
    def _assess_edge_suitability(self, accuracy: float, fps: float, size_mb: float, 
                                memory_mb: float, security_score: float) -> Dict:
        """Assess suitability for edge deployment."""
        requirements = self.security_criteria['real_time_requirements']['edge']
        
        suitability = {
            'suitable': False,
            'score': 0.0,
            'requirements_met': {},
            'limitations': []
        }
        
        # Check requirements
        fps_ok = fps >= requirements
        size_ok = size_mb <= 100  # 100MB limit for edge
        memory_ok = memory_mb <= 500  # 500MB memory increase limit
        accuracy_ok = accuracy >= 0.6  # Minimum accuracy for edge
        security_ok = security_score >= 0.7  # Security threshold
        
        suitability['requirements_met'] = {
            'real_time_performance': fps_ok,
            'model_size': size_ok,
            'memory_usage': memory_ok,
            'accuracy_threshold': accuracy_ok,
            'security_threshold': security_ok
        }
        
        # Calculate suitability score
        requirements_score = sum(suitability['requirements_met'].values()) / len(suitability['requirements_met'])
        performance_bonus = min(1.0, fps / (requirements * 2))  # Bonus for exceeding requirements
        
        suitability['score'] = (requirements_score * 0.8 + performance_bonus * 0.2)
        suitability['suitable'] = all(suitability['requirements_met'].values())
        
        # Identify limitations
        if not fps_ok:
            suitability['limitations'].append(f"Insufficient FPS: {fps:.1f} < {requirements}")
        if not size_ok:
            suitability['limitations'].append(f"Model too large: {size_mb:.1f}MB > 100MB")
        if not memory_ok:
            suitability['limitations'].append(f"High memory usage: {memory_mb:.1f}MB > 500MB")
        if not accuracy_ok:
            suitability['limitations'].append(f"Low accuracy: {accuracy:.3f} < 0.6")
        if not security_ok:
            suitability['limitations'].append(f"Low security score: {security_score:.3f} < 0.7")
        
        return suitability
    
    def _assess_server_suitability(self, accuracy: float, fps: float, security_score: float) -> Dict:
        """Assess suitability for server deployment."""
        requirements = self.security_criteria['real_time_requirements']['server']
        
        suitability = {
            'suitable': False,
            'score': 0.0,
            'requirements_met': {},
            'limitations': []
        }
        
        # Check requirements (more relaxed for server)
        fps_ok = fps >= requirements
        accuracy_ok = accuracy >= 0.7  # Higher accuracy requirement for server
        security_ok = security_score >= 0.8  # Higher security requirement
        
        suitability['requirements_met'] = {
            'high_performance': fps_ok,
            'high_accuracy': accuracy_ok,
            'high_security': security_ok
        }
        
        # Calculate suitability score
        requirements_score = sum(suitability['requirements_met'].values()) / len(suitability['requirements_met'])
        accuracy_bonus = min(1.0, accuracy / 0.9)  # Bonus for high accuracy
        
        suitability['score'] = (requirements_score * 0.7 + accuracy_bonus * 0.3)
        suitability['suitable'] = all(suitability['requirements_met'].values())
        
        # Identify limitations
        if not fps_ok:
            suitability['limitations'].append(f"Insufficient FPS: {fps:.1f} < {requirements}")
        if not accuracy_ok:
            suitability['limitations'].append(f"Low accuracy: {accuracy:.3f} < 0.7")
        if not security_ok:
            suitability['limitations'].append(f"Low security score: {security_score:.3f} < 0.8")
        
        return suitability
    
    def _assess_mobile_suitability(self, accuracy: float, fps: float, size_mb: float, 
                                  memory_mb: float) -> Dict:
        """Assess suitability for mobile deployment."""
        requirements = self.security_criteria['real_time_requirements']['mobile']
        
        suitability = {
            'suitable': False,
            'score': 0.0,
            'requirements_met': {},
            'limitations': []
        }
        
        # Check requirements (strict size/memory limits)
        fps_ok = fps >= requirements
        size_ok = size_mb <= 50  # 50MB limit for mobile
        memory_ok = memory_mb <= 200  # 200MB memory increase limit
        accuracy_ok = accuracy >= 0.5  # Lower accuracy acceptable for mobile
        
        suitability['requirements_met'] = {
            'mobile_performance': fps_ok,
            'mobile_size': size_ok,
            'mobile_memory': memory_ok,
            'acceptable_accuracy': accuracy_ok
        }
        
        # Calculate suitability score (heavily weighted towards size/memory)
        size_score = max(0, (50 - size_mb) / 50)
        memory_score = max(0, (200 - memory_mb) / 200)
        performance_score = min(1.0, fps / requirements)
        accuracy_score = min(1.0, accuracy / 0.7)
        
        suitability['score'] = (size_score * 0.4 + memory_score * 0.3 + 
                               performance_score * 0.2 + accuracy_score * 0.1)
        suitability['suitable'] = all(suitability['requirements_met'].values())
        
        # Identify limitations
        if not fps_ok:
            suitability['limitations'].append(f"Insufficient FPS: {fps:.1f} < {requirements}")
        if not size_ok:
            suitability['limitations'].append(f"Model too large: {size_mb:.1f}MB > 50MB")
        if not memory_ok:
            suitability['limitations'].append(f"High memory usage: {memory_mb:.1f}MB > 200MB")
        if not accuracy_ok:
            suitability['limitations'].append(f"Low accuracy: {accuracy:.3f} < 0.5")
        
        return suitability
    
    def _assess_production_readiness(self, accuracy: float, fps: float, security_score: float, 
                                   security_data: Dict) -> Dict:
        """Assess overall production readiness."""
        readiness = {
            'ready': False,
            'score': 0.0,
            'criteria_met': {},
            'blockers': []
        }
        
        # Production readiness criteria
        accuracy_ok = accuracy >= 0.7
        performance_ok = fps >= 15
        security_ok = security_score >= 0.8
        
        # Check false positive rates
        fp_analysis = security_data.get('false_positive_analysis', {})
        fp_acceptable = False
        
        for threshold_data in fp_analysis.values():
            if isinstance(threshold_data, dict) and threshold_data.get('meets_acceptable_threshold', False):
                fp_acceptable = True
                break
        
        readiness['criteria_met'] = {
            'production_accuracy': accuracy_ok,
            'real_time_performance': performance_ok,
            'security_performance': security_ok,
            'acceptable_false_positives': fp_acceptable
        }
        
        # Calculate readiness score
        criteria_score = sum(readiness['criteria_met'].values()) / len(readiness['criteria_met'])
        quality_bonus = min(1.0, (accuracy + security_score) / 2)
        
        readiness['score'] = (criteria_score * 0.8 + quality_bonus * 0.2)
        readiness['ready'] = all(readiness['criteria_met'].values())
        
        # Identify blockers
        if not accuracy_ok:
            readiness['blockers'].append(f"Low accuracy: {accuracy:.3f} < 0.7")
        if not performance_ok:
            readiness['blockers'].append(f"Insufficient performance: {fps:.1f} FPS < 15")
        if not security_ok:
            readiness['blockers'].append(f"Low security score: {security_score:.3f} < 0.8")
        if not fp_acceptable:
            readiness['blockers'].append("High false positive rate (>30%)")
        
        return readiness
    
    def _generate_variant_rankings(self, comparison_results: Dict) -> Dict:
        """Generate rankings for different use cases."""
        rankings = {
            'overall_best': {},
            'accuracy_ranking': {},
            'performance_ranking': {},
            'efficiency_ranking': {},
            'security_ranking': {},
            'deployment_rankings': {}
        }
        
        try:
            variant_names = list(self.yolo_variants.keys())
            
            # Overall ranking (weighted combination)
            overall_scores = {}
            for variant_name in variant_names:
                accuracy_data = comparison_results.get('accuracy_comparison', {}).get(variant_name, {})
                performance_data = comparison_results.get('performance_comparison', {}).get(variant_name, {})
                security_data = comparison_results.get('security_comparison', {}).get(variant_name, {})
                
                mAP50 = accuracy_data.get('mAP50', 0)
                fps = performance_data.get('summary', {}).get('best_fps', 0)
                security_score = security_data.get('security_score', 0)
                
                # Weighted overall score (security-focused)
                overall_score = (mAP50 * 0.3 + min(1.0, fps / 30) * 0.2 + security_score * 0.5)
                overall_scores[variant_name] = overall_score
            
            rankings['overall_best'] = self._rank_variants(overall_scores)
            
            # Accuracy ranking
            accuracy_scores = {
                name: comparison_results.get('accuracy_comparison', {}).get(name, {}).get('mAP50', 0)
                for name in variant_names
            }
            rankings['accuracy_ranking'] = self._rank_variants(accuracy_scores)
            
            # Performance ranking
            performance_scores = {
                name: comparison_results.get('performance_comparison', {}).get(name, {}).get('summary', {}).get('best_fps', 0)
                for name in variant_names
            }
            rankings['performance_ranking'] = self._rank_variants(performance_scores)
            
            # Security ranking
            security_scores = {
                name: comparison_results.get('security_comparison', {}).get(name, {}).get('security_score', 0)
                for name in variant_names
            }
            rankings['security_ranking'] = self._rank_variants(security_scores)
            
            # Efficiency ranking
            efficiency_scores = {
                name: comparison_results.get('efficiency_comparison', {}).get(name, {}).get('efficiency_score', 0)
                for name in variant_names
            }
            rankings['efficiency_ranking'] = self._rank_variants(efficiency_scores)
            
            # Deployment-specific rankings
            deployment_data = comparison_results.get('deployment_suitability', {})
            
            for deployment_type in ['edge_deployment', 'server_deployment', 'mobile_deployment']:
                deployment_scores = {
                    name: deployment_data.get(name, {}).get(deployment_type, {}).get('score', 0)
                    for name in variant_names
                }
                rankings['deployment_rankings'][deployment_type] = self._rank_variants(deployment_scores)
            
        except Exception as e:
            rankings['error'] = str(e)
        
        return rankings
    
    def _rank_variants(self, scores: Dict[str, float]) -> List[Dict]:
        """Rank variants by score."""
        sorted_variants = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings = []
        for rank, (variant_name, score) in enumerate(sorted_variants, 1):
            rankings.append({
                'rank': rank,
                'variant': variant_name,
                'score': score
            })
        
        return rankings
    
    def _generate_variant_recommendations(self, comparison_results: Dict) -> Dict:
        """Generate recommendations for YOLO variant selection."""
        recommendations = {
            'best_for_edge': None,
            'best_for_server': None,
            'best_for_mobile': None,
            'best_overall': None,
            'variant_analysis': {},
            'selection_guide': []
        }
        
        try:
            rankings = comparison_results.get('variant_rankings', {})
            deployment_suitability = comparison_results.get('deployment_suitability', {})
            
            # Best for each deployment type
            deployment_rankings = rankings.get('deployment_rankings', {})
            
            if deployment_rankings.get('edge_deployment'):
                best_edge = deployment_rankings['edge_deployment'][0]
                recommendations['best_for_edge'] = {
                    'variant': best_edge['variant'],
                    'score': best_edge['score'],
                    'reason': self._get_deployment_reason('edge', best_edge['variant'], comparison_results)
                }
            
            if deployment_rankings.get('server_deployment'):
                best_server = deployment_rankings['server_deployment'][0]
                recommendations['best_for_server'] = {
                    'variant': best_server['variant'],
                    'score': best_server['score'],
                    'reason': self._get_deployment_reason('server', best_server['variant'], comparison_results)
                }
            
            if deployment_rankings.get('mobile_deployment'):
                best_mobile = deployment_rankings['mobile_deployment'][0]
                recommendations['best_for_mobile'] = {
                    'variant': best_mobile['variant'],
                    'score': best_mobile['score'],
                    'reason': self._get_deployment_reason('mobile', best_mobile['variant'], comparison_results)
                }
            
            # Best overall
            overall_ranking = rankings.get('overall_best', [])
            if overall_ranking:
                best_overall = overall_ranking[0]
                recommendations['best_overall'] = {
                    'variant': best_overall['variant'],
                    'score': best_overall['score'],
                    'reason': "Best balanced performance across accuracy, speed, and security metrics"
                }
            
            # Variant analysis
            for variant_name in self.yolo_variants.keys():
                variant_info = self.yolo_variants[variant_name]['variant_info']
                
                # Find strengths and weaknesses
                strengths = []
                weaknesses = []
                
                # Check rankings
                for ranking_type, ranking_data in rankings.items():
                    if ranking_type == 'deployment_rankings':
                        continue
                    
                    if isinstance(ranking_data, list):
                        variant_rank = next((item['rank'] for item in ranking_data 
                                           if item['variant'] == variant_name), None)
                        
                        if variant_rank == 1:
                            strengths.append(f"Best {ranking_type.replace('_', ' ')}")
                        elif variant_rank and variant_rank <= 2:
                            strengths.append(f"Top {ranking_type.replace('_', ' ')}")
                        elif variant_rank and variant_rank >= len(self.yolo_variants) - 1:
                            weaknesses.append(f"Poor {ranking_type.replace('_', ' ')}")
                
                # Check deployment suitability
                deployment_data = deployment_suitability.get(variant_name, {})
                suitable_deployments = []
                
                for deploy_type, deploy_data in deployment_data.items():
                    if isinstance(deploy_data, dict) and deploy_data.get('suitable', False):
                        suitable_deployments.append(deploy_type.replace('_', ' '))
                
                recommendations['variant_analysis'][variant_name] = {
                    'variant_type': variant_info.get('variant_type', 'unknown'),
                    'size_category': variant_info.get('size_category', 'unknown'),
                    'strengths': strengths,
                    'weaknesses': weaknesses,
                    'suitable_deployments': suitable_deployments,
                    'recommendation': self._generate_variant_recommendation(
                        variant_name, strengths, weaknesses, suitable_deployments
                    )
                }
            
            # Selection guide
            recommendations['selection_guide'] = self._generate_selection_guide(comparison_results)
            
        except Exception as e:
            recommendations['error'] = str(e)
        
        return recommendations
    
    def _get_deployment_reason(self, deployment_type: str, variant_name: str, 
                              comparison_results: Dict) -> str:
        """Get reason why variant is best for deployment type."""
        try:
            accuracy_data = comparison_results.get('accuracy_comparison', {}).get(variant_name, {})
            performance_data = comparison_results.get('performance_comparison', {}).get(variant_name, {})
            variant_info = self.yolo_variants[variant_name]['variant_info']
            
            mAP50 = accuracy_data.get('mAP50', 0)
            fps = performance_data.get('summary', {}).get('best_fps', 0)
            size_mb = variant_info.get('size_mb', 0)
            
            if deployment_type == 'edge':
                return f"Optimal balance of speed ({fps:.1f} FPS), accuracy ({mAP50:.3f}), and size ({size_mb:.1f}MB) for edge deployment"
            elif deployment_type == 'server':
                return f"Highest accuracy ({mAP50:.3f}) with excellent performance ({fps:.1f} FPS) for server deployment"
            elif deployment_type == 'mobile':
                return f"Smallest size ({size_mb:.1f}MB) with acceptable performance ({fps:.1f} FPS) for mobile deployment"
            else:
                return "Best overall performance metrics"
                
        except Exception:
            return "Recommended based on performance analysis"
    
    def _generate_variant_recommendation(self, variant_name: str, strengths: List[str], 
                                       weaknesses: List[str], suitable_deployments: List[str]) -> str:
        """Generate specific recommendation for a variant."""
        if not suitable_deployments:
            return "Not recommended for production deployment due to performance limitations"
        
        if len(strengths) >= 2:
            return f"Highly recommended for {', '.join(suitable_deployments)} - excels in {', '.join(strengths[:2])}"
        elif strengths:
            return f"Recommended for {', '.join(suitable_deployments)} - strong {strengths[0]}"
        else:
            return f"Suitable for {', '.join(suitable_deployments)} with acceptable performance"
    
    def _generate_selection_guide(self, comparison_results: Dict) -> List[str]:
        """Generate general selection guide."""
        guide = [
            "YOLO Variant Selection Guide:",
            "",
            "For Edge Deployment:",
            "- Prioritize YOLOv8n or YOLOv8s for real-time performance",
            "- Ensure model size < 100MB and FPS ≥ 15",
            "- Accept slightly lower accuracy for speed",
            "",
            "For Server Deployment:",
            "- Use YOLOv8m or YOLOv8l for highest accuracy",
            "- Target FPS ≥ 30 with mAP@0.5 ≥ 0.7",
            "- Model size less critical",
            "",
            "For Mobile Deployment:",
            "- Use YOLOv8n exclusively",
            "- Require model size < 50MB",
            "- Optimize for battery efficiency",
            "",
            "Security Considerations:",
            "- Prioritize variants with security score ≥ 0.8",
            "- Ensure false positive rate ≤ 30%",
            "- Test critical class detection thoroughly"
        ]
        
        return guide
    
    def _save_comparison_results(self, results: Dict) -> None:
        """Save YOLO variant comparison results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.comparison_dir / f"yolo_comparison_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / 'yolo_variant_comparison.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate comparison report
        self._generate_comparison_report(results, results_dir)
        
        # Generate variant selection guide
        self._generate_selection_guide_document(results, results_dir)
        
        # Generate comparison plots
        self._generate_comparison_plots(results, results_dir)
        
        logger.info(f"YOLO variant comparison results saved to: {results_dir}")
    
    def _generate_comparison_report(self, results: Dict, output_dir: Path) -> None:
        """Generate comprehensive comparison report."""
        report_file = output_dir / 'yolo_variant_comparison_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# YOLO Variant Comparison Report\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            recommendations = results.get('recommendations', {})
            
            if recommendations.get('best_overall'):
                best = recommendations['best_overall']
                f.write(f"**Best Overall Variant**: {best['variant']} (Score: {best['score']:.3f})\n\n")
            
            # Deployment Recommendations
            f.write("## Deployment Recommendations\n\n")
            
            deployment_types = [
                ('best_for_edge', 'Edge Deployment'),
                ('best_for_server', 'Server Deployment'),
                ('best_for_mobile', 'Mobile Deployment')
            ]
            
            for rec_key, rec_name in deployment_types:
                if recommendations.get(rec_key):
                    rec = recommendations[rec_key]
                    f.write(f"### {rec_name}\n")
                    f.write(f"**Recommended**: {rec['variant']} (Score: {rec['score']:.3f})\n")
                    f.write(f"**Reason**: {rec['reason']}\n\n")
            
            # Variant Comparison Table
            f.write("## Variant Comparison\n\n")
            f.write("| Variant | Type | Size (MB) | mAP@0.5 | FPS | Security Score | Edge | Server | Mobile |\n")
            f.write("|---------|------|-----------|---------|-----|----------------|------|--------|--------|\n")
            
            for variant_name in self.yolo_variants.keys():
                variant_info = results.get('variant_info', {}).get(variant_name, {})
                accuracy = results.get('accuracy_comparison', {}).get(variant_name, {})
                performance = results.get('performance_comparison', {}).get(variant_name, {})
                security = results.get('security_comparison', {}).get(variant_name, {})
                deployment = results.get('deployment_suitability', {}).get(variant_name, {})
                
                variant_type = variant_info.get('variant_type', 'unknown')
                size_mb = variant_info.get('size_mb', 0)
                mAP50 = accuracy.get('mAP50', 0)
                fps = performance.get('summary', {}).get('best_fps', 0)
                security_score = security.get('security_score', 0)
                
                edge_suitable = '✅' if deployment.get('edge_deployment', {}).get('suitable', False) else '❌'
                server_suitable = '✅' if deployment.get('server_deployment', {}).get('suitable', False) else '❌'
                mobile_suitable = '✅' if deployment.get('mobile_deployment', {}).get('suitable', False) else '❌'
                
                f.write(f"| {variant_name} | {variant_type} | {size_mb:.1f} | {mAP50:.3f} | {fps:.1f} | {security_score:.3f} | {edge_suitable} | {server_suitable} | {mobile_suitable} |\n")
            
            f.write("\n")
            
            # Detailed Analysis
            f.write("## Detailed Variant Analysis\n\n")
            variant_analysis = recommendations.get('variant_analysis', {})
            
            for variant_name, analysis in variant_analysis.items():
                f.write(f"### {variant_name}\n")
                f.write(f"**Type**: {analysis.get('variant_type', 'unknown')}\n")
                f.write(f"**Category**: {analysis.get('size_category', 'unknown')}\n")
                
                if analysis.get('strengths'):
                    f.write(f"**Strengths**: {', '.join(analysis['strengths'])}\n")
                
                if analysis.get('weaknesses'):
                    f.write(f"**Weaknesses**: {', '.join(analysis['weaknesses'])}\n")
                
                if analysis.get('suitable_deployments'):
                    f.write(f"**Suitable For**: {', '.join(analysis['suitable_deployments'])}\n")
                
                f.write(f"**Recommendation**: {analysis.get('recommendation', 'No specific recommendation')}\n\n")
            
            # Selection Guide
            f.write("## Selection Guide\n\n")
            selection_guide = recommendations.get('selection_guide', [])
            for line in selection_guide:
                f.write(f"{line}\n")
    
    def _generate_selection_guide_document(self, results: Dict, output_dir: Path) -> None:
        """Generate standalone selection guide document."""
        guide_file = output_dir / 'yolo_variant_selection_guide.md'
        
        with open(guide_file, 'w') as f:
            f.write("# YOLO Variant Selection Guide for Campus Security\n\n")
            
            f.write("## Quick Selection Matrix\n\n")
            f.write("| Use Case | Recommended Variant | Key Criteria |\n")
            f.write("|----------|-------------------|---------------|\n")
            
            recommendations = results.get('recommendations', {})
            
            if recommendations.get('best_for_edge'):
                edge_rec = recommendations['best_for_edge']
                f.write(f"| Edge Deployment | {edge_rec['variant']} | Real-time performance, small size |\n")
            
            if recommendations.get('best_for_server'):
                server_rec = recommendations['best_for_server']
                f.write(f"| Server Deployment | {server_rec['variant']} | High accuracy, excellent performance |\n")
            
            if recommendations.get('best_for_mobile'):
                mobile_rec = recommendations['best_for_mobile']
                f.write(f"| Mobile Deployment | {mobile_rec['variant']} | Minimal size, battery efficiency |\n")
            
            f.write("\n")
            
            # Detailed selection criteria
            f.write("## Detailed Selection Criteria\n\n")
            
            f.write("### Edge Deployment Requirements\n")
            f.write("- **Performance**: ≥15 FPS for real-time processing\n")
            f.write("- **Model Size**: ≤100 MB for edge device storage\n")
            f.write("- **Memory Usage**: ≤500 MB additional memory\n")
            f.write("- **Accuracy**: ≥60% mAP@0.5 (acceptable trade-off)\n")
            f.write("- **Security**: ≥70% security score\n\n")
            
            f.write("### Server Deployment Requirements\n")
            f.write("- **Performance**: ≥30 FPS for high throughput\n")
            f.write("- **Accuracy**: ≥70% mAP@0.5 (high accuracy priority)\n")
            f.write("- **Security**: ≥80% security score\n")
            f.write("- **Model Size**: Not critical (server resources available)\n\n")
            
            f.write("### Mobile Deployment Requirements\n")
            f.write("- **Model Size**: ≤50 MB for mobile app distribution\n")
            f.write("- **Memory Usage**: ≤200 MB additional memory\n")
            f.write("- **Performance**: ≥10 FPS (acceptable for mobile)\n")
            f.write("- **Accuracy**: ≥50% mAP@0.5 (mobile trade-off)\n\n")
            
            # Variant characteristics
            f.write("## YOLO Variant Characteristics\n\n")
            
            variant_chars = {
                'yolov8n': {
                    'name': 'YOLOv8 Nano',
                    'best_for': 'Mobile and edge deployment',
                    'pros': ['Smallest size', 'Fastest inference', 'Low memory usage'],
                    'cons': ['Lower accuracy', 'Limited detection capability']
                },
                'yolov8s': {
                    'name': 'YOLOv8 Small',
                    'best_for': 'Edge deployment with balanced performance',
                    'pros': ['Good speed-accuracy balance', 'Reasonable size', 'Edge-friendly'],
                    'cons': ['Moderate accuracy', 'May struggle with small objects']
                },
                'yolov8m': {
                    'name': 'YOLOv8 Medium',
                    'best_for': 'Server deployment',
                    'pros': ['High accuracy', 'Good detection capability', 'Robust performance'],
                    'cons': ['Larger size', 'Higher memory usage', 'Slower inference']
                },
                'yolov8l': {
                    'name': 'YOLOv8 Large',
                    'best_for': 'High-accuracy server deployment',
                    'pros': ['Highest accuracy', 'Excellent detection', 'Best security performance'],
                    'cons': ['Large size', 'High memory usage', 'Slow inference']
                },
                'yolov8x': {
                    'name': 'YOLOv8 Extra Large',
                    'best_for': 'Research and maximum accuracy scenarios',
                    'pros': ['Maximum accuracy', 'Best possible detection'],
                    'cons': ['Very large size', 'Very slow', 'High resource usage']
                }
            }
            
            for variant_key, char in variant_chars.items():
                f.write(f"### {char['name']}\n")
                f.write(f"**Best For**: {char['best_for']}\n\n")
                f.write("**Pros**:\n")
                for pro in char['pros']:
                    f.write(f"- {pro}\n")
                f.write("\n**Cons**:\n")
                for con in char['cons']:
                    f.write(f"- {con}\n")
                f.write("\n")
    
    def _generate_comparison_plots(self, results: Dict, output_dir: Path) -> None:
        """Generate comparison visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Prepare data
            variant_names = list(self.yolo_variants.keys())
            
            # Extract metrics for plotting
            accuracy_data = results.get('accuracy_comparison', {})
            performance_data = results.get('performance_comparison', {})
            variant_info = results.get('variant_info', {})
            
            plot_data = []
            for variant_name in variant_names:
                accuracy = accuracy_data.get(variant_name, {})
                performance = performance_data.get(variant_name, {})
                info = variant_info.get(variant_name, {})
                
                plot_data.append({
                    'Variant': variant_name,
                    'mAP@0.5': accuracy.get('mAP50', 0),
                    'FPS': performance.get('summary', {}).get('best_fps', 0),
                    'Size (MB)': info.get('size_mb', 0),
                    'Parameters (M)': info.get('parameters_millions', 0)
                })
            
            df = pd.DataFrame(plot_data)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy vs Speed
            axes[0, 0].scatter(df['FPS'], df['mAP@0.5'], s=100, alpha=0.7)
            for i, variant in enumerate(df['Variant']):
                axes[0, 0].annotate(variant, (df['FPS'].iloc[i], df['mAP@0.5'].iloc[i]), 
                                   xytext=(5, 5), textcoords='offset points')
            axes[0, 0].set_xlabel('FPS')
            axes[0, 0].set_ylabel('mAP@0.5')
            axes[0, 0].set_title('Accuracy vs Speed Trade-off')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Model Size vs Accuracy
            axes[0, 1].scatter(df['Size (MB)'], df['mAP@0.5'], s=100, alpha=0.7)
            for i, variant in enumerate(df['Variant']):
                axes[0, 1].annotate(variant, (df['Size (MB)'].iloc[i], df['mAP@0.5'].iloc[i]), 
                                   xytext=(5, 5), textcoords='offset points')
            axes[0, 1].set_xlabel('Model Size (MB)')
            axes[0, 1].set_ylabel('mAP@0.5')
            axes[0, 1].set_title('Model Size vs Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Performance comparison bar chart
            axes[1, 0].bar(df['Variant'], df['FPS'])
            axes[1, 0].set_ylabel('FPS')
            axes[1, 0].set_title('Performance Comparison')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Accuracy comparison bar chart
            axes[1, 1].bar(df['Variant'], df['mAP@0.5'])
            axes[1, 1].set_ylabel('mAP@0.5')
            axes[1, 1].set_title('Accuracy Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'yolo_variant_comparison_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create radar chart for top variants
            self._create_radar_chart(results, output_dir)
            
        except Exception as e:
            logger.error(f"Failed to generate comparison plots: {e}")
    
    def _create_radar_chart(self, results: Dict, output_dir: Path) -> None:
        """Create radar chart comparing top variants."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get top 3 variants
            rankings = results.get('variant_rankings', {})
            overall_ranking = rankings.get('overall_best', [])
            
            if len(overall_ranking) < 2:
                return
            
            top_variants = [item['variant'] for item in overall_ranking[:3]]
            
            # Metrics for radar chart
            metrics = ['Accuracy', 'Speed', 'Security', 'Efficiency', 'Edge Suitability']
            
            # Prepare data
            radar_data = []
            for variant in top_variants:
                accuracy = results.get('accuracy_comparison', {}).get(variant, {}).get('mAP50', 0)
                speed = min(1.0, results.get('performance_comparison', {}).get(variant, {}).get('summary', {}).get('best_fps', 0) / 30)
                security = results.get('security_comparison', {}).get(variant, {}).get('security_score', 0)
                efficiency = min(1.0, results.get('efficiency_comparison', {}).get(variant, {}).get('efficiency_score', 0) / 10)
                edge_suit = results.get('deployment_suitability', {}).get(variant, {}).get('edge_deployment', {}).get('score', 0)
                
                radar_data.append([accuracy, speed, security, efficiency, edge_suit])
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = ['red', 'blue', 'green']
            
            for i, (variant, data) in enumerate(zip(top_variants, radar_data)):
                data += data[:1]  # Complete the circle
                ax.plot(angles, data, 'o-', linewidth=2, label=variant, color=colors[i])
                ax.fill(angles, data, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('YOLO Variant Performance Comparison', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.savefig(output_dir / 'yolo_variant_radar_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create radar chart: {e}")


def main():
    """Main function for YOLO variant comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Variant Comparator')
    parser.add_argument('--config', default='../config/dataset_config.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model paths in format name:path (e.g., nano:yolov8n.pt small:yolov8s.pt)')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = YOLOVariantComparator(args.config)
    
    # Add models
    for model_spec in args.models:
        if ':' in model_spec:
            name, path = model_spec.split(':', 1)
            comparator.add_yolo_variant(name, path)
        else:
            logger.error(f"Invalid model specification: {model_spec}. Use format name:path")
            return
    
    # Run comparison
    results = comparator.run_comprehensive_comparison()
    
    # Print summary
    recommendations = results.get('recommendations', {})
    
    print("\n=== YOLO Variant Comparison Results ===")
    
    if recommendations.get('best_overall'):
        best = recommendations['best_overall']
        print(f"Best Overall: {best['variant']} (Score: {best['score']:.3f})")
    
    print("\nDeployment Recommendations:")
    for deploy_type in ['best_for_edge', 'best_for_server', 'best_for_mobile']:
        if recommendations.get(deploy_type):
            rec = recommendations[deploy_type]
            deploy_name = deploy_type.replace('best_for_', '').title()
            print(f"  {deploy_name}: {rec['variant']} (Score: {rec['score']:.3f})")
    
    return results


if __name__ == "__main__":
    main()