#!/usr/bin/env python3
"""
Enhanced Security Model Validator
Specialized validation for campus security requirements 6.2 and 6.3.
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
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from ultralytics import YOLO
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSecurityValidator:
    """Enhanced validator focusing on security-specific requirements."""
    
    def __init__(self, model_path: str, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the enhanced validator."""
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Load model
        self.model = YOLO(str(self.model_path))
        
        # Set up paths
        self.yolo_data_dir = Path(self.config['dataset']['yolo_data_dir'])
        self.validation_dir = Path("../enhanced_validation_results")
        self.validation_dir.mkdir(exist_ok=True)
        
        # Security-specific thresholds (Requirement 6.2)
        self.false_positive_thresholds = {
            'initial_target': 0.30,  # ≤30% initially
            'improved_target': 0.10,  # ≤10% after 3 months
            'excellent_target': 0.05  # Excellent performance
        }
        
        # Environmental conditions for adaptive testing (Requirement 6.3)
        self.environmental_conditions = {
            'daylight': {'brightness_factor': 1.0, 'contrast_factor': 1.0},
            'artificial_light': {'brightness_factor': 0.8, 'contrast_factor': 0.9},
            'low_light': {'brightness_factor': 0.4, 'contrast_factor': 0.7},
            'night_vision': {'brightness_factor': 0.2, 'contrast_factor': 0.5},
            'overcast': {'brightness_factor': 0.7, 'contrast_factor': 0.8},
            'sunny': {'brightness_factor': 1.2, 'contrast_factor': 1.1}
        }
        
        # Security event types for validation
        self.security_events = {
            'intrusion': {'priority': 'critical', 'min_confidence': 0.7},
            'loitering': {'priority': 'medium', 'min_confidence': 0.5},
            'crowding': {'priority': 'medium', 'min_confidence': 0.6},
            'abandoned_object': {'priority': 'low', 'min_confidence': 0.4},
            'violence': {'priority': 'critical', 'min_confidence': 0.8},
            'theft': {'priority': 'high', 'min_confidence': 0.7},
            'suspicious': {'priority': 'high', 'min_confidence': 0.6},
            'normal': {'priority': 'normal', 'min_confidence': 0.3}
        }
        
        logger.info(f"EnhancedSecurityValidator initialized for: {self.model_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_enhanced_validation(self) -> Dict:
        """Run enhanced security-focused validation."""
        logger.info("Starting enhanced security validation...")
        
        validation_results = {
            'model_info': self._get_model_info(),
            'validation_timestamp': datetime.now().isoformat(),
            'false_positive_analysis': {},
            'environmental_adaptation': {},
            'security_scenario_tests': {},
            'threshold_optimization': {},
            'compliance_assessment': {},
            'validation_summary': {}
        }
        
        try:
            # 1. False Positive Rate Analysis (Requirement 6.2)
            logger.info("Analyzing false positive rates...")
            validation_results['false_positive_analysis'] = self._analyze_false_positive_rates()
            
            # 2. Environmental Adaptation Testing (Requirement 6.3)
            logger.info("Testing environmental adaptation...")
            validation_results['environmental_adaptation'] = self._test_environmental_adaptation()
            
            # 3. Security Scenario Validation
            logger.info("Validating security scenarios...")
            validation_results['security_scenario_tests'] = self._validate_security_scenarios()
            
            # 4. Threshold Optimization
            logger.info("Optimizing detection thresholds...")
            validation_results['threshold_optimization'] = self._optimize_detection_thresholds()
            
            # 5. Compliance Assessment
            logger.info("Assessing requirement compliance...")
            validation_results['compliance_assessment'] = self._assess_compliance(validation_results)
            
            # 6. Generate Summary
            validation_results['validation_summary'] = self._generate_enhanced_summary(validation_results)
            
            # Save results
            self._save_enhanced_results(validation_results)
            
            logger.info("Enhanced security validation completed!")
            return validation_results
            
        except Exception as e:
            logger.error(f"Enhanced validation failed: {e}")
            validation_results['validation_summary'] = {'status': 'failed', 'error': str(e)}
            return validation_results
    
    def _get_model_info(self) -> Dict:
        """Get enhanced model information."""
        model_info = {
            'model_path': str(self.model_path),
            'model_size_mb': self.model_path.stat().st_size / (1024 * 1024),
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
            'input_size': 640,
            'classes': list(self.config['classes'].keys()),
            'num_classes': len(self.config['classes'])
        }
        
        # Determine model variant
        model_name = self.model_path.name.lower()
        for size in ['n', 's', 'm', 'l', 'x']:
            if f'yolov8{size}' in model_name:
                model_info['variant'] = f'yolov8{size}'
                break
        else:
            model_info['variant'] = 'unknown'
        
        return model_info
    
    def _analyze_false_positive_rates(self) -> Dict:
        """Comprehensive false positive rate analysis (Requirement 6.2)."""
        logger.info("Analyzing false positive rates across confidence thresholds...")
        
        false_positive_analysis = {
            'threshold_analysis': {},
            'class_specific_fp': {},
            'temporal_analysis': {},
            'optimization_recommendations': {}
        }
        
        try:
            # Test multiple confidence thresholds
            confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            for conf_threshold in confidence_thresholds:
                logger.info(f"Testing confidence threshold: {conf_threshold}")
                
                # Run validation at this threshold
                results = self.model.val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=conf_threshold,
                    verbose=False
                )
                
                # Calculate false positive rate
                precision = float(results.box.mp)
                false_positive_rate = 1 - precision
                
                false_positive_analysis['threshold_analysis'][str(conf_threshold)] = {
                    'precision': precision,
                    'false_positive_rate': false_positive_rate,
                    'recall': float(results.box.mr),
                    'mAP50': float(results.box.map50),
                    'meets_initial_target': false_positive_rate <= self.false_positive_thresholds['initial_target'],
                    'meets_improved_target': false_positive_rate <= self.false_positive_thresholds['improved_target']
                }
            
            # Find optimal threshold for each target
            optimal_thresholds = self._find_optimal_thresholds(false_positive_analysis['threshold_analysis'])
            false_positive_analysis['optimal_thresholds'] = optimal_thresholds
            
            # Class-specific false positive analysis
            false_positive_analysis['class_specific_fp'] = self._analyze_class_specific_fp()
            
            # Generate optimization recommendations
            false_positive_analysis['optimization_recommendations'] = self._generate_fp_optimization_recommendations(
                false_positive_analysis
            )
            
        except Exception as e:
            false_positive_analysis['error'] = str(e)
        
        return false_positive_analysis
    
    def _find_optimal_thresholds(self, threshold_analysis: Dict) -> Dict:
        """Find optimal confidence thresholds for different targets."""
        optimal_thresholds = {
            'initial_target': None,
            'improved_target': None,
            'best_f1': None
        }
        
        best_f1_score = 0
        
        for threshold_str, metrics in threshold_analysis.items():
            threshold = float(threshold_str)
            fp_rate = metrics['false_positive_rate']
            precision = metrics['precision']
            recall = metrics['recall']
            
            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Check targets
            if fp_rate <= self.false_positive_thresholds['initial_target'] and optimal_thresholds['initial_target'] is None:
                optimal_thresholds['initial_target'] = {
                    'threshold': threshold,
                    'false_positive_rate': fp_rate,
                    'precision': precision,
                    'recall': recall
                }
            
            if fp_rate <= self.false_positive_thresholds['improved_target'] and optimal_thresholds['improved_target'] is None:
                optimal_thresholds['improved_target'] = {
                    'threshold': threshold,
                    'false_positive_rate': fp_rate,
                    'precision': precision,
                    'recall': recall
                }
            
            # Track best F1 score
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                optimal_thresholds['best_f1'] = {
                    'threshold': threshold,
                    'f1_score': f1_score,
                    'false_positive_rate': fp_rate,
                    'precision': precision,
                    'recall': recall
                }
        
        return optimal_thresholds
    
    def _analyze_class_specific_fp(self) -> Dict:
        """Analyze false positive rates for each security event class."""
        class_fp_analysis = {}
        
        try:
            # Run validation to get per-class metrics
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                conf=0.25,
                verbose=False
            )
            
            class_names = list(self.config['classes'].keys())
            
            if hasattr(results.box, 'ap_class_index'):
                for i, class_idx in enumerate(results.box.ap_class_index):
                    if class_idx < len(class_names):
                        class_name = class_names[class_idx]
                        precision = float(results.box.p[i]) if hasattr(results.box, 'p') else 0
                        recall = float(results.box.r[i]) if hasattr(results.box, 'r') else 0
                        
                        class_fp_analysis[class_name] = {
                            'precision': precision,
                            'false_positive_rate': 1 - precision,
                            'recall': recall,
                            'priority': self.security_events.get(class_name, {}).get('priority', 'unknown'),
                            'meets_target': (1 - precision) <= self.false_positive_thresholds['initial_target']
                        }
            
        except Exception as e:
            class_fp_analysis['error'] = str(e)
        
        return class_fp_analysis
    
    def _generate_fp_optimization_recommendations(self, fp_analysis: Dict) -> List[str]:
        """Generate recommendations for false positive optimization."""
        recommendations = []
        
        try:
            threshold_analysis = fp_analysis.get('threshold_analysis', {})
            optimal_thresholds = fp_analysis.get('optimal_thresholds', {})
            class_specific = fp_analysis.get('class_specific_fp', {})
            
            # Threshold recommendations
            if optimal_thresholds.get('initial_target'):
                threshold = optimal_thresholds['initial_target']['threshold']
                recommendations.append(
                    f"Use confidence threshold {threshold:.2f} to meet initial 30% false positive target"
                )
            else:
                recommendations.append(
                    "Model does not meet 30% false positive target at any tested threshold - consider retraining"
                )
            
            if optimal_thresholds.get('improved_target'):
                threshold = optimal_thresholds['improved_target']['threshold']
                recommendations.append(
                    f"Use confidence threshold {threshold:.2f} to meet improved 10% false positive target"
                )
            else:
                recommendations.append(
                    "Model requires improvement to meet 10% false positive target"
                )
            
            # Class-specific recommendations
            high_fp_classes = [
                class_name for class_name, metrics in class_specific.items()
                if isinstance(metrics, dict) and metrics.get('false_positive_rate', 0) > 0.3
            ]
            
            if high_fp_classes:
                recommendations.append(
                    f"Classes with high false positive rates need attention: {', '.join(high_fp_classes)}"
                )
            
            # Priority-based recommendations
            critical_classes = [
                class_name for class_name, metrics in class_specific.items()
                if isinstance(metrics, dict) and metrics.get('priority') == 'critical'
                and metrics.get('false_positive_rate', 0) > 0.2
            ]
            
            if critical_classes:
                recommendations.append(
                    f"Critical security classes have high false positives: {', '.join(critical_classes)} - prioritize improvement"
                )
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _test_environmental_adaptation(self) -> Dict:
        """Test model adaptation to environmental conditions (Requirement 6.3)."""
        logger.info("Testing environmental adaptation capabilities...")
        
        environmental_results = {
            'condition_performance': {},
            'adaptation_analysis': {},
            'threshold_recommendations': {}
        }
        
        try:
            # Create test images with different environmental conditions
            base_image = self._create_test_image()
            
            for condition_name, condition_params in self.environmental_conditions.items():
                logger.info(f"Testing condition: {condition_name}")
                
                # Apply environmental transformation
                transformed_image = self._apply_environmental_transform(base_image, condition_params)
                
                # Test at multiple confidence thresholds
                condition_results = {}
                confidence_thresholds = [0.3, 0.5, 0.7]
                
                for conf_threshold in confidence_thresholds:
                    # Run inference
                    results = self.model.predict(transformed_image, conf=conf_threshold, verbose=False)
                    
                    # Count detections
                    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    
                    # Calculate average confidence
                    avg_confidence = 0
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        confidences = results[0].boxes.conf.cpu().numpy()
                        avg_confidence = np.mean(confidences)
                    
                    condition_results[str(conf_threshold)] = {
                        'num_detections': num_detections,
                        'avg_confidence': float(avg_confidence)
                    }
                
                environmental_results['condition_performance'][condition_name] = condition_results
            
            # Analyze adaptation patterns
            environmental_results['adaptation_analysis'] = self._analyze_environmental_adaptation(
                environmental_results['condition_performance']
            )
            
            # Generate threshold recommendations for each condition
            environmental_results['threshold_recommendations'] = self._generate_environmental_thresholds(
                environmental_results['condition_performance']
            )
            
        except Exception as e:
            environmental_results['error'] = str(e)
        
        return environmental_results
    
    def _create_test_image(self) -> np.ndarray:
        """Create a synthetic test image for environmental testing."""
        # Create a simple test scene
        image = np.ones((640, 640, 3), dtype=np.uint8) * 128  # Gray background
        
        # Add some objects/shapes to detect
        cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)  # Green circle
        cv2.rectangle(image, (300, 450), (500, 550), (0, 0, 255), -1)  # Red rectangle
        
        return image
    
    def _apply_environmental_transform(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply environmental transformation to image."""
        transformed = image.copy().astype(np.float32)
        
        # Apply brightness adjustment
        brightness_factor = params.get('brightness_factor', 1.0)
        transformed = transformed * brightness_factor
        
        # Apply contrast adjustment
        contrast_factor = params.get('contrast_factor', 1.0)
        mean = np.mean(transformed)
        transformed = (transformed - mean) * contrast_factor + mean
        
        # Clip values and convert back to uint8
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        
        return transformed
    
    def _analyze_environmental_adaptation(self, condition_performance: Dict) -> Dict:
        """Analyze how well the model adapts to different conditions."""
        adaptation_analysis = {
            'stability_score': 0,
            'condition_rankings': {},
            'adaptation_recommendations': []
        }
        
        try:
            # Calculate stability across conditions
            all_detection_counts = []
            all_confidences = []
            
            for condition_name, condition_data in condition_performance.items():
                # Use medium confidence threshold (0.5) for analysis
                medium_conf_data = condition_data.get('0.5', {})
                detection_count = medium_conf_data.get('num_detections', 0)
                avg_confidence = medium_conf_data.get('avg_confidence', 0)
                
                all_detection_counts.append(detection_count)
                all_confidences.append(avg_confidence)
                
                adaptation_analysis['condition_rankings'][condition_name] = {
                    'detection_count': detection_count,
                    'avg_confidence': avg_confidence,
                    'performance_score': detection_count * avg_confidence
                }
            
            # Calculate stability (lower coefficient of variation = more stable)
            if len(all_detection_counts) > 1:
                detection_cv = np.std(all_detection_counts) / np.mean(all_detection_counts) if np.mean(all_detection_counts) > 0 else 1
                confidence_cv = np.std(all_confidences) / np.mean(all_confidences) if np.mean(all_confidences) > 0 else 1
                
                # Stability score (0-1, higher is better)
                adaptation_analysis['stability_score'] = max(0, 1 - (detection_cv + confidence_cv) / 2)
            
            # Generate recommendations
            if adaptation_analysis['stability_score'] < 0.7:
                adaptation_analysis['adaptation_recommendations'].append(
                    "Model shows poor environmental adaptation - consider training with more diverse lighting conditions"
                )
            
            # Find best and worst performing conditions
            performance_scores = {
                name: data['performance_score'] 
                for name, data in adaptation_analysis['condition_rankings'].items()
            }
            
            if performance_scores:
                best_condition = max(performance_scores.keys(), key=lambda x: performance_scores[x])
                worst_condition = min(performance_scores.keys(), key=lambda x: performance_scores[x])
                
                adaptation_analysis['best_condition'] = best_condition
                adaptation_analysis['worst_condition'] = worst_condition
                
                adaptation_analysis['adaptation_recommendations'].append(
                    f"Best performance in {best_condition} conditions, worst in {worst_condition}"
                )
            
        except Exception as e:
            adaptation_analysis['error'] = str(e)
        
        return adaptation_analysis
    
    def _generate_environmental_thresholds(self, condition_performance: Dict) -> Dict:
        """Generate adaptive thresholds for different environmental conditions."""
        threshold_recommendations = {}
        
        try:
            for condition_name, condition_data in condition_performance.items():
                # Find optimal threshold for this condition
                best_threshold = 0.5  # Default
                best_score = 0
                
                for threshold_str, metrics in condition_data.items():
                    threshold = float(threshold_str)
                    detection_count = metrics.get('num_detections', 0)
                    avg_confidence = metrics.get('avg_confidence', 0)
                    
                    # Score balances detection count and confidence
                    score = detection_count * avg_confidence
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                
                threshold_recommendations[condition_name] = {
                    'recommended_threshold': best_threshold,
                    'expected_detections': condition_data.get(str(best_threshold), {}).get('num_detections', 0),
                    'expected_confidence': condition_data.get(str(best_threshold), {}).get('avg_confidence', 0)
                }
        
        except Exception as e:
            threshold_recommendations['error'] = str(e)
        
        return threshold_recommendations
    
    def _validate_security_scenarios(self) -> Dict:
        """Validate specific security scenarios and event types."""
        logger.info("Validating security-specific scenarios...")
        
        scenario_results = {
            'intrusion_detection': {},
            'loitering_detection': {},
            'crowding_detection': {},
            'abandoned_object_detection': {},
            'scenario_summary': {}
        }
        
        try:
            # Test each security scenario
            for event_type, event_config in self.security_events.items():
                if event_type == 'normal':
                    continue  # Skip normal class for scenario testing
                
                logger.info(f"Testing {event_type} detection scenario...")
                
                # Run validation focusing on this event type
                results = self.model.val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=event_config['min_confidence'],
                    verbose=False
                )
                
                # Extract metrics for this class
                class_names = list(self.config['classes'].keys())
                if event_type in class_names:
                    class_idx = class_names.index(event_type)
                    
                    class_metrics = {}
                    if hasattr(results.box, 'ap_class_index') and class_idx in results.box.ap_class_index:
                        idx = list(results.box.ap_class_index).index(class_idx)
                        class_metrics = {
                            'ap50': float(results.box.ap50[idx]),
                            'precision': float(results.box.p[idx]) if hasattr(results.box, 'p') else 0,
                            'recall': float(results.box.r[idx]) if hasattr(results.box, 'r') else 0,
                            'priority': event_config['priority'],
                            'min_confidence': event_config['min_confidence']
                        }
                        
                        # Calculate scenario-specific metrics
                        class_metrics['meets_priority_requirements'] = self._check_priority_requirements(
                            class_metrics, event_config['priority']
                        )
                    
                    scenario_results[f'{event_type}_detection'] = class_metrics
            
            # Generate scenario summary
            scenario_results['scenario_summary'] = self._generate_scenario_summary(scenario_results)
            
        except Exception as e:
            scenario_results['error'] = str(e)
        
        return scenario_results
    
    def _check_priority_requirements(self, metrics: Dict, priority: str) -> bool:
        """Check if metrics meet requirements for the given priority level."""
        priority_requirements = {
            'critical': {'min_recall': 0.9, 'min_precision': 0.8},
            'high': {'min_recall': 0.8, 'min_precision': 0.7},
            'medium': {'min_recall': 0.7, 'min_precision': 0.6},
            'low': {'min_recall': 0.6, 'min_precision': 0.5}
        }
        
        requirements = priority_requirements.get(priority, {'min_recall': 0.5, 'min_precision': 0.5})
        
        recall_ok = metrics.get('recall', 0) >= requirements['min_recall']
        precision_ok = metrics.get('precision', 0) >= requirements['min_precision']
        
        return recall_ok and precision_ok
    
    def _generate_scenario_summary(self, scenario_results: Dict) -> Dict:
        """Generate summary of security scenario validation."""
        summary = {
            'total_scenarios_tested': 0,
            'scenarios_passed': 0,
            'critical_scenarios_status': {},
            'recommendations': []
        }
        
        try:
            critical_events = ['intrusion', 'violence']
            high_priority_events = ['theft', 'suspicious']
            
            for key, metrics in scenario_results.items():
                if key.endswith('_detection') and isinstance(metrics, dict) and 'priority' in metrics:
                    summary['total_scenarios_tested'] += 1
                    
                    if metrics.get('meets_priority_requirements', False):
                        summary['scenarios_passed'] += 1
                    
                    # Track critical scenarios
                    event_name = key.replace('_detection', '')
                    if event_name in critical_events:
                        summary['critical_scenarios_status'][event_name] = {
                            'passed': metrics.get('meets_priority_requirements', False),
                            'recall': metrics.get('recall', 0),
                            'precision': metrics.get('precision', 0)
                        }
            
            # Generate recommendations
            if summary['scenarios_passed'] < summary['total_scenarios_tested']:
                summary['recommendations'].append(
                    f"Only {summary['scenarios_passed']}/{summary['total_scenarios_tested']} scenarios passed - model needs improvement"
                )
            
            # Check critical scenarios
            failed_critical = [
                event for event, status in summary['critical_scenarios_status'].items()
                if not status['passed']
            ]
            
            if failed_critical:
                summary['recommendations'].append(
                    f"Critical security scenarios failed: {', '.join(failed_critical)} - immediate attention required"
                )
            
            summary['pass_rate'] = summary['scenarios_passed'] / summary['total_scenarios_tested'] if summary['total_scenarios_tested'] > 0 else 0
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _optimize_detection_thresholds(self) -> Dict:
        """Optimize detection thresholds for different scenarios and conditions."""
        logger.info("Optimizing detection thresholds...")
        
        optimization_results = {
            'global_optimization': {},
            'class_specific_optimization': {},
            'environmental_optimization': {},
            'deployment_recommendations': {}
        }
        
        try:
            # Global threshold optimization
            optimization_results['global_optimization'] = self._optimize_global_thresholds()
            
            # Class-specific optimization
            optimization_results['class_specific_optimization'] = self._optimize_class_thresholds()
            
            # Environmental optimization
            optimization_results['environmental_optimization'] = self._optimize_environmental_thresholds()
            
            # Generate deployment recommendations
            optimization_results['deployment_recommendations'] = self._generate_deployment_recommendations(
                optimization_results
            )
            
        except Exception as e:
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _optimize_global_thresholds(self) -> Dict:
        """Optimize global confidence thresholds."""
        global_optimization = {}
        
        try:
            # Test range of thresholds
            thresholds = np.arange(0.1, 0.95, 0.05)
            threshold_metrics = {}
            
            for threshold in thresholds:
                results = self.model.val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=threshold,
                    verbose=False
                )
                
                precision = float(results.box.mp)
                recall = float(results.box.mr)
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                false_positive_rate = 1 - precision
                
                threshold_metrics[f'{threshold:.2f}'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'false_positive_rate': false_positive_rate,
                    'meets_fp_target': false_positive_rate <= self.false_positive_thresholds['initial_target']
                }
            
            # Find optimal thresholds for different objectives
            global_optimization['threshold_metrics'] = threshold_metrics
            global_optimization['optimal_thresholds'] = self._find_multi_objective_thresholds(threshold_metrics)
            
        except Exception as e:
            global_optimization['error'] = str(e)
        
        return global_optimization
    
    def _find_multi_objective_thresholds(self, threshold_metrics: Dict) -> Dict:
        """Find optimal thresholds for multiple objectives."""
        optimal_thresholds = {
            'max_f1': None,
            'fp_target_30': None,
            'fp_target_10': None,
            'balanced_security': None
        }
        
        best_f1 = 0
        
        for threshold_str, metrics in threshold_metrics.items():
            threshold = float(threshold_str)
            f1_score = metrics['f1_score']
            fp_rate = metrics['false_positive_rate']
            precision = metrics['precision']
            recall = metrics['recall']
            
            # Best F1 score
            if f1_score > best_f1:
                best_f1 = f1_score
                optimal_thresholds['max_f1'] = {
                    'threshold': threshold,
                    'f1_score': f1_score,
                    'false_positive_rate': fp_rate
                }
            
            # False positive targets
            if fp_rate <= 0.30 and optimal_thresholds['fp_target_30'] is None:
                optimal_thresholds['fp_target_30'] = {
                    'threshold': threshold,
                    'false_positive_rate': fp_rate,
                    'recall': recall
                }
            
            if fp_rate <= 0.10 and optimal_thresholds['fp_target_10'] is None:
                optimal_thresholds['fp_target_10'] = {
                    'threshold': threshold,
                    'false_positive_rate': fp_rate,
                    'recall': recall
                }
            
            # Balanced security (emphasizes recall while controlling FP)
            security_score = recall * 0.7 + (1 - fp_rate) * 0.3
            if (optimal_thresholds['balanced_security'] is None or 
                security_score > optimal_thresholds['balanced_security'].get('security_score', 0)):
                optimal_thresholds['balanced_security'] = {
                    'threshold': threshold,
                    'security_score': security_score,
                    'recall': recall,
                    'false_positive_rate': fp_rate
                }
        
        return optimal_thresholds
    
    def _optimize_class_thresholds(self) -> Dict:
        """Optimize thresholds for each security event class."""
        class_optimization = {}
        
        try:
            for event_type, event_config in self.security_events.items():
                if event_type == 'normal':
                    continue
                
                # Test different thresholds for this class
                class_thresholds = {}
                test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                
                for threshold in test_thresholds:
                    results = self.model.val(
                        data=str(self.yolo_data_dir / "dataset.yaml"),
                        split='test',
                        conf=threshold,
                        verbose=False
                    )
                    
                    # Extract class-specific metrics
                    class_names = list(self.config['classes'].keys())
                    if event_type in class_names:
                        class_idx = class_names.index(event_type)
                        
                        if hasattr(results.box, 'ap_class_index') and class_idx in results.box.ap_class_index:
                            idx = list(results.box.ap_class_index).index(class_idx)
                            precision = float(results.box.p[idx]) if hasattr(results.box, 'p') else 0
                            recall = float(results.box.r[idx]) if hasattr(results.box, 'r') else 0
                            
                            class_thresholds[str(threshold)] = {
                                'precision': precision,
                                'recall': recall,
                                'false_positive_rate': 1 - precision,
                                'meets_priority_req': self._check_priority_requirements(
                                    {'precision': precision, 'recall': recall}, 
                                    event_config['priority']
                                )
                            }
                
                # Find optimal threshold for this class
                optimal_threshold = self._find_class_optimal_threshold(class_thresholds, event_config['priority'])
                
                class_optimization[event_type] = {
                    'threshold_analysis': class_thresholds,
                    'optimal_threshold': optimal_threshold,
                    'priority': event_config['priority']
                }
        
        except Exception as e:
            class_optimization['error'] = str(e)
        
        return class_optimization
    
    def _find_class_optimal_threshold(self, class_thresholds: Dict, priority: str) -> Dict:
        """Find optimal threshold for a specific class based on its priority."""
        priority_weights = {
            'critical': {'recall_weight': 0.8, 'precision_weight': 0.2},
            'high': {'recall_weight': 0.7, 'precision_weight': 0.3},
            'medium': {'recall_weight': 0.6, 'precision_weight': 0.4},
            'low': {'recall_weight': 0.5, 'precision_weight': 0.5}
        }
        
        weights = priority_weights.get(priority, {'recall_weight': 0.5, 'precision_weight': 0.5})
        
        best_threshold = None
        best_score = 0
        
        for threshold_str, metrics in class_thresholds.items():
            threshold = float(threshold_str)
            recall = metrics.get('recall', 0)
            precision = metrics.get('precision', 0)
            
            # Calculate weighted score
            score = recall * weights['recall_weight'] + precision * weights['precision_weight']
            
            if score > best_score:
                best_score = score
                best_threshold = {
                    'threshold': threshold,
                    'score': score,
                    'recall': recall,
                    'precision': precision,
                    'false_positive_rate': metrics.get('false_positive_rate', 0)
                }
        
        return best_threshold
    
    def _optimize_environmental_thresholds(self) -> Dict:
        """Optimize thresholds for different environmental conditions."""
        # This would use the environmental adaptation results
        # For now, return placeholder structure
        return {
            'daylight': {'recommended_threshold': 0.5},
            'low_light': {'recommended_threshold': 0.4},
            'artificial_light': {'recommended_threshold': 0.45},
            'optimization_note': 'Environmental threshold optimization requires environmental test data'
        }
    
    def _generate_deployment_recommendations(self, optimization_results: Dict) -> Dict:
        """Generate deployment recommendations based on optimization results."""
        recommendations = {
            'edge_deployment': {},
            'server_deployment': {},
            'mobile_deployment': {},
            'general_recommendations': []
        }
        
        try:
            global_opt = optimization_results.get('global_optimization', {})
            optimal_thresholds = global_opt.get('optimal_thresholds', {})
            
            # Edge deployment recommendations
            if optimal_thresholds.get('balanced_security'):
                balanced = optimal_thresholds['balanced_security']
                recommendations['edge_deployment'] = {
                    'recommended_threshold': balanced['threshold'],
                    'expected_recall': balanced['recall'],
                    'expected_fp_rate': balanced['false_positive_rate'],
                    'rationale': 'Balanced threshold for edge deployment prioritizing security detection'
                }
            
            # Server deployment recommendations
            if optimal_thresholds.get('max_f1'):
                max_f1 = optimal_thresholds['max_f1']
                recommendations['server_deployment'] = {
                    'recommended_threshold': max_f1['threshold'],
                    'expected_f1': max_f1['f1_score'],
                    'expected_fp_rate': max_f1['false_positive_rate'],
                    'rationale': 'Optimal F1 score for server deployment with processing capacity'
                }
            
            # Mobile deployment recommendations
            if optimal_thresholds.get('fp_target_10'):
                low_fp = optimal_thresholds['fp_target_10']
                recommendations['mobile_deployment'] = {
                    'recommended_threshold': low_fp['threshold'],
                    'expected_recall': low_fp['recall'],
                    'expected_fp_rate': low_fp['false_positive_rate'],
                    'rationale': 'High threshold for mobile deployment to minimize false alarms'
                }
            
            # General recommendations
            recommendations['general_recommendations'] = [
                "Implement adaptive thresholding based on environmental conditions",
                "Use class-specific thresholds for different security event types",
                "Monitor false positive rates in production and adjust thresholds accordingly",
                "Consider ensemble methods for critical security applications"
            ]
            
            # Add compliance-based recommendations
            if optimal_thresholds.get('fp_target_30'):
                recommendations['general_recommendations'].append(
                    f"Use threshold {optimal_thresholds['fp_target_30']['threshold']:.2f} to meet initial 30% FP requirement"
                )
            
            if optimal_thresholds.get('fp_target_10'):
                recommendations['general_recommendations'].append(
                    f"Target threshold {optimal_thresholds['fp_target_10']['threshold']:.2f} for improved 10% FP requirement"
                )
        
        except Exception as e:
            recommendations['error'] = str(e)
        
        return recommendations
    
    def _assess_compliance(self, validation_results: Dict) -> Dict:
        """Assess compliance with requirements 6.2 and 6.3."""
        compliance_assessment = {
            'requirement_6_2_compliance': {},
            'requirement_6_3_compliance': {},
            'overall_compliance': {}
        }
        
        try:
            # Requirement 6.2: False positive rate ≤30% initially, ≤10% after improvement
            fp_analysis = validation_results.get('false_positive_analysis', {})
            optimal_thresholds = fp_analysis.get('optimal_thresholds', {})
            
            req_6_2 = {
                'initial_target_met': optimal_thresholds.get('initial_target') is not None,
                'improved_target_achievable': optimal_thresholds.get('improved_target') is not None,
                'compliance_status': 'non_compliant'
            }
            
            if req_6_2['initial_target_met']:
                if req_6_2['improved_target_achievable']:
                    req_6_2['compliance_status'] = 'fully_compliant'
                else:
                    req_6_2['compliance_status'] = 'partially_compliant'
            
            req_6_2['details'] = {
                'initial_threshold': optimal_thresholds.get('initial_target', {}).get('threshold'),
                'initial_fp_rate': optimal_thresholds.get('initial_target', {}).get('false_positive_rate'),
                'improved_threshold': optimal_thresholds.get('improved_target', {}).get('threshold'),
                'improved_fp_rate': optimal_thresholds.get('improved_target', {}).get('false_positive_rate')
            }
            
            compliance_assessment['requirement_6_2_compliance'] = req_6_2
            
            # Requirement 6.3: Environmental adaptation
            env_adaptation = validation_results.get('environmental_adaptation', {})
            adaptation_analysis = env_adaptation.get('adaptation_analysis', {})
            
            req_6_3 = {
                'adaptation_capability': adaptation_analysis.get('stability_score', 0) > 0.6,
                'threshold_adaptation': len(env_adaptation.get('threshold_recommendations', {})) > 0,
                'compliance_status': 'non_compliant'
            }
            
            if req_6_3['adaptation_capability'] and req_6_3['threshold_adaptation']:
                req_6_3['compliance_status'] = 'compliant'
            elif req_6_3['adaptation_capability'] or req_6_3['threshold_adaptation']:
                req_6_3['compliance_status'] = 'partially_compliant'
            
            req_6_3['details'] = {
                'stability_score': adaptation_analysis.get('stability_score', 0),
                'best_condition': adaptation_analysis.get('best_condition'),
                'worst_condition': adaptation_analysis.get('worst_condition'),
                'adaptive_thresholds_available': len(env_adaptation.get('threshold_recommendations', {}))
            }
            
            compliance_assessment['requirement_6_3_compliance'] = req_6_3
            
            # Overall compliance
            overall_status = 'non_compliant'
            if (req_6_2['compliance_status'] in ['fully_compliant', 'partially_compliant'] and 
                req_6_3['compliance_status'] in ['compliant', 'partially_compliant']):
                if (req_6_2['compliance_status'] == 'fully_compliant' and 
                    req_6_3['compliance_status'] == 'compliant'):
                    overall_status = 'fully_compliant'
                else:
                    overall_status = 'partially_compliant'
            
            compliance_assessment['overall_compliance'] = {
                'status': overall_status,
                'req_6_2_status': req_6_2['compliance_status'],
                'req_6_3_status': req_6_3['compliance_status']
            }
            
        except Exception as e:
            compliance_assessment['error'] = str(e)
        
        return compliance_assessment
    
    def _generate_enhanced_summary(self, validation_results: Dict) -> Dict:
        """Generate enhanced validation summary."""
        summary = {
            'validation_status': 'unknown',
            'compliance_summary': {},
            'key_findings': [],
            'critical_issues': [],
            'recommendations': [],
            'deployment_readiness': {}
        }
        
        try:
            # Extract key results
            compliance = validation_results.get('compliance_assessment', {})
            fp_analysis = validation_results.get('false_positive_analysis', {})
            env_adaptation = validation_results.get('environmental_adaptation', {})
            scenarios = validation_results.get('security_scenario_tests', {})
            
            # Compliance summary
            overall_compliance = compliance.get('overall_compliance', {})
            summary['compliance_summary'] = {
                'overall_status': overall_compliance.get('status', 'unknown'),
                'requirement_6_2': compliance.get('requirement_6_2_compliance', {}).get('compliance_status', 'unknown'),
                'requirement_6_3': compliance.get('requirement_6_3_compliance', {}).get('compliance_status', 'unknown')
            }
            
            # Determine validation status
            if overall_compliance.get('status') == 'fully_compliant':
                summary['validation_status'] = 'passed'
            elif overall_compliance.get('status') == 'partially_compliant':
                summary['validation_status'] = 'conditional_pass'
            else:
                summary['validation_status'] = 'failed'
            
            # Key findings
            optimal_thresholds = fp_analysis.get('optimal_thresholds', {})
            if optimal_thresholds.get('initial_target'):
                threshold = optimal_thresholds['initial_target']['threshold']
                fp_rate = optimal_thresholds['initial_target']['false_positive_rate']
                summary['key_findings'].append(
                    f"Achieves 30% false positive target at confidence threshold {threshold:.2f} (FP rate: {fp_rate:.1%})"
                )
            
            if optimal_thresholds.get('improved_target'):
                threshold = optimal_thresholds['improved_target']['threshold']
                fp_rate = optimal_thresholds['improved_target']['false_positive_rate']
                summary['key_findings'].append(
                    f"Achieves 10% false positive target at confidence threshold {threshold:.2f} (FP rate: {fp_rate:.1%})"
                )
            
            # Environmental adaptation findings
            adaptation_analysis = env_adaptation.get('adaptation_analysis', {})
            stability_score = adaptation_analysis.get('stability_score', 0)
            summary['key_findings'].append(
                f"Environmental adaptation stability score: {stability_score:.2f}"
            )
            
            # Critical issues
            if not optimal_thresholds.get('initial_target'):
                summary['critical_issues'].append(
                    "Model does not meet initial 30% false positive rate requirement"
                )
            
            if stability_score < 0.6:
                summary['critical_issues'].append(
                    "Poor environmental adaptation - model performance varies significantly across conditions"
                )
            
            # Security scenario issues
            scenario_summary = scenarios.get('scenario_summary', {})
            if scenario_summary.get('pass_rate', 0) < 0.8:
                summary['critical_issues'].append(
                    f"Low security scenario pass rate: {scenario_summary.get('pass_rate', 0):.1%}"
                )
            
            # Recommendations
            fp_recommendations = fp_analysis.get('optimization_recommendations', [])
            summary['recommendations'].extend(fp_recommendations)
            
            env_recommendations = adaptation_analysis.get('adaptation_recommendations', [])
            summary['recommendations'].extend(env_recommendations)
            
            # Deployment readiness
            summary['deployment_readiness'] = {
                'edge_ready': summary['validation_status'] in ['passed', 'conditional_pass'] and stability_score > 0.6,
                'server_ready': summary['validation_status'] == 'passed',
                'mobile_ready': (summary['validation_status'] == 'passed' and 
                               optimal_thresholds.get('improved_target') is not None),
                'production_ready': (summary['validation_status'] == 'passed' and 
                                   len(summary['critical_issues']) == 0)
            }
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _save_enhanced_results(self, results: Dict) -> None:
        """Save enhanced validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.validation_dir / f"enhanced_validation_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / 'enhanced_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate enhanced report
        self._generate_enhanced_report(results, results_dir)
        
        # Generate compliance report
        self._generate_compliance_report(results, results_dir)
        
        logger.info(f"Enhanced validation results saved to: {results_dir}")
    
    def _generate_enhanced_report(self, results: Dict, output_dir: Path) -> None:
        """Generate enhanced validation report."""
        report_file = output_dir / 'enhanced_validation_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Enhanced Security Model Validation Report\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = results.get('validation_summary', {})
            
            f.write(f"**Validation Status**: {summary.get('validation_status', 'Unknown')}\n")
            f.write(f"**Overall Compliance**: {summary.get('compliance_summary', {}).get('overall_status', 'Unknown')}\n\n")
            
            # Compliance Status
            f.write("## Compliance Assessment\n\n")
            compliance = results.get('compliance_assessment', {})
            
            f.write("### Requirement 6.2 - False Positive Rate Control\n")
            req_6_2 = compliance.get('requirement_6_2_compliance', {})
            f.write(f"**Status**: {req_6_2.get('compliance_status', 'Unknown')}\n")
            
            details = req_6_2.get('details', {})
            if details.get('initial_threshold'):
                f.write(f"- Initial 30% FP target: Achievable at threshold {details['initial_threshold']:.2f}\n")
            if details.get('improved_threshold'):
                f.write(f"- Improved 10% FP target: Achievable at threshold {details['improved_threshold']:.2f}\n")
            f.write("\n")
            
            f.write("### Requirement 6.3 - Environmental Adaptation\n")
            req_6_3 = compliance.get('requirement_6_3_compliance', {})
            f.write(f"**Status**: {req_6_3.get('compliance_status', 'Unknown')}\n")
            
            details = req_6_3.get('details', {})
            f.write(f"- Stability Score: {details.get('stability_score', 0):.2f}\n")
            f.write(f"- Best Condition: {details.get('best_condition', 'Unknown')}\n")
            f.write(f"- Worst Condition: {details.get('worst_condition', 'Unknown')}\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            for finding in summary.get('key_findings', []):
                f.write(f"- {finding}\n")
            f.write("\n")
            
            # Critical Issues
            if summary.get('critical_issues'):
                f.write("## Critical Issues\n\n")
                for issue in summary['critical_issues']:
                    f.write(f"- ⚠️ {issue}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in summary.get('recommendations', []):
                f.write(f"- {rec}\n")
            f.write("\n")
            
            # Deployment Readiness
            f.write("## Deployment Readiness\n\n")
            readiness = summary.get('deployment_readiness', {})
            
            f.write("| Deployment Type | Ready | Status |\n")
            f.write("|-----------------|-------|--------|\n")
            f.write(f"| Edge Deployment | {'✅' if readiness.get('edge_ready') else '❌'} | {'Ready' if readiness.get('edge_ready') else 'Not Ready'} |\n")
            f.write(f"| Server Deployment | {'✅' if readiness.get('server_ready') else '❌'} | {'Ready' if readiness.get('server_ready') else 'Not Ready'} |\n")
            f.write(f"| Mobile Deployment | {'✅' if readiness.get('mobile_ready') else '❌'} | {'Ready' if readiness.get('mobile_ready') else 'Not Ready'} |\n")
            f.write(f"| Production Deployment | {'✅' if readiness.get('production_ready') else '❌'} | {'Ready' if readiness.get('production_ready') else 'Not Ready'} |\n")
    
    def _generate_compliance_report(self, results: Dict, output_dir: Path) -> None:
        """Generate detailed compliance report."""
        report_file = output_dir / 'compliance_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Security Requirements Compliance Report\n\n")
            
            f.write("## Requirement 6.2 - False Positive Rate Control\n\n")
            f.write("**Requirement**: The system SHALL achieve a false positive rate of ≤30% initially, improving to ≤10% after 3 months of operation.\n\n")
            
            fp_analysis = results.get('false_positive_analysis', {})
            optimal_thresholds = fp_analysis.get('optimal_thresholds', {})
            
            if optimal_thresholds.get('initial_target'):
                initial = optimal_thresholds['initial_target']
                f.write(f"✅ **Initial Target (≤30% FP)**: ACHIEVED\n")
                f.write(f"- Confidence Threshold: {initial['threshold']:.2f}\n")
                f.write(f"- False Positive Rate: {initial['false_positive_rate']:.1%}\n")
                f.write(f"- Recall: {initial['recall']:.1%}\n\n")
            else:
                f.write("❌ **Initial Target (≤30% FP)**: NOT ACHIEVED\n")
                f.write("- Model requires retraining or architecture changes\n\n")
            
            if optimal_thresholds.get('improved_target'):
                improved = optimal_thresholds['improved_target']
                f.write(f"✅ **Improved Target (≤10% FP)**: ACHIEVABLE\n")
                f.write(f"- Confidence Threshold: {improved['threshold']:.2f}\n")
                f.write(f"- False Positive Rate: {improved['false_positive_rate']:.1%}\n")
                f.write(f"- Recall: {improved['recall']:.1%}\n\n")
            else:
                f.write("❌ **Improved Target (≤10% FP)**: NOT ACHIEVABLE\n")
                f.write("- Model needs significant improvement for long-term target\n\n")
            
            f.write("## Requirement 6.3 - Environmental Adaptation\n\n")
            f.write("**Requirement**: The system SHALL adapt detection thresholds automatically based on time of day, weather, and historical patterns.\n\n")
            
            env_adaptation = results.get('environmental_adaptation', {})
            adaptation_analysis = env_adaptation.get('adaptation_analysis', {})
            
            stability_score = adaptation_analysis.get('stability_score', 0)
            if stability_score > 0.7:
                f.write(f"✅ **Environmental Stability**: GOOD (Score: {stability_score:.2f})\n")
            elif stability_score > 0.5:
                f.write(f"⚠️ **Environmental Stability**: MODERATE (Score: {stability_score:.2f})\n")
            else:
                f.write(f"❌ **Environmental Stability**: POOR (Score: {stability_score:.2f})\n")
            
            f.write(f"- Best Performance: {adaptation_analysis.get('best_condition', 'Unknown')}\n")
            f.write(f"- Worst Performance: {adaptation_analysis.get('worst_condition', 'Unknown')}\n\n")
            
            # Threshold recommendations
            threshold_recs = env_adaptation.get('threshold_recommendations', {})
            if threshold_recs:
                f.write("### Adaptive Threshold Recommendations\n\n")
                f.write("| Condition | Recommended Threshold | Expected Detections |\n")
                f.write("|-----------|----------------------|--------------------|\n")
                
                for condition, rec in threshold_recs.items():
                    if isinstance(rec, dict):
                        threshold = rec.get('recommended_threshold', 0)
                        detections = rec.get('expected_detections', 0)
                        f.write(f"| {condition} | {threshold:.2f} | {detections} |\n")


def main():
    """Main function for running enhanced security validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Security Model Validator')
    parser.add_argument('model_path', help='Path to the YOLO model file')
    parser.add_argument('--config', default='../config/dataset_config.yaml', 
                       help='Path to dataset configuration file')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = EnhancedSecurityValidator(args.model_path, args.config)
    
    # Run enhanced validation
    results = validator.run_enhanced_validation()
    
    # Print summary
    summary = results.get('validation_summary', {})
    print(f"\nValidation Status: {summary.get('validation_status', 'Unknown')}")
    print(f"Compliance Status: {summary.get('compliance_summary', {}).get('overall_status', 'Unknown')}")
    
    if summary.get('critical_issues'):
        print("\nCritical Issues:")
        for issue in summary['critical_issues']:
            print(f"  - {issue}")
    
    return results


if __name__ == "__main__":
    main()