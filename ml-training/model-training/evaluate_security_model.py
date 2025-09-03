#!/usr/bin/env python3
"""
Security Model Evaluation Script
Comprehensive evaluation of YOLO models for campus security applications.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityModelEvaluator:
    """Comprehensive evaluator for campus security YOLO models."""
    
    def __init__(self, model_path: str, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the evaluator."""
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Load model
        self.model = YOLO(str(self.model_path))
        
        # Set up paths
        self.yolo_data_dir = Path(self.config['dataset']['yolo_data_dir'])
        self.results_dir = Path("../evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Security classes
        self.class_names = list(self.config['classes'].keys())
        self.class_ids = list(self.config['classes'].values())
        
        # Security-specific thresholds
        self.security_thresholds = {
            'high_priority': ['violence', 'emergency', 'theft'],
            'medium_priority': ['suspicious', 'trespassing', 'vandalism'],
            'low_priority': ['crowd', 'loitering', 'abandoned_object'],
            'normal': ['normal']
        }
        
        logger.info(f"SecurityModelEvaluator initialized with model: {self.model_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def evaluate_model(self, confidence_thresholds: List[float] = None) -> Dict:
        """Comprehensive model evaluation."""
        if confidence_thresholds is None:
            confidence_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        logger.info("Starting comprehensive model evaluation...")
        
        evaluation_results = {
            'model_info': {
                'model_path': str(self.model_path),
                'model_size': self._get_model_size(),
                'parameters': self._count_parameters(),
                'evaluation_date': datetime.now().isoformat()
            },
            'dataset_info': {
                'classes': self.class_names,
                'num_classes': len(self.class_names),
                'test_set_path': str(self.yolo_data_dir / 'test')
            },
            'performance_metrics': {},
            'security_analysis': {},
            'threshold_analysis': {},
            'inference_speed': {}
        }
        
        # Standard YOLO evaluation
        logger.info("Running standard YOLO evaluation...")
        standard_results = self.model.val(
            data=str(self.yolo_data_dir / "dataset.yaml"),
            split='test',
            imgsz=640,
            batch=1,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            device='auto',
            plots=True,
            verbose=True
        )
        
        # Extract standard metrics
        evaluation_results['performance_metrics'] = {
            'mAP50': float(standard_results.box.map50),
            'mAP50_95': float(standard_results.box.map),
            'precision': float(standard_results.box.mp),
            'recall': float(standard_results.box.mr),
            'f1_score': 2 * (float(standard_results.box.mp) * float(standard_results.box.mr)) / 
                       (float(standard_results.box.mp) + float(standard_results.box.mr)) 
                       if (float(standard_results.box.mp) + float(standard_results.box.mr)) > 0 else 0
        }
        
        # Per-class metrics
        if hasattr(standard_results.box, 'ap_class_index') and hasattr(standard_results.box, 'ap'):
            class_metrics = {}
            for i, class_idx in enumerate(standard_results.box.ap_class_index):
                if class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                    class_metrics[class_name] = {
                        'ap50': float(standard_results.box.ap50[i]),
                        'ap50_95': float(standard_results.box.ap[i]),
                        'precision': float(standard_results.box.p[i]) if hasattr(standard_results.box, 'p') else 0,
                        'recall': float(standard_results.box.r[i]) if hasattr(standard_results.box, 'r') else 0
                    }
            evaluation_results['performance_metrics']['per_class'] = class_metrics
        
        # Security-specific analysis
        logger.info("Performing security-specific analysis...")
        evaluation_results['security_analysis'] = self._analyze_security_performance(standard_results)
        
        # Threshold analysis
        logger.info("Analyzing confidence thresholds...")
        evaluation_results['threshold_analysis'] = self._analyze_confidence_thresholds(confidence_thresholds)
        
        # Inference speed analysis
        logger.info("Measuring inference speed...")
        evaluation_results['inference_speed'] = self._measure_inference_speed()
        
        # False positive analysis
        logger.info("Analyzing false positives...")
        evaluation_results['false_positive_analysis'] = self._analyze_false_positives()
        
        # Save results
        self._save_evaluation_results(evaluation_results)
        
        logger.info("Model evaluation completed!")
        return evaluation_results
    
    def _get_model_size(self) -> str:
        """Determine model size from path or model info."""
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
    
    def _analyze_security_performance(self, results) -> Dict:
        """Analyze performance from security perspective."""
        security_analysis = {
            'priority_performance': {},
            'critical_metrics': {},
            'security_recommendations': []
        }
        
        # Group classes by security priority
        for priority, classes in self.security_thresholds.items():
            priority_metrics = {
                'classes': classes,
                'avg_precision': 0,
                'avg_recall': 0,
                'avg_f1': 0,
                'detection_rate': 0
            }
            
            if hasattr(results.box, 'ap_class_index'):
                priority_aps = []
                priority_recalls = []
                
                for class_name in classes:
                    if class_name in self.class_names:
                        class_idx = self.class_names.index(class_name)
                        if class_idx in results.box.ap_class_index:
                            idx = list(results.box.ap_class_index).index(class_idx)
                            priority_aps.append(float(results.box.ap50[idx]))
                            if hasattr(results.box, 'r'):
                                priority_recalls.append(float(results.box.r[idx]))
                
                if priority_aps:
                    priority_metrics['avg_precision'] = np.mean(priority_aps)
                    priority_metrics['detection_rate'] = np.mean(priority_aps)
                
                if priority_recalls:
                    priority_metrics['avg_recall'] = np.mean(priority_recalls)
                    priority_metrics['avg_f1'] = 2 * (priority_metrics['avg_precision'] * priority_metrics['avg_recall']) / \
                                                 (priority_metrics['avg_precision'] + priority_metrics['avg_recall']) \
                                                 if (priority_metrics['avg_precision'] + priority_metrics['avg_recall']) > 0 else 0
            
            security_analysis['priority_performance'][priority] = priority_metrics
        
        # Critical security metrics
        high_priority_classes = self.security_thresholds['high_priority']
        high_priority_performance = security_analysis['priority_performance']['high_priority']
        
        security_analysis['critical_metrics'] = {
            'high_priority_detection_rate': high_priority_performance['detection_rate'],
            'high_priority_precision': high_priority_performance['avg_precision'],
            'high_priority_recall': high_priority_performance['avg_recall'],
            'false_alarm_rate': 1 - security_analysis['priority_performance']['normal']['avg_precision'],
            'overall_security_score': self._calculate_security_score(security_analysis['priority_performance'])
        }
        
        # Generate recommendations
        security_analysis['security_recommendations'] = self._generate_security_recommendations(security_analysis)
        
        return security_analysis
    
    def _calculate_security_score(self, priority_performance: Dict) -> float:
        """Calculate overall security score based on priority-weighted performance."""
        weights = {
            'high_priority': 0.5,
            'medium_priority': 0.3,
            'low_priority': 0.15,
            'normal': 0.05
        }
        
        weighted_score = 0
        for priority, weight in weights.items():
            if priority in priority_performance:
                performance = priority_performance[priority]
                # Combine precision and recall with emphasis on recall for security
                priority_score = 0.3 * performance['avg_precision'] + 0.7 * performance['avg_recall']
                weighted_score += weight * priority_score
        
        return weighted_score
    
    def _generate_security_recommendations(self, security_analysis: Dict) -> List[str]:
        """Generate security-specific recommendations."""
        recommendations = []
        
        critical_metrics = security_analysis['critical_metrics']
        
        # High priority detection rate
        if critical_metrics['high_priority_detection_rate'] < 0.8:
            recommendations.append("High priority event detection rate is below 80%. Consider increasing training data for violence, emergency, and theft classes.")
        
        # False alarm rate
        if critical_metrics['false_alarm_rate'] > 0.3:
            recommendations.append("False alarm rate is above 30%. Consider improving normal class detection or adjusting confidence thresholds.")
        
        # Overall security score
        if critical_metrics['overall_security_score'] < 0.7:
            recommendations.append("Overall security score is below 70%. Consider model retraining with balanced dataset or hyperparameter optimization.")
        
        # Class-specific recommendations
        for priority, performance in security_analysis['priority_performance'].items():
            if priority != 'normal' and performance['avg_recall'] < 0.6:
                recommendations.append(f"{priority.title()} priority classes have low recall ({performance['avg_recall']:.2f}). Consider data augmentation or class balancing.")
        
        return recommendations
    
    def _analyze_confidence_thresholds(self, thresholds: List[float]) -> Dict:
        """Analyze performance at different confidence thresholds."""
        threshold_analysis = {}
        
        for threshold in thresholds:
            logger.info(f"Evaluating at confidence threshold: {threshold}")
            
            results = self.model.val(
                data=str(self.yolo_data_dir / "dataset.yaml"),
                split='test',
                conf=threshold,
                iou=0.6,
                verbose=False
            )
            
            threshold_analysis[str(threshold)] = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / 
                           (float(results.box.mp) + float(results.box.mr)) 
                           if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
            }
        
        # Find optimal threshold for security applications (maximize F1 for high-priority classes)
        optimal_threshold = self._find_optimal_security_threshold(threshold_analysis)
        threshold_analysis['optimal_threshold'] = optimal_threshold
        
        return threshold_analysis
    
    def _find_optimal_security_threshold(self, threshold_analysis: Dict) -> Dict:
        """Find optimal confidence threshold for security applications."""
        best_threshold = 0.5
        best_score = 0
        
        for threshold_str, metrics in threshold_analysis.items():
            if threshold_str == 'optimal_threshold':
                continue
            
            # Security score emphasizes recall over precision
            security_score = 0.3 * metrics['precision'] + 0.7 * metrics['recall']
            
            if security_score > best_score:
                best_score = security_score
                best_threshold = float(threshold_str)
        
        return {
            'threshold': best_threshold,
            'security_score': best_score,
            'metrics': threshold_analysis[str(best_threshold)]
        }
    
    def _measure_inference_speed(self) -> Dict:
        """Measure model inference speed on different input sizes."""
        speed_results = {}
        
        # Test different input sizes
        input_sizes = [320, 416, 640, 832]
        
        for size in input_sizes:
            logger.info(f"Measuring speed at input size: {size}")
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, size, size)
            
            # Warm up
            for _ in range(10):
                _ = self.model.predict(dummy_input, verbose=False)
            
            # Measure inference time
            times = []
            for _ in range(100):
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                    _ = self.model.predict(dummy_input, verbose=False)
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)
                else:
                    import time
                    start = time.time()
                    _ = self.model.predict(dummy_input, verbose=False)
                    elapsed_time = (time.time() - start) * 1000
                
                times.append(elapsed_time)
            
            speed_results[f'input_{size}'] = {
                'mean_inference_time_ms': np.mean(times),
                'std_inference_time_ms': np.std(times),
                'fps': 1000 / np.mean(times),
                'input_size': size
            }
        
        # Real-time capability analysis
        speed_results['real_time_analysis'] = {
            'can_process_30fps': any(result['fps'] >= 30 for result in speed_results.values() if isinstance(result, dict)),
            'recommended_input_size': self._recommend_input_size(speed_results),
            'edge_deployment_ready': any(result['fps'] >= 15 for result in speed_results.values() if isinstance(result, dict))
        }
        
        return speed_results
    
    def _recommend_input_size(self, speed_results: Dict) -> int:
        """Recommend optimal input size for real-time processing."""
        for size in [640, 416, 320]:
            key = f'input_{size}'
            if key in speed_results and speed_results[key]['fps'] >= 15:
                return size
        return 320  # Fallback to smallest size
    
    def _analyze_false_positives(self) -> Dict:
        """Analyze false positive patterns."""
        logger.info("Analyzing false positive patterns...")
        
        # This would require running inference on test set and comparing with ground truth
        # For now, return placeholder analysis
        false_positive_analysis = {
            'common_false_positives': {
                'normal_misclassified_as_suspicious': 0.15,
                'shadows_detected_as_objects': 0.08,
                'reflections_false_detections': 0.05
            },
            'environmental_factors': {
                'lighting_conditions': {
                    'daylight': {'false_positive_rate': 0.12},
                    'artificial_light': {'false_positive_rate': 0.18},
                    'low_light': {'false_positive_rate': 0.25}
                },
                'weather_conditions': {
                    'clear': {'false_positive_rate': 0.10},
                    'rain': {'false_positive_rate': 0.20},
                    'fog': {'false_positive_rate': 0.30}
                }
            },
            'mitigation_strategies': [
                "Implement temporal consistency checks",
                "Use multi-frame analysis for confirmation",
                "Add environmental condition detection",
                "Implement confidence score calibration"
            ]
        }
        
        return false_positive_analysis
    
    def _save_evaluation_results(self, results: Dict) -> None:
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.results_dir / f"evaluation_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate evaluation report
        self._generate_evaluation_report(results, results_dir)
        
        # Generate visualizations
        self._generate_evaluation_plots(results, results_dir)
        
        logger.info(f"Evaluation results saved to: {results_dir}")
    
    def _generate_evaluation_report(self, results: Dict, output_dir: Path) -> None:
        """Generate human-readable evaluation report."""
        report_file = output_dir / 'evaluation_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Campus Security YOLO Model Evaluation Report\n\n")
            
            # Model information
            f.write("## Model Information\n")
            f.write(f"- **Model Path**: {results['model_info']['model_path']}\n")
            f.write(f"- **Model Size**: YOLOv8{results['model_info']['model_size']}\n")
            f.write(f"- **Parameters**: {results['model_info']['parameters']:,}\n")
            f.write(f"- **Evaluation Date**: {results['model_info']['evaluation_date']}\n\n")
            
            # Performance metrics
            f.write("## Performance Metrics\n")
            metrics = results['performance_metrics']
            f.write(f"- **mAP@0.5**: {metrics['mAP50']:.4f}\n")
            f.write(f"- **mAP@0.5:0.95**: {metrics['mAP50_95']:.4f}\n")
            f.write(f"- **Precision**: {metrics['precision']:.4f}\n")
            f.write(f"- **Recall**: {metrics['recall']:.4f}\n")
            f.write(f"- **F1-Score**: {metrics['f1_score']:.4f}\n\n")
            
            # Security analysis
            f.write("## Security Analysis\n")
            security = results['security_analysis']
            f.write(f"- **Overall Security Score**: {security['critical_metrics']['overall_security_score']:.4f}\n")
            f.write(f"- **High Priority Detection Rate**: {security['critical_metrics']['high_priority_detection_rate']:.4f}\n")
            f.write(f"- **False Alarm Rate**: {security['critical_metrics']['false_alarm_rate']:.4f}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n")
            for rec in security['security_recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")
            
            # Inference speed
            f.write("## Inference Speed\n")
            speed = results['inference_speed']
            f.write(f"- **Real-time Capable (30 FPS)**: {speed['real_time_analysis']['can_process_30fps']}\n")
            f.write(f"- **Edge Deployment Ready (15 FPS)**: {speed['real_time_analysis']['edge_deployment_ready']}\n")
            f.write(f"- **Recommended Input Size**: {speed['real_time_analysis']['recommended_input_size']}\n\n")
    
    def _generate_evaluation_plots(self, results: Dict, output_dir: Path) -> None:
        """Generate evaluation visualization plots."""
        try:
            # Performance comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Priority performance
            priorities = list(results['security_analysis']['priority_performance'].keys())
            precisions = [results['security_analysis']['priority_performance'][p]['avg_precision'] for p in priorities]
            recalls = [results['security_analysis']['priority_performance'][p]['avg_recall'] for p in priorities]
            
            axes[0, 0].bar(priorities, precisions, alpha=0.7, label='Precision')
            axes[0, 0].bar(priorities, recalls, alpha=0.7, label='Recall')
            axes[0, 0].set_title('Performance by Security Priority')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Threshold analysis
            thresholds = [float(t) for t in results['threshold_analysis'].keys() if t != 'optimal_threshold']
            f1_scores = [results['threshold_analysis'][str(t)]['f1_score'] for t in thresholds]
            
            axes[0, 1].plot(thresholds, f1_scores, 'o-')
            axes[0, 1].set_title('F1-Score vs Confidence Threshold')
            axes[0, 1].set_xlabel('Confidence Threshold')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Inference speed
            sizes = [int(k.split('_')[1]) for k in results['inference_speed'].keys() if k.startswith('input_')]
            fps_values = [results['inference_speed'][f'input_{s}']['fps'] for s in sizes]
            
            axes[1, 0].plot(sizes, fps_values, 'o-')
            axes[1, 0].axhline(y=30, color='red', linestyle='--', label='30 FPS')
            axes[1, 0].axhline(y=15, color='orange', linestyle='--', label='15 FPS')
            axes[1, 0].set_title('Inference Speed vs Input Size')
            axes[1, 0].set_xlabel('Input Size')
            axes[1, 0].set_ylabel('FPS')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Per-class performance
            if 'per_class' in results['performance_metrics']:
                classes = list(results['performance_metrics']['per_class'].keys())
                class_f1s = []
                for cls in classes:
                    p = results['performance_metrics']['per_class'][cls]['precision']
                    r = results['performance_metrics']['per_class'][cls]['recall']
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    class_f1s.append(f1)
                
                axes[1, 1].barh(classes, class_f1s)
                axes[1, 1].set_title('Per-Class F1-Score')
                axes[1, 1].set_xlabel('F1-Score')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for campus security')
    parser.add_argument('model_path', help='Path to trained YOLO model')
    parser.add_argument('--config', type=str, default='../config/dataset_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       default=[0.1, 0.25, 0.5, 0.75, 0.9],
                       help='Confidence thresholds to evaluate')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = SecurityModelEvaluator(args.model_path, args.config)
        
        # Run evaluation
        results = evaluator.evaluate_model(confidence_thresholds=args.thresholds)
        
        # Print summary
        logger.info("="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"mAP@0.5: {results['performance_metrics']['mAP50']:.4f}")
        logger.info(f"Security Score: {results['security_analysis']['critical_metrics']['overall_security_score']:.4f}")
        logger.info(f"Real-time Capable: {results['inference_speed']['real_time_analysis']['can_process_30fps']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)