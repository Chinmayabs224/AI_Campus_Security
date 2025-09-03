#!/usr/bin/env python3
"""
Model Comparison Framework
Compare multiple YOLO models for campus security applications.
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparisonFramework:
    """Framework for comparing multiple YOLO models."""
    
    def __init__(self, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the comparison framework."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up paths
        self.yolo_data_dir = Path(self.config['dataset']['yolo_data_dir'])
        self.comparison_dir = Path("../comparison_results")
        self.comparison_dir.mkdir(exist_ok=True)
        
        # Models to compare
        self.models = {}
        self.comparison_results = {}
        
        # Comparison metrics
        self.comparison_metrics = [
            'mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score',
            'inference_time_ms', 'fps', 'model_size_mb', 'parameters'
        ]
        
        logger.info("ModelComparisonFramework initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def add_model(self, model_name: str, model_path: str, description: str = "") -> None:
        """Add a model to the comparison."""
        try:
            model = YOLO(model_path)
            self.models[model_name] = {
                'model': model,
                'path': model_path,
                'description': description,
                'size': self._get_model_size(model_path),
                'parameters': self._count_parameters(model)
            }
            logger.info(f"Added model '{model_name}' to comparison")
        except Exception as e:
            logger.error(f"Failed to add model '{model_name}': {e}")
    
    def _get_model_size(self, model_path: str) -> float:
        """Get model file size in MB."""
        try:
            return Path(model_path).stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def _count_parameters(self, model) -> int:
        """Count model parameters."""
        try:
            return sum(p.numel() for p in model.model.parameters())
        except:
            return 0
    
    def run_comparison(self) -> Dict:
        """Run comprehensive comparison of all models."""
        logger.info(f"Starting comparison of {len(self.models)} models...")
        
        comparison_results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models_compared': list(self.models.keys()),
            'accuracy_comparison': {},
            'speed_comparison': {},
            'efficiency_comparison': {},
            'security_comparison': {},
            'summary': {}
        }
        
        # Run accuracy comparison
        logger.info("Running accuracy comparison...")
        comparison_results['accuracy_comparison'] = self._compare_accuracy()
        
        # Run speed comparison
        logger.info("Running speed comparison...")
        comparison_results['speed_comparison'] = self._compare_speed()
        
        # Run efficiency comparison
        logger.info("Running efficiency comparison...")
        comparison_results['efficiency_comparison'] = self._compare_efficiency()
        
        # Run security-specific comparison
        logger.info("Running security-specific comparison...")
        comparison_results['security_comparison'] = self._compare_security_performance()
        
        # Generate summary and recommendations
        comparison_results['summary'] = self._generate_comparison_summary(comparison_results)
        
        # Save results
        self._save_comparison_results(comparison_results)
        
        logger.info("Model comparison completed!")
        return comparison_results
    
    def _compare_accuracy(self) -> Dict:
        """Compare accuracy metrics across models."""
        accuracy_results = {}
        
        for model_name, model_info in self.models.items():
            logger.info(f"Evaluating accuracy for {model_name}...")
            
            try:
                # Run validation
                results = model_info['model'].val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    verbose=False
                )
                
                accuracy_results[model_name] = {
                    'mAP50': float(results.box.map50),
                    'mAP50_95': float(results.box.map),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                    'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / 
                               (float(results.box.mp) + float(results.box.mr)) 
                               if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
                }
                
                # Per-class accuracy
                if hasattr(results.box, 'ap_class_index'):
                    class_metrics = {}
                    class_names = list(self.config['classes'].keys())
                    
                    for i, class_idx in enumerate(results.box.ap_class_index):
                        if class_idx < len(class_names):
                            class_name = class_names[class_idx]
                            class_metrics[class_name] = {
                                'ap50': float(results.box.ap50[i]),
                                'ap50_95': float(results.box.ap[i])
                            }
                    
                    accuracy_results[model_name]['per_class'] = class_metrics
                
            except Exception as e:
                logger.error(f"Accuracy evaluation failed for {model_name}: {e}")
                accuracy_results[model_name] = {'error': str(e)}
        
        return accuracy_results
    
    def _compare_speed(self) -> Dict:
        """Compare inference speed across models."""
        speed_results = {}
        
        # Test input sizes
        input_sizes = [320, 640, 832]
        
        for model_name, model_info in self.models.items():
            logger.info(f"Evaluating speed for {model_name}...")
            
            model_speed_results = {}
            
            for size in input_sizes:
                try:
                    # Create test input
                    test_input = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                    
                    # Warm up
                    for _ in range(5):
                        _ = model_info['model'].predict(test_input, verbose=False)
                    
                    # Measure inference time
                    times = []
                    for _ in range(20):
                        start_time = time.time()
                        _ = model_info['model'].predict(test_input, verbose=False)
                        end_time = time.time()
                        times.append((end_time - start_time) * 1000)
                    
                    model_speed_results[f'input_{size}'] = {
                        'mean_time_ms': np.mean(times),
                        'std_time_ms': np.std(times),
                        'fps': 1000 / np.mean(times)
                    }
                    
                except Exception as e:
                    model_speed_results[f'input_{size}'] = {'error': str(e)}
            
            # Overall speed metrics
            try:
                best_fps = max([result.get('fps', 0) for result in model_speed_results.values() 
                               if isinstance(result, dict) and 'fps' in result])
                model_speed_results['best_fps'] = best_fps
                model_speed_results['real_time_capable'] = best_fps >= 15
            except:
                model_speed_results['best_fps'] = 0
                model_speed_results['real_time_capable'] = False
            
            speed_results[model_name] = model_speed_results
        
        return speed_results
    
    def _compare_efficiency(self) -> Dict:
        """Compare efficiency (accuracy vs speed vs size) across models."""
        efficiency_results = {}
        
        for model_name, model_info in self.models.items():
            try:
                # Get accuracy from previous comparison
                accuracy_data = self.comparison_results.get('accuracy_comparison', {}).get(model_name, {})
                speed_data = self.comparison_results.get('speed_comparison', {}).get(model_name, {})
                
                mAP50 = accuracy_data.get('mAP50', 0)
                best_fps = speed_data.get('best_fps', 0)
                model_size_mb = model_info['size']
                parameters = model_info['parameters']
                
                # Calculate efficiency metrics
                efficiency_score = 0
                if best_fps > 0 and model_size_mb > 0:
                    # Efficiency = (Accuracy * Speed) / Size
                    efficiency_score = (mAP50 * best_fps) / model_size_mb
                
                accuracy_per_param = mAP50 / (parameters / 1e6) if parameters > 0 else 0  # mAP per million parameters
                
                efficiency_results[model_name] = {
                    'efficiency_score': efficiency_score,
                    'accuracy_per_param': accuracy_per_param,
                    'fps_per_mb': best_fps / model_size_mb if model_size_mb > 0 else 0,
                    'model_size_mb': model_size_mb,
                    'parameters_millions': parameters / 1e6 if parameters > 0 else 0
                }
                
            except Exception as e:
                efficiency_results[model_name] = {'error': str(e)}
        
        return efficiency_results
    
    def _compare_security_performance(self) -> Dict:
        """Compare security-specific performance metrics."""
        security_results = {}
        
        # Security priority classes
        security_priorities = {
            'critical': ['violence', 'emergency'],
            'high': ['theft', 'suspicious'],
            'medium': ['trespassing', 'vandalism'],
            'low': ['crowd', 'loitering', 'abandoned_object'],
            'normal': ['normal']
        }
        
        for model_name, model_info in self.models.items():
            logger.info(f"Evaluating security performance for {model_name}...")
            
            try:
                # Run validation with lower confidence to catch more detections
                results = model_info['model'].val(
                    data=str(self.yolo_data_dir / "dataset.yaml"),
                    split='test',
                    conf=0.25,
                    verbose=False
                )
                
                # Calculate priority-based performance
                priority_performance = {}
                class_names = list(self.config['classes'].keys())
                
                for priority, classes in security_priorities.items():
                    priority_metrics = []
                    
                    if hasattr(results.box, 'ap_class_index'):
                        for class_name in classes:
                            if class_name in class_names:
                                class_idx = class_names.index(class_name)
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
                
                # Calculate security score
                critical_recall = priority_performance.get('critical', {}).get('avg_recall', 0)
                high_recall = priority_performance.get('high', {}).get('avg_recall', 0)
                normal_precision = priority_performance.get('normal', {}).get('avg_precision', 0)
                
                # Security score emphasizes critical/high recall and normal precision
                security_score = (0.4 * critical_recall + 0.3 * high_recall + 0.3 * normal_precision)
                
                security_results[model_name] = {
                    'priority_performance': priority_performance,
                    'security_score': security_score,
                    'critical_detection_rate': critical_recall,
                    'false_alarm_rate': 1 - normal_precision
                }
                
            except Exception as e:
                security_results[model_name] = {'error': str(e)}
        
        return security_results  
  
    def _generate_comparison_summary(self, comparison_results: Dict) -> Dict:
        """Generate comparison summary and recommendations."""
        summary = {
            'best_overall': None,
            'best_accuracy': None,
            'best_speed': None,
            'best_efficiency': None,
            'best_security': None,
            'recommendations': {}
        }
        
        try:
            # Find best models in each category
            
            # Best accuracy
            accuracy_results = comparison_results.get('accuracy_comparison', {})
            if accuracy_results:
                best_accuracy_model = max(accuracy_results.keys(), 
                                        key=lambda x: accuracy_results[x].get('mAP50', 0))
                summary['best_accuracy'] = {
                    'model': best_accuracy_model,
                    'mAP50': accuracy_results[best_accuracy_model].get('mAP50', 0)
                }
            
            # Best speed
            speed_results = comparison_results.get('speed_comparison', {})
            if speed_results:
                best_speed_model = max(speed_results.keys(),
                                     key=lambda x: speed_results[x].get('best_fps', 0))
                summary['best_speed'] = {
                    'model': best_speed_model,
                    'fps': speed_results[best_speed_model].get('best_fps', 0)
                }
            
            # Best efficiency
            efficiency_results = comparison_results.get('efficiency_comparison', {})
            if efficiency_results:
                best_efficiency_model = max(efficiency_results.keys(),
                                          key=lambda x: efficiency_results[x].get('efficiency_score', 0))
                summary['best_efficiency'] = {
                    'model': best_efficiency_model,
                    'efficiency_score': efficiency_results[best_efficiency_model].get('efficiency_score', 0)
                }
            
            # Best security
            security_results = comparison_results.get('security_comparison', {})
            if security_results:
                best_security_model = max(security_results.keys(),
                                        key=lambda x: security_results[x].get('security_score', 0))
                summary['best_security'] = {
                    'model': best_security_model,
                    'security_score': security_results[best_security_model].get('security_score', 0)
                }
            
            # Overall best (weighted combination)
            overall_scores = {}
            for model_name in self.models.keys():
                accuracy_score = accuracy_results.get(model_name, {}).get('mAP50', 0)
                speed_score = min(speed_results.get(model_name, {}).get('best_fps', 0) / 30, 1.0)  # Normalize to 30 FPS
                efficiency_score = efficiency_results.get(model_name, {}).get('efficiency_score', 0)
                security_score = security_results.get(model_name, {}).get('security_score', 0)
                
                # Weighted overall score (security emphasis)
                overall_score = (0.3 * accuracy_score + 0.2 * speed_score + 
                               0.2 * efficiency_score + 0.3 * security_score)
                overall_scores[model_name] = overall_score
            
            if overall_scores:
                best_overall_model = max(overall_scores.keys(), key=lambda x: overall_scores[x])
                summary['best_overall'] = {
                    'model': best_overall_model,
                    'overall_score': overall_scores[best_overall_model]
                }
            
            # Generate recommendations
            summary['recommendations'] = self._generate_deployment_recommendations(comparison_results)
            
        except Exception as e:
            logger.error(f"Failed to generate comparison summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def _generate_deployment_recommendations(self, comparison_results: Dict) -> Dict:
        """Generate deployment recommendations based on comparison results."""
        recommendations = {
            'edge_deployment': None,
            'server_deployment': None,
            'mobile_deployment': None,
            'high_accuracy_deployment': None,
            'general_recommendations': []
        }
        
        try:
            speed_results = comparison_results.get('speed_comparison', {})
            accuracy_results = comparison_results.get('accuracy_comparison', {})
            efficiency_results = comparison_results.get('efficiency_comparison', {})
            security_results = comparison_results.get('security_comparison', {})
            
            # Edge deployment (prioritize speed and size)
            edge_candidates = {}
            for model_name in self.models.keys():
                fps = speed_results.get(model_name, {}).get('best_fps', 0)
                size_mb = self.models[model_name]['size']
                accuracy = accuracy_results.get(model_name, {}).get('mAP50', 0)
                
                if fps >= 15 and size_mb <= 50:  # Real-time capable and reasonable size
                    edge_score = fps * accuracy / size_mb
                    edge_candidates[model_name] = edge_score
            
            if edge_candidates:
                best_edge = max(edge_candidates.keys(), key=lambda x: edge_candidates[x])
                recommendations['edge_deployment'] = {
                    'model': best_edge,
                    'reason': f"Best balance of speed ({speed_results[best_edge]['best_fps']:.1f} FPS) and accuracy ({accuracy_results[best_edge]['mAP50']:.3f} mAP@0.5)"
                }
            
            # Server deployment (prioritize accuracy)
            if accuracy_results:
                best_accuracy_model = max(accuracy_results.keys(), 
                                        key=lambda x: accuracy_results[x].get('mAP50', 0))
                recommendations['server_deployment'] = {
                    'model': best_accuracy_model,
                    'reason': f"Highest accuracy ({accuracy_results[best_accuracy_model]['mAP50']:.3f} mAP@0.5)"
                }
            
            # Mobile deployment (prioritize size and efficiency)
            mobile_candidates = {}
            for model_name in self.models.keys():
                size_mb = self.models[model_name]['size']
                efficiency = efficiency_results.get(model_name, {}).get('efficiency_score', 0)
                
                if size_mb <= 25:  # Mobile-friendly size
                    mobile_candidates[model_name] = efficiency
            
            if mobile_candidates:
                best_mobile = max(mobile_candidates.keys(), key=lambda x: mobile_candidates[x])
                recommendations['mobile_deployment'] = {
                    'model': best_mobile,
                    'reason': f"Smallest size ({self.models[best_mobile]['size']:.1f} MB) with good efficiency"
                }
            
            # High accuracy deployment (security critical)
            if security_results:
                best_security_model = max(security_results.keys(),
                                        key=lambda x: security_results[x].get('security_score', 0))
                recommendations['high_accuracy_deployment'] = {
                    'model': best_security_model,
                    'reason': f"Best security performance (score: {security_results[best_security_model]['security_score']:.3f})"
                }
            
            # General recommendations
            recommendations['general_recommendations'] = [
                "Use ensemble methods for critical security applications",
                "Implement confidence threshold optimization for each deployment",
                "Consider model quantization for edge deployment",
                "Monitor performance in production and retrain as needed"
            ]
            
        except Exception as e:
            recommendations['error'] = str(e)
        
        return recommendations
    
    def _save_comparison_results(self, results: Dict) -> None:
        """Save comparison results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.comparison_dir / f"comparison_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = results_dir / 'comparison_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate comparison report
        self._generate_comparison_report(results, results_dir)
        
        # Generate comparison plots
        self._generate_comparison_plots(results, results_dir)
        
        logger.info(f"Comparison results saved to: {results_dir}")
    
    def _generate_comparison_report(self, results: Dict, output_dir: Path) -> None:
        """Generate human-readable comparison report."""
        report_file = output_dir / 'comparison_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = results.get('summary', {})
            
            if summary.get('best_overall'):
                best = summary['best_overall']
                f.write(f"**Best Overall Model**: {best['model']} (Score: {best['overall_score']:.3f})\n\n")
            
            # Category Winners
            f.write("## Category Winners\n\n")
            categories = ['best_accuracy', 'best_speed', 'best_efficiency', 'best_security']
            category_names = ['Accuracy', 'Speed', 'Efficiency', 'Security']
            
            for category, name in zip(categories, category_names):
                if summary.get(category):
                    winner = summary[category]
                    f.write(f"- **{name}**: {winner['model']}\n")
            f.write("\n")
            
            # Model Comparison Table
            f.write("## Model Comparison\n\n")
            f.write("| Model | mAP@0.5 | FPS | Size (MB) | Security Score | Efficiency |\n")
            f.write("|-------|---------|-----|-----------|----------------|------------|\n")
            
            for model_name in self.models.keys():
                accuracy = results.get('accuracy_comparison', {}).get(model_name, {})
                speed = results.get('speed_comparison', {}).get(model_name, {})
                efficiency = results.get('efficiency_comparison', {}).get(model_name, {})
                security = results.get('security_comparison', {}).get(model_name, {})
                
                mAP50 = accuracy.get('mAP50', 0)
                fps = speed.get('best_fps', 0)
                size_mb = self.models[model_name]['size']
                security_score = security.get('security_score', 0)
                efficiency_score = efficiency.get('efficiency_score', 0)
                
                f.write(f"| {model_name} | {mAP50:.3f} | {fps:.1f} | {size_mb:.1f} | {security_score:.3f} | {efficiency_score:.3f} |\n")
            
            f.write("\n")
            
            # Deployment Recommendations
            f.write("## Deployment Recommendations\n\n")
            recommendations = summary.get('recommendations', {})
            
            deployment_types = [
                ('edge_deployment', 'Edge Deployment'),
                ('server_deployment', 'Server Deployment'),
                ('mobile_deployment', 'Mobile Deployment'),
                ('high_accuracy_deployment', 'High Accuracy Deployment')
            ]
            
            for deploy_key, deploy_name in deployment_types:
                if recommendations.get(deploy_key):
                    rec = recommendations[deploy_key]
                    f.write(f"### {deploy_name}\n")
                    f.write(f"**Recommended Model**: {rec['model']}\n")
                    f.write(f"**Reason**: {rec['reason']}\n\n")
            
            # General Recommendations
            if recommendations.get('general_recommendations'):
                f.write("### General Recommendations\n")
                for rec in recommendations['general_recommendations']:
                    f.write(f"- {rec}\n")
                f.write("\n")
    
    def _generate_comparison_plots(self, results: Dict, output_dir: Path) -> None:
        """Generate comparison visualization plots."""
        try:
            # Prepare data for plotting
            model_names = list(self.models.keys())
            
            # Extract metrics
            accuracy_data = results.get('accuracy_comparison', {})
            speed_data = results.get('speed_comparison', {})
            efficiency_data = results.get('efficiency_comparison', {})
            security_data = results.get('security_comparison', {})
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Model': model_names,
                'mAP@0.5': [accuracy_data.get(m, {}).get('mAP50', 0) for m in model_names],
                'FPS': [speed_data.get(m, {}).get('best_fps', 0) for m in model_names],
                'Size (MB)': [self.models[m]['size'] for m in model_names],
                'Security Score': [security_data.get(m, {}).get('security_score', 0) for m in model_names],
                'Efficiency': [efficiency_data.get(m, {}).get('efficiency_score', 0) for m in model_names]
            })
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Accuracy comparison
            axes[0, 0].bar(comparison_df['Model'], comparison_df['mAP@0.5'])
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('mAP@0.5')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Speed comparison
            axes[0, 1].bar(comparison_df['Model'], comparison_df['FPS'])
            axes[0, 1].set_title('Model Speed Comparison')
            axes[0, 1].set_ylabel('FPS')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Size comparison
            axes[0, 2].bar(comparison_df['Model'], comparison_df['Size (MB)'])
            axes[0, 2].set_title('Model Size Comparison')
            axes[0, 2].set_ylabel('Size (MB)')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # Security score comparison
            axes[1, 0].bar(comparison_df['Model'], comparison_df['Security Score'])
            axes[1, 0].set_title('Security Performance Comparison')
            axes[1, 0].set_ylabel('Security Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Efficiency comparison
            axes[1, 1].bar(comparison_df['Model'], comparison_df['Efficiency'])
            axes[1, 1].set_title('Efficiency Comparison')
            axes[1, 1].set_ylabel('Efficiency Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Accuracy vs Speed scatter plot
            axes[1, 2].scatter(comparison_df['FPS'], comparison_df['mAP@0.5'], 
                              s=comparison_df['Size (MB)'] * 10, alpha=0.7)
            axes[1, 2].set_xlabel('FPS')
            axes[1, 2].set_ylabel('mAP@0.5')
            axes[1, 2].set_title('Accuracy vs Speed (Size = bubble size)')
            
            # Add model labels to scatter plot
            for i, model in enumerate(comparison_df['Model']):
                axes[1, 2].annotate(model, 
                                   (comparison_df['FPS'].iloc[i], comparison_df['mAP@0.5'].iloc[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'model_comparison_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create radar chart for overall comparison
            self._create_radar_chart(comparison_df, output_dir)
            
        except Exception as e:
            logger.warning(f"Could not generate comparison plots: {e}")
    
    def _create_radar_chart(self, comparison_df: pd.DataFrame, output_dir: Path) -> None:
        """Create radar chart for model comparison."""
        try:
            import matplotlib.pyplot as plt
            from math import pi
            
            # Normalize metrics to 0-1 scale
            metrics = ['mAP@0.5', 'FPS', 'Security Score', 'Efficiency']
            normalized_df = comparison_df.copy()
            
            for metric in metrics:
                max_val = normalized_df[metric].max()
                if max_val > 0:
                    normalized_df[metric] = normalized_df[metric] / max_val
            
            # Size is inverted (smaller is better)
            max_size = normalized_df['Size (MB)'].max()
            if max_size > 0:
                normalized_df['Size (Inverted)'] = 1 - (normalized_df['Size (MB)'] / max_size)
            
            radar_metrics = ['mAP@0.5', 'FPS', 'Security Score', 'Efficiency', 'Size (Inverted)']
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Calculate angles for each metric
            angles = [n / float(len(radar_metrics)) * 2 * pi for n in range(len(radar_metrics))]
            angles += angles[:1]  # Complete the circle
            
            # Plot each model
            colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
            
            for i, (_, row) in enumerate(normalized_df.iterrows()):
                values = [row[metric] for metric in radar_metrics]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Radar Chart', size=16, y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.tight_layout()
            plt.savefig(output_dir / 'model_radar_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create radar chart: {e}")

def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare multiple YOLO security models')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model paths in format name:path')
    parser.add_argument('--config', type=str, default='../config/dataset_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize comparison framework
        framework = ModelComparisonFramework(config_path=args.config)
        
        # Add models
        for model_spec in args.models:
            if ':' in model_spec:
                name, path = model_spec.split(':', 1)
                framework.add_model(name, path)
            else:
                logger.error(f"Invalid model specification: {model_spec}. Use format 'name:path'")
                return 1
        
        # Run comparison
        results = framework.run_comparison()
        
        # Print summary
        summary = results.get('summary', {})
        logger.info("="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        if summary.get('best_overall'):
            best = summary['best_overall']
            logger.info(f"Best Overall: {best['model']} (Score: {best['overall_score']:.3f})")
        
        categories = ['best_accuracy', 'best_speed', 'best_efficiency', 'best_security']
        category_names = ['Accuracy', 'Speed', 'Efficiency', 'Security']
        
        for category, name in zip(categories, category_names):
            if summary.get(category):
                winner = summary[category]
                logger.info(f"Best {name}: {winner['model']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)