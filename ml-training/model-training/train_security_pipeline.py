#!/usr/bin/env python3
"""
Complete Security Model Training Pipeline
Orchestrates the entire YOLO training workflow for campus security applications.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "data-processing"))

from train_yolo_security import SecurityYOLOTrainer
from optimize_hyperparameters import SecurityHyperparameterOptimizer
from evaluate_security_model import SecurityModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityTrainingPipeline:
    """Complete training pipeline for campus security YOLO models."""
    
    def __init__(self, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the training pipeline."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up directories
        self.models_dir = Path("../models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Pipeline configuration
        self.pipeline_config = {
            'model_sizes': ['n', 's', 'm'],  # Train multiple model sizes
            'optimization_trials': 30,
            'final_epochs': 100,
            'export_formats': ['onnx', 'torchscript'],
            'evaluation_thresholds': [0.1, 0.25, 0.5, 0.75, 0.9]
        }
        
        # Results tracking
        self.pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'models_trained': {},
            'optimization_results': {},
            'evaluation_results': {},
            'best_model': None,
            'pipeline_summary': {}
        }
        
        logger.info("SecurityTrainingPipeline initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_complete_pipeline(self, 
                            optimize_hyperparameters: bool = True,
                            train_multiple_sizes: bool = True,
                            export_models: bool = True,
                            evaluate_models: bool = True) -> Dict:
        """Run the complete training pipeline."""
        
        logger.info("="*80)
        logger.info("STARTING COMPLETE SECURITY MODEL TRAINING PIPELINE")
        logger.info("="*80)
        
        try:
            # Step 1: Hyperparameter Optimization (optional)
            if optimize_hyperparameters:
                logger.info("STEP 1: Hyperparameter Optimization")
                self._run_hyperparameter_optimization()
            else:
                logger.info("STEP 1: Skipping hyperparameter optimization")
            
            # Step 2: Train models
            logger.info("STEP 2: Model Training")
            if train_multiple_sizes:
                self._train_multiple_model_sizes()
            else:
                self._train_single_model()
            
            # Step 3: Export models (optional)
            if export_models:
                logger.info("STEP 3: Model Export")
                self._export_trained_models()
            else:
                logger.info("STEP 3: Skipping model export")
            
            # Step 4: Evaluate models (optional)
            if evaluate_models:
                logger.info("STEP 4: Model Evaluation")
                self._evaluate_trained_models()
            else:
                logger.info("STEP 4: Skipping model evaluation")
            
            # Step 5: Generate final report
            logger.info("STEP 5: Generating Final Report")
            self._generate_pipeline_report()
            
            # Step 6: Select best model
            logger.info("STEP 6: Selecting Best Model")
            self._select_best_model()
            
            self.pipeline_results['end_time'] = datetime.now().isoformat()
            self.pipeline_results['status'] = 'completed'
            
            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.pipeline_results['status'] = 'failed'
            self.pipeline_results['error'] = str(e)
            raise
    
    def _run_hyperparameter_optimization(self) -> None:
        """Run hyperparameter optimization."""
        logger.info("Starting hyperparameter optimization...")
        
        try:
            optimizer = SecurityHyperparameterOptimizer(config_path=str(self.config_path))
            
            # Run optimization
            study = optimizer.optimize(
                n_trials=self.pipeline_config['optimization_trials'],
                timeout=3600 * 6  # 6 hours max
            )
            
            # Store optimization results
            self.pipeline_results['optimization_results'] = {
                'best_trial': study.best_trial.number,
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials)
            }
            
            # Update config with optimized parameters
            self._update_config_with_optimized_params(study.best_params)
            
            logger.info(f"Hyperparameter optimization completed. Best mAP@0.5: {study.best_value:.4f}")
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            # Continue with default parameters
            self.pipeline_results['optimization_results'] = {'status': 'failed', 'error': str(e)}
    
    def _update_config_with_optimized_params(self, best_params: Dict) -> None:
        """Update training configuration with optimized parameters."""
        # Update YOLO config with optimized parameters
        if 'yolo' not in self.config:
            self.config['yolo'] = {}
        
        # Map optimization parameters to config
        param_mapping = {
            'learning_rate': 'learning_rate',
            'batch_size': 'batch_size',
            'weight_decay': 'weight_decay',
            'momentum': 'momentum'
        }
        
        for opt_param, config_param in param_mapping.items():
            if opt_param in best_params:
                self.config['yolo'][config_param] = best_params[opt_param]
        
        logger.info("Configuration updated with optimized parameters")
    
    def _train_multiple_model_sizes(self) -> None:
        """Train models of different sizes."""
        logger.info("Training multiple model sizes...")
        
        for model_size in self.pipeline_config['model_sizes']:
            logger.info(f"Training YOLOv8{model_size} model...")
            
            try:
                # Initialize trainer
                trainer = SecurityYOLOTrainer(config_path=str(self.config_path))
                trainer.model_size = model_size
                
                # Update trainer config if optimization was performed
                if 'optimization_results' in self.pipeline_results and 'best_params' in self.pipeline_results['optimization_results']:
                    best_params = self.pipeline_results['optimization_results']['best_params']
                    trainer.batch_size = best_params.get('batch_size', trainer.batch_size)
                    trainer.learning_rate = best_params.get('learning_rate', trainer.learning_rate)
                
                # Setup model
                trainer.setup_model(pretrained=True)
                
                # Train model
                training_params = {
                    'epochs': self.pipeline_config['final_epochs'],
                    'name': f'security_yolo_{model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                }
                
                results = trainer.train_model(**training_params)
                
                # Store results
                self.pipeline_results['models_trained'][model_size] = {
                    'model_path': results['model_path'],
                    'results_dir': results['results_dir'],
                    'final_metrics': results['final_metrics'],
                    'training_params': results['training_params']
                }
                
                logger.info(f"YOLOv8{model_size} training completed successfully")
                
            except Exception as e:
                logger.error(f"Training YOLOv8{model_size} failed: {e}")
                self.pipeline_results['models_trained'][model_size] = {'status': 'failed', 'error': str(e)}
    
    def _train_single_model(self) -> None:
        """Train a single model (default size)."""
        logger.info("Training single model...")
        
        model_size = 'n'  # Default to nano for faster training
        
        try:
            trainer = SecurityYOLOTrainer(config_path=str(self.config_path))
            trainer.model_size = model_size
            
            # Setup and train
            trainer.setup_model(pretrained=True)
            results = trainer.train_model(epochs=self.pipeline_config['final_epochs'])
            
            # Store results
            self.pipeline_results['models_trained'][model_size] = {
                'model_path': results['model_path'],
                'results_dir': results['results_dir'],
                'final_metrics': results['final_metrics'],
                'training_params': results['training_params']
            }
            
            logger.info("Single model training completed successfully")
            
        except Exception as e:
            logger.error(f"Single model training failed: {e}")
            self.pipeline_results['models_trained'][model_size] = {'status': 'failed', 'error': str(e)}
    
    def _export_trained_models(self) -> None:
        """Export trained models to deployment formats."""
        logger.info("Exporting trained models...")
        
        for model_size, model_info in self.pipeline_results['models_trained'].items():
            if 'model_path' not in model_info:
                continue
            
            try:
                logger.info(f"Exporting YOLOv8{model_size} model...")
                
                # Initialize trainer for export
                trainer = SecurityYOLOTrainer(config_path=str(self.config_path))
                
                # Export model
                export_results = trainer.export_model(
                    model_path=model_info['model_path'],
                    formats=self.pipeline_config['export_formats']
                )
                
                # Store export results
                model_info['export_results'] = export_results
                
                logger.info(f"YOLOv8{model_size} export completed")
                
            except Exception as e:
                logger.error(f"Export failed for YOLOv8{model_size}: {e}")
                model_info['export_results'] = {'status': 'failed', 'error': str(e)}
    
    def _evaluate_trained_models(self) -> None:
        """Evaluate all trained models."""
        logger.info("Evaluating trained models...")
        
        for model_size, model_info in self.pipeline_results['models_trained'].items():
            if 'model_path' not in model_info:
                continue
            
            try:
                logger.info(f"Evaluating YOLOv8{model_size} model...")
                
                # Initialize evaluator
                evaluator = SecurityModelEvaluator(
                    model_path=model_info['model_path'],
                    config_path=str(self.config_path)
                )
                
                # Run evaluation
                evaluation_results = evaluator.evaluate_model(
                    confidence_thresholds=self.pipeline_config['evaluation_thresholds']
                )
                
                # Store evaluation results
                self.pipeline_results['evaluation_results'][model_size] = evaluation_results
                
                logger.info(f"YOLOv8{model_size} evaluation completed")
                
            except Exception as e:
                logger.error(f"Evaluation failed for YOLOv8{model_size}: {e}")
                self.pipeline_results['evaluation_results'][model_size] = {'status': 'failed', 'error': str(e)}
    
    def _select_best_model(self) -> None:
        """Select the best model based on security metrics."""
        logger.info("Selecting best model...")
        
        best_model = None
        best_score = 0
        
        for model_size, eval_results in self.pipeline_results['evaluation_results'].items():
            if 'security_analysis' not in eval_results:
                continue
            
            # Calculate composite security score
            security_metrics = eval_results['security_analysis']['critical_metrics']
            
            # Weighted score: emphasize high-priority detection and low false alarms
            composite_score = (
                0.4 * security_metrics.get('high_priority_detection_rate', 0) +
                0.3 * security_metrics.get('overall_security_score', 0) +
                0.2 * eval_results['performance_metrics'].get('mAP50', 0) +
                0.1 * (1 - security_metrics.get('false_alarm_rate', 1))
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = {
                    'model_size': model_size,
                    'model_path': self.pipeline_results['models_trained'][model_size]['model_path'],
                    'composite_score': composite_score,
                    'security_score': security_metrics.get('overall_security_score', 0),
                    'mAP50': eval_results['performance_metrics'].get('mAP50', 0),
                    'high_priority_detection': security_metrics.get('high_priority_detection_rate', 0),
                    'false_alarm_rate': security_metrics.get('false_alarm_rate', 1)
                }
        
        self.pipeline_results['best_model'] = best_model
        
        if best_model:
            logger.info(f"Best model: YOLOv8{best_model['model_size']}")
            logger.info(f"Composite score: {best_model['composite_score']:.4f}")
            logger.info(f"Security score: {best_model['security_score']:.4f}")
            logger.info(f"mAP@0.5: {best_model['mAP50']:.4f}")
        else:
            logger.warning("No best model could be determined")
    
    def _generate_pipeline_report(self) -> None:
        """Generate comprehensive pipeline report."""
        logger.info("Generating pipeline report...")
        
        # Create pipeline summary
        summary = {
            'pipeline_duration': self._calculate_pipeline_duration(),
            'models_trained': len([m for m in self.pipeline_results['models_trained'].values() if 'model_path' in m]),
            'models_evaluated': len([e for e in self.pipeline_results['evaluation_results'].values() if 'performance_metrics' in e]),
            'optimization_performed': 'optimization_results' in self.pipeline_results and 'best_params' in self.pipeline_results['optimization_results'],
            'export_performed': any('export_results' in m for m in self.pipeline_results['models_trained'].values() if isinstance(m, dict))
        }
        
        self.pipeline_results['pipeline_summary'] = summary
        
        # Save complete results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.models_dir / f'pipeline_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.pipeline_results, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(timestamp)
        
        logger.info(f"Pipeline report saved to: {results_file}")
    
    def _calculate_pipeline_duration(self) -> str:
        """Calculate total pipeline duration."""
        if 'end_time' in self.pipeline_results:
            start = datetime.fromisoformat(self.pipeline_results['start_time'])
            end = datetime.fromisoformat(self.pipeline_results['end_time'])
            duration = end - start
            return str(duration)
        return "In progress"
    
    def _generate_markdown_report(self, timestamp: str) -> None:
        """Generate markdown report."""
        report_file = self.models_dir / f'pipeline_report_{timestamp}.md'
        
        with open(report_file, 'w') as f:
            f.write("# Campus Security YOLO Training Pipeline Report\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = self.pipeline_results['pipeline_summary']
            f.write(f"- **Pipeline Duration**: {summary['pipeline_duration']}\n")
            f.write(f"- **Models Trained**: {summary['models_trained']}\n")
            f.write(f"- **Models Evaluated**: {summary['models_evaluated']}\n")
            f.write(f"- **Hyperparameter Optimization**: {'Yes' if summary['optimization_performed'] else 'No'}\n")
            f.write(f"- **Model Export**: {'Yes' if summary['export_performed'] else 'No'}\n\n")
            
            # Best Model
            if self.pipeline_results['best_model']:
                best = self.pipeline_results['best_model']
                f.write("## Best Model\n\n")
                f.write(f"- **Model**: YOLOv8{best['model_size']}\n")
                f.write(f"- **Composite Score**: {best['composite_score']:.4f}\n")
                f.write(f"- **Security Score**: {best['security_score']:.4f}\n")
                f.write(f"- **mAP@0.5**: {best['mAP50']:.4f}\n")
                f.write(f"- **High Priority Detection Rate**: {best['high_priority_detection']:.4f}\n")
                f.write(f"- **False Alarm Rate**: {best['false_alarm_rate']:.4f}\n")
                f.write(f"- **Model Path**: `{best['model_path']}`\n\n")
            
            # Model Comparison
            f.write("## Model Comparison\n\n")
            f.write("| Model | mAP@0.5 | Security Score | High Priority Detection | False Alarm Rate |\n")
            f.write("|-------|---------|----------------|------------------------|------------------|\n")
            
            for model_size, eval_results in self.pipeline_results['evaluation_results'].items():
                if 'performance_metrics' in eval_results:
                    mAP50 = eval_results['performance_metrics'].get('mAP50', 0)
                    security_score = eval_results['security_analysis']['critical_metrics'].get('overall_security_score', 0)
                    high_priority = eval_results['security_analysis']['critical_metrics'].get('high_priority_detection_rate', 0)
                    false_alarm = eval_results['security_analysis']['critical_metrics'].get('false_alarm_rate', 1)
                    
                    f.write(f"| YOLOv8{model_size} | {mAP50:.4f} | {security_score:.4f} | {high_priority:.4f} | {false_alarm:.4f} |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Deployment Recommendations\n\n")
            if self.pipeline_results['best_model']:
                best_size = self.pipeline_results['best_model']['model_size']
                f.write(f"1. **Deploy YOLOv8{best_size}** as the primary security model\n")
                f.write("2. **Configure confidence threshold** based on evaluation results\n")
                f.write("3. **Implement temporal consistency** checks to reduce false positives\n")
                f.write("4. **Monitor performance** in production and retrain as needed\n")
                f.write("5. **Consider ensemble methods** for critical security applications\n\n")
        
        logger.info(f"Markdown report saved to: {report_file}")

def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Run complete security model training pipeline')
    parser.add_argument('--config', type=str, default='../config/dataset_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='Skip hyperparameter optimization')
    parser.add_argument('--single-model', action='store_true',
                       help='Train only single model size')
    parser.add_argument('--skip-export', action='store_true',
                       help='Skip model export')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip model evaluation')
    parser.add_argument('--model-sizes', nargs='+', choices=['n', 's', 'm', 'l', 'x'],
                       default=['n', 's', 'm'], help='Model sizes to train')
    parser.add_argument('--optimization-trials', type=int, default=30,
                       help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = SecurityTrainingPipeline(config_path=args.config)
        
        # Update pipeline configuration
        pipeline.pipeline_config['model_sizes'] = args.model_sizes
        pipeline.pipeline_config['optimization_trials'] = args.optimization_trials
        pipeline.pipeline_config['final_epochs'] = args.epochs
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(
            optimize_hyperparameters=not args.skip_optimization,
            train_multiple_sizes=not args.single_model,
            export_models=not args.skip_export,
            evaluate_models=not args.skip_evaluation
        )
        
        # Print final summary
        logger.info("="*80)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*80)
        
        if results['best_model']:
            best = results['best_model']
            logger.info(f"Best Model: YOLOv8{best['model_size']}")
            logger.info(f"Security Score: {best['security_score']:.4f}")
            logger.info(f"mAP@0.5: {best['mAP50']:.4f}")
            logger.info(f"Model Path: {best['model_path']}")
        
        logger.info(f"Pipeline Duration: {results['pipeline_summary']['pipeline_duration']}")
        logger.info(f"Models Trained: {results['pipeline_summary']['models_trained']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)