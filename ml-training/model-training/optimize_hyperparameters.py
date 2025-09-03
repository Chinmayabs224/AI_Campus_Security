#!/usr/bin/env python3
"""
Hyperparameter Optimization for Campus Security YOLO Models
Advanced hyperparameter tuning using Optuna for security-specific scenarios.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import optuna
import torch
import numpy as np
from ultralytics import YOLO
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityHyperparameterOptimizer:
    """Hyperparameter optimizer for campus security YOLO models."""
    
    def __init__(self, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the optimizer."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up paths
        self.yolo_data_dir = Path(self.config['dataset']['yolo_data_dir'])
        self.models_dir = Path("../models/optimization")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization settings
        self.n_trials = 50
        self.timeout = 3600 * 12  # 12 hours
        self.study_name = f"security_yolo_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Security-specific parameter ranges
        self.param_ranges = {
            'learning_rate': (1e-4, 1e-1),
            'batch_size': [8, 16, 32],
            'weight_decay': (1e-6, 1e-2),
            'momentum': (0.8, 0.99),
            'warmup_epochs': (1, 10),
            'box_loss_gain': (0.5, 2.0),
            'cls_loss_gain': (0.5, 2.0),
            'dfl_loss_gain': (0.5, 2.0),
            'hsv_h': (0.0, 0.1),
            'hsv_s': (0.0, 0.9),
            'hsv_v': (0.0, 0.9),
            'degrees': (0.0, 45.0),
            'translate': (0.0, 0.5),
            'scale': (0.0, 0.9),
            'shear': (0.0, 10.0),
            'perspective': (0.0, 0.001),
            'flipud': (0.0, 1.0),
            'fliplr': (0.0, 1.0),
            'mosaic': (0.0, 1.0),
            'mixup': (0.0, 0.5),
            'copy_paste': (0.0, 0.5)
        }
        
        logger.info("SecurityHyperparameterOptimizer initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        
        # Suggest hyperparameters
        params = self._suggest_hyperparameters(trial)
        
        # Create unique run name
        run_name = f"trial_{trial.number:03d}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            # Initialize model
            model = YOLO('yolov8n.pt')  # Start with nano for faster optimization
            
            # Training parameters
            train_params = {
                'data': str(self.yolo_data_dir / "dataset.yaml"),
                'epochs': 30,  # Shorter epochs for optimization
                'batch': params['batch_size'],
                'imgsz': 640,
                'lr0': params['learning_rate'],
                'weight_decay': params['weight_decay'],
                'momentum': params['momentum'],
                'warmup_epochs': params['warmup_epochs'],
                'box': params['box_loss_gain'],
                'cls': params['cls_loss_gain'],
                'dfl': params['dfl_loss_gain'],
                'hsv_h': params['hsv_h'],
                'hsv_s': params['hsv_s'],
                'hsv_v': params['hsv_v'],
                'degrees': params['degrees'],
                'translate': params['translate'],
                'scale': params['scale'],
                'shear': params['shear'],
                'perspective': params['perspective'],
                'flipud': params['flipud'],
                'fliplr': params['fliplr'],
                'mosaic': params['mosaic'],
                'mixup': params['mixup'],
                'copy_paste': params['copy_paste'],
                'project': str(self.models_dir),
                'name': run_name,
                'exist_ok': True,
                'save': False,  # Don't save models during optimization
                'plots': False,  # Disable plots for speed
                'verbose': False,
                'patience': 10,  # Early stopping
                'cache': True,
                'device': 'auto',
                'workers': 4,
                'seed': 42,
                'deterministic': True,
                'val': True
            }
            
            # Train model
            results = model.train(**train_params)
            
            # Extract validation mAP@0.5 as objective
            val_map50 = float(results.results_dict.get('metrics/mAP50(B)', 0.0))
            
            # Log trial results
            logger.info(f"Trial {trial.number}: mAP@0.5 = {val_map50:.4f}")
            
            # Report intermediate values for pruning
            trial.report(val_map50, step=30)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return val_map50
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return poor score for failed trials
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for the trial."""
        
        params = {}
        
        # Learning parameters
        params['learning_rate'] = trial.suggest_float('learning_rate', *self.param_ranges['learning_rate'], log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', self.param_ranges['batch_size'])
        params['weight_decay'] = trial.suggest_float('weight_decay', *self.param_ranges['weight_decay'], log=True)
        params['momentum'] = trial.suggest_float('momentum', *self.param_ranges['momentum'])
        params['warmup_epochs'] = trial.suggest_int('warmup_epochs', *self.param_ranges['warmup_epochs'])
        
        # Loss function weights
        params['box_loss_gain'] = trial.suggest_float('box_loss_gain', *self.param_ranges['box_loss_gain'])
        params['cls_loss_gain'] = trial.suggest_float('cls_loss_gain', *self.param_ranges['cls_loss_gain'])
        params['dfl_loss_gain'] = trial.suggest_float('dfl_loss_gain', *self.param_ranges['dfl_loss_gain'])
        
        # Data augmentation parameters
        params['hsv_h'] = trial.suggest_float('hsv_h', *self.param_ranges['hsv_h'])
        params['hsv_s'] = trial.suggest_float('hsv_s', *self.param_ranges['hsv_s'])
        params['hsv_v'] = trial.suggest_float('hsv_v', *self.param_ranges['hsv_v'])
        params['degrees'] = trial.suggest_float('degrees', *self.param_ranges['degrees'])
        params['translate'] = trial.suggest_float('translate', *self.param_ranges['translate'])
        params['scale'] = trial.suggest_float('scale', *self.param_ranges['scale'])
        params['shear'] = trial.suggest_float('shear', *self.param_ranges['shear'])
        params['perspective'] = trial.suggest_float('perspective', *self.param_ranges['perspective'])
        params['flipud'] = trial.suggest_float('flipud', *self.param_ranges['flipud'])
        params['fliplr'] = trial.suggest_float('fliplr', *self.param_ranges['fliplr'])
        params['mosaic'] = trial.suggest_float('mosaic', *self.param_ranges['mosaic'])
        params['mixup'] = trial.suggest_float('mixup', *self.param_ranges['mixup'])
        params['copy_paste'] = trial.suggest_float('copy_paste', *self.param_ranges['copy_paste'])
        
        return params
    
    def optimize(self, n_trials: Optional[int] = None, timeout: Optional[int] = None) -> optuna.Study:
        """Run hyperparameter optimization."""
        
        if n_trials is None:
            n_trials = self.n_trials
        if timeout is None:
            timeout = self.timeout
        
        logger.info(f"Starting hyperparameter optimization...")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            ),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self._log_callback]
        )
        
        # Save results
        self._save_optimization_results(study)
        
        logger.info("Hyperparameter optimization completed!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best mAP@0.5: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study
    
    def _log_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback function for logging trial results."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} completed with value: {trial.value:.4f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"Trial {trial.number} pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            logger.warning(f"Trial {trial.number} failed")
    
    def _save_optimization_results(self, study: optuna.Study) -> None:
        """Save optimization results to files."""
        
        results_dir = self.models_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir.mkdir(exist_ok=True)
        
        # Save study summary
        summary = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'optimization_history': [
                {
                    'trial': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        summary_file = results_dir / 'optimization_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save best parameters as YAML config
        best_config = {
            'optimization_results': {
                'best_trial': study.best_trial.number,
                'best_mAP50': study.best_value,
                'optimization_date': datetime.now().isoformat()
            },
            'optimized_parameters': study.best_params
        }
        
        config_file = results_dir / 'best_hyperparameters.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        # Generate optimization plots
        try:
            import matplotlib.pyplot as plt
            
            # Optimization history plot
            values = [trial.value for trial in study.trials if trial.value is not None]
            trials = [trial.number for trial in study.trials if trial.value is not None]
            
            plt.figure(figsize=(10, 6))
            plt.plot(trials, values, 'b-', alpha=0.7)
            plt.scatter(trials, values, c='red', s=20)
            plt.axhline(y=study.best_value, color='green', linestyle='--', label=f'Best: {study.best_value:.4f}')
            plt.xlabel('Trial')
            plt.ylabel('mAP@0.5')
            plt.title('Hyperparameter Optimization History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(results_dir / 'optimization_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parameter importance plot
            if len(study.trials) > 10:
                importance = optuna.importance.get_param_importances(study)
                
                plt.figure(figsize=(10, 8))
                params = list(importance.keys())
                values = list(importance.values())
                
                plt.barh(params, values)
                plt.xlabel('Importance')
                plt.title('Parameter Importance')
                plt.tight_layout()
                plt.savefig(results_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        
        logger.info(f"Optimization results saved to: {results_dir}")
    
    def train_with_best_params(self, study: optuna.Study, model_size: str = 'n', epochs: int = 100) -> Dict:
        """Train final model with best hyperparameters."""
        
        logger.info("Training final model with optimized hyperparameters...")
        
        best_params = study.best_params
        
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Training parameters with best hyperparameters
        train_params = {
            'data': str(self.yolo_data_dir / "dataset.yaml"),
            'epochs': epochs,
            'batch': best_params['batch_size'],
            'imgsz': 640,
            'lr0': best_params['learning_rate'],
            'weight_decay': best_params['weight_decay'],
            'momentum': best_params['momentum'],
            'warmup_epochs': best_params['warmup_epochs'],
            'box': best_params['box_loss_gain'],
            'cls': best_params['cls_loss_gain'],
            'dfl': best_params['dfl_loss_gain'],
            'hsv_h': best_params['hsv_h'],
            'hsv_s': best_params['hsv_s'],
            'hsv_v': best_params['hsv_v'],
            'degrees': best_params['degrees'],
            'translate': best_params['translate'],
            'scale': best_params['scale'],
            'shear': best_params['shear'],
            'perspective': best_params['perspective'],
            'flipud': best_params['flipud'],
            'fliplr': best_params['fliplr'],
            'mosaic': best_params['mosaic'],
            'mixup': best_params['mixup'],
            'copy_paste': best_params['copy_paste'],
            'project': str(self.models_dir),
            'name': f'optimized_yolo_{model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'save': True,
            'plots': True,
            'verbose': True,
            'patience': 50,
            'cache': True,
            'device': 'auto',
            'workers': 8,
            'seed': 42,
            'deterministic': True,
            'val': True
        }
        
        # Train model
        results = model.train(**train_params)
        
        # Validate model
        val_results = model.val(
            data=str(self.yolo_data_dir / "dataset.yaml"),
            split='test'
        )
        
        training_results = {
            'model_path': str(results.save_dir / 'weights' / 'best.pt'),
            'results_dir': str(results.save_dir),
            'best_hyperparameters': best_params,
            'optimization_mAP50': study.best_value,
            'final_mAP50': float(val_results.box.map50),
            'final_mAP50_95': float(val_results.box.map),
            'final_precision': float(val_results.box.mp),
            'final_recall': float(val_results.box.mr)
        }
        
        logger.info("Optimized model training completed!")
        logger.info(f"Optimization mAP@0.5: {training_results['optimization_mAP50']:.4f}")
        logger.info(f"Final mAP@0.5: {training_results['final_mAP50']:.4f}")
        logger.info(f"Final mAP@0.5:0.95: {training_results['final_mAP50_95']:.4f}")
        
        return training_results

def main():
    """Main optimization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize YOLO hyperparameters for campus security')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=43200, help='Optimization timeout in seconds (default: 12h)')
    parser.add_argument('--train-final', action='store_true', help='Train final model with best parameters')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='Model size for final training')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for final training')
    parser.add_argument('--config', type=str, default='../config/dataset_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize optimizer
        optimizer = SecurityHyperparameterOptimizer(config_path=args.config)
        
        # Run optimization
        study = optimizer.optimize(n_trials=args.trials, timeout=args.timeout)
        
        # Train final model if requested
        if args.train_final:
            final_results = optimizer.train_with_best_params(
                study, 
                model_size=args.model_size, 
                epochs=args.epochs
            )
            logger.info(f"Final model saved to: {final_results['model_path']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)