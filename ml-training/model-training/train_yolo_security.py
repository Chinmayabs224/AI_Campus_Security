#!/usr/bin/env python3
"""
YOLO Model Training for Campus Security
Trains custom YOLOv8 models for security event detection using the processed UCF Crime Dataset.
"""

import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityYOLOTrainer:
    """Custom YOLO trainer for campus security applications."""
    
    def __init__(self, config_path: str = "../config/dataset_config.yaml"):
        """Initialize the trainer with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.training_results = {}
        
        # Set up paths
        self.data_dir = Path(self.config['dataset']['processed_data_dir'])
        self.yolo_data_dir = Path(self.config['dataset']['yolo_data_dir'])
        self.models_dir = Path("../models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.model_size = self.config['yolo']['model_size']
        self.epochs = self.config['yolo']['epochs']
        self.batch_size = self.config['yolo']['batch_size']
        self.learning_rate = self.config['yolo']['learning_rate']
        self.patience = self.config['yolo']['patience']
        self.input_size = self.config['yolo']['input_size']
        
        logger.info(f"Initialized SecurityYOLOTrainer with model size: {self.model_size}")
    
    def _load_config(self) -> Dict:
        """Load training configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_model(self, pretrained: bool = True) -> None:
        """Initialize YOLO model with security-specific configuration."""
        logger.info(f"Setting up YOLOv8{self.model_size} model...")
        
        if pretrained:
            # Load pretrained COCO model
            model_name = f"yolov8{self.model_size}.pt"
            self.model = YOLO(model_name)
            logger.info(f"Loaded pretrained model: {model_name}")
        else:
            # Load architecture only
            model_name = f"yolov8{self.model_size}.yaml"
            self.model = YOLO(model_name)
            logger.info(f"Loaded model architecture: {model_name}")
        
        # Verify dataset configuration
        dataset_yaml = self.yolo_data_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset configuration not found: {dataset_yaml}")
        
        logger.info(f"Using dataset configuration: {dataset_yaml}")
    
    def configure_training_params(self, **kwargs) -> Dict:
        """Configure training parameters with security-specific optimizations."""
        
        # Base training parameters
        train_params = {
            'data': str(self.yolo_data_dir / "dataset.yaml"),
            'epochs': kwargs.get('epochs', self.epochs),
            'batch': kwargs.get('batch', self.batch_size),
            'imgsz': kwargs.get('imgsz', self.input_size),
            'lr0': kwargs.get('lr0', self.learning_rate),
            'patience': kwargs.get('patience', self.patience),
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'cache': True,      # Cache images for faster training
            'device': kwargs.get('device', 'auto'),
            'workers': kwargs.get('workers', 8),
            'project': str(self.models_dir),
            'name': kwargs.get('name', f'security_yolo_{self.model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            'exist_ok': True,
            'pretrained': kwargs.get('pretrained', True),
            'optimizer': kwargs.get('optimizer', 'AdamW'),
            'verbose': True,
            'seed': kwargs.get('seed', 42),
            'deterministic': kwargs.get('deterministic', True),
            'single_cls': False,  # Multi-class detection
            'rect': False,        # Rectangular training for better performance
            'cos_lr': True,       # Cosine learning rate scheduler
            'close_mosaic': 10,   # Disable mosaic augmentation in last 10 epochs
            'resume': kwargs.get('resume', False),
            'amp': kwargs.get('amp', True),  # Automatic Mixed Precision
            'fraction': kwargs.get('fraction', 1.0),  # Dataset fraction to use
            'profile': False,     # Profile ONNX and TensorRT speeds during validation
            'freeze': kwargs.get('freeze', None),  # Freeze layers: backbone=10, all=24
            'multi_scale': kwargs.get('multi_scale', False),  # Multi-scale training
            'overlap_mask': True,  # Masks should overlap during training
            'mask_ratio': 4,      # Mask downsample ratio
            'dropout': kwargs.get('dropout', 0.0),  # Use dropout regularization
            'val': True,          # Validate/test during training
        }
        
        # Security-specific augmentations
        security_augmentations = {
            'hsv_h': 0.015,      # Hue augmentation (±1.5%)
            'hsv_s': 0.7,        # Saturation augmentation (±70%)
            'hsv_v': 0.4,        # Value augmentation (±40%)
            'degrees': 10.0,     # Rotation (±10 degrees)
            'translate': 0.1,    # Translation (±10%)
            'scale': 0.5,        # Scale (±50%)
            'shear': 2.0,        # Shear (±2 degrees)
            'perspective': 0.0,  # Perspective transformation (0-0.001)
            'flipud': 0.0,       # Vertical flip probability
            'fliplr': 0.5,       # Horizontal flip probability
            'mosaic': 1.0,       # Mosaic augmentation probability
            'mixup': 0.1,        # MixUp augmentation probability
            'copy_paste': 0.1,   # Copy-paste augmentation probability
        }
        
        train_params.update(security_augmentations)
        
        # Add custom parameters
        train_params.update(kwargs)
        
        logger.info("Training parameters configured for security applications")
        return train_params
    
    def setup_wandb_logging(self, project_name: str = "campus-security-yolo") -> None:
        """Setup Weights & Biases logging for experiment tracking."""
        try:
            wandb.init(
                project=project_name,
                name=f"yolo_{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'model_size': self.model_size,
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'input_size': self.input_size,
                    'dataset': 'ucf_crime_security'
                }
            )
            logger.info("Weights & Biases logging initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B logging: {e}")
    
    def train_model(self, **kwargs) -> Dict:
        """Train the YOLO model with security-specific configuration."""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        logger.info("Starting YOLO model training for campus security...")
        
        # Configure training parameters
        train_params = self.configure_training_params(**kwargs)
        
        # Log training configuration
        logger.info("Training Configuration:")
        for key, value in train_params.items():
            logger.info(f"  {key}: {value}")
        
        try:
            # Start training
            results = self.model.train(**train_params)
            
            # Store training results
            self.training_results = {
                'model_path': str(results.save_dir / 'weights' / 'best.pt'),
                'last_model_path': str(results.save_dir / 'weights' / 'last.pt'),
                'results_dir': str(results.save_dir),
                'training_params': train_params,
                'final_metrics': self._extract_final_metrics(results),
                'training_time': results.speed if hasattr(results, 'speed') else None
            }
            
            logger.info("Training completed successfully!")
            logger.info(f"Best model saved to: {self.training_results['model_path']}")
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _extract_final_metrics(self, results) -> Dict:
        """Extract final training metrics from results."""
        try:
            metrics = {}
            
            # Extract metrics from results if available
            if hasattr(results, 'results_dict'):
                metrics.update(results.results_dict)
            
            # Try to read metrics from results file
            results_dir = Path(results.save_dir)
            results_csv = results_dir / 'results.csv'
            
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                if not df.empty:
                    last_row = df.iloc[-1]
                    metrics.update({
                        'final_epoch': last_row.get('epoch', 0),
                        'train_loss': last_row.get('train/box_loss', 0),
                        'val_loss': last_row.get('val/box_loss', 0),
                        'mAP50': last_row.get('metrics/mAP50(B)', 0),
                        'mAP50_95': last_row.get('metrics/mAP50-95(B)', 0),
                        'precision': last_row.get('metrics/precision(B)', 0),
                        'recall': last_row.get('metrics/recall(B)', 0)
                    })
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not extract final metrics: {e}")
            return {}
    
    def validate_model(self, model_path: Optional[str] = None) -> Dict:
        """Validate the trained model on test set."""
        if model_path is None:
            if not self.training_results:
                raise ValueError("No trained model available. Train model first or provide model_path.")
            model_path = self.training_results['model_path']
        
        logger.info(f"Validating model: {model_path}")
        
        # Load model for validation
        model = YOLO(model_path)
        
        # Run validation
        dataset_yaml = self.yolo_data_dir / "dataset.yaml"
        results = model.val(
            data=str(dataset_yaml),
            split='test',
            imgsz=self.input_size,
            batch=1,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            device='auto',
            dnn=False,
            plots=True,
            verbose=True
        )
        
        # Extract validation metrics
        validation_metrics = {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0,
            'class_metrics': {}
        }
        
        # Per-class metrics
        if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
            class_names = self.config['classes']
            for i, class_idx in enumerate(results.box.ap_class_index):
                if class_idx < len(class_names):
                    class_name = list(class_names.keys())[class_idx]
                    validation_metrics['class_metrics'][class_name] = {
                        'ap50': float(results.box.ap50[i]),
                        'ap50_95': float(results.box.ap[i])
                    }
        
        logger.info("Validation Results:")
        logger.info(f"  mAP@0.5: {validation_metrics['mAP50']:.4f}")
        logger.info(f"  mAP@0.5:0.95: {validation_metrics['mAP50_95']:.4f}")
        logger.info(f"  Precision: {validation_metrics['precision']:.4f}")
        logger.info(f"  Recall: {validation_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {validation_metrics['f1_score']:.4f}")
        
        return validation_metrics
    
    def export_model(self, model_path: Optional[str] = None, formats: List[str] = None) -> Dict:
        """Export trained model to various formats for deployment."""
        if model_path is None:
            if not self.training_results:
                raise ValueError("No trained model available. Train model first or provide model_path.")
            model_path = self.training_results['model_path']
        
        if formats is None:
            formats = ['onnx', 'torchscript', 'tflite']
        
        logger.info(f"Exporting model to formats: {formats}")
        
        # Load model
        model = YOLO(model_path)
        
        export_results = {}
        
        for format_name in formats:
            try:
                logger.info(f"Exporting to {format_name.upper()}...")
                
                export_path = model.export(
                    format=format_name,
                    imgsz=self.input_size,
                    keras=False,
                    optimize=True,
                    half=True,
                    int8=False,
                    dynamic=False,
                    simplify=True,
                    opset=11,
                    workspace=4,
                    nms=True
                )
                
                export_results[format_name] = str(export_path)
                logger.info(f"✓ {format_name.upper()} export successful: {export_path}")
                
            except Exception as e:
                logger.error(f"✗ {format_name.upper()} export failed: {e}")
                export_results[format_name] = None
        
        return export_results
    
    def hyperparameter_optimization(self, iterations: int = 30) -> Dict:
        """Perform hyperparameter optimization using Ultralytics tuning."""
        logger.info(f"Starting hyperparameter optimization with {iterations} iterations...")
        
        if self.model is None:
            self.setup_model()
        
        # Run hyperparameter tuning
        results = self.model.tune(
            data=str(self.yolo_data_dir / "dataset.yaml"),
            epochs=30,  # Shorter epochs for tuning
            iterations=iterations,
            optimizer='AdamW',
            plots=True,
            save=True,
            val=True
        )
        
        logger.info("Hyperparameter optimization completed")
        return results
    
    def generate_training_report(self) -> Dict:
        """Generate comprehensive training report."""
        if not self.training_results:
            raise ValueError("No training results available. Train model first.")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_configuration': {
                'model_size': self.model_size,
                'input_size': self.input_size,
                'classes': self.config['classes']
            },
            'training_configuration': self.training_results['training_params'],
            'training_results': self.training_results['final_metrics'],
            'model_paths': {
                'best_model': self.training_results['model_path'],
                'last_model': self.training_results['last_model_path'],
                'results_directory': self.training_results['results_dir']
            }
        }
        
        # Add validation results if available
        try:
            validation_metrics = self.validate_model()
            report['validation_results'] = validation_metrics
        except Exception as e:
            logger.warning(f"Could not add validation results to report: {e}")
        
        # Save report
        report_path = Path(self.training_results['results_dir']) / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to: {report_path}")
        return report

def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description='Train YOLO model for campus security')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='YOLO model size (default: n)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (default: auto)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights (default: True)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--export', action='store_true',
                       help='Export model after training')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--config', type=str, default='../config/dataset_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = SecurityYOLOTrainer(config_path=args.config)
        
        # Override config with command line arguments
        trainer.model_size = args.model_size
        trainer.epochs = args.epochs
        trainer.batch_size = args.batch_size
        trainer.learning_rate = args.learning_rate
        trainer.patience = args.patience
        
        # Setup model
        trainer.setup_model(pretrained=args.pretrained)
        
        # Setup logging
        if args.wandb:
            trainer.setup_wandb_logging()
        
        # Hyperparameter tuning
        if args.tune:
            logger.info("Performing hyperparameter optimization...")
            tune_results = trainer.hyperparameter_optimization()
            logger.info("Hyperparameter tuning completed")
            return 0
        
        # Train model
        training_params = {
            'epochs': args.epochs,
            'batch': args.batch_size,
            'lr0': args.learning_rate,
            'patience': args.patience,
            'device': args.device,
            'resume': args.resume
        }
        
        results = trainer.train_model(**training_params)
        
        # Validate model
        logger.info("Validating trained model...")
        validation_results = trainer.validate_model()
        
        # Export model
        if args.export:
            logger.info("Exporting model...")
            export_results = trainer.export_model()
            logger.info(f"Export results: {export_results}")
        
        # Generate report
        report = trainer.generate_training_report()
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Best model: {results['model_path']}")
        logger.info(f"mAP@0.5: {validation_results.get('mAP50', 'N/A'):.4f}")
        logger.info(f"mAP@0.5:0.95: {validation_results.get('mAP50_95', 'N/A'):.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)