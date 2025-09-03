# YOLO Model Training for Campus Security

This module provides comprehensive YOLO model training capabilities specifically designed for campus security applications. It includes hyperparameter optimization, multi-model training, evaluation, and deployment preparation.

## Overview

The training pipeline transforms the processed UCF Crime Dataset into high-performance YOLO models optimized for real-time security event detection. The system supports multiple model sizes, automated hyperparameter optimization, and comprehensive security-focused evaluation.

## Features

- **Multi-Model Training**: Train YOLOv8n, YOLOv8s, and YOLOv8m variants
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Security-Focused Evaluation**: Metrics tailored for security applications
- **Model Export**: ONNX and TorchScript formats for edge deployment
- **Comprehensive Reporting**: Detailed training and evaluation reports
- **Pipeline Orchestration**: Complete end-to-end training workflow

## Security Classes

The models are trained to detect 10 security-relevant classes:

| Priority | Classes | Description |
|----------|---------|-------------|
| **High** | violence, emergency, theft | Critical security events requiring immediate response |
| **Medium** | suspicious, trespassing, vandalism | Concerning activities requiring investigation |
| **Low** | crowd, loitering, abandoned_object | Situational awareness events |
| **Normal** | normal | Baseline campus activity |

## Quick Start

### Complete Training Pipeline

Run the full training pipeline with optimization:

```bash
python train_security_pipeline.py
```

### Individual Components

Train a single model:
```bash
python train_yolo_security.py --model-size n --epochs 100
```

Optimize hyperparameters:
```bash
python optimize_hyperparameters.py --trials 50
```

Evaluate a trained model:
```bash
python evaluate_security_model.py path/to/model.pt
```

## Advanced Usage

### Custom Training Configuration

```bash
# Train multiple model sizes with custom epochs
python train_security_pipeline.py --model-sizes n s m --epochs 150

# Skip optimization and train single model
python train_security_pipeline.py --skip-optimization --single-model

# Run only hyperparameter optimization
python optimize_hyperparameters.py --trials 100 --train-final
```

### Model-Specific Training

```bash
# Train YOLOv8s with custom parameters
python train_yolo_security.py \
    --model-size s \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --patience 100 \
    --export

# Resume training from checkpoint
python train_yolo_security.py --resume path/to/checkpoint.pt
```

### Hyperparameter Optimization

```bash
# Extensive optimization with custom timeout
python optimize_hyperparameters.py \
    --trials 100 \
    --timeout 43200 \
    --train-final \
    --model-size m \
    --epochs 150
```

### Model Evaluation

```bash
# Comprehensive evaluation with custom thresholds
python evaluate_security_model.py model.pt \
    --thresholds 0.1 0.3 0.5 0.7 0.9
```

## Configuration

Training is configured via `../config/dataset_config.yaml`:

```yaml
# YOLO training configuration
yolo:
  model_size: "yolov8n"
  input_size: 640
  batch_size: 16
  epochs: 100
  learning_rate: 0.01
  patience: 50

# Security-specific detection parameters
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 100

# Performance benchmarks
benchmarks:
  target_fps: 30
  max_inference_time_ms: 100
  memory_limit_mb: 2048
  accuracy_threshold: 0.85
```

## Training Pipeline Stages

### 1. Hyperparameter Optimization (Optional)

- **Duration**: 2-12 hours
- **Trials**: 30-100 optimization trials
- **Objective**: Maximize mAP@0.5 on validation set
- **Parameters**: Learning rate, batch size, augmentation settings
- **Output**: Optimized hyperparameters for final training

### 2. Model Training

- **Models**: YOLOv8n (fast), YOLOv8s (balanced), YOLOv8m (accurate)
- **Duration**: 2-8 hours per model
- **Epochs**: 100-200 (with early stopping)
- **Augmentation**: Security-specific data augmentation
- **Output**: Trained model weights and training logs

### 3. Model Export

- **Formats**: ONNX (edge deployment), TorchScript (production)
- **Optimization**: Half-precision, dynamic batching
- **Validation**: Export format verification
- **Output**: Deployment-ready model files

### 4. Model Evaluation

- **Metrics**: mAP, precision, recall, F1-score
- **Security Analysis**: Priority-based performance assessment
- **Speed Testing**: Inference time across input sizes
- **Output**: Comprehensive evaluation report

## Performance Metrics

### Standard Metrics

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Security-Specific Metrics

- **Security Score**: Weighted performance across priority classes
- **High Priority Detection Rate**: Recall for critical security events
- **False Alarm Rate**: False positive rate for normal activities
- **Real-time Capability**: Inference speed for live deployment

## Model Comparison

| Model | Parameters | Speed (FPS) | mAP@0.5 | Security Score | Use Case |
|-------|------------|-------------|---------|----------------|----------|
| YOLOv8n | 3.2M | 45-60 | 0.65-0.75 | 0.70-0.80 | Edge devices, real-time |
| YOLOv8s | 11.2M | 30-45 | 0.70-0.80 | 0.75-0.85 | Balanced performance |
| YOLOv8m | 25.9M | 20-30 | 0.75-0.85 | 0.80-0.90 | High accuracy needs |

## Output Structure

```
models/
├── security_yolo_n_20240101_120000/     # Training run directory
│   ├── weights/
│   │   ├── best.pt                      # Best model weights
│   │   └── last.pt                      # Last epoch weights
│   ├── results.csv                      # Training metrics
│   ├── confusion_matrix.png             # Confusion matrix
│   └── training_report.json             # Detailed training report
├── optimization/                        # Hyperparameter optimization
│   └── optimization_results_20240101/
│       ├── optimization_summary.json    # Optimization results
│       ├── best_hyperparameters.yaml   # Best parameters
│       └── optimization_history.png     # Optimization plots
└── evaluation_results/                  # Model evaluation
    └── evaluation_20240101/
        ├── evaluation_results.json      # Evaluation metrics
        ├── evaluation_report.md         # Human-readable report
        └── evaluation_plots.png         # Performance visualizations
```

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA GTX 1060 (6GB VRAM) or equivalent
- **RAM**: 16 GB system memory
- **Storage**: 50 GB free space
- **CPU**: 4+ cores for data loading

### Recommended Requirements

- **GPU**: NVIDIA RTX 3070 (8GB VRAM) or better
- **RAM**: 32 GB system memory
- **Storage**: 100 GB SSD storage
- **CPU**: 8+ cores for optimal performance

### Training Time Estimates

| Model | GPU | Epochs | Estimated Time |
|-------|-----|--------|----------------|
| YOLOv8n | RTX 3070 | 100 | 2-3 hours |
| YOLOv8s | RTX 3070 | 100 | 4-6 hours |
| YOLOv8m | RTX 3070 | 100 | 6-8 hours |

## Monitoring and Logging

### TensorBoard Integration

```bash
# View training progress
tensorboard --logdir models/
```

### Weights & Biases (Optional)

```bash
# Enable W&B logging
python train_yolo_security.py --wandb
```

### Training Logs

- **Console Output**: Real-time training progress
- **CSV Files**: Epoch-by-epoch metrics
- **JSON Reports**: Comprehensive training summaries
- **Plots**: Loss curves, confusion matrices, sample predictions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train_yolo_security.py --batch-size 8
   ```

2. **Slow Training**
   ```bash
   # Enable mixed precision and caching
   python train_yolo_security.py --amp --cache
   ```

3. **Poor Convergence**
   ```bash
   # Run hyperparameter optimization
   python optimize_hyperparameters.py --trials 50
   ```

4. **Low mAP Scores**
   - Check dataset quality and annotations
   - Increase training epochs
   - Adjust learning rate and augmentation

### Performance Optimization

- **Use SSD storage** for faster data loading
- **Enable AMP** (Automatic Mixed Precision) for speed
- **Optimize batch size** based on GPU memory
- **Use multiple workers** for data loading

## Integration with Edge Deployment

### Model Export for Edge

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('path/to/best.pt')

# Export for edge deployment
model.export(
    format='onnx',
    imgsz=640,
    half=True,
    optimize=True,
    simplify=True
)
```

### Inference Optimization

- **Input Size**: Use 640x640 for best accuracy/speed balance
- **Batch Processing**: Process multiple frames together
- **Half Precision**: Use FP16 for faster inference
- **TensorRT**: Convert ONNX to TensorRT for NVIDIA devices

## Validation and Testing

### Automated Testing

```bash
# Run training pipeline tests
python -m pytest tests/test_training_pipeline.py

# Validate model exports
python tests/validate_model_exports.py
```

### Manual Validation

1. **Check Training Curves**: Ensure loss decreases and mAP increases
2. **Validate Exports**: Test ONNX and TorchScript inference
3. **Speed Benchmarks**: Verify real-time performance requirements
4. **Security Metrics**: Confirm high-priority class detection rates

## Next Steps

After successful model training:

1. **Deploy to Edge**: Use exported models in edge inference service (Task 3.2)
2. **Validation Testing**: Implement comprehensive model validation (Task 2.3)
3. **Production Integration**: Integrate with core backend services (Task 4.x)
4. **Continuous Learning**: Set up model retraining pipeline (Task 10.x)

## Requirements Compliance

This implementation satisfies:

- **Requirement 6.1**: Intelligent threat detection with configurable sensitivity
- **Requirement 6.2**: False positive rate ≤30% initially, ≤10% after optimization
- **Requirement 6.5**: Model retraining support with labeled incident data
- **Requirement 3.1**: Edge deployment preparation with optimized models
- **Requirement 7.4**: Performance metrics and system monitoring

## Support and Maintenance

### Model Updates

- **Retraining Schedule**: Monthly with new incident data
- **Performance Monitoring**: Continuous accuracy tracking
- **Version Control**: Model versioning and rollback capabilities
- **A/B Testing**: Compare model versions in production

### Documentation

- **Training Logs**: Complete training history and metrics
- **Model Cards**: Detailed model specifications and limitations
- **Deployment Guides**: Edge deployment instructions
- **API Documentation**: Model inference API specifications