# Model Evaluation and Validation Framework

This module provides comprehensive evaluation, validation, and benchmarking tools for campus security YOLO models. It includes security-specific metrics, robustness testing, and deployment readiness assessment.

## Overview

The evaluation framework ensures that trained YOLO models meet the stringent requirements for campus security applications. It provides multi-dimensional assessment covering accuracy, performance, robustness, and security-specific criteria.

## Components

### 1. Security Model Validator (`security_model_validator.py`)
Comprehensive validation framework with functional, performance, security, and robustness tests.

**Features:**
- Functional testing (model loading, inference, input validation)
- Performance testing (speed, memory, throughput, scalability)
- Security-specific testing (critical event detection, false positive analysis)
- Robustness testing (noise, lighting, resolution variations)
- Automated validation reporting

### 2. Enhanced Security Validator (`enhanced_security_validator.py`) **NEW**
Specialized validator focusing on campus security requirements 6.2 and 6.3.

**Features:**
- **Requirement 6.2 Compliance**: False positive rate analysis (≤30% initially, ≤10% improved)
- **Requirement 6.3 Compliance**: Environmental adaptation testing and threshold optimization
- Security scenario validation (intrusion, loitering, crowding, abandoned objects)
- Adaptive threshold optimization for different conditions
- Comprehensive compliance assessment and reporting

### 3. YOLO Variant Comparator (`yolo_variant_comparator.py`) **NEW**
Specialized framework for comparing different YOLO model variants (YOLOv8n, s, m, l, x).

**Features:**
- Multi-dimensional variant comparison (accuracy, performance, efficiency, security)
- Deployment suitability assessment (edge, server, mobile)
- Security-focused performance analysis
- Variant ranking and selection recommendations
- Comprehensive deployment guidance

### 4. Model Comparison Framework (`model_comparison_framework.py`)
Compare multiple YOLO models across different metrics and deployment scenarios.

**Features:**
- Multi-model accuracy comparison
- Performance benchmarking across models
- Efficiency analysis (accuracy vs speed vs size)
- Security performance comparison
- Deployment recommendations

### 5. Benchmark Suite (`benchmark_suite.py`)
Standardized benchmarking suite for consistent model evaluation.

**Features:**
- Comprehensive accuracy benchmarks
- Performance benchmarks (inference speed, memory usage)
- Stress testing (duration, memory, concurrent load)
- Robustness benchmarks (noise, lighting, blur)
- Security-specific benchmarks

## Quick Start

### Enhanced Security Validation (Requirements 6.2 & 6.3)

```bash
# Run enhanced security validation
python enhanced_security_validator.py path/to/model.pt

# Validate with custom config
python enhanced_security_validator.py path/to/model.pt --config custom_config.yaml
```

### YOLO Variant Comparison

```bash
# Compare YOLO variants
python yolo_variant_comparator.py \
    --models nano:models/yolov8n.pt small:models/yolov8s.pt medium:models/yolov8m.pt large:models/yolov8l.pt

# Compare with custom configuration
python yolo_variant_comparator.py \
    --config custom_config.yaml \
    --models nano:yolov8n.pt small:yolov8s.pt
```

### Single Model Validation

```bash
# Run comprehensive validation
python security_model_validator.py path/to/model.pt

# Validate with custom config
python security_model_validator.py path/to/model.pt --config custom_config.yaml
```

### Model Comparison

```bash
# Compare multiple models
python model_comparison_framework.py \
    --models nano:models/yolov8n.pt small:models/yolov8s.pt medium:models/yolov8m.pt
```

### Comprehensive Benchmarking

```bash
# Run full benchmark suite
python benchmark_suite.py path/to/model.pt

# Benchmark with custom configuration
python benchmark_suite.py path/to/model.pt --config benchmark_config.yaml
```

## Validation Categories

### Functional Tests
- **Model Loading**: Verify model loads correctly
- **Basic Inference**: Test single image inference
- **Batch Inference**: Test batch processing capability
- **Input Validation**: Test error handling for invalid inputs
- **Output Format**: Verify output format consistency

### Performance Tests
- **Inference Speed**: Measure inference time across input sizes
- **Memory Usage**: Monitor memory consumption and stability
- **Throughput**: Test processing capacity under load
- **Scalability**: Evaluate batch processing efficiency

### Security Tests
- **Critical Event Detection**: Validate detection of high-priority security events
- **False Positive Analysis**: Measure false alarm rates
- **Confidence Calibration**: Test confidence score reliability
- **Priority Class Performance**: Evaluate performance by security priority

### Robustness Tests
- **Noise Robustness**: Test performance with image noise
- **Lighting Conditions**: Evaluate under various lighting scenarios
- **Resolution Variations**: Test across different input resolutions
- **Stress Conditions**: Validate under high-load scenarios

## Security-Specific Metrics

### Priority-Based Performance
Models are evaluated based on security event priorities:

| Priority | Classes | Weight | Target Recall |
|----------|---------|--------|---------------|
| **Critical** | violence, emergency | 40% | ≥90% |
| **High** | theft, suspicious | 30% | ≥80% |
| **Medium** | trespassing, vandalism | 20% | ≥70% |
| **Low** | crowd, loitering, abandoned_object | 10% | ≥60% |

### Security Score Calculation
```
Security Score = 0.4 × Critical_Recall + 0.3 × High_Recall + 0.2 × Medium_Recall + 0.1 × Low_Recall
```

### False Positive Thresholds
- **Acceptable**: ≤30% false positive rate
- **Good**: ≤20% false positive rate  
- **Excellent**: ≤10% false positive rate

## Benchmark Standards

### Accuracy Benchmarks
- **mAP@0.5**: ≥70% for production deployment
- **mAP@0.5:0.95**: ≥50% for comprehensive evaluation
- **Precision**: ≥80% to minimize false alarms
- **Recall**: ≥70% to ensure threat detection

### Performance Benchmarks
- **Real-time Processing**: ≥15 FPS minimum, ≥30 FPS target
- **Inference Time**: ≤100ms per frame
- **Memory Usage**: ≤2GB for edge deployment
- **Model Size**: ≤50MB for edge, ≤25MB for mobile

### Robustness Standards
- **Noise Tolerance**: ≥70% detection retention with moderate noise
- **Lighting Adaptation**: Functional across daylight/artificial/low-light
- **Resolution Flexibility**: Support 320px to 832px input sizes

## Output Structure

```
validation_results/
├── validation_YYYYMMDD_HHMMSS/
│   ├── validation_results.json          # Complete validation data
│   ├── validation_report.md             # Human-readable report
│   └── validation_plots.png             # Performance visualizations
├── comparison_YYYYMMDD_HHMMSS/
│   ├── comparison_results.json          # Model comparison data
│   ├── comparison_report.md             # Comparison analysis
│   ├── model_comparison_plots.png       # Comparison charts
│   └── model_radar_chart.png           # Radar performance chart
└── benchmark_YYYYMMDD_HHMMSS/
    ├── benchmark_results.json           # Comprehensive benchmarks
    └── benchmark_report.md              # Benchmark analysis
```

## Validation Thresholds

### Production Readiness Criteria
A model is considered production-ready if it meets:
- **Overall Score**: ≥0.7 (70%)
- **Security Score**: ≥0.8 (80%)
- **Real-time Capability**: ≥15 FPS
- **Acceptable False Positive Rate**: ≤30%
- **Critical Event Detection**: ≥80% recall

### Deployment Suitability

#### Edge Deployment
- **Performance**: ≥15 FPS
- **Size**: ≤50 MB
- **Accuracy**: ≥60% mAP@0.5
- **Memory**: ≤2 GB usage

#### Server Deployment  
- **Accuracy**: ≥70% mAP@0.5
- **Precision**: ≥70%
- **Security Score**: ≥0.8

#### Mobile Deployment
- **Size**: ≤25 MB
- **Performance**: ≥10 FPS
- **Memory**: ≤1 GB usage

## Integration with Training Pipeline

### Automated Validation
```python
from security_model_validator import SecurityModelValidator

# Validate after training
validator = SecurityModelValidator('path/to/trained_model.pt')
results = validator.run_comprehensive_validation()

# Check if model meets production standards
if results['validation_summary']['overall_status'] in ['excellent', 'good']:
    print("Model ready for deployment")
else:
    print("Model needs improvement")
```

### Continuous Benchmarking
```python
from benchmark_suite import SecurityModelBenchmark

# Regular benchmarking
benchmark = SecurityModelBenchmark('production_model.pt')
results = benchmark.run_full_benchmark_suite()

# Monitor performance degradation
if results['benchmark_summary']['overall_score'] < 0.7:
    print("Model performance degraded - consider retraining")
```

## Custom Validation Configuration

Create custom validation thresholds:

```yaml
# custom_validation_config.yaml
validation_thresholds:
  min_accuracy: 0.75
  max_false_positive_rate: 0.25
  min_critical_recall: 0.85
  max_inference_time_ms: 80
  min_fps: 20

security_priorities:
  critical: ['violence', 'emergency', 'weapon']
  high: ['theft', 'suspicious', 'intrusion']
  medium: ['trespassing', 'vandalism']
  low: ['crowd', 'loitering']

benchmark_config:
  performance_tests:
    input_sizes: [416, 640, 832]
    test_iterations: 100
  stress_tests:
    duration_minutes: 15
    concurrent_threads: [2, 4, 8]
```

## Troubleshooting

### Common Issues

1. **Validation Failures**
   ```bash
   # Check model compatibility
   python -c "from ultralytics import YOLO; YOLO('model.pt').info()"
   ```

2. **Performance Issues**
   ```bash
   # Test with smaller input size
   python security_model_validator.py model.pt --input-size 416
   ```

3. **Memory Errors**
   ```bash
   # Reduce batch size for testing
   python benchmark_suite.py model.pt --max-batch-size 4
   ```

### Validation Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run validation with debug output
validator = SecurityModelValidator('model.pt')
results = validator.run_comprehensive_validation()
```

## Performance Optimization

### Speed Optimization
- Use smaller model variants (YOLOv8n vs YOLOv8m)
- Optimize input resolution (640px vs 832px)
- Enable half-precision inference
- Use TensorRT optimization for NVIDIA GPUs

### Memory Optimization
- Reduce batch sizes
- Use gradient checkpointing
- Clear cache between inferences
- Monitor memory leaks

### Accuracy Optimization
- Adjust confidence thresholds per class
- Implement ensemble methods
- Use test-time augmentation
- Fine-tune on domain-specific data

## Requirements Compliance

This evaluation framework satisfies:

- **Requirement 6.2**: False positive rate measurement and optimization (≤30% initially, ≤10% target)
  - Enhanced Security Validator provides comprehensive false positive analysis
  - Automatic threshold optimization for meeting FP rate targets
  - Compliance assessment with detailed reporting
- **Requirement 6.3**: Environmental adaptation and model comparison framework
  - Environmental condition testing (daylight, low-light, artificial light, etc.)
  - Adaptive threshold recommendations for different conditions
  - YOLO Variant Comparator for comprehensive model comparison
- **Requirement 1.1**: Real-time alert capability validation (≤5s response time)
- **Requirement 3.1**: Edge deployment readiness assessment
- **Requirement 7.4**: Performance metrics and system monitoring

## Integration Examples

### CI/CD Pipeline Integration
```yaml
# .github/workflows/model-validation.yml
- name: Validate Model
  run: |
    python ml-training/model-evaluation/security_model_validator.py \
      models/latest_model.pt \
      --config config/production_validation.yaml
    
    # Fail if validation doesn't meet standards
    if [ $? -ne 0 ]; then
      echo "Model validation failed"
      exit 1
    fi
```

### Production Monitoring
```python
# Monitor deployed model performance
import schedule
import time

def daily_benchmark():
    benchmark = SecurityModelBenchmark('production_model.pt')
    results = benchmark.run_full_benchmark_suite()
    
    # Alert if performance degrades
    if results['benchmark_summary']['overall_score'] < 0.7:
        send_alert("Model performance degraded")

schedule.every().day.at("02:00").do(daily_benchmark)
```

## Enhanced Validation Features

### Requirements-Focused Validation

The enhanced validation framework specifically addresses campus security requirements:

#### Requirement 6.2 - False Positive Rate Control
- **Initial Target**: ≤30% false positive rate validation
- **Improved Target**: ≤10% false positive rate optimization
- **Threshold Analysis**: Multi-threshold false positive analysis
- **Class-Specific FP**: Per-class false positive measurement
- **Optimization Recommendations**: Automated threshold optimization

#### Requirement 6.3 - Environmental Adaptation
- **Condition Testing**: Daylight, low-light, artificial light, night vision scenarios
- **Stability Analysis**: Performance consistency across conditions
- **Adaptive Thresholds**: Condition-specific threshold recommendations
- **Environmental Scoring**: Quantitative adaptation capability assessment

### YOLO Variant Analysis

Comprehensive comparison framework for YOLO model variants:

#### Deployment Suitability Assessment
- **Edge Deployment**: Real-time performance, size constraints, memory usage
- **Server Deployment**: High accuracy, throughput optimization
- **Mobile Deployment**: Size optimization, battery efficiency
- **Production Readiness**: Comprehensive deployment criteria

#### Multi-Dimensional Comparison
- **Accuracy Metrics**: mAP@0.5, precision, recall, F1-score
- **Performance Metrics**: FPS, inference time, memory usage, throughput
- **Efficiency Metrics**: Accuracy per parameter, FPS per MB, efficiency scores
- **Security Metrics**: Critical class performance, false positive rates
- **Deployment Metrics**: Size, parameters, suitability scores

### Enhanced Reporting

#### Compliance Reports
- **Requirement Compliance**: Detailed assessment against requirements 6.2 and 6.3
- **Validation Status**: Pass/fail determination with detailed reasoning
- **Critical Issues**: Identification of deployment blockers
- **Recommendations**: Actionable improvement suggestions

#### Deployment Guidance
- **Variant Selection**: Optimal YOLO variant for specific deployment scenarios
- **Threshold Configuration**: Recommended confidence thresholds per condition
- **Performance Expectations**: Expected accuracy, speed, and resource usage
- **Risk Assessment**: Potential issues and mitigation strategies

## Future Enhancements

- **Adversarial Testing**: Evaluate robustness against adversarial attacks
- **Fairness Evaluation**: Assess bias across different demographic groups
- **Explainability Analysis**: Generate model interpretation reports
- **A/B Testing Framework**: Compare model versions in production
- **Automated Retraining Triggers**: Initiate retraining based on performance metrics
- **Real-time Monitoring**: Continuous validation in production environments
- **Federated Validation**: Multi-site validation and comparison