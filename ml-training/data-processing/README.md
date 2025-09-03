# UCF Crime Dataset Processing for Campus Security

This module provides a complete pipeline for downloading, preprocessing, and preparing the UCF Crime Dataset for YOLO model training in campus security applications.

## Overview

The UCF Crime Dataset processing pipeline transforms raw video data into a YOLO-compatible format suitable for training security event detection models. The pipeline includes:

- Dataset download from Kaggle
- Video frame extraction and preprocessing
- Security-focused class mapping
- Balanced train/validation/test splits
- YOLO annotation generation
- Dataset validation and quality assurance

## Security Classes

The pipeline maps UCF Crime categories to campus security-focused classes:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | normal | Normal campus activity |
| 1 | suspicious | Suspicious behavior patterns |
| 2 | violence | Physical altercations, fights |
| 3 | theft | Stealing, pickpocketing |
| 4 | vandalism | Property damage, graffiti |
| 5 | trespassing | Unauthorized area access |
| 6 | crowd | Large gatherings, protests |
| 7 | abandoned_object | Unattended bags, packages |
| 8 | loitering | Prolonged presence in restricted areas |
| 9 | emergency | Medical emergencies, accidents |

## Usage

### Quick Start

Run the complete processing pipeline:

```bash
python process_ucf_dataset.py
```

### Advanced Options

```bash
# Skip dataset download (use existing data)
python process_ucf_dataset.py --skip-download

# Skip frame extraction (use existing frames)
python process_ucf_dataset.py --skip-extraction

# Limit frames per video (for faster processing)
python process_ucf_dataset.py --max-frames 25

# Only validate existing dataset
python process_ucf_dataset.py --validate-only

# Use custom data directory
python process_ucf_dataset.py --data-dir /path/to/data
```

### Individual Components

Run specific processing steps:

```bash
# Download and analyze dataset only
python download_dataset.py

# Validate processed dataset
python validate_dataset.py

# Test pipeline with mock data
python test_dataset_processing.py
```

## Prerequisites

### Python Dependencies

Install required packages:

```bash
pip install -r ../requirements.txt
```

Key dependencies:
- `kagglehub` - Dataset download
- `opencv-python` - Video processing
- `ultralytics` - YOLO framework
- `PyYAML` - Configuration files
- `numpy` - Numerical operations

### Kaggle API Setup

1. Create a Kaggle account and generate API credentials
2. Place `kaggle.json` in `~/.kaggle/` directory
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Output Structure

The pipeline creates the following directory structure:

```
data/
├── raw/
│   └── ucf-crime/                    # Raw downloaded dataset
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
└── processed/
    └── ucf-crime/
        ├── dataset_analysis.json     # Dataset structure analysis
        ├── frame_extraction_metadata.json  # Frame extraction details
        ├── extracted_frames/         # Individual video frames
        │   ├── normal_video1_frame_0001.jpg
        │   ├── violence_video2_frame_0001.jpg
        │   └── ...
        └── yolo_format/             # YOLO-compatible dataset
            ├── dataset.yaml         # YOLO dataset configuration
            ├── classes.txt          # Class definitions
            ├── split_statistics.json # Dataset split statistics
            ├── validation_report.json # Quality validation report
            ├── train/
            │   ├── images/          # Training images
            │   └── labels/          # Training annotations
            ├── val/
            │   ├── images/          # Validation images
            │   └── labels/          # Validation annotations
            └── test/
                ├── images/          # Test images
                └── labels/          # Test annotations
```

## Configuration

Dataset processing is configured via `../config/dataset_config.yaml`:

```yaml
# Frame extraction settings
frame_extraction:
  max_frames_per_video: 100
  frame_interval_strategy: "uniform"
  output_format: "jpg"
  resize_dimensions: [640, 640]

# Dataset splits
splits:
  train: 0.7
  validation: 0.2
  test: 0.1

# YOLO training parameters
yolo:
  model_size: "yolov8n"
  input_size: 640
  batch_size: 16
  epochs: 100
```

## Quality Assurance

The pipeline includes comprehensive validation:

### Automatic Validation

- Directory structure verification
- YOLO configuration validation
- Image-label pair consistency
- Annotation format compliance
- Class distribution analysis
- Image property validation

### Validation Report

Each processing run generates a detailed validation report (`validation_report.json`) containing:

- Overall validation status
- Individual test results
- Dataset statistics
- Class distribution
- Quality metrics

## Performance Considerations

### Processing Time

- **Full dataset**: 2-4 hours (depending on hardware)
- **Frame extraction**: Most time-consuming step
- **Annotation generation**: Fast (synthetic annotations)

### Storage Requirements

- **Raw dataset**: ~5-10 GB
- **Extracted frames**: ~2-5 GB (depends on max_frames setting)
- **YOLO dataset**: ~1-3 GB

### Memory Usage

- **Peak memory**: ~2-4 GB during frame extraction
- **Recommended RAM**: 8 GB minimum
- **GPU memory**: Not required for preprocessing

## Troubleshooting

### Common Issues

1. **Kaggle API Error**
   ```
   Solution: Verify kaggle.json credentials and permissions
   ```

2. **OpenCV Import Error**
   ```bash
   pip install opencv-python
   ```

3. **Memory Error During Processing**
   ```bash
   # Reduce frames per video
   python process_ucf_dataset.py --max-frames 25
   ```

4. **Validation Failures**
   ```bash
   # Check validation report for specific issues
   python validate_dataset.py
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with YOLO Training

The processed dataset is ready for YOLO training:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train on processed dataset
model.train(
    data='data/processed/ucf-crime/yolo_format/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## Testing

Run the test suite to verify pipeline functionality:

```bash
python test_dataset_processing.py
```

The test creates mock data and validates all processing steps without requiring the full dataset download.

## Next Steps

After successful dataset processing:

1. **Model Training**: Use the YOLO dataset for training (Task 2.2)
2. **Model Validation**: Implement performance benchmarking (Task 2.3)
3. **Edge Deployment**: Prepare models for edge inference (Task 3.2)

## Requirements Compliance

This implementation satisfies the following requirements:

- **Requirement 6.1**: Intelligent threat detection with configurable sensitivity
- **Requirement 6.2**: False positive rate optimization through proper training data
- **Requirement 3.1**: Edge computing preparation with optimized data formats
- **Requirement 5.1**: Integration readiness with existing camera infrastructure

## Support

For issues or questions:

1. Check the validation report for specific errors
2. Review the troubleshooting section
3. Run the test suite to verify installation
4. Check logs for detailed error information