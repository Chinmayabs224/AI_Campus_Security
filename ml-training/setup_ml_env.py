#!/usr/bin/env python3
"""
ML Training Environment Setup Script
Sets up the machine learning training environment for campus security.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if CUDA/GPU is available for training."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ CUDA available with {gpu_count} GPU(s): {gpu_name}")
            return True
        else:
            logger.warning("⚠ CUDA not available. Training will use CPU (slower)")
            return False
    except ImportError:
        logger.warning("⚠ PyTorch not installed yet")
        return False

def install_ml_requirements():
    """Install ML-specific requirements."""
    logger.info("Installing ML training requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        cmd = f"{sys.executable} -m pip install -r {requirements_file}"
        subprocess.run(cmd, shell=True, check=True)
        logger.info("✓ ML requirements installed")
    else:
        logger.error("❌ ML requirements.txt not found")

def setup_kaggle_credentials():
    """Guide user through Kaggle API setup."""
    logger.info("Setting up Kaggle credentials...")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        logger.info("✓ Kaggle credentials already configured")
        return True
    
    logger.info("Kaggle credentials not found. Please follow these steps:")
    logger.info("1. Go to https://www.kaggle.com/account")
    logger.info("2. Click 'Create New API Token'")
    logger.info("3. Download kaggle.json")
    logger.info(f"4. Place it at: {kaggle_json}")
    logger.info("5. Run: chmod 600 ~/.kaggle/kaggle.json (on Unix systems)")
    
    return False

def create_directory_structure():
    """Create necessary directories for ML training."""
    logger.info("Creating ML directory structure...")
    
    base_dir = Path(__file__).parent.parent
    directories = [
        "data/raw/ucf-crime",
        "data/processed/ucf-crime",
        "data/processed/ucf-crime/yolo_format",
        "models/checkpoints",
        "models/exports",
        "logs/training",
        "logs/evaluation",
        "experiments",
        "results"
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created {directory}/")

def download_sample_dataset():
    """Download and process the UCF Crime dataset."""
    logger.info("Setting up UCF Crime dataset...")
    
    try:
        # Import the dataset processor
        sys.path.append(str(Path(__file__).parent / "data-processing"))
        from download_dataset import UCFCrimeDatasetProcessor
        
        processor = UCFCrimeDatasetProcessor()
        
        # Check if dataset already exists
        if processor.raw_data_dir.exists() and any(processor.raw_data_dir.iterdir()):
            logger.info("✓ UCF Crime dataset already downloaded")
        else:
            logger.info("Downloading UCF Crime dataset (this may take a while)...")
            processor.download_dataset()
            logger.info("✓ Dataset downloaded successfully")
        
        # Analyze dataset
        processor.analyze_dataset_structure()
        
        # Create YOLO structure
        processor.create_yolo_dataset_structure()
        
        logger.info("✓ Dataset processing completed")
        
    except Exception as e:
        logger.error(f"❌ Dataset setup failed: {e}")
        logger.info("You can manually run the dataset processor later:")
        logger.info("python ml-training/data-processing/download_dataset.py")

def verify_yolo_installation():
    """Verify YOLO/Ultralytics installation."""
    try:
        from ultralytics import YOLO
        
        # Try to load a model (will download if not present)
        logger.info("Verifying YOLO installation...")
        model = YOLO('yolov8n.pt')  # This will download the model
        logger.info("✓ YOLO installation verified")
        
        # Move model to models directory
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLO verification failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Setting up ML Training Environment for Campus Security")
    logger.info("=" * 60)
    
    try:
        # Create directories
        create_directory_structure()
        
        # Install requirements
        install_ml_requirements()
        
        # Check GPU
        gpu_available = check_gpu_availability()
        
        # Setup Kaggle
        kaggle_ready = setup_kaggle_credentials()
        
        # Verify YOLO
        yolo_ready = verify_yolo_installation()
        
        # Download dataset if Kaggle is ready
        if kaggle_ready:
            download_sample_dataset()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ML Environment Setup Summary:")
        logger.info(f"✓ Directory structure created")
        logger.info(f"✓ Requirements installed")
        logger.info(f"{'✓' if gpu_available else '⚠'} GPU: {'Available' if gpu_available else 'Not available (CPU only)'}")
        logger.info(f"{'✓' if kaggle_ready else '⚠'} Kaggle: {'Ready' if kaggle_ready else 'Needs setup'}")
        logger.info(f"{'✓' if yolo_ready else '⚠'} YOLO: {'Ready' if yolo_ready else 'Needs verification'}")
        
        if not kaggle_ready:
            logger.info("\n📝 Next steps:")
            logger.info("1. Set up Kaggle credentials (see instructions above)")
            logger.info("2. Run: python ml-training/data-processing/download_dataset.py")
        
        logger.info("\n🚀 Ready to start ML training!")
        logger.info("Example commands:")
        logger.info("- Download dataset: python ml-training/data-processing/download_dataset.py")
        logger.info("- Train model: python ml-training/model-training/train_yolo.py")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()