#!/usr/bin/env python3
"""
Test script for UCF Crime Dataset processing without downloading large dataset.
Creates mock data to test the processing pipeline.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from download_dataset import UCFCrimeDatasetProcessor
from validate_dataset import DatasetValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_dataset(base_dir: Path) -> None:
    """Create a mock UCF Crime dataset for testing."""
    logger.info("Creating mock dataset for testing...")
    
    raw_dir = base_dir / "raw" / "ucf-crime"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock video files (empty files for testing)
    mock_videos = [
        "normal_walking_001.mp4",
        "violence_fighting_001.mp4", 
        "theft_stealing_001.mp4",
        "suspicious_behavior_001.mp4",
        "crowd_gathering_001.mp4"
    ]
    
    for video_name in mock_videos:
        video_path = raw_dir / video_name
        video_path.touch()  # Create empty file
    
    logger.info(f"Created {len(mock_videos)} mock video files")

def test_dataset_processing():
    """Test the dataset processing pipeline with mock data."""
    logger.info("Starting dataset processing test...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"Using temporary directory: {temp_path}")
        
        # Create mock dataset
        create_mock_dataset(temp_path)
        
        # Initialize processor with temp directory
        processor = UCFCrimeDatasetProcessor(base_data_dir=str(temp_path))
        
        try:
            # Test 1: Analyze dataset structure (skip download)
            logger.info("Test 1: Analyzing mock dataset structure...")
            analysis = processor.analyze_dataset_structure()
            
            assert analysis['total_files'] == 5, f"Expected 5 files, got {analysis['total_files']}"
            assert len(analysis['video_files']) == 5, f"Expected 5 video files, got {len(analysis['video_files'])}"
            logger.info("✓ Dataset analysis test passed")
            
            # Test 2: Create YOLO structure
            logger.info("Test 2: Creating YOLO dataset structure...")
            processor.create_yolo_dataset_structure()
            
            yolo_dir = processor.processed_data_dir / "yolo_format"
            assert yolo_dir.exists(), "YOLO directory not created"
            assert (yolo_dir / "dataset.yaml").exists(), "Dataset config not created"
            logger.info("✓ YOLO structure creation test passed")
            
            # Test 3: Create balanced splits (without frame extraction)
            logger.info("Test 3: Creating balanced dataset splits...")
            
            # Create some mock frames for testing splits
            frames_dir = processor.processed_data_dir / "extracted_frames"
            frames_dir.mkdir(exist_ok=True)
            
            mock_frames = [
                "normal_walking_001_frame_0001.jpg",
                "violence_fighting_001_frame_0001.jpg",
                "theft_stealing_001_frame_0001.jpg",
                "suspicious_behavior_001_frame_0001.jpg",
                "crowd_gathering_001_frame_0001.jpg"
            ]
            
            for frame_name in mock_frames:
                (frames_dir / frame_name).touch()
            
            # Create mock metadata
            import json
            metadata = {
                "extraction_params": {"max_frames_per_video": 1},
                "processed_videos": {},
                "class_distribution": {},
                "total_frames": 5
            }
            
            for i, frame_name in enumerate(mock_frames):
                video_name = frame_name.replace("_frame_0001.jpg", ".mp4")
                class_name = frame_name.split('_')[0]
                
                metadata["processed_videos"][video_name] = {
                    "class": class_name,
                    "frames": [{"filename": frame_name, "class": class_name}]
                }
                
                if class_name not in metadata["class_distribution"]:
                    metadata["class_distribution"][class_name] = 0
                metadata["class_distribution"][class_name] += 1
            
            metadata_file = processor.processed_data_dir / "frame_extraction_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Now test the splits
            processor.create_balanced_dataset_splits()
            
            # Verify splits were created
            for split in ['train', 'val', 'test']:
                images_dir = yolo_dir / split / 'images'
                labels_dir = yolo_dir / split / 'labels'
                assert images_dir.exists(), f"{split} images directory not created"
                assert labels_dir.exists(), f"{split} labels directory not created"
            
            logger.info("✓ Dataset splits test passed")
            
            # Test 4: Validation
            logger.info("Test 4: Running dataset validation...")
            validator = DatasetValidator(base_data_dir=str(temp_path))
            
            # Run individual validations
            assert validator.validate_directory_structure(), "Directory structure validation failed"
            assert validator.validate_dataset_config(), "Dataset config validation failed"
            
            logger.info("✓ Dataset validation test passed")
            
            logger.info("="*60)
            logger.info("ALL TESTS PASSED SUCCESSFULLY!")
            logger.info("Dataset processing pipeline is working correctly.")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False

def main():
    """Main test function."""
    try:
        success = test_dataset_processing()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)