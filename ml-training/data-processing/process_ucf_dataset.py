#!/usr/bin/env python3
"""
Complete UCF Crime Dataset Processing Pipeline
Orchestrates the entire dataset processing workflow for campus security training.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from download_dataset import UCFCrimeDatasetProcessor
from validate_dataset import DatasetValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description='Process UCF Crime Dataset for campus security')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip dataset download (use existing data)')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip frame extraction (use existing frames)')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames per video (default: 50)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing dataset')
    parser.add_argument('--data-dir', type=str, default='../../data',
                       help='Base data directory (default: ../../data)')
    
    args = parser.parse_args()
    
    try:
        logger.info("="*80)
        logger.info("UCF CRIME DATASET PROCESSING PIPELINE")
        logger.info("="*80)
        
        # Initialize processor and validator
        processor = UCFCrimeDatasetProcessor(base_data_dir=args.data_dir)
        validator = DatasetValidator(base_data_dir=args.data_dir)
        
        if args.validate_only:
            logger.info("Running validation only...")
            report = validator.generate_validation_report()
            
            if report['summary']['overall_status'] == 'PASS':
                logger.info("✓ Dataset validation PASSED!")
                return 0
            else:
                logger.error("✗ Dataset validation FAILED!")
                return 1
        
        # Step 1: Download dataset (if not skipped)
        if not args.skip_download:
            logger.info("STEP 1: Downloading UCF Crime Dataset...")
            try:
                dataset_path = processor.download_dataset()
                logger.info(f"✓ Dataset downloaded to: {dataset_path}")
            except Exception as e:
                logger.error(f"✗ Download failed: {e}")
                return 1
        else:
            logger.info("STEP 1: Skipping download (using existing data)")
        
        # Step 2: Analyze dataset structure
        logger.info("STEP 2: Analyzing dataset structure...")
        try:
            analysis = processor.analyze_dataset_structure()
            logger.info(f"✓ Found {analysis['total_files']} files, {len(analysis['video_files'])} videos")
        except Exception as e:
            logger.error(f"✗ Analysis failed: {e}")
            return 1
        
        # Step 3: Create YOLO structure
        logger.info("STEP 3: Creating YOLO dataset structure...")
        try:
            processor.create_yolo_dataset_structure()
            logger.info("✓ YOLO structure created")
        except Exception as e:
            logger.error(f"✗ YOLO structure creation failed: {e}")
            return 1
        
        # Step 4: Extract frames (if not skipped)
        if not args.skip_extraction:
            logger.info(f"STEP 4: Extracting frames (max {args.max_frames} per video)...")
            try:
                processor.extract_frames_from_videos(
                    max_frames_per_video=args.max_frames,
                    resize_dims=(640, 640)
                )
                logger.info("✓ Frame extraction completed")
            except Exception as e:
                logger.error(f"✗ Frame extraction failed: {e}")
                return 1
        else:
            logger.info("STEP 4: Skipping frame extraction (using existing frames)")
        
        # Step 5: Create balanced splits
        logger.info("STEP 5: Creating balanced dataset splits...")
        try:
            processor.create_balanced_dataset_splits()
            logger.info("✓ Dataset splits created")
        except Exception as e:
            logger.error(f"✗ Dataset splitting failed: {e}")
            return 1
        
        # Step 6: Validate dataset
        logger.info("STEP 6: Validating processed dataset...")
        try:
            report = validator.generate_validation_report()
            
            if report['summary']['overall_status'] == 'PASS':
                logger.info("✓ Dataset validation PASSED!")
            else:
                logger.warning("⚠ Dataset validation had issues - check validation report")
                
                # Print summary of issues
                for validation_name, result in report['validation_results'].items():
                    if not result['passed']:
                        logger.warning(f"  - {validation_name}: {result['status']}")
        
        except Exception as e:
            logger.error(f"✗ Validation failed: {e}")
            return 1
        
        # Final summary
        logger.info("="*80)
        logger.info("PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Print dataset statistics
        yolo_dir = processor.processed_data_dir / "yolo_format"
        stats_file = yolo_dir / "split_statistics.json"
        
        if stats_file.exists():
            import json
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            logger.info("Dataset Summary:")
            total_images = sum(split_stats['total'] for split_stats in stats.values())
            logger.info(f"  Total Images: {total_images}")
            
            for split_name, split_stats in stats.items():
                logger.info(f"  {split_name.upper()}: {split_stats['total']} images")
                
                # Show class distribution for training set
                if split_name == 'train' and 'classes' in split_stats:
                    logger.info("  Training Class Distribution:")
                    for class_name, count in split_stats['classes'].items():
                        percentage = (count / split_stats['total']) * 100
                        logger.info(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        logger.info(f"Dataset Location: {yolo_dir}")
        logger.info("Ready for YOLO model training!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)