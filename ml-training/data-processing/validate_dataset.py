#!/usr/bin/env python3
"""
Dataset Validation Script for UCF Crime Dataset
Validates the processed dataset for YOLO training compatibility.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validates the processed UCF Crime dataset for YOLO training."""
    
    def __init__(self, base_data_dir: str = "../../data"):
        self.base_data_dir = Path(base_data_dir)
        self.processed_data_dir = self.base_data_dir / "processed" / "ucf-crime"
        self.yolo_dir = self.processed_data_dir / "yolo_format"
        
    def validate_directory_structure(self) -> bool:
        """Validate that all required directories exist."""
        logger.info("Validating directory structure...")
        
        required_dirs = [
            self.yolo_dir,
            self.yolo_dir / "train" / "images",
            self.yolo_dir / "train" / "labels",
            self.yolo_dir / "val" / "images", 
            self.yolo_dir / "val" / "labels",
            self.yolo_dir / "test" / "images",
            self.yolo_dir / "test" / "labels"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            logger.error(f"Missing directories: {missing_dirs}")
            return False
        
        logger.info("✓ Directory structure is valid")
        return True
    
    def validate_dataset_config(self) -> bool:
        """Validate the YOLO dataset configuration file."""
        logger.info("Validating dataset configuration...")
        
        config_file = self.yolo_dir / "dataset.yaml"
        if not config_file.exists():
            logger.error("Dataset configuration file not found")
            return False
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            required_keys = ['path', 'train', 'val', 'test', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                logger.error(f"Missing configuration keys: {missing_keys}")
                return False
            
            # Validate class count
            if config['nc'] != len(config['names']):
                logger.error(f"Class count mismatch: nc={config['nc']}, names={len(config['names'])}")
                return False
            
            logger.info(f"✓ Dataset config valid: {config['nc']} classes")
            return True
            
        except Exception as e:
            logger.error(f"Error reading dataset config: {e}")
            return False
    
    def validate_image_label_pairs(self) -> Dict[str, Dict]:
        """Validate that each image has a corresponding label file."""
        logger.info("Validating image-label pairs...")
        
        results = {}
        
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_dir / split / "images"
            labels_dir = self.yolo_dir / split / "labels"
            
            image_files = set(f.stem for f in images_dir.glob("*.jpg"))
            label_files = set(f.stem for f in labels_dir.glob("*.txt"))
            
            missing_labels = image_files - label_files
            orphaned_labels = label_files - image_files
            
            results[split] = {
                'total_images': len(image_files),
                'total_labels': len(label_files),
                'missing_labels': len(missing_labels),
                'orphaned_labels': len(orphaned_labels),
                'valid_pairs': len(image_files & label_files)
            }
            
            if missing_labels:
                logger.warning(f"{split}: {len(missing_labels)} images missing labels")
            if orphaned_labels:
                logger.warning(f"{split}: {len(orphaned_labels)} orphaned labels")
            
            logger.info(f"✓ {split}: {results[split]['valid_pairs']} valid image-label pairs")
        
        return results
    
    def validate_annotation_format(self, sample_size: int = 10) -> bool:
        """Validate YOLO annotation format in label files."""
        logger.info("Validating annotation format...")
        
        valid_annotations = 0
        invalid_annotations = 0
        
        for split in ['train', 'val', 'test']:
            labels_dir = self.yolo_dir / split / "labels"
            label_files = list(labels_dir.glob("*.txt"))
            
            # Sample files for validation
            sample_files = label_files[:min(sample_size, len(label_files))]
            
            for label_file in sample_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            logger.error(f"Invalid annotation format in {label_file}:{line_num}")
                            invalid_annotations += 1
                            continue
                        
                        # Validate class ID
                        try:
                            class_id = int(parts[0])
                            if class_id < 0 or class_id > 9:  # 10 classes (0-9)
                                logger.error(f"Invalid class ID {class_id} in {label_file}:{line_num}")
                                invalid_annotations += 1
                                continue
                        except ValueError:
                            logger.error(f"Non-integer class ID in {label_file}:{line_num}")
                            invalid_annotations += 1
                            continue
                        
                        # Validate bounding box coordinates
                        try:
                            coords = [float(x) for x in parts[1:]]
                            if not all(0.0 <= coord <= 1.0 for coord in coords):
                                logger.error(f"Invalid coordinates in {label_file}:{line_num}")
                                invalid_annotations += 1
                                continue
                        except ValueError:
                            logger.error(f"Non-numeric coordinates in {label_file}:{line_num}")
                            invalid_annotations += 1
                            continue
                        
                        valid_annotations += 1
                
                except Exception as e:
                    logger.error(f"Error reading {label_file}: {e}")
                    invalid_annotations += 1
        
        total_annotations = valid_annotations + invalid_annotations
        if total_annotations > 0:
            success_rate = (valid_annotations / total_annotations) * 100
            logger.info(f"✓ Annotation validation: {success_rate:.1f}% valid ({valid_annotations}/{total_annotations})")
            return success_rate > 95  # 95% success rate threshold
        
        logger.warning("No annotations found to validate")
        return False
    
    def validate_class_distribution(self) -> Dict[str, Dict]:
        """Validate class distribution across splits."""
        logger.info("Validating class distribution...")
        
        class_names = [
            'normal', 'suspicious', 'violence', 'theft', 'vandalism',
            'trespassing', 'crowd', 'abandoned_object', 'loitering', 'emergency'
        ]
        
        distribution = {}
        
        for split in ['train', 'val', 'test']:
            labels_dir = self.yolo_dir / split / "labels"
            split_distribution = {class_name: 0 for class_name in class_names}
            
            for label_file in labels_dir.glob("*.txt"):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            if 0 <= class_id < len(class_names):
                                split_distribution[class_names[class_id]] += 1
                
                except Exception as e:
                    logger.warning(f"Error processing {label_file}: {e}")
            
            distribution[split] = split_distribution
            
            # Log distribution for this split
            total_objects = sum(split_distribution.values())
            logger.info(f"{split} split - Total objects: {total_objects}")
            for class_name, count in split_distribution.items():
                if count > 0:
                    percentage = (count / total_objects) * 100 if total_objects > 0 else 0
                    logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return distribution
    
    def validate_image_properties(self, sample_size: int = 5) -> bool:
        """Validate image properties (dimensions, format, etc.)."""
        logger.info("Validating image properties...")
        
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, skipping image validation")
            return True
        
        valid_images = 0
        invalid_images = 0
        
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_dir / split / "images"
            image_files = list(images_dir.glob("*.jpg"))
            
            # Sample files for validation
            sample_files = image_files[:min(sample_size, len(image_files))]
            
            for image_file in sample_files:
                try:
                    img = cv2.imread(str(image_file))
                    if img is None:
                        logger.error(f"Could not read image: {image_file}")
                        invalid_images += 1
                        continue
                    
                    height, width = img.shape[:2]
                    
                    # Check if image dimensions are reasonable
                    if width < 32 or height < 32:
                        logger.error(f"Image too small: {image_file} ({width}x{height})")
                        invalid_images += 1
                        continue
                    
                    # Check if image is square (YOLO training expectation)
                    if width != height:
                        logger.warning(f"Non-square image: {image_file} ({width}x{height})")
                    
                    valid_images += 1
                
                except Exception as e:
                    logger.error(f"Error validating image {image_file}: {e}")
                    invalid_images += 1
        
        total_images = valid_images + invalid_images
        if total_images > 0:
            success_rate = (valid_images / total_images) * 100
            logger.info(f"✓ Image validation: {success_rate:.1f}% valid ({valid_images}/{total_images})")
            return success_rate > 90
        
        return True
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'validation_results': {},
            'summary': {}
        }
        
        # Run all validations
        validations = [
            ('directory_structure', self.validate_directory_structure),
            ('dataset_config', self.validate_dataset_config),
            ('annotation_format', lambda: self.validate_annotation_format(sample_size=20)),
            ('image_properties', lambda: self.validate_image_properties(sample_size=10))
        ]
        
        passed_validations = 0
        
        for validation_name, validation_func in validations:
            try:
                result = validation_func()
                report['validation_results'][validation_name] = {
                    'passed': result,
                    'status': 'PASS' if result else 'FAIL'
                }
                if result:
                    passed_validations += 1
            except Exception as e:
                report['validation_results'][validation_name] = {
                    'passed': False,
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Add detailed results
        report['image_label_pairs'] = self.validate_image_label_pairs()
        report['class_distribution'] = self.validate_class_distribution()
        
        # Summary
        report['summary'] = {
            'total_validations': len(validations),
            'passed_validations': passed_validations,
            'success_rate': (passed_validations / len(validations)) * 100,
            'overall_status': 'PASS' if passed_validations == len(validations) else 'FAIL'
        }
        
        # Save report
        report_file = self.yolo_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to: {report_file}")
        logger.info(f"Overall validation status: {report['summary']['overall_status']}")
        logger.info(f"Success rate: {report['summary']['success_rate']:.1f}%")
        
        return report

def main():
    """Main function to run dataset validation."""
    validator = DatasetValidator()
    
    try:
        logger.info("Starting dataset validation...")
        report = validator.generate_validation_report()
        
        if report['summary']['overall_status'] == 'PASS':
            logger.info("✓ Dataset validation completed successfully!")
            logger.info("Dataset is ready for YOLO training.")
        else:
            logger.error("✗ Dataset validation failed!")
            logger.error("Please fix the issues before proceeding with training.")
            
            # Print failed validations
            for validation_name, result in report['validation_results'].items():
                if not result['passed']:
                    logger.error(f"  - {validation_name}: {result['status']}")
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()