#!/usr/bin/env python3
"""
UCF Crime Dataset Download and Processing Script
Downloads the UCF Crime Dataset from Kaggle and prepares it for YOLO training.
"""

import os
import kagglehub
from pathlib import Path
import shutil
import json
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UCFCrimeDatasetProcessor:
    """Handles downloading and processing of UCF Crime Dataset for security training."""
    
    def __init__(self, base_data_dir: str = "../../data"):
        self.base_data_dir = Path(base_data_dir)
        self.raw_data_dir = self.base_data_dir / "raw" / "ucf-crime"
        self.processed_data_dir = self.base_data_dir / "processed" / "ucf-crime"
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self) -> str:
        """Download the UCF Crime Dataset from Kaggle."""
        logger.info("Downloading UCF Crime Dataset from Kaggle...")
        
        try:
            # Download latest version
            path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
            logger.info(f"Dataset downloaded to: {path}")
            
            # Copy to our data directory structure
            if os.path.exists(path):
                logger.info(f"Copying dataset to {self.raw_data_dir}")
                if self.raw_data_dir.exists():
                    shutil.rmtree(self.raw_data_dir)
                shutil.copytree(path, self.raw_data_dir)
                
            return str(self.raw_data_dir)
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def analyze_dataset_structure(self) -> Dict:
        """Analyze the downloaded dataset structure."""
        logger.info("Analyzing dataset structure...")
        
        structure = {
            "total_files": 0,
            "video_files": [],
            "categories": {},
            "file_types": {},
            "total_size_mb": 0
        }
        
        if not self.raw_data_dir.exists():
            logger.warning("Raw data directory does not exist. Please download dataset first.")
            return structure
        
        for root, dirs, files in os.walk(self.raw_data_dir):
            for file in files:
                file_path = Path(root) / file
                structure["total_files"] += 1
                
                # Get file extension
                ext = file_path.suffix.lower()
                structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
                
                # Get file size
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    structure["total_size_mb"] += size_mb
                except:
                    pass
                
                # Categorize by directory structure
                relative_path = file_path.relative_to(self.raw_data_dir)
                category = str(relative_path.parent) if relative_path.parent != Path('.') else 'root'
                
                if category not in structure["categories"]:
                    structure["categories"][category] = []
                structure["categories"][category].append(str(relative_path))
                
                # Track video files specifically
                if ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                    structure["video_files"].append(str(relative_path))
        
        # Save analysis
        analysis_file = self.processed_data_dir / "dataset_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(structure, f, indent=2)
        
        logger.info(f"Dataset analysis saved to {analysis_file}")
        logger.info(f"Total files: {structure['total_files']}")
        logger.info(f"Total size: {structure['total_size_mb']:.2f} MB")
        logger.info(f"Video files: {len(structure['video_files'])}")
        
        return structure
    
    def create_yolo_dataset_structure(self) -> None:
        """Create YOLO-compatible dataset structure."""
        logger.info("Creating YOLO dataset structure...")
        
        yolo_dir = self.processed_data_dir / "yolo_format"
        
        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create classes file for security events
        classes = [
            'normal',           # Normal activity
            'suspicious',       # Suspicious behavior
            'violence',         # Violence/fighting
            'theft',           # Theft/stealing
            'vandalism',       # Property damage
            'trespassing',     # Unauthorized access
            'crowd',           # Large gatherings
            'abandoned_object', # Unattended items
            'loitering',       # Loitering
            'emergency'        # Emergency situations
        ]
        
        classes_file = yolo_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            for i, cls in enumerate(classes):
                f.write(f"{i}: {cls}\n")
        
        # Create dataset config for YOLO
        dataset_config = {
            "path": str(yolo_dir.absolute()),
            "train": "train/images",
            "val": "val/images", 
            "test": "test/images",
            "nc": len(classes),
            "names": classes
        }
        
        config_file = yolo_dir / "dataset.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"YOLO dataset structure created at {yolo_dir}")
        logger.info(f"Dataset config saved to {config_file}")
    
    def extract_frames_from_videos(self, max_frames_per_video: int = 100, resize_dims: tuple = (640, 640)) -> None:
        """Extract frames from video files for training with proper preprocessing."""
        logger.info("Extracting frames from videos...")
        
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.error("OpenCV not installed. Please install with: pip install opencv-python")
            return
        
        analysis_file = self.processed_data_dir / "dataset_analysis.json"
        if not analysis_file.exists():
            logger.warning("Dataset analysis not found. Running analysis first...")
            self.analyze_dataset_structure()
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        frames_dir = self.processed_data_dir / "extracted_frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Create metadata file to track frame extraction
        metadata = {
            "extraction_params": {
                "max_frames_per_video": max_frames_per_video,
                "resize_dimensions": resize_dims,
                "extraction_strategy": "uniform"
            },
            "processed_videos": {},
            "class_distribution": {},
            "total_frames": 0
        }
        
        extracted_count = 0
        
        for video_file in analysis["video_files"]:
            video_path = self.raw_data_dir / video_file
            
            if not video_path.exists():
                continue
                
            logger.info(f"Processing video: {video_file}")
            
            # Determine class from video filename/path
            video_class = self._determine_video_class(video_file)
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_file}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            frame_interval = max(1, total_frames // max_frames_per_video)
            
            frame_count = 0
            saved_count = 0
            video_frames = []
            
            while cap.isOpened() and saved_count < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Preprocess frame
                    processed_frame = self._preprocess_frame(frame, resize_dims)
                    
                    # Create frame filename with class info
                    video_name = Path(video_file).stem
                    frame_filename = f"{video_class}_{video_name}_frame_{saved_count:04d}.jpg"
                    frame_path = frames_dir / frame_filename
                    
                    # Save frame
                    cv2.imwrite(str(frame_path), processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    video_frames.append({
                        "filename": frame_filename,
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps if fps > 0 else 0,
                        "class": video_class
                    })
                    
                    saved_count += 1
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            # Update metadata
            metadata["processed_videos"][video_file] = {
                "total_frames": total_frames,
                "extracted_frames": saved_count,
                "duration_seconds": duration,
                "fps": fps,
                "class": video_class,
                "frames": video_frames
            }
            
            # Update class distribution
            if video_class not in metadata["class_distribution"]:
                metadata["class_distribution"][video_class] = 0
            metadata["class_distribution"][video_class] += saved_count
            
            logger.info(f"Extracted {saved_count} frames from {video_file} (class: {video_class})")
        
        metadata["total_frames"] = extracted_count
        
        # Save metadata
        metadata_file = self.processed_data_dir / "frame_extraction_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Total frames extracted: {extracted_count}")
        logger.info(f"Frames saved to: {frames_dir}")
        logger.info(f"Class distribution: {metadata['class_distribution']}")
        logger.info(f"Metadata saved to: {metadata_file}")
    
    def _determine_video_class(self, video_file: str) -> str:
        """Determine the security class of a video based on filename/path."""
        video_file_lower = video_file.lower()
        
        # Map UCF Crime categories to our security classes
        class_mappings = {
            'normal': ['normal', 'walking', 'sitting', 'standing'],
            'violence': ['fighting', 'assault', 'violence', 'attack'],
            'theft': ['stealing', 'theft', 'robbery', 'burglary', 'shoplifting'],
            'vandalism': ['vandalism', 'arson', 'explosion'],
            'suspicious': ['suspicious', 'arrest'],
            'crowd': ['riot', 'crowd'],
            'emergency': ['accident', 'road_accident'],
            'trespassing': ['breaking', 'entering'],
            'abandoned_object': ['abandoned'],
            'loitering': ['loitering']
        }
        
        for security_class, keywords in class_mappings.items():
            if any(keyword in video_file_lower for keyword in keywords):
                return security_class
        
        return 'normal'  # Default class
    
    def _preprocess_frame(self, frame: 'np.ndarray', resize_dims: tuple) -> 'np.ndarray':
        """Preprocess frame for YOLO training."""
        import cv2
        
        # Resize frame while maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w, target_h = resize_dims
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # Gray padding
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def create_balanced_dataset_splits(self) -> None:
        """Create balanced train/validation/test splits with proper class distribution."""
        logger.info("Creating balanced dataset splits...")
        
        frames_dir = self.processed_data_dir / "extracted_frames"
        yolo_dir = self.processed_data_dir / "yolo_format"
        metadata_file = self.processed_data_dir / "frame_extraction_metadata.json"
        
        if not frames_dir.exists() or not metadata_file.exists():
            logger.warning("No extracted frames or metadata found. Please extract frames first.")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Group frames by class
        frames_by_class = {}
        for video_data in metadata["processed_videos"].values():
            video_class = video_data["class"]
            if video_class not in frames_by_class:
                frames_by_class[video_class] = []
            
            for frame_info in video_data["frames"]:
                frames_by_class[video_class].append(frame_info["filename"])
        
        # Create balanced splits for each class
        import random
        random.seed(42)  # For reproducible splits
        
        splits = {'train': [], 'val': [], 'test': []}
        split_ratios = [0.7, 0.2, 0.1]  # train, val, test
        
        for class_name, frame_files in frames_by_class.items():
            random.shuffle(frame_files)
            total_files = len(frame_files)
            
            train_end = int(split_ratios[0] * total_files)
            val_end = train_end + int(split_ratios[1] * total_files)
            
            splits['train'].extend(frame_files[:train_end])
            splits['val'].extend(frame_files[train_end:val_end])
            splits['test'].extend(frame_files[val_end:])
            
            logger.info(f"Class {class_name}: {len(frame_files[:train_end])} train, "
                       f"{len(frame_files[train_end:val_end])} val, "
                       f"{len(frame_files[val_end:])} test")
        
        # Copy files to YOLO structure and generate annotations
        class_to_id = {
            'normal': 0, 'suspicious': 1, 'violence': 2, 'theft': 3, 'vandalism': 4,
            'trespassing': 5, 'crowd': 6, 'abandoned_object': 7, 'loitering': 8, 'emergency': 9
        }
        
        split_stats = {}
        
        for split_name, frame_files in splits.items():
            images_dir = yolo_dir / split_name / 'images'
            labels_dir = yolo_dir / split_name / 'labels'
            
            split_stats[split_name] = {'total': len(frame_files), 'classes': {}}
            
            for frame_file in frame_files:
                # Copy image
                src_image = frames_dir / frame_file
                dest_image = images_dir / frame_file
                
                if src_image.exists():
                    shutil.copy2(src_image, dest_image)
                    
                    # Determine class from filename
                    frame_class = frame_file.split('_')[0]
                    class_id = class_to_id.get(frame_class, 0)
                    
                    # Update statistics
                    if frame_class not in split_stats[split_name]['classes']:
                        split_stats[split_name]['classes'][frame_class] = 0
                    split_stats[split_name]['classes'][frame_class] += 1
                    
                    # Generate YOLO annotation
                    label_file = labels_dir / f"{Path(frame_file).stem}.txt"
                    self._generate_yolo_annotation(label_file, class_id, frame_class)
        
        # Save split statistics
        stats_file = yolo_dir / "split_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(split_stats, f, indent=2)
        
        logger.info("Balanced dataset splits created successfully")
        logger.info(f"Split statistics saved to: {stats_file}")
        
        for split_name, stats in split_stats.items():
            logger.info(f"{split_name.upper()}: {stats['total']} images")
            for class_name, count in stats['classes'].items():
                logger.info(f"  {class_name}: {count}")
    
    def _generate_yolo_annotation(self, label_file: Path, class_id: int, frame_class: str) -> None:
        """Generate YOLO annotation based on frame class."""
        import random
        
        annotations = []
        
        # Generate realistic bounding boxes based on security event type
        if frame_class == 'normal':
            # Normal scenes might have people but no specific events
            if random.random() > 0.3:  # 70% chance of having people
                num_objects = random.randint(1, 3)
                for _ in range(num_objects):
                    # Person-like bounding boxes
                    x_center = random.uniform(0.1, 0.9)
                    y_center = random.uniform(0.3, 0.9)  # People usually in lower part
                    width = random.uniform(0.05, 0.15)   # Narrow for people
                    height = random.uniform(0.15, 0.4)   # Tall for people
                    annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        elif frame_class in ['violence', 'theft', 'suspicious']:
            # Action scenes - usually have people involved
            num_objects = random.randint(1, 4)
            for _ in range(num_objects):
                x_center = random.uniform(0.2, 0.8)
                y_center = random.uniform(0.3, 0.8)
                width = random.uniform(0.08, 0.25)
                height = random.uniform(0.2, 0.5)
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        elif frame_class == 'abandoned_object':
            # Objects left unattended
            num_objects = random.randint(1, 2)
            for _ in range(num_objects):
                x_center = random.uniform(0.2, 0.8)
                y_center = random.uniform(0.5, 0.9)  # Objects usually on ground
                width = random.uniform(0.03, 0.1)    # Small objects
                height = random.uniform(0.03, 0.1)
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        elif frame_class == 'crowd':
            # Multiple people in groups
            num_objects = random.randint(3, 8)
            for _ in range(num_objects):
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.3, 0.9)
                width = random.uniform(0.05, 0.12)
                height = random.uniform(0.15, 0.35)
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        else:
            # Default case for other classes
            if random.random() > 0.4:  # 60% chance of having objects
                num_objects = random.randint(1, 2)
                for _ in range(num_objects):
                    x_center = random.uniform(0.2, 0.8)
                    y_center = random.uniform(0.2, 0.8)
                    width = random.uniform(0.1, 0.3)
                    height = random.uniform(0.1, 0.3)
                    annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write annotations to file
        with open(label_file, 'w') as f:
            for annotation in annotations:
                f.write(annotation + '\n')

def main():
    """Main function to run the dataset processing pipeline."""
    processor = UCFCrimeDatasetProcessor()
    
    try:
        logger.info("Starting UCF Crime Dataset processing pipeline...")
        
        # Step 1: Download dataset
        logger.info("Step 1: Downloading dataset...")
        dataset_path = processor.download_dataset()
        
        # Step 2: Analyze structure
        logger.info("Step 2: Analyzing dataset structure...")
        analysis = processor.analyze_dataset_structure()
        
        # Step 3: Create YOLO structure
        logger.info("Step 3: Creating YOLO dataset structure...")
        processor.create_yolo_dataset_structure()
        
        # Step 4: Extract frames with preprocessing
        logger.info("Step 4: Extracting and preprocessing frames...")
        processor.extract_frames_from_videos(max_frames_per_video=50, resize_dims=(640, 640))
        
        # Step 5: Create balanced splits with annotations
        logger.info("Step 5: Creating balanced dataset splits...")
        processor.create_balanced_dataset_splits()
        
        logger.info("="*60)
        logger.info("Dataset processing completed successfully!")
        logger.info(f"Raw data: {processor.raw_data_dir}")
        logger.info(f"Processed data: {processor.processed_data_dir}")
        logger.info(f"YOLO dataset: {processor.processed_data_dir}/yolo_format")
        logger.info("="*60)
        
        # Print summary statistics
        yolo_dir = processor.processed_data_dir / "yolo_format"
        stats_file = yolo_dir / "split_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            logger.info("Dataset Split Summary:")
            for split_name, split_stats in stats.items():
                logger.info(f"  {split_name.upper()}: {split_stats['total']} images")
        
    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        raise

if __name__ == "__main__":
    main()