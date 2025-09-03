"""
Automated Labeling Workflow for Training Data
Provides semi-automated labeling with human validation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import cv2
import numpy as np
from dataclasses import dataclass, asdict
import yaml
from ultralytics import YOLO
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import hashlib

@dataclass
class LabelingTask:
    """Structure for labeling tasks"""
    task_id: str
    incident_id: str
    frame_path: str
    predicted_labels: List[Dict]
    confidence_scores: List[float]
    status: str = "pending"  # pending, labeled, validated, rejected
    human_labels: Optional[List[Dict]] = None
    validation_notes: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None

@dataclass
class AutoLabel:
    """Structure for automated labels"""
    class_id: int
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2] normalized
    confidence: float
    source: str  # "model", "clustering", "temporal"

class AutomatedLabelingWorkflow:
    """Automated labeling workflow with human validation"""
    
    def __init__(self, config_path: str = "config/labeling_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize paths
        self.labeling_dir = Path(self.config['labeling']['output_dir'])
        self.labeling_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.current_model = None
        self.ensemble_models = []
        
        # Initialize database for task tracking
        self.db_path = self.labeling_dir / "labeling_tasks.db"
        self._init_database()
        
        # Load class mappings
        self.class_mapping = self.config['classes']
        self.security_classes = self.config['security_priority_classes']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load labeling configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'labeling': {
                    'output_dir': '../../data/labeling',
                    'confidence_threshold': 0.25,
                    'nms_threshold': 0.45,
                    'ensemble_agreement_threshold': 0.6,
                    'temporal_consistency_frames': 5,
                    'clustering_eps': 0.3,
                    'min_samples': 3
                },
                'classes': {
                    0: 'person',
                    1: 'suspicious_activity',
                    2: 'violence',
                    3: 'theft',
                    4: 'abandoned_object',
                    5: 'crowding',
                    6: 'loitering'
                },
                'security_priority_classes': [1, 2, 3, 4],
                'models': {
                    'primary_model': '../models/best_security_model.pt',
                    'ensemble_models': []
                },
                'validation': {
                    'require_human_validation': True,
                    'auto_approve_threshold': 0.9,
                    'batch_size': 50
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for labeling workflow"""
        logger = logging.getLogger('automated_labeling')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for task tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS labeling_tasks (
                    task_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    frame_path TEXT NOT NULL,
                    predicted_labels TEXT,
                    confidence_scores TEXT,
                    status TEXT DEFAULT 'pending',
                    human_labels TEXT,
                    validation_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS label_statistics (
                    date TEXT PRIMARY KEY,
                    total_tasks INTEGER,
                    completed_tasks INTEGER,
                    auto_approved INTEGER,
                    human_validated INTEGER,
                    rejected INTEGER,
                    avg_confidence REAL
                )
            ''')
    
    async def load_models(self):
        """Load YOLO models for ensemble labeling"""
        try:
            # Load primary model
            primary_model_path = self.config['models']['primary_model']
            if Path(primary_model_path).exists():
                self.current_model = YOLO(primary_model_path)
                self.logger.info(f"Loaded primary model: {primary_model_path}")
            else:
                # Use pretrained model as fallback
                self.current_model = YOLO('yolov8n.pt')
                self.logger.warning("Primary model not found, using pretrained YOLOv8n")
            
            # Load ensemble models
            for model_path in self.config['models'].get('ensemble_models', []):
                if Path(model_path).exists():
                    model = YOLO(model_path)
                    self.ensemble_models.append(model)
                    self.logger.info(f"Loaded ensemble model: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    async def create_labeling_tasks(self, incident_data_list: List[Dict]) -> List[LabelingTask]:
        """Create labeling tasks from incident data"""
        tasks = []
        
        for incident_data in incident_data_list:
            try:
                # Extract frames if not already done
                frame_paths = await self._get_incident_frames(incident_data)
                
                for frame_path in frame_paths:
                    # Generate predictions
                    predicted_labels = await self._generate_predictions(frame_path)
                    
                    # Create task
                    task_id = self._generate_task_id(incident_data['incident_id'], frame_path)
                    task = LabelingTask(
                        task_id=task_id,
                        incident_id=incident_data['incident_id'],
                        frame_path=frame_path,
                        predicted_labels=predicted_labels,
                        confidence_scores=[label['confidence'] for label in predicted_labels],
                        created_at=datetime.now()
                    )
                    
                    tasks.append(task)
                    
                    # Save to database
                    await self._save_task_to_db(task)
                    
            except Exception as e:
                self.logger.error(f"Error creating labeling task for incident {incident_data.get('incident_id', 'unknown')}: {e}")
        
        self.logger.info(f"Created {len(tasks)} labeling tasks")
        return tasks
    
    async def _get_incident_frames(self, incident_data: Dict) -> List[str]:
        """Get frame paths for incident"""
        video_path = Path(incident_data['video_path'])
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        
        if frames_dir.exists():
            return [str(f) for f in frames_dir.glob("*.jpg")]
        else:
            # Extract frames if not already done
            return await self._extract_frames(str(video_path))
    
    async def _extract_frames(self, video_path: str) -> List[str]:
        """Extract frames from video"""
        video_path = Path(video_path)
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_paths = []
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, frame_count // 20)  # Extract ~20 frames per video
            
            for frame_idx in range(0, frame_count, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_filename = f"frame_{frame_idx:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
        
        finally:
            cap.release()
        
        return frame_paths
    
    async def _generate_predictions(self, frame_path: str) -> List[Dict]:
        """Generate predictions using ensemble of models"""
        predictions = []
        
        try:
            # Primary model predictions
            primary_preds = await self._predict_with_model(self.current_model, frame_path)
            
            # Ensemble predictions
            ensemble_preds = []
            for model in self.ensemble_models:
                preds = await self._predict_with_model(model, frame_path)
                ensemble_preds.append(preds)
            
            # Combine predictions using ensemble voting
            if ensemble_preds:
                predictions = await self._ensemble_predictions(primary_preds, ensemble_preds)
            else:
                predictions = primary_preds
            
            # Apply temporal consistency if available
            predictions = await self._apply_temporal_consistency(frame_path, predictions)
            
            # Apply clustering-based refinement
            predictions = await self._apply_clustering_refinement(frame_path, predictions)
            
        except Exception as e:
            self.logger.error(f"Error generating predictions for {frame_path}: {e}")
        
        return predictions
    
    async def _predict_with_model(self, model: YOLO, frame_path: str) -> List[Dict]:
        """Generate predictions with a single model"""
        try:
            results = model(frame_path, conf=self.config['labeling']['confidence_threshold'])
            predictions = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxyn[i].cpu().numpy()  # Normalized coordinates
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        predictions.append({
                            'class_id': cls_id,
                            'class_name': self.class_mapping.get(cls_id, f'class_{cls_id}'),
                            'bbox': bbox.tolist(),
                            'confidence': conf,
                            'source': 'model'
                        })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Model prediction error: {e}")
            return []
    
    async def _ensemble_predictions(self, primary_preds: List[Dict], ensemble_preds: List[List[Dict]]) -> List[Dict]:
        """Combine predictions using ensemble voting"""
        if not ensemble_preds:
            return primary_preds
        
        # Simple ensemble: average confidence scores for overlapping detections
        all_predictions = [primary_preds] + ensemble_preds
        combined_predictions = []
        
        # Group predictions by spatial overlap
        for pred in primary_preds:
            overlapping_preds = [pred]
            
            for ensemble_pred_list in ensemble_preds:
                for ensemble_pred in ensemble_pred_list:
                    if (pred['class_id'] == ensemble_pred['class_id'] and 
                        self._calculate_iou(pred['bbox'], ensemble_pred['bbox']) > 0.5):
                        overlapping_preds.append(ensemble_pred)
            
            if len(overlapping_preds) >= len(all_predictions) * self.config['labeling']['ensemble_agreement_threshold']:
                # Average confidence scores
                avg_confidence = np.mean([p['confidence'] for p in overlapping_preds])
                
                combined_pred = pred.copy()
                combined_pred['confidence'] = avg_confidence
                combined_pred['source'] = 'ensemble'
                combined_predictions.append(combined_pred)
        
        return combined_predictions
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    async def _apply_temporal_consistency(self, frame_path: str, predictions: List[Dict]) -> List[Dict]:
        """Apply temporal consistency checks"""
        # For now, return predictions as-is
        # In full implementation, would check consistency across video frames
        return predictions
    
    async def _apply_clustering_refinement(self, frame_path: str, predictions: List[Dict]) -> List[Dict]:
        """Apply clustering-based refinement"""
        if len(predictions) < 2:
            return predictions
        
        try:
            # Extract features for clustering (bbox centers and sizes)
            features = []
            for pred in predictions:
                bbox = pred['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                features.append([center_x, center_y, width, height, pred['confidence']])
            
            # Apply DBSCAN clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            clustering = DBSCAN(
                eps=self.config['labeling']['clustering_eps'],
                min_samples=self.config['labeling']['min_samples']
            )
            cluster_labels = clustering.fit_predict(features_scaled)
            
            # Filter out noise points and low-confidence clusters
            refined_predictions = []
            for i, pred in enumerate(predictions):
                if cluster_labels[i] != -1:  # Not noise
                    refined_predictions.append(pred)
            
            return refined_predictions
            
        except Exception as e:
            self.logger.error(f"Clustering refinement error: {e}")
            return predictions
    
    def _generate_task_id(self, incident_id: str, frame_path: str) -> str:
        """Generate unique task ID"""
        content = f"{incident_id}_{frame_path}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def _save_task_to_db(self, task: LabelingTask):
        """Save labeling task to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO labeling_tasks 
                (task_id, incident_id, frame_path, predicted_labels, confidence_scores, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id,
                task.incident_id,
                task.frame_path,
                json.dumps(task.predicted_labels),
                json.dumps(task.confidence_scores),
                task.status,
                task.created_at.isoformat()
            ))
    
    async def process_labeling_batch(self, batch_size: int = None) -> Dict:
        """Process a batch of labeling tasks"""
        if batch_size is None:
            batch_size = self.config['validation']['batch_size']
        
        # Get pending tasks
        pending_tasks = await self._get_pending_tasks(batch_size)
        
        if not pending_tasks:
            self.logger.info("No pending labeling tasks")
            return {'processed': 0, 'auto_approved': 0, 'requires_validation': 0}
        
        auto_approved = 0
        requires_validation = 0
        
        for task in pending_tasks:
            try:
                # Check if task can be auto-approved
                if await self._can_auto_approve(task):
                    await self._auto_approve_task(task)
                    auto_approved += 1
                else:
                    await self._mark_for_human_validation(task)
                    requires_validation += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing task {task.task_id}: {e}")
        
        # Update statistics
        await self._update_statistics(len(pending_tasks), auto_approved, requires_validation)
        
        return {
            'processed': len(pending_tasks),
            'auto_approved': auto_approved,
            'requires_validation': requires_validation
        }
    
    async def _get_pending_tasks(self, limit: int) -> List[LabelingTask]:
        """Get pending labeling tasks from database"""
        tasks = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT task_id, incident_id, frame_path, predicted_labels, confidence_scores, created_at
                FROM labeling_tasks 
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
            ''', (limit,))
            
            for row in cursor.fetchall():
                task = LabelingTask(
                    task_id=row[0],
                    incident_id=row[1],
                    frame_path=row[2],
                    predicted_labels=json.loads(row[3]),
                    confidence_scores=json.loads(row[4]),
                    created_at=datetime.fromisoformat(row[5])
                )
                tasks.append(task)
        
        return tasks
    
    async def _can_auto_approve(self, task: LabelingTask) -> bool:
        """Check if task can be auto-approved"""
        if not self.config['validation']['require_human_validation']:
            return True
        
        # Auto-approve if all predictions have high confidence
        auto_approve_threshold = self.config['validation']['auto_approve_threshold']
        
        if not task.confidence_scores:
            return False
        
        # Check if all confidence scores are above threshold
        high_confidence = all(conf >= auto_approve_threshold for conf in task.confidence_scores)
        
        # Additional checks for security-critical classes
        has_security_class = any(
            pred['class_id'] in self.security_classes 
            for pred in task.predicted_labels
        )
        
        # Require human validation for security-critical detections
        if has_security_class:
            return False
        
        return high_confidence
    
    async def _auto_approve_task(self, task: LabelingTask):
        """Auto-approve a labeling task"""
        task.status = "auto_approved"
        task.completed_at = datetime.now()
        
        # Convert predictions to YOLO format and save
        await self._save_yolo_labels(task)
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE labeling_tasks 
                SET status = ?, completed_at = ?
                WHERE task_id = ?
            ''', (task.status, task.completed_at.isoformat(), task.task_id))
    
    async def _mark_for_human_validation(self, task: LabelingTask):
        """Mark task for human validation"""
        task.status = "requires_validation"
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE labeling_tasks 
                SET status = ?
                WHERE task_id = ?
            ''', (task.status, task.task_id))
    
    async def _save_yolo_labels(self, task: LabelingTask):
        """Save labels in YOLO format"""
        frame_path = Path(task.frame_path)
        label_path = frame_path.parent / f"{frame_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for pred in task.predicted_labels:
                bbox = pred['bbox']
                # Convert to YOLO format: class_id center_x center_y width height
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                f.write(f"{pred['class_id']} {center_x} {center_y} {width} {height}\n")
    
    async def _update_statistics(self, processed: int, auto_approved: int, requires_validation: int):
        """Update labeling statistics"""
        today = datetime.now().date().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get existing stats
            cursor = conn.execute('SELECT * FROM label_statistics WHERE date = ?', (today,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                conn.execute('''
                    UPDATE label_statistics 
                    SET total_tasks = total_tasks + ?, 
                        completed_tasks = completed_tasks + ?,
                        auto_approved = auto_approved + ?
                    WHERE date = ?
                ''', (processed, auto_approved, auto_approved, today))
            else:
                # Create new record
                conn.execute('''
                    INSERT INTO label_statistics 
                    (date, total_tasks, completed_tasks, auto_approved, human_validated, rejected)
                    VALUES (?, ?, ?, ?, 0, 0)
                ''', (today, processed, auto_approved, auto_approved))
    
    async def get_validation_queue(self, limit: int = 50) -> List[LabelingTask]:
        """Get tasks requiring human validation"""
        tasks = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT task_id, incident_id, frame_path, predicted_labels, confidence_scores, created_at
                FROM labeling_tasks 
                WHERE status = 'requires_validation'
                ORDER BY created_at ASC
                LIMIT ?
            ''', (limit,))
            
            for row in cursor.fetchall():
                task = LabelingTask(
                    task_id=row[0],
                    incident_id=row[1],
                    frame_path=row[2],
                    predicted_labels=json.loads(row[3]),
                    confidence_scores=json.loads(row[4]),
                    created_at=datetime.fromisoformat(row[5])
                )
                tasks.append(task)
        
        return tasks
    
    async def submit_human_validation(self, task_id: str, human_labels: List[Dict], validation_notes: str = ""):
        """Submit human validation for a task"""
        task_status = "validated" if human_labels else "rejected"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE labeling_tasks 
                SET status = ?, human_labels = ?, validation_notes = ?, completed_at = ?
                WHERE task_id = ?
            ''', (
                task_status,
                json.dumps(human_labels) if human_labels else None,
                validation_notes,
                datetime.now().isoformat(),
                task_id
            ))
        
        # If validated, save YOLO labels
        if human_labels:
            # Get task info
            cursor = conn.execute('SELECT frame_path FROM labeling_tasks WHERE task_id = ?', (task_id,))
            frame_path = cursor.fetchone()[0]
            
            # Create temporary task for label saving
            temp_task = LabelingTask(
                task_id=task_id,
                incident_id="",
                frame_path=frame_path,
                predicted_labels=human_labels,
                confidence_scores=[]
            )
            await self._save_yolo_labels(temp_task)
    
    async def get_labeling_statistics(self) -> Dict:
        """Get labeling workflow statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_tasks,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'auto_approved' THEN 1 ELSE 0 END) as auto_approved,
                    SUM(CASE WHEN status = 'validated' THEN 1 ELSE 0 END) as validated,
                    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN status = 'requires_validation' THEN 1 ELSE 0 END) as requires_validation
                FROM labeling_tasks
            ''')
            
            overall_stats = cursor.fetchone()
            
            # Daily statistics
            cursor = conn.execute('''
                SELECT date, total_tasks, completed_tasks, auto_approved, human_validated, rejected
                FROM label_statistics
                ORDER BY date DESC
                LIMIT 30
            ''')
            
            daily_stats = cursor.fetchall()
            
            return {
                'overall': {
                    'total_tasks': overall_stats[0],
                    'pending': overall_stats[1],
                    'auto_approved': overall_stats[2],
                    'validated': overall_stats[3],
                    'rejected': overall_stats[4],
                    'requires_validation': overall_stats[5]
                },
                'daily': [
                    {
                        'date': row[0],
                        'total_tasks': row[1],
                        'completed_tasks': row[2],
                        'auto_approved': row[3],
                        'human_validated': row[4],
                        'rejected': row[5]
                    }
                    for row in daily_stats
                ]
            }

async def main():
    """Example usage of automated labeling workflow"""
    workflow = AutomatedLabelingWorkflow()
    
    # Load models
    await workflow.load_models()
    
    # Example incident data
    incident_data = [
        {
            'incident_id': 'test_incident_1',
            'video_path': '../../data/retraining/suspicious/incident_1_20241203_143022.mp4'
        }
    ]
    
    # Create labeling tasks
    tasks = await workflow.create_labeling_tasks(incident_data)
    print(f"Created {len(tasks)} labeling tasks")
    
    # Process batch
    results = await workflow.process_labeling_batch(batch_size=10)
    print(f"Processed batch: {results}")
    
    # Get statistics
    stats = await workflow.get_labeling_statistics()
    print(f"Labeling statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(main())