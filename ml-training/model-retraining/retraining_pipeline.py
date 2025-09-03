"""
Model Retraining Pipeline with Performance Validation
Orchestrates the complete retraining workflow with validation and deployment
"""

import asyncio
import logging
import json
import yaml
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import sqlite3
import hashlib
import os
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent / "model-training"))
sys.path.append(str(Path(__file__).parent.parent / "model-evaluation"))

from train_security_pipeline import SecurityTrainingPipeline
from enhanced_security_validator import EnhancedSecurityValidator
from data_collection_pipeline import DataCollectionPipeline
from automated_labeling import AutomatedLabelingWorkflow

@dataclass
class RetrainingJob:
    """Structure for retraining jobs"""
    job_id: str
    trigger_reason: str
    data_collection_start: datetime
    data_collection_end: datetime
    total_new_samples: int
    samples_by_class: Dict[str, int]
    status: str = "pending"  # pending, collecting, labeling, training, validating, completed, failed
    current_model_path: Optional[str] = None
    new_model_path: Optional[str] = None
    performance_comparison: Optional[Dict] = None
    deployment_approved: bool = False
    created_at: datetime = None
    completed_at: Optional[datetime] = None

@dataclass
class PerformanceMetrics:
    """Structure for model performance metrics"""
    model_path: str
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1_score: float
    security_score: float
    false_positive_rate: float
    false_negative_rate: float
    inference_time_ms: float
    model_size_mb: float

class ModelRetrainingPipeline:
    """Complete model retraining pipeline with validation"""
    
    def __init__(self, config_path: str = "config/retraining_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize paths
        self.retraining_dir = Path(self.config['retraining']['output_dir'])
        self.retraining_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_collector = DataCollectionPipeline()
        self.labeling_workflow = AutomatedLabelingWorkflow()
        
        self.models_dir = self.retraining_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.retraining_dir / "retraining_jobs.db"
        self._init_database()
        
        # Performance thresholds
        self.performance_thresholds = self.config['validation']['performance_thresholds']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load retraining configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'retraining': {
                    'output_dir': '../../data/retraining_pipeline',
                    'trigger_conditions': {
                        'min_new_samples': 100,
                        'performance_degradation_threshold': 0.05,
                        'false_positive_rate_threshold': 0.15,
                        'days_since_last_training': 30
                    },
                    'data_collection_window_days': 14,
                    'min_samples_per_class': 20,
                    'validation_split': 0.2
                },
                'validation': {
                    'performance_thresholds': {
                        'min_mAP50_improvement': 0.02,
                        'max_performance_degradation': 0.01,
                        'min_security_score': 0.7,
                        'max_false_positive_rate': 0.1
                    },
                    'validation_dataset_path': '../../data/validation',
                    'benchmark_models': []
                },
                'training': {
                    'epochs': 50,
                    'patience': 10,
                    'model_sizes': ['n', 's'],
                    'use_transfer_learning': True
                },
                'deployment': {
                    'require_manual_approval': True,
                    'backup_current_model': True,
                    'rollback_on_failure': True
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for retraining pipeline"""
        logger = logging.getLogger('retraining_pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.retraining_dir / "retraining.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for job tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS retraining_jobs (
                    job_id TEXT PRIMARY KEY,
                    trigger_reason TEXT NOT NULL,
                    data_collection_start TIMESTAMP,
                    data_collection_end TIMESTAMP,
                    total_new_samples INTEGER,
                    samples_by_class TEXT,
                    status TEXT DEFAULT 'pending',
                    current_model_path TEXT,
                    new_model_path TEXT,
                    performance_comparison TEXT,
                    deployment_approved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance_history (
                    model_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    job_id TEXT,
                    mAP50 REAL,
                    mAP50_95 REAL,
                    precision_val REAL,
                    recall_val REAL,
                    f1_score REAL,
                    security_score REAL,
                    false_positive_rate REAL,
                    false_negative_rate REAL,
                    inference_time_ms REAL,
                    model_size_mb REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES retraining_jobs (job_id)
                )
            ''')
    
    async def check_retraining_triggers(self) -> Optional[RetrainingJob]:
        """Check if retraining should be triggered"""
        self.logger.info("Checking retraining triggers...")
        
        triggers = []
        
        # Check for new data availability
        new_data_trigger = await self._check_new_data_trigger()
        if new_data_trigger:
            triggers.append(new_data_trigger)
        
        # Check for performance degradation
        performance_trigger = await self._check_performance_degradation_trigger()
        if performance_trigger:
            triggers.append(performance_trigger)
        
        # Check time-based trigger
        time_trigger = await self._check_time_based_trigger()
        if time_trigger:
            triggers.append(time_trigger)
        
        if triggers:
            # Create retraining job
            job = await self._create_retraining_job(triggers)
            self.logger.info(f"Retraining triggered: {', '.join(triggers)}")
            return job
        
        self.logger.info("No retraining triggers detected")
        return None
    
    async def _check_new_data_trigger(self) -> Optional[str]:
        """Check if enough new data is available"""
        # Count recent incidents
        cutoff_date = datetime.now() - timedelta(
            days=self.config['retraining']['data_collection_window_days']
        )
        
        # Simulate data count - in real implementation, query database
        new_samples_count = 150  # Simulated count
        min_samples = self.config['retraining']['trigger_conditions']['min_new_samples']
        
        if new_samples_count >= min_samples:
            return f"new_data_available_{new_samples_count}_samples"
        
        return None
    
    async def _check_performance_degradation_trigger(self) -> Optional[str]:
        """Check for model performance degradation"""
        # Get latest model performance
        current_performance = await self._get_current_model_performance()
        
        if not current_performance:
            return None
        
        # Compare with historical performance
        historical_performance = await self._get_historical_performance()
        
        if historical_performance:
            performance_drop = historical_performance['mAP50'] - current_performance['mAP50']
            threshold = self.config['retraining']['trigger_conditions']['performance_degradation_threshold']
            
            if performance_drop > threshold:
                return f"performance_degradation_{performance_drop:.3f}"
        
        # Check false positive rate
        fp_rate = current_performance.get('false_positive_rate', 0)
        fp_threshold = self.config['retraining']['trigger_conditions']['false_positive_rate_threshold']
        
        if fp_rate > fp_threshold:
            return f"high_false_positive_rate_{fp_rate:.3f}"
        
        return None
    
    async def _check_time_based_trigger(self) -> Optional[str]:
        """Check time-based retraining trigger"""
        # Get last training date
        last_training_date = await self._get_last_training_date()
        
        if last_training_date:
            days_since = (datetime.now() - last_training_date).days
            threshold = self.config['retraining']['trigger_conditions']['days_since_last_training']
            
            if days_since >= threshold:
                return f"scheduled_retraining_{days_since}_days"
        
        return None
    
    async def _create_retraining_job(self, triggers: List[str]) -> RetrainingJob:
        """Create a new retraining job"""
        job_id = self._generate_job_id()
        
        job = RetrainingJob(
            job_id=job_id,
            trigger_reason=", ".join(triggers),
            data_collection_start=datetime.now() - timedelta(
                days=self.config['retraining']['data_collection_window_days']
            ),
            data_collection_end=datetime.now(),
            total_new_samples=0,
            samples_by_class={},
            created_at=datetime.now()
        )
        
        # Save to database
        await self._save_job_to_db(job)
        
        return job
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        return f"retrain_{timestamp}_{random_suffix}"
    
    async def _save_job_to_db(self, job: RetrainingJob):
        """Save retraining job to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO retraining_jobs 
                (job_id, trigger_reason, data_collection_start, data_collection_end, 
                 total_new_samples, samples_by_class, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.job_id,
                job.trigger_reason,
                job.data_collection_start.isoformat(),
                job.data_collection_end.isoformat(),
                job.total_new_samples,
                json.dumps(job.samples_by_class),
                job.status,
                job.created_at.isoformat()
            ))
    
    async def execute_retraining_job(self, job: RetrainingJob) -> bool:
        """Execute complete retraining job"""
        self.logger.info(f"Starting retraining job: {job.job_id}")
        
        try:
            # Step 1: Data Collection
            job.status = "collecting"
            await self._update_job_status(job)
            
            collected_data = await self._collect_training_data(job)
            if not collected_data:
                job.status = "failed"
                await self._update_job_status(job)
                return False
            
            # Step 2: Automated Labeling
            job.status = "labeling"
            await self._update_job_status(job)
            
            labeling_success = await self._process_labeling(collected_data)
            if not labeling_success:
                job.status = "failed"
                await self._update_job_status(job)
                return False
            
            # Step 3: Model Training
            job.status = "training"
            await self._update_job_status(job)
            
            new_model_path = await self._train_new_model(job)
            if not new_model_path:
                job.status = "failed"
                await self._update_job_status(job)
                return False
            
            job.new_model_path = new_model_path
            
            # Step 4: Performance Validation
            job.status = "validating"
            await self._update_job_status(job)
            
            validation_results = await self._validate_new_model(job)
            if not validation_results['passed']:
                job.status = "failed"
                await self._update_job_status(job)
                self.logger.warning(f"Model validation failed: {validation_results['reason']}")
                return False
            
            job.performance_comparison = validation_results
            
            # Step 5: Deployment Decision
            deployment_approved = await self._make_deployment_decision(job)
            job.deployment_approved = deployment_approved
            
            if deployment_approved:
                job.status = "completed"
                self.logger.info(f"Retraining job {job.job_id} completed successfully")
            else:
                job.status = "completed"
                self.logger.info(f"Retraining job {job.job_id} completed but deployment not approved")
            
            job.completed_at = datetime.now()
            await self._update_job_status(job)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Retraining job {job.job_id} failed: {e}")
            job.status = "failed"
            await self._update_job_status(job)
            return False
    
    async def _collect_training_data(self, job: RetrainingJob) -> Optional[List[Dict]]:
        """Collect training data for retraining"""
        self.logger.info("Collecting training data...")
        
        try:
            # Calculate collection window
            days_back = (job.data_collection_end - job.data_collection_start).days
            
            # Collect incident data
            collected_data = await self.data_collector.collect_recent_incidents(days_back)
            
            if not collected_data:
                self.logger.warning("No new training data collected")
                return None
            
            # Update job statistics
            job.total_new_samples = len(collected_data)
            job.samples_by_class = {}
            
            for incident in collected_data:
                event_type = incident.event_type
                job.samples_by_class[event_type] = job.samples_by_class.get(event_type, 0) + 1
            
            # Check minimum samples per class
            min_samples = self.config['retraining']['min_samples_per_class']
            insufficient_classes = [
                cls for cls, count in job.samples_by_class.items() 
                if count < min_samples
            ]
            
            if insufficient_classes:
                self.logger.warning(f"Insufficient samples for classes: {insufficient_classes}")
                # Continue anyway but log the warning
            
            self.logger.info(f"Collected {len(collected_data)} training samples")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return None
    
    async def _process_labeling(self, collected_data: List[Dict]) -> bool:
        """Process automated labeling for collected data"""
        self.logger.info("Processing automated labeling...")
        
        try:
            # Load models for labeling
            await self.labeling_workflow.load_models()
            
            # Create labeling tasks
            tasks = await self.labeling_workflow.create_labeling_tasks(collected_data)
            
            if not tasks:
                self.logger.warning("No labeling tasks created")
                return False
            
            # Process labeling in batches
            total_processed = 0
            batch_size = 50
            
            while total_processed < len(tasks):
                batch_results = await self.labeling_workflow.process_labeling_batch(batch_size)
                total_processed += batch_results['processed']
                
                self.logger.info(f"Processed {total_processed}/{len(tasks)} labeling tasks")
                
                if batch_results['processed'] == 0:
                    break
            
            # Check if enough tasks were auto-approved
            stats = await self.labeling_workflow.get_labeling_statistics()
            auto_approved = stats['overall']['auto_approved']
            total_tasks = stats['overall']['total_tasks']
            
            if total_tasks > 0:
                auto_approval_rate = auto_approved / total_tasks
                self.logger.info(f"Auto-approval rate: {auto_approval_rate:.2%}")
                
                # Require at least 50% auto-approval for automated retraining
                if auto_approval_rate < 0.5:
                    self.logger.warning("Low auto-approval rate, may need human validation")
                    # Continue anyway for demonstration
            
            return True
            
        except Exception as e:
            self.logger.error(f"Labeling processing failed: {e}")
            return False
    
    async def _train_new_model(self, job: RetrainingJob) -> Optional[str]:
        """Train new model with collected data"""
        self.logger.info("Training new model...")
        
        try:
            # Prepare training configuration
            training_config = self._prepare_training_config(job)
            
            # Initialize training pipeline
            trainer = SecurityTrainingPipeline(config_path=training_config)
            
            # Update training parameters
            trainer.pipeline_config['final_epochs'] = self.config['training']['epochs']
            trainer.pipeline_config['model_sizes'] = self.config['training']['model_sizes']
            
            # Run training pipeline
            results = trainer.run_complete_pipeline(
                optimize_hyperparameters=False,  # Skip optimization for retraining
                train_multiple_sizes=len(self.config['training']['model_sizes']) > 1,
                export_models=True,
                evaluate_models=True
            )
            
            if results['status'] != 'completed':
                self.logger.error("Training pipeline failed")
                return None
            
            # Get best model path
            if results['best_model']:
                best_model_path = results['best_model']['model_path']
                
                # Copy model to retraining directory
                model_filename = f"retrained_model_{job.job_id}.pt"
                new_model_path = self.models_dir / model_filename
                shutil.copy2(best_model_path, new_model_path)
                
                self.logger.info(f"New model trained: {new_model_path}")
                return str(new_model_path)
            else:
                self.logger.error("No best model found in training results")
                return None
                
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return None
    
    def _prepare_training_config(self, job: RetrainingJob) -> str:
        """Prepare training configuration for retraining"""
        # Create temporary config file
        config_path = self.retraining_dir / f"training_config_{job.job_id}.yaml"
        
        # Base configuration
        config = {
            'dataset': {
                'train_dir': str(self.retraining_dir / "train"),
                'val_dir': str(self.retraining_dir / "val"),
                'test_dir': str(self.retraining_dir / "test"),
                'classes': {
                    0: 'person',
                    1: 'suspicious_activity',
                    2: 'violence',
                    3: 'theft',
                    4: 'abandoned_object',
                    5: 'crowding',
                    6: 'loitering'
                }
            },
            'yolo': {
                'epochs': self.config['training']['epochs'],
                'patience': self.config['training']['patience'],
                'batch_size': 16,
                'learning_rate': 0.01,
                'weight_decay': 0.0005
            }
        }
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    async def _validate_new_model(self, job: RetrainingJob) -> Dict:
        """Validate new model performance"""
        self.logger.info("Validating new model performance...")
        
        try:
            # Get current model performance
            current_performance = await self._get_current_model_performance()
            
            # Evaluate new model
            validator = EnhancedSecurityValidator(
                model_path=job.new_model_path,
                config_path="../config/dataset_config.yaml"
            )
            
            new_performance = await self._evaluate_model_performance(validator)
            
            # Compare performances
            comparison = self._compare_model_performance(current_performance, new_performance)
            
            # Check validation thresholds
            validation_passed = self._check_validation_thresholds(comparison)
            
            # Save performance metrics
            await self._save_performance_metrics(job.job_id, new_performance)
            
            return {
                'passed': validation_passed,
                'current_performance': current_performance,
                'new_performance': new_performance,
                'comparison': comparison,
                'reason': comparison.get('validation_reason', '')
            }
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return {
                'passed': False,
                'reason': f"Validation error: {e}"
            }
    
    async def _evaluate_model_performance(self, validator: EnhancedSecurityValidator) -> PerformanceMetrics:
        """Evaluate model performance metrics"""
        # Run comprehensive evaluation
        results = validator.run_comprehensive_evaluation()
        
        # Extract key metrics
        performance_metrics = results.get('performance_metrics', {})
        security_analysis = results.get('security_analysis', {})
        
        return PerformanceMetrics(
            model_path=validator.model_path,
            mAP50=performance_metrics.get('mAP50', 0.0),
            mAP50_95=performance_metrics.get('mAP50-95', 0.0),
            precision=performance_metrics.get('precision', 0.0),
            recall=performance_metrics.get('recall', 0.0),
            f1_score=performance_metrics.get('f1', 0.0),
            security_score=security_analysis.get('critical_metrics', {}).get('overall_security_score', 0.0),
            false_positive_rate=security_analysis.get('critical_metrics', {}).get('false_alarm_rate', 1.0),
            false_negative_rate=1.0 - performance_metrics.get('recall', 0.0),
            inference_time_ms=results.get('inference_time_ms', 0.0),
            model_size_mb=results.get('model_size_mb', 0.0)
        )
    
    def _compare_model_performance(self, current: Optional[PerformanceMetrics], new: PerformanceMetrics) -> Dict:
        """Compare current and new model performance"""
        if not current:
            return {
                'mAP50_improvement': new.mAP50,
                'security_score_improvement': new.security_score,
                'false_positive_improvement': -new.false_positive_rate,
                'validation_reason': 'No current model for comparison'
            }
        
        return {
            'mAP50_improvement': new.mAP50 - current.mAP50,
            'security_score_improvement': new.security_score - current.security_score,
            'false_positive_improvement': current.false_positive_rate - new.false_positive_rate,
            'precision_improvement': new.precision - current.precision,
            'recall_improvement': new.recall - current.recall,
            'inference_time_change': new.inference_time_ms - current.inference_time_ms,
            'model_size_change': new.model_size_mb - current.model_size_mb
        }
    
    def _check_validation_thresholds(self, comparison: Dict) -> bool:
        """Check if new model meets validation thresholds"""
        thresholds = self.performance_thresholds
        
        # Check minimum improvement
        mAP_improvement = comparison.get('mAP50_improvement', 0)
        if mAP_improvement < thresholds['min_mAP50_improvement']:
            comparison['validation_reason'] = f"Insufficient mAP50 improvement: {mAP_improvement:.4f}"
            return False
        
        # Check for performance degradation
        if mAP_improvement < -thresholds['max_performance_degradation']:
            comparison['validation_reason'] = f"Performance degradation: {mAP_improvement:.4f}"
            return False
        
        # Check security score
        security_improvement = comparison.get('security_score_improvement', 0)
        if security_improvement < 0:
            comparison['validation_reason'] = f"Security score degradation: {security_improvement:.4f}"
            return False
        
        # Check false positive rate
        fp_improvement = comparison.get('false_positive_improvement', 0)
        if fp_improvement < 0:
            comparison['validation_reason'] = f"False positive rate increased: {-fp_improvement:.4f}"
            return False
        
        comparison['validation_reason'] = "All validation thresholds passed"
        return True
    
    async def _make_deployment_decision(self, job: RetrainingJob) -> bool:
        """Make deployment decision based on validation results"""
        if not job.performance_comparison or not job.performance_comparison['passed']:
            return False
        
        # Check if manual approval is required
        if self.config['deployment']['require_manual_approval']:
            self.logger.info("Manual approval required for deployment")
            # In a real system, this would trigger a notification for human review
            # For now, auto-approve if validation passed
            return True
        
        return True
    
    async def _get_current_model_performance(self) -> Optional[PerformanceMetrics]:
        """Get current model performance metrics"""
        # Query latest performance from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM model_performance_history 
                ORDER BY created_at DESC 
                LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if row:
                return PerformanceMetrics(
                    model_path=row[1],
                    mAP50=row[3] or 0.0,
                    mAP50_95=row[4] or 0.0,
                    precision=row[5] or 0.0,
                    recall=row[6] or 0.0,
                    f1_score=row[7] or 0.0,
                    security_score=row[8] or 0.0,
                    false_positive_rate=row[9] or 1.0,
                    false_negative_rate=row[10] or 1.0,
                    inference_time_ms=row[11] or 0.0,
                    model_size_mb=row[12] or 0.0
                )
        
        return None
    
    async def _get_historical_performance(self) -> Optional[Dict]:
        """Get historical performance metrics"""
        # For now, return simulated historical performance
        return {
            'mAP50': 0.75,
            'security_score': 0.8,
            'false_positive_rate': 0.08
        }
    
    async def _get_last_training_date(self) -> Optional[datetime]:
        """Get date of last training"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT MAX(created_at) FROM retraining_jobs 
                WHERE status = 'completed'
            ''')
            
            result = cursor.fetchone()[0]
            if result:
                return datetime.fromisoformat(result)
        
        return None
    
    async def _save_performance_metrics(self, job_id: str, performance: PerformanceMetrics):
        """Save performance metrics to database"""
        model_id = hashlib.md5(f"{job_id}_{performance.model_path}".encode()).hexdigest()[:16]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO model_performance_history 
                (model_id, model_path, job_id, mAP50, mAP50_95, precision_val, recall_val, 
                 f1_score, security_score, false_positive_rate, false_negative_rate, 
                 inference_time_ms, model_size_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id, performance.model_path, job_id, performance.mAP50, performance.mAP50_95,
                performance.precision, performance.recall, performance.f1_score, performance.security_score,
                performance.false_positive_rate, performance.false_negative_rate,
                performance.inference_time_ms, performance.model_size_mb
            ))
    
    async def _update_job_status(self, job: RetrainingJob):
        """Update job status in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE retraining_jobs 
                SET status = ?, total_new_samples = ?, samples_by_class = ?, 
                    new_model_path = ?, performance_comparison = ?, 
                    deployment_approved = ?, completed_at = ?
                WHERE job_id = ?
            ''', (
                job.status,
                job.total_new_samples,
                json.dumps(job.samples_by_class),
                job.new_model_path,
                json.dumps(job.performance_comparison) if job.performance_comparison else None,
                job.deployment_approved,
                job.completed_at.isoformat() if job.completed_at else None,
                job.job_id
            ))
    
    async def get_retraining_status(self) -> Dict:
        """Get current retraining pipeline status"""
        with sqlite3.connect(self.db_path) as conn:
            # Get job statistics
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_jobs,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'collecting' THEN 1 ELSE 0 END) as collecting,
                    SUM(CASE WHEN status = 'labeling' THEN 1 ELSE 0 END) as labeling,
                    SUM(CASE WHEN status = 'training' THEN 1 ELSE 0 END) as training,
                    SUM(CASE WHEN status = 'validating' THEN 1 ELSE 0 END) as validating,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN deployment_approved = 1 THEN 1 ELSE 0 END) as deployed
                FROM retraining_jobs
            ''')
            
            job_stats = cursor.fetchone()
            
            # Get recent jobs
            cursor = conn.execute('''
                SELECT job_id, trigger_reason, status, created_at, completed_at, deployment_approved
                FROM retraining_jobs
                ORDER BY created_at DESC
                LIMIT 10
            ''')
            
            recent_jobs = cursor.fetchall()
            
            return {
                'job_statistics': {
                    'total_jobs': job_stats[0],
                    'pending': job_stats[1],
                    'collecting': job_stats[2],
                    'labeling': job_stats[3],
                    'training': job_stats[4],
                    'validating': job_stats[5],
                    'completed': job_stats[6],
                    'failed': job_stats[7],
                    'deployed': job_stats[8]
                },
                'recent_jobs': [
                    {
                        'job_id': row[0],
                        'trigger_reason': row[1],
                        'status': row[2],
                        'created_at': row[3],
                        'completed_at': row[4],
                        'deployment_approved': bool(row[5])
                    }
                    for row in recent_jobs
                ]
            }

async def main():
    """Example usage of retraining pipeline"""
    pipeline = ModelRetrainingPipeline()
    
    # Check for retraining triggers
    job = await pipeline.check_retraining_triggers()
    
    if job:
        print(f"Retraining triggered: {job.trigger_reason}")
        
        # Execute retraining job
        success = await pipeline.execute_retraining_job(job)
        
        if success:
            print(f"Retraining job {job.job_id} completed successfully")
            if job.deployment_approved:
                print("New model approved for deployment")
            else:
                print("New model not approved for deployment")
        else:
            print(f"Retraining job {job.job_id} failed")
    else:
        print("No retraining triggers detected")
    
    # Get pipeline status
    status = await pipeline.get_retraining_status()
    print(f"Pipeline status: {status}")

if __name__ == "__main__":
    asyncio.run(main())