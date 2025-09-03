"""
Model Drift Detection System for Edge Deployment

This module monitors model performance drift and triggers retraining/redeployment
when performance degrades beyond acceptable thresholds.
"""

import asyncio
import logging
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import yaml
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class DriftMetrics:
    """Model drift detection metrics"""
    device_id: str
    model_version_id: str
    timestamp: datetime
    
    # Performance metrics
    accuracy_drift: float
    precision_drift: float
    recall_drift: float
    latency_drift: float
    
    # Statistical drift measures
    psi_score: float  # Population Stability Index
    kl_divergence: float  # Kullback-Leibler divergence
    js_divergence: float  # Jensen-Shannon divergence
    
    # Drift severity
    drift_severity: str  # low, medium, high, critical
    drift_detected: bool
    
    # Additional context
    sample_size: int
    confidence_interval: Tuple[float, float]


@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    device_id: str
    model_version_id: str
    drift_type: str  # performance, statistical, concept
    severity: str
    metrics: DriftMetrics
    recommended_action: str
    created_at: datetime
    acknowledged: bool = False


class ModelDriftDetector:
    """
    Advanced model drift detection system that monitors performance
    degradation and statistical drift in edge deployments
    """
    
    def __init__(self, config_path: str = "config/drift_detection_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize paths
        self.drift_dir = Path(self.config['drift_detection']['output_dir'])
        self.drift_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Initialize database
        self.db_path = self.drift_dir / "drift_detection.db"
        self._init_database()
        
        # Drift detection parameters
        self.drift_thresholds = self.config['thresholds']
        self.baseline_window_days = self.config['drift_detection']['baseline_window_days']
        self.detection_window_days = self.config['drift_detection']['detection_window_days']
        
        # Background monitoring
        self._monitoring_active = False
        self._background_tasks = set()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load drift detection configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'drift_detection': {
                    'output_dir': '../../data/drift_detection',
                    'monitoring_interval_hours': 6,
                    'baseline_window_days': 7,
                    'detection_window_days': 1,
                    'min_samples_for_detection': 100,
                    'enable_statistical_tests': True,
                    'enable_visualization': True
                },
                'thresholds': {
                    'performance_drift': {
                        'accuracy_threshold': 0.05,
                        'precision_threshold': 0.05,
                        'recall_threshold': 0.05,
                        'latency_threshold': 0.2
                    },
                    'statistical_drift': {
                        'psi_threshold': 0.2,
                        'kl_divergence_threshold': 0.1,
                        'js_divergence_threshold': 0.1
                    },
                    'severity_levels': {
                        'low': 0.05,
                        'medium': 0.1,
                        'high': 0.2,
                        'critical': 0.3
                    }
                },
                'alerts': {
                    'enable_notifications': True,
                    'notification_channels': ['email', 'webhook'],
                    'escalation_hours': 24,
                    'auto_trigger_retraining': False
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for drift detection"""
        logger = logging.getLogger('model_drift_detector')
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
            log_file = self.drift_dir / "drift_detection.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for drift tracking"""
        with sqlite3.connect(self.db_path) as conn:
            # Drift metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS drift_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    model_version_id TEXT NOT NULL,
                    timestamp TIMESTAMP,
                    accuracy_drift REAL,
                    precision_drift REAL,
                    recall_drift REAL,
                    latency_drift REAL,
                    psi_score REAL,
                    kl_divergence REAL,
                    js_divergence REAL,
                    drift_severity TEXT,
                    drift_detected BOOLEAN,
                    sample_size INTEGER,
                    confidence_lower REAL,
                    confidence_upper REAL
                )
            ''')
            
            # Drift alerts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    alert_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    model_version_id TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    recommended_action TEXT,
                    created_at TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    metrics_id INTEGER,
                    FOREIGN KEY (metrics_id) REFERENCES drift_metrics (id)
                )
            ''')
            
            # Baseline performance table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS baseline_performance (
                    device_id TEXT,
                    model_version_id TEXT,
                    metric_name TEXT,
                    baseline_value REAL,
                    baseline_std REAL,
                    sample_count INTEGER,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (device_id, model_version_id, metric_name)
                )
            ''')
    
    async def start_drift_monitoring(self):
        """Start continuous drift monitoring"""
        if self._monitoring_active:
            self.logger.warning("Drift monitoring already active")
            return
        
        self._monitoring_active = True
        self.logger.info("Starting drift monitoring")
        
        async def monitoring_loop():
            while self._monitoring_active:
                try:
                    await self._run_drift_detection_cycle()
                    
                    # Wait for next monitoring cycle
                    interval_hours = self.config['drift_detection']['monitoring_interval_hours']
                    await asyncio.sleep(interval_hours * 3600)
                    
                except Exception as e:
                    self.logger.error(f"Drift monitoring error: {e}")
                    await asyncio.sleep(300)  # Retry after 5 minutes
        
        task = asyncio.create_task(monitoring_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _run_drift_detection_cycle(self):
        """Run complete drift detection cycle for all devices"""
        self.logger.info("Running drift detection cycle")
        
        # Get active devices and models
        active_deployments = await self._get_active_deployments()
        
        for device_id, model_version_id in active_deployments:
            try:
                await self._detect_drift_for_device(device_id, model_version_id)
            except Exception as e:
                self.logger.error(f"Drift detection failed for device {device_id}: {e}")
    
    async def _get_active_deployments(self) -> List[Tuple[str, str]]:
        """Get list of active device-model deployments"""
        # This would typically query the deployment database
        # For now, return simulated active deployments
        return [
            ("edge_001", "v20241203_120000_abc12345"),
            ("edge_002", "v20241203_120000_abc12345"),
            ("edge_003", "v20241203_120000_abc12345")
        ]
    
    async def _detect_drift_for_device(self, device_id: str, model_version_id: str):
        """Detect drift for a specific device and model"""
        self.logger.info(f"Detecting drift for device {device_id}, model {model_version_id}")
        
        # Get baseline performance
        baseline_metrics = await self._get_baseline_performance(device_id, model_version_id)
        if not baseline_metrics:
            self.logger.warning(f"No baseline metrics found for {device_id}, establishing baseline")
            await self._establish_baseline(device_id, model_version_id)
            return
        
        # Get recent performance data
        recent_metrics = await self._get_recent_performance(device_id, model_version_id)
        if not recent_metrics or len(recent_metrics) < self.config['drift_detection']['min_samples_for_detection']:
            self.logger.warning(f"Insufficient recent data for drift detection: {len(recent_metrics) if recent_metrics else 0} samples")
            return
        
        # Calculate drift metrics
        drift_metrics = await self._calculate_drift_metrics(
            device_id, model_version_id, baseline_metrics, recent_metrics
        )
        
        # Store drift metrics
        await self._store_drift_metrics(drift_metrics)
        
        # Check for drift and generate alerts
        if drift_metrics.drift_detected:
            await self._generate_drift_alert(drift_metrics)
        
        # Update visualizations if enabled
        if self.config['drift_detection']['enable_visualization']:
            await self._update_drift_visualizations(device_id, model_version_id)
    
    async def _get_baseline_performance(self, device_id: str, model_version_id: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Get baseline performance metrics"""
        baseline_metrics = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT metric_name, baseline_value, baseline_std, sample_count
                FROM baseline_performance 
                WHERE device_id = ? AND model_version_id = ?
            ''', (device_id, model_version_id))
            
            for row in cursor.fetchall():
                metric_name, baseline_value, baseline_std, sample_count = row
                baseline_metrics[metric_name] = {
                    'mean': baseline_value,
                    'std': baseline_std,
                    'count': sample_count
                }
        
        return baseline_metrics if baseline_metrics else None
    
    async def _establish_baseline(self, device_id: str, model_version_id: str):
        """Establish baseline performance metrics"""
        self.logger.info(f"Establishing baseline for device {device_id}")
        
        # Get historical performance data for baseline period
        baseline_data = await self._get_historical_performance(
            device_id, model_version_id, self.baseline_window_days
        )
        
        if not baseline_data or len(baseline_data) < 50:  # Minimum samples for reliable baseline
            self.logger.warning(f"Insufficient historical data for baseline: {len(baseline_data) if baseline_data else 0} samples")
            return
        
        # Calculate baseline statistics for each metric
        metrics_to_track = ['accuracy', 'precision', 'recall', 'inference_time_ms', 'error_rate']
        
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics_to_track:
                if metric in baseline_data[0]:  # Check if metric exists in data
                    values = [sample[metric] for sample in baseline_data if sample.get(metric) is not None]
                    
                    if values:
                        baseline_mean = np.mean(values)
                        baseline_std = np.std(values)
                        sample_count = len(values)
                        
                        # Store baseline
                        conn.execute('''
                            INSERT OR REPLACE INTO baseline_performance 
                            (device_id, model_version_id, metric_name, baseline_value, 
                             baseline_std, sample_count, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            device_id, model_version_id, metric, baseline_mean,
                            baseline_std, sample_count, datetime.now().isoformat(),
                            datetime.now().isoformat()
                        ))
                        
                        self.logger.info(f"Established baseline for {metric}: {baseline_mean:.4f} ± {baseline_std:.4f}")
    
    async def _get_historical_performance(self, device_id: str, model_version_id: str, days: int) -> List[Dict[str, float]]:
        """Get historical performance data"""
        # This would typically query the performance monitoring database
        # For demonstration, return simulated data
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Simulate historical performance data
        np.random.seed(42)  # For reproducible results
        num_samples = days * 24  # Hourly samples
        
        historical_data = []
        for i in range(num_samples):
            timestamp = cutoff_date + timedelta(hours=i)
            
            # Simulate gradually degrading performance
            degradation_factor = 1.0 - (i / num_samples) * 0.1  # 10% degradation over time
            
            sample = {
                'timestamp': timestamp.isoformat(),
                'accuracy': 0.85 * degradation_factor + np.random.normal(0, 0.02),
                'precision': 0.82 * degradation_factor + np.random.normal(0, 0.02),
                'recall': 0.78 * degradation_factor + np.random.normal(0, 0.02),
                'inference_time_ms': 45 / degradation_factor + np.random.normal(0, 5),
                'error_rate': (1 - degradation_factor) * 0.05 + np.random.normal(0, 0.01)
            }
            
            # Ensure realistic bounds
            sample['accuracy'] = max(0, min(1, sample['accuracy']))
            sample['precision'] = max(0, min(1, sample['precision']))
            sample['recall'] = max(0, min(1, sample['recall']))
            sample['inference_time_ms'] = max(10, sample['inference_time_ms'])
            sample['error_rate'] = max(0, min(1, sample['error_rate']))
            
            historical_data.append(sample)
        
        return historical_data
    
    async def _get_recent_performance(self, device_id: str, model_version_id: str) -> List[Dict[str, float]]:
        """Get recent performance data for drift detection"""
        # Get data from the last detection window
        recent_data = await self._get_historical_performance(
            device_id, model_version_id, self.detection_window_days
        )
        
        # Return only the most recent portion
        if recent_data:
            recent_samples = int(len(recent_data) * 0.2)  # Last 20% of samples
            return recent_data[-recent_samples:] if recent_samples > 0 else recent_data
        
        return []
    
    async def _calculate_drift_metrics(self, device_id: str, model_version_id: str,
                                     baseline_metrics: Dict[str, Dict[str, float]],
                                     recent_data: List[Dict[str, float]]) -> DriftMetrics:
        """Calculate comprehensive drift metrics"""
        
        # Calculate performance drift
        performance_drifts = {}
        statistical_measures = {}
        
        for metric_name in ['accuracy', 'precision', 'recall', 'inference_time_ms']:
            if metric_name in baseline_metrics:
                baseline = baseline_metrics[metric_name]
                recent_values = [sample[metric_name] for sample in recent_data if metric_name in sample]
                
                if recent_values:
                    recent_mean = np.mean(recent_values)
                    baseline_mean = baseline['mean']
                    
                    # Calculate relative drift
                    if baseline_mean != 0:
                        drift = abs(recent_mean - baseline_mean) / abs(baseline_mean)
                    else:
                        drift = abs(recent_mean - baseline_mean)
                    
                    performance_drifts[f"{metric_name}_drift"] = drift
                    
                    # Calculate statistical measures
                    if len(recent_values) > 10:  # Minimum for statistical tests
                        # Population Stability Index (PSI)
                        psi = self._calculate_psi(baseline_mean, baseline['std'], recent_values)
                        statistical_measures['psi_score'] = psi
                        
                        # KL Divergence (approximated)
                        kl_div = self._calculate_kl_divergence(baseline_mean, baseline['std'], recent_values)
                        statistical_measures['kl_divergence'] = kl_div
                        
                        # Jensen-Shannon Divergence
                        js_div = self._calculate_js_divergence(baseline_mean, baseline['std'], recent_values)
                        statistical_measures['js_divergence'] = js_div
        
        # Determine overall drift severity
        max_performance_drift = max(performance_drifts.values()) if performance_drifts else 0
        max_statistical_drift = max(statistical_measures.values()) if statistical_measures else 0
        
        drift_severity, drift_detected = self._assess_drift_severity(max_performance_drift, max_statistical_drift)
        
        # Calculate confidence interval
        if recent_data:
            accuracy_values = [sample.get('accuracy', 0) for sample in recent_data]
            if accuracy_values:
                confidence_interval = self._calculate_confidence_interval(accuracy_values)
            else:
                confidence_interval = (0.0, 0.0)
        else:
            confidence_interval = (0.0, 0.0)
        
        return DriftMetrics(
            device_id=device_id,
            model_version_id=model_version_id,
            timestamp=datetime.now(),
            accuracy_drift=performance_drifts.get('accuracy_drift', 0),
            precision_drift=performance_drifts.get('precision_drift', 0),
            recall_drift=performance_drifts.get('recall_drift', 0),
            latency_drift=performance_drifts.get('inference_time_ms_drift', 0),
            psi_score=statistical_measures.get('psi_score', 0),
            kl_divergence=statistical_measures.get('kl_divergence', 0),
            js_divergence=statistical_measures.get('js_divergence', 0),
            drift_severity=drift_severity,
            drift_detected=drift_detected,
            sample_size=len(recent_data),
            confidence_interval=confidence_interval
        )
    
    def _calculate_psi(self, baseline_mean: float, baseline_std: float, recent_values: List[float]) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins for PSI calculation
            recent_mean = np.mean(recent_values)
            recent_std = np.std(recent_values)
            
            # Simple PSI approximation based on mean and std changes
            mean_change = abs(recent_mean - baseline_mean) / (baseline_std + 1e-8)
            std_change = abs(recent_std - baseline_std) / (baseline_std + 1e-8)
            
            psi = mean_change + std_change
            return min(psi, 2.0)  # Cap at 2.0
            
        except Exception as e:
            self.logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def _calculate_kl_divergence(self, baseline_mean: float, baseline_std: float, recent_values: List[float]) -> float:
        """Calculate approximate KL divergence"""
        try:
            recent_mean = np.mean(recent_values)
            recent_std = np.std(recent_values)
            
            # Approximate KL divergence for normal distributions
            if recent_std > 0 and baseline_std > 0:
                kl_div = np.log(recent_std / baseline_std) + \
                        (baseline_std**2 + (baseline_mean - recent_mean)**2) / (2 * recent_std**2) - 0.5
                return max(0, kl_div)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"KL divergence calculation failed: {e}")
            return 0.0
    
    def _calculate_js_divergence(self, baseline_mean: float, baseline_std: float, recent_values: List[float]) -> float:
        """Calculate Jensen-Shannon divergence"""
        try:
            # Simplified JS divergence calculation
            kl_div = self._calculate_kl_divergence(baseline_mean, baseline_std, recent_values)
            js_div = kl_div / 2  # Simplified approximation
            return js_div
            
        except Exception as e:
            self.logger.warning(f"JS divergence calculation failed: {e}")
            return 0.0
    
    def _assess_drift_severity(self, performance_drift: float, statistical_drift: float) -> Tuple[str, bool]:
        """Assess overall drift severity"""
        max_drift = max(performance_drift, statistical_drift)
        
        severity_levels = self.drift_thresholds['severity_levels']
        
        if max_drift >= severity_levels['critical']:
            return 'critical', True
        elif max_drift >= severity_levels['high']:
            return 'high', True
        elif max_drift >= severity_levels['medium']:
            return 'medium', True
        elif max_drift >= severity_levels['low']:
            return 'low', True
        else:
            return 'none', False
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for values"""
        try:
            if len(values) < 2:
                return (0.0, 0.0)
            
            mean = np.mean(values)
            std_err = stats.sem(values)
            
            # Calculate confidence interval
            h = std_err * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
            
            return (mean - h, mean + h)
            
        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            return (0.0, 0.0)
    
    async def _store_drift_metrics(self, drift_metrics: DriftMetrics):
        """Store drift metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO drift_metrics 
                (device_id, model_version_id, timestamp, accuracy_drift, precision_drift,
                 recall_drift, latency_drift, psi_score, kl_divergence, js_divergence,
                 drift_severity, drift_detected, sample_size, confidence_lower, confidence_upper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                drift_metrics.device_id,
                drift_metrics.model_version_id,
                drift_metrics.timestamp.isoformat(),
                drift_metrics.accuracy_drift,
                drift_metrics.precision_drift,
                drift_metrics.recall_drift,
                drift_metrics.latency_drift,
                drift_metrics.psi_score,
                drift_metrics.kl_divergence,
                drift_metrics.js_divergence,
                drift_metrics.drift_severity,
                drift_metrics.drift_detected,
                drift_metrics.sample_size,
                drift_metrics.confidence_interval[0],
                drift_metrics.confidence_interval[1]
            ))
            
            # Get the inserted row ID
            metrics_id = cursor.lastrowid
            
        self.logger.info(f"Stored drift metrics for device {drift_metrics.device_id}: {drift_metrics.drift_severity}")
        return metrics_id
    
    async def _generate_drift_alert(self, drift_metrics: DriftMetrics):
        """Generate drift alert"""
        alert_id = f"drift_{drift_metrics.device_id}_{int(drift_metrics.timestamp.timestamp())}"
        
        # Determine drift type and recommended action
        drift_type = self._classify_drift_type(drift_metrics)
        recommended_action = self._get_recommended_action(drift_metrics)
        
        alert = DriftAlert(
            alert_id=alert_id,
            device_id=drift_metrics.device_id,
            model_version_id=drift_metrics.model_version_id,
            drift_type=drift_type,
            severity=drift_metrics.drift_severity,
            metrics=drift_metrics,
            recommended_action=recommended_action,
            created_at=drift_metrics.timestamp
        )
        
        # Store alert
        await self._store_drift_alert(alert)
        
        # Send notifications if enabled
        if self.config['alerts']['enable_notifications']:
            await self._send_drift_notification(alert)
        
        self.logger.warning(f"Drift alert generated: {alert_id} - {drift_type} ({drift_metrics.drift_severity})")
    
    def _classify_drift_type(self, drift_metrics: DriftMetrics) -> str:
        """Classify the type of drift detected"""
        performance_threshold = 0.05
        statistical_threshold = 0.1
        
        has_performance_drift = (
            drift_metrics.accuracy_drift > performance_threshold or
            drift_metrics.precision_drift > performance_threshold or
            drift_metrics.recall_drift > performance_threshold
        )
        
        has_statistical_drift = (
            drift_metrics.psi_score > statistical_threshold or
            drift_metrics.kl_divergence > statistical_threshold
        )
        
        if has_performance_drift and has_statistical_drift:
            return "concept_drift"
        elif has_performance_drift:
            return "performance_drift"
        elif has_statistical_drift:
            return "statistical_drift"
        else:
            return "unknown_drift"
    
    def _get_recommended_action(self, drift_metrics: DriftMetrics) -> str:
        """Get recommended action based on drift severity"""
        if drift_metrics.drift_severity == "critical":
            return "immediate_model_rollback_and_retraining"
        elif drift_metrics.drift_severity == "high":
            return "schedule_model_retraining"
        elif drift_metrics.drift_severity == "medium":
            return "increase_monitoring_frequency"
        else:
            return "continue_monitoring"
    
    async def _store_drift_alert(self, alert: DriftAlert):
        """Store drift alert in database"""
        # First store the metrics and get the ID
        metrics_id = await self._store_drift_metrics(alert.metrics)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO drift_alerts 
                (alert_id, device_id, model_version_id, drift_type, severity,
                 recommended_action, created_at, metrics_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.device_id,
                alert.model_version_id,
                alert.drift_type,
                alert.severity,
                alert.recommended_action,
                alert.created_at.isoformat(),
                metrics_id
            ))
    
    async def _send_drift_notification(self, alert: DriftAlert):
        """Send drift notification"""
        # In a real system, this would send notifications via configured channels
        notification_data = {
            'alert_id': alert.alert_id,
            'device_id': alert.device_id,
            'drift_type': alert.drift_type,
            'severity': alert.severity,
            'recommended_action': alert.recommended_action,
            'timestamp': alert.created_at.isoformat()
        }
        
        # Store notification for later processing
        notification_file = self.drift_dir / "drift_notifications.jsonl"
        with open(notification_file, 'a') as f:
            f.write(json.dumps(notification_data) + '\n')
        
        self.logger.info(f"Drift notification queued: {alert.alert_id}")
    
    async def _update_drift_visualizations(self, device_id: str, model_version_id: str):
        """Update drift visualization charts"""
        try:
            # Get drift history
            drift_history = await self._get_drift_history(device_id, model_version_id, days=30)
            
            if not drift_history:
                return
            
            # Create visualization directory
            viz_dir = self.drift_dir / "visualizations" / device_id
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot drift trends
            await self._plot_drift_trends(drift_history, viz_dir / f"drift_trends_{model_version_id}.png")
            
            # Plot performance metrics
            await self._plot_performance_trends(drift_history, viz_dir / f"performance_trends_{model_version_id}.png")
            
        except Exception as e:
            self.logger.error(f"Visualization update failed: {e}")
    
    async def _get_drift_history(self, device_id: str, model_version_id: str, days: int) -> List[Dict]:
        """Get drift history for visualization"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, accuracy_drift, precision_drift, recall_drift,
                       latency_drift, psi_score, drift_severity, drift_detected
                FROM drift_metrics 
                WHERE device_id = ? AND model_version_id = ? AND timestamp > ?
                ORDER BY timestamp
            ''', (device_id, model_version_id, cutoff_date.isoformat()))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'timestamp': datetime.fromisoformat(row[0]),
                    'accuracy_drift': row[1],
                    'precision_drift': row[2],
                    'recall_drift': row[3],
                    'latency_drift': row[4],
                    'psi_score': row[5],
                    'drift_severity': row[6],
                    'drift_detected': row[7]
                })
            
            return history
    
    async def _plot_drift_trends(self, drift_history: List[Dict], output_path: Path):
        """Plot drift trends over time"""
        if not drift_history:
            return
        
        plt.figure(figsize=(12, 8))
        
        timestamps = [h['timestamp'] for h in drift_history]
        
        # Plot different drift metrics
        plt.subplot(2, 2, 1)
        plt.plot(timestamps, [h['accuracy_drift'] for h in drift_history], label='Accuracy Drift')
        plt.plot(timestamps, [h['precision_drift'] for h in drift_history], label='Precision Drift')
        plt.plot(timestamps, [h['recall_drift'] for h in drift_history], label='Recall Drift')
        plt.title('Performance Drift Trends')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.plot(timestamps, [h['psi_score'] for h in drift_history], label='PSI Score', color='red')
        plt.title('Statistical Drift (PSI)')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        plt.plot(timestamps, [h['latency_drift'] for h in drift_history], label='Latency Drift', color='orange')
        plt.title('Latency Drift')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        # Drift detection events
        drift_events = [h['timestamp'] for h in drift_history if h['drift_detected']]
        if drift_events:
            plt.scatter(drift_events, [1] * len(drift_events), color='red', s=50, label='Drift Detected')
        plt.title('Drift Detection Events')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    async def _plot_performance_trends(self, drift_history: List[Dict], output_path: Path):
        """Plot performance trends"""
        # This would create additional performance visualizations
        # Implementation similar to drift trends but focusing on absolute performance metrics
        pass
    
    async def get_drift_summary(self, device_id: str, days: int = 7) -> Dict[str, Any]:
        """Get drift summary for a device"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get recent drift metrics
            cursor = conn.execute('''
                SELECT COUNT(*) as total_checks,
                       SUM(CASE WHEN drift_detected = 1 THEN 1 ELSE 0 END) as drift_detections,
                       AVG(accuracy_drift) as avg_accuracy_drift,
                       MAX(psi_score) as max_psi_score,
                       drift_severity
                FROM drift_metrics 
                WHERE device_id = ? AND timestamp > ?
                GROUP BY drift_severity
                ORDER BY timestamp DESC
            ''', (device_id, cutoff_date.isoformat()))
            
            drift_summary = {
                'device_id': device_id,
                'period_days': days,
                'total_checks': 0,
                'drift_detections': 0,
                'severity_breakdown': {},
                'avg_accuracy_drift': 0,
                'max_psi_score': 0
            }
            
            for row in cursor.fetchall():
                drift_summary['total_checks'] += row[0]
                drift_summary['drift_detections'] += row[1]
                drift_summary['avg_accuracy_drift'] = max(drift_summary['avg_accuracy_drift'], row[2] or 0)
                drift_summary['max_psi_score'] = max(drift_summary['max_psi_score'], row[3] or 0)
                
                if row[4]:  # drift_severity
                    drift_summary['severity_breakdown'][row[4]] = row[1]
            
            # Get active alerts
            cursor = conn.execute('''
                SELECT COUNT(*) FROM drift_alerts 
                WHERE device_id = ? AND created_at > ? AND acknowledged = 0
            ''', (device_id, cutoff_date.isoformat()))
            
            drift_summary['active_alerts'] = cursor.fetchone()[0]
            
            return drift_summary
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a drift alert"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                UPDATE drift_alerts SET acknowledged = 1 
                WHERE alert_id = ?
            ''', (alert_id,))
            
            return cursor.rowcount > 0
    
    async def stop_drift_monitoring(self):
        """Stop drift monitoring"""
        self._monitoring_active = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Drift monitoring stopped")


# Example usage
async def main():
    """Example usage of model drift detector"""
    
    # Initialize drift detector
    drift_detector = ModelDriftDetector()
    
    # Start monitoring
    await drift_detector.start_drift_monitoring()
    
    # Let it run for a while
    await asyncio.sleep(10)
    
    # Get drift summary
    summary = await drift_detector.get_drift_summary("edge_001", days=7)
    print(f"Drift summary: {summary}")
    
    # Stop monitoring
    await drift_detector.stop_drift_monitoring()


if __name__ == "__main__":
    asyncio.run(main())