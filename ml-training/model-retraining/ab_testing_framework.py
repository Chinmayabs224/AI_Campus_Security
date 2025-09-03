"""
A/B Testing Framework for Model Deployment
Manages gradual rollout and performance comparison
"""

import asyncio
import logging
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, asdict
import sqlite3
import hashlib
import random
from enum import Enum

class DeploymentStrategy(Enum):
    """Deployment strategies for A/B testing"""
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    GRADUAL_ROLLOUT = "gradual_rollout"
    SHADOW_TESTING = "shadow_testing"

@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_id: str
    model_a_path: str  # Current/control model
    model_b_path: str  # New/treatment model
    strategy: DeploymentStrategy
    traffic_split: Dict[str, float]  # {"model_a": 0.8, "model_b": 0.2}
    duration_hours: int
    success_criteria: Dict[str, float]
    rollback_criteria: Dict[str, float]
    edge_nodes: List[str]
    created_at: datetime = None

@dataclass
class ABTestMetrics:
    """Metrics for A/B test evaluation"""
    model_id: str
    test_id: str
    timestamp: datetime
    inference_count: int
    avg_confidence: float
    detection_rate: float
    false_positive_rate: float
    false_negative_rate: float
    avg_inference_time_ms: float
    error_rate: float
    user_feedback_score: Optional[float] = None

class ABTestingFramework:
    """A/B testing framework for model deployment"""
    
    def __init__(self, config_path: str = "config/ab_testing_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize paths
        self.testing_dir = Path(self.config['ab_testing']['output_dir'])
        self.testing_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        
        # Initialize database
        self.db_path = self.testing_dir / "ab_tests.db"
        self._init_database()
        
        # Active tests tracking
        self.active_tests: Dict[str, ABTestConfig] = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load A/B testing configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'ab_testing': {
                    'output_dir': '../../data/ab_testing',
                    'default_duration_hours': 72,
                    'min_sample_size': 1000,
                    'confidence_level': 0.95,
                    'statistical_power': 0.8
                },
                'deployment': {
                    'edge_nodes': ['edge_node_1', 'edge_node_2', 'edge_node_3'],
                    'rollback_timeout_minutes': 30,
                    'health_check_interval_seconds': 60
                },
                'success_criteria': {
                    'min_detection_rate_improvement': 0.02,
                    'max_false_positive_increase': 0.01,
                    'max_inference_time_increase_ms': 50,
                    'min_confidence_improvement': 0.05
                },
                'rollback_criteria': {
                    'max_error_rate': 0.05,
                    'max_false_positive_rate': 0.15,
                    'max_inference_time_ms': 500,
                    'min_detection_rate': 0.7
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for A/B testing"""
        logger = logging.getLogger('ab_testing')
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
            log_file = self.testing_dir / "ab_testing.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for A/B test tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id TEXT PRIMARY KEY,
                    model_a_path TEXT NOT NULL,
                    model_b_path TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    traffic_split TEXT NOT NULL,
                    duration_hours INTEGER,
                    success_criteria TEXT,
                    rollback_criteria TEXT,
                    edge_nodes TEXT,
                    status TEXT DEFAULT 'created',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    result TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    test_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    inference_count INTEGER,
                    avg_confidence REAL,
                    detection_rate REAL,
                    false_positive_rate REAL,
                    false_negative_rate REAL,
                    avg_inference_time_ms REAL,
                    error_rate REAL,
                    user_feedback_score REAL,
                    FOREIGN KEY (test_id) REFERENCES ab_tests (test_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployment_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    edge_node TEXT,
                    model_version TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,
                    FOREIGN KEY (test_id) REFERENCES ab_tests (test_id)
                )
            ''')
    
    async def create_ab_test(
        self,
        model_a_path: str,
        model_b_path: str,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        traffic_split: Dict[str, float] = None,
        duration_hours: int = None
    ) -> ABTestConfig:
        """Create a new A/B test configuration"""
        
        if traffic_split is None:
            traffic_split = {"model_a": 0.9, "model_b": 0.1}  # Default canary
        
        if duration_hours is None:
            duration_hours = self.config['ab_testing']['default_duration_hours']
        
        test_id = self._generate_test_id()
        
        test_config = ABTestConfig(
            test_id=test_id,
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            strategy=strategy,
            traffic_split=traffic_split,
            duration_hours=duration_hours,
            success_criteria=self.config['success_criteria'],
            rollback_criteria=self.config['rollback_criteria'],
            edge_nodes=self.config['deployment']['edge_nodes'],
            created_at=datetime.now()
        )
        
        # Save to database
        await self._save_test_config(test_config)
        
        self.logger.info(f"Created A/B test {test_id} with strategy {strategy.value}")
        return test_config
    
    def _generate_test_id(self) -> str:
        """Generate unique test ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        return f"ab_test_{timestamp}_{random_suffix}"
    
    async def _save_test_config(self, test_config: ABTestConfig):
        """Save test configuration to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO ab_tests 
                (test_id, model_a_path, model_b_path, strategy, traffic_split, 
                 duration_hours, success_criteria, rollback_criteria, edge_nodes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_config.test_id,
                test_config.model_a_path,
                test_config.model_b_path,
                test_config.strategy.value,
                json.dumps(test_config.traffic_split),
                test_config.duration_hours,
                json.dumps(test_config.success_criteria),
                json.dumps(test_config.rollback_criteria),
                json.dumps(test_config.edge_nodes),
                test_config.created_at.isoformat()
            ))
    
    async def start_ab_test(self, test_id: str) -> bool:
        """Start an A/B test deployment"""
        try:
            # Load test configuration
            test_config = await self._load_test_config(test_id)
            if not test_config:
                self.logger.error(f"Test configuration not found: {test_id}")
                return False
            
            self.logger.info(f"Starting A/B test {test_id}")
            
            # Deploy models based on strategy
            deployment_success = await self._deploy_models(test_config)
            if not deployment_success:
                self.logger.error(f"Model deployment failed for test {test_id}")
                return False
            
            # Update test status
            await self._update_test_status(test_id, "running", started_at=datetime.now())
            
            # Add to active tests
            self.active_tests[test_id] = test_config
            
            # Start monitoring
            asyncio.create_task(self._monitor_test(test_config))
            
            self.logger.info(f"A/B test {test_id} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start A/B test {test_id}: {e}")
            await self._update_test_status(test_id, "failed")
            return False
    
    async def _load_test_config(self, test_id: str) -> Optional[ABTestConfig]:
        """Load test configuration from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT test_id, model_a_path, model_b_path, strategy, traffic_split,
                       duration_hours, success_criteria, rollback_criteria, edge_nodes, created_at
                FROM ab_tests WHERE test_id = ?
            ''', (test_id,))
            
            row = cursor.fetchone()
            if row:
                return ABTestConfig(
                    test_id=row[0],
                    model_a_path=row[1],
                    model_b_path=row[2],
                    strategy=DeploymentStrategy(row[3]),
                    traffic_split=json.loads(row[4]),
                    duration_hours=row[5],
                    success_criteria=json.loads(row[6]),
                    rollback_criteria=json.loads(row[7]),
                    edge_nodes=json.loads(row[8]),
                    created_at=datetime.fromisoformat(row[9])
                )
        
        return None
    
    async def _deploy_models(self, test_config: ABTestConfig) -> bool:
        """Deploy models to edge nodes based on strategy"""
        try:
            if test_config.strategy == DeploymentStrategy.CANARY:
                return await self._deploy_canary(test_config)
            elif test_config.strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._deploy_blue_green(test_config)
            elif test_config.strategy == DeploymentStrategy.GRADUAL_ROLLOUT:
                return await self._deploy_gradual_rollout(test_config)
            elif test_config.strategy == DeploymentStrategy.SHADOW_TESTING:
                return await self._deploy_shadow_testing(test_config)
            else:
                self.logger.error(f"Unknown deployment strategy: {test_config.strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False
    
    async def _deploy_canary(self, test_config: ABTestConfig) -> bool:
        """Deploy using canary strategy"""
        self.logger.info("Deploying canary release")
        
        # Calculate node allocation
        total_nodes = len(test_config.edge_nodes)
        model_b_nodes = max(1, int(total_nodes * test_config.traffic_split['model_b']))
        
        # Deploy model B to canary nodes
        canary_nodes = test_config.edge_nodes[:model_b_nodes]
        production_nodes = test_config.edge_nodes[model_b_nodes:]
        
        # Simulate deployment to canary nodes
        for node in canary_nodes:
            await self._deploy_model_to_node(test_config.test_id, node, "model_b", test_config.model_b_path)
        
        # Keep model A on production nodes
        for node in production_nodes:
            await self._deploy_model_to_node(test_config.test_id, node, "model_a", test_config.model_a_path)
        
        return True
    
    async def _deploy_blue_green(self, test_config: ABTestConfig) -> bool:
        """Deploy using blue-green strategy"""
        self.logger.info("Deploying blue-green release")
        
        # Split nodes into blue and green environments
        mid_point = len(test_config.edge_nodes) // 2
        blue_nodes = test_config.edge_nodes[:mid_point]
        green_nodes = test_config.edge_nodes[mid_point:]
        
        # Deploy model A to blue environment
        for node in blue_nodes:
            await self._deploy_model_to_node(test_config.test_id, node, "model_a", test_config.model_a_path)
        
        # Deploy model B to green environment
        for node in green_nodes:
            await self._deploy_model_to_node(test_config.test_id, node, "model_b", test_config.model_b_path)
        
        return True
    
    async def _deploy_gradual_rollout(self, test_config: ABTestConfig) -> bool:
        """Deploy using gradual rollout strategy"""
        self.logger.info("Starting gradual rollout")
        
        # Start with small percentage for model B
        initial_split = min(0.1, test_config.traffic_split['model_b'])
        
        # Deploy initial split
        total_nodes = len(test_config.edge_nodes)
        model_b_nodes = max(1, int(total_nodes * initial_split))
        
        for i, node in enumerate(test_config.edge_nodes):
            if i < model_b_nodes:
                await self._deploy_model_to_node(test_config.test_id, node, "model_b", test_config.model_b_path)
            else:
                await self._deploy_model_to_node(test_config.test_id, node, "model_a", test_config.model_a_path)
        
        # Schedule gradual increase
        asyncio.create_task(self._gradual_rollout_scheduler(test_config))
        
        return True
    
    async def _deploy_shadow_testing(self, test_config: ABTestConfig) -> bool:
        """Deploy using shadow testing strategy"""
        self.logger.info("Deploying shadow testing")
        
        # Deploy model A to all nodes for production traffic
        for node in test_config.edge_nodes:
            await self._deploy_model_to_node(test_config.test_id, node, "model_a", test_config.model_a_path)
        
        # Deploy model B for shadow testing (parallel inference without affecting results)
        for node in test_config.edge_nodes:
            await self._deploy_shadow_model_to_node(test_config.test_id, node, "model_b", test_config.model_b_path)
        
        return True
    
    async def _deploy_model_to_node(self, test_id: str, node: str, model_version: str, model_path: str):
        """Deploy model to specific edge node"""
        # Simulate model deployment
        self.logger.info(f"Deploying {model_version} to {node}: {model_path}")
        
        # Log deployment event
        await self._log_deployment_event(test_id, "model_deployed", node, model_version, 
                                       f"Deployed model from {model_path}")
        
        # In real implementation, this would:
        # 1. Copy model file to edge node
        # 2. Update edge node configuration
        # 3. Restart inference service
        # 4. Verify deployment health
        
        await asyncio.sleep(1)  # Simulate deployment time
    
    async def _deploy_shadow_model_to_node(self, test_id: str, node: str, model_version: str, model_path: str):
        """Deploy shadow model for parallel testing"""
        self.logger.info(f"Deploying shadow {model_version} to {node}: {model_path}")
        
        await self._log_deployment_event(test_id, "shadow_model_deployed", node, model_version,
                                       f"Deployed shadow model from {model_path}")
        
        await asyncio.sleep(1)  # Simulate deployment time
    
    async def _gradual_rollout_scheduler(self, test_config: ABTestConfig):
        """Schedule gradual rollout increases"""
        target_split = test_config.traffic_split['model_b']
        current_split = 0.1
        increment = 0.1
        interval_hours = test_config.duration_hours // 10  # 10 steps
        
        while current_split < target_split:
            await asyncio.sleep(interval_hours * 3600)  # Wait for interval
            
            # Check if test is still active
            if test_config.test_id not in self.active_tests:
                break
            
            # Increase traffic to model B
            current_split = min(current_split + increment, target_split)
            await self._update_traffic_split(test_config, current_split)
            
            self.logger.info(f"Increased model B traffic to {current_split:.1%}")
    
    async def _update_traffic_split(self, test_config: ABTestConfig, new_split: float):
        """Update traffic split during gradual rollout"""
        total_nodes = len(test_config.edge_nodes)
        model_b_nodes = int(total_nodes * new_split)
        
        # Redeploy models with new split
        for i, node in enumerate(test_config.edge_nodes):
            if i < model_b_nodes:
                await self._deploy_model_to_node(test_config.test_id, node, "model_b", test_config.model_b_path)
            else:
                await self._deploy_model_to_node(test_config.test_id, node, "model_a", test_config.model_a_path)
    
    async def _log_deployment_event(self, test_id: str, event_type: str, edge_node: str, 
                                  model_version: str, details: str):
        """Log deployment event"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO deployment_events 
                (test_id, event_type, edge_node, model_version, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (test_id, event_type, edge_node, model_version, details))
    
    async def _monitor_test(self, test_config: ABTestConfig):
        """Monitor A/B test progress and metrics"""
        self.logger.info(f"Starting monitoring for test {test_config.test_id}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=test_config.duration_hours)
        
        while datetime.now() < end_time and test_config.test_id in self.active_tests:
            try:
                # Collect metrics from edge nodes
                await self._collect_test_metrics(test_config)
                
                # Check rollback criteria
                if await self._should_rollback(test_config):
                    await self._rollback_test(test_config.test_id, "rollback_criteria_met")
                    break
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config['deployment']['health_check_interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Monitoring error for test {test_config.test_id}: {e}")
        
        # Test completed naturally
        if test_config.test_id in self.active_tests:
            await self._complete_test(test_config.test_id)
    
    async def _collect_test_metrics(self, test_config: ABTestConfig):
        """Collect metrics from edge nodes"""
        # Simulate metric collection
        for node in test_config.edge_nodes:
            # Generate simulated metrics
            model_a_metrics = self._generate_simulated_metrics("model_a", test_config.test_id)
            model_b_metrics = self._generate_simulated_metrics("model_b", test_config.test_id)
            
            # Save metrics
            await self._save_test_metrics(model_a_metrics)
            await self._save_test_metrics(model_b_metrics)
    
    def _generate_simulated_metrics(self, model_id: str, test_id: str) -> ABTestMetrics:
        """Generate simulated metrics for testing"""
        # Simulate model B being slightly better
        base_performance = {
            "inference_count": random.randint(50, 200),
            "avg_confidence": 0.75 + random.uniform(-0.1, 0.1),
            "detection_rate": 0.85 + random.uniform(-0.05, 0.05),
            "false_positive_rate": 0.08 + random.uniform(-0.02, 0.02),
            "false_negative_rate": 0.12 + random.uniform(-0.02, 0.02),
            "avg_inference_time_ms": 150 + random.uniform(-20, 20),
            "error_rate": 0.01 + random.uniform(0, 0.02)
        }
        
        # Model B gets slight improvement
        if model_id == "model_b":
            base_performance["avg_confidence"] += 0.03
            base_performance["detection_rate"] += 0.02
            base_performance["false_positive_rate"] -= 0.01
        
        return ABTestMetrics(
            model_id=model_id,
            test_id=test_id,
            timestamp=datetime.now(),
            **base_performance
        )
    
    async def _save_test_metrics(self, metrics: ABTestMetrics):
        """Save test metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO ab_test_metrics 
                (model_id, test_id, inference_count, avg_confidence, detection_rate,
                 false_positive_rate, false_negative_rate, avg_inference_time_ms, error_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.model_id,
                metrics.test_id,
                metrics.inference_count,
                metrics.avg_confidence,
                metrics.detection_rate,
                metrics.false_positive_rate,
                metrics.false_negative_rate,
                metrics.avg_inference_time_ms,
                metrics.error_rate
            ))
    
    async def _should_rollback(self, test_config: ABTestConfig) -> bool:
        """Check if test should be rolled back"""
        # Get recent metrics for model B
        recent_metrics = await self._get_recent_metrics(test_config.test_id, "model_b", hours=1)
        
        if not recent_metrics:
            return False
        
        # Check rollback criteria
        rollback_criteria = test_config.rollback_criteria
        
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        avg_fp_rate = np.mean([m.false_positive_rate for m in recent_metrics])
        avg_inference_time = np.mean([m.avg_inference_time_ms for m in recent_metrics])
        avg_detection_rate = np.mean([m.detection_rate for m in recent_metrics])
        
        if (avg_error_rate > rollback_criteria['max_error_rate'] or
            avg_fp_rate > rollback_criteria['max_false_positive_rate'] or
            avg_inference_time > rollback_criteria['max_inference_time_ms'] or
            avg_detection_rate < rollback_criteria['min_detection_rate']):
            
            self.logger.warning(f"Rollback criteria met for test {test_config.test_id}")
            return True
        
        return False
    
    async def _get_recent_metrics(self, test_id: str, model_id: str, hours: int) -> List[ABTestMetrics]:
        """Get recent metrics for a model"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        metrics = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT model_id, test_id, timestamp, inference_count, avg_confidence,
                       detection_rate, false_positive_rate, false_negative_rate,
                       avg_inference_time_ms, error_rate
                FROM ab_test_metrics 
                WHERE test_id = ? AND model_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (test_id, model_id, cutoff_time.isoformat()))
            
            for row in cursor.fetchall():
                metrics.append(ABTestMetrics(
                    model_id=row[0],
                    test_id=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    inference_count=row[3],
                    avg_confidence=row[4],
                    detection_rate=row[5],
                    false_positive_rate=row[6],
                    false_negative_rate=row[7],
                    avg_inference_time_ms=row[8],
                    error_rate=row[9]
                ))
        
        return metrics
    
    async def _rollback_test(self, test_id: str, reason: str):
        """Rollback A/B test"""
        self.logger.warning(f"Rolling back test {test_id}: {reason}")
        
        test_config = self.active_tests.get(test_id)
        if not test_config:
            return
        
        # Deploy model A to all nodes
        for node in test_config.edge_nodes:
            await self._deploy_model_to_node(test_id, node, "model_a", test_config.model_a_path)
        
        # Update test status
        await self._update_test_status(test_id, "rolled_back", completed_at=datetime.now(), 
                                     result=f"Rolled back: {reason}")
        
        # Remove from active tests
        if test_id in self.active_tests:
            del self.active_tests[test_id]
        
        await self._log_deployment_event(test_id, "test_rolled_back", "all", "model_a", reason)
    
    async def _complete_test(self, test_id: str):
        """Complete A/B test and analyze results"""
        self.logger.info(f"Completing test {test_id}")
        
        # Analyze test results
        results = await self._analyze_test_results(test_id)
        
        # Make deployment decision
        decision = await self._make_deployment_decision(test_id, results)
        
        # Update test status
        await self._update_test_status(test_id, "completed", completed_at=datetime.now(), 
                                     result=json.dumps({"analysis": results, "decision": decision}))
        
        # Remove from active tests
        if test_id in self.active_tests:
            del self.active_tests[test_id]
        
        self.logger.info(f"Test {test_id} completed with decision: {decision}")
    
    async def _analyze_test_results(self, test_id: str) -> Dict:
        """Analyze A/B test results"""
        # Get metrics for both models
        model_a_metrics = await self._get_recent_metrics(test_id, "model_a", hours=24)
        model_b_metrics = await self._get_recent_metrics(test_id, "model_b", hours=24)
        
        if not model_a_metrics or not model_b_metrics:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate aggregate metrics
        def aggregate_metrics(metrics_list):
            return {
                "avg_confidence": np.mean([m.avg_confidence for m in metrics_list]),
                "detection_rate": np.mean([m.detection_rate for m in metrics_list]),
                "false_positive_rate": np.mean([m.false_positive_rate for m in metrics_list]),
                "avg_inference_time_ms": np.mean([m.avg_inference_time_ms for m in metrics_list]),
                "error_rate": np.mean([m.error_rate for m in metrics_list]),
                "total_inferences": sum([m.inference_count for m in metrics_list])
            }
        
        model_a_agg = aggregate_metrics(model_a_metrics)
        model_b_agg = aggregate_metrics(model_b_metrics)
        
        # Calculate improvements
        improvements = {
            "confidence_improvement": model_b_agg["avg_confidence"] - model_a_agg["avg_confidence"],
            "detection_rate_improvement": model_b_agg["detection_rate"] - model_a_agg["detection_rate"],
            "false_positive_improvement": model_a_agg["false_positive_rate"] - model_b_agg["false_positive_rate"],
            "inference_time_change": model_b_agg["avg_inference_time_ms"] - model_a_agg["avg_inference_time_ms"],
            "error_rate_change": model_b_agg["error_rate"] - model_a_agg["error_rate"]
        }
        
        return {
            "model_a_metrics": model_a_agg,
            "model_b_metrics": model_b_agg,
            "improvements": improvements,
            "sample_sizes": {
                "model_a": len(model_a_metrics),
                "model_b": len(model_b_metrics)
            }
        }
    
    async def _make_deployment_decision(self, test_id: str, results: Dict) -> str:
        """Make deployment decision based on test results"""
        if "error" in results:
            return "insufficient_data"
        
        test_config = await self._load_test_config(test_id)
        if not test_config:
            return "config_error"
        
        improvements = results["improvements"]
        success_criteria = test_config.success_criteria
        
        # Check success criteria
        criteria_met = (
            improvements["detection_rate_improvement"] >= success_criteria["min_detection_rate_improvement"] and
            improvements["false_positive_improvement"] >= -success_criteria["max_false_positive_increase"] and
            improvements["inference_time_change"] <= success_criteria["max_inference_time_increase_ms"] and
            improvements["confidence_improvement"] >= success_criteria["min_confidence_improvement"]
        )
        
        if criteria_met:
            return "deploy_model_b"
        else:
            return "keep_model_a"
    
    async def _update_test_status(self, test_id: str, status: str, started_at: datetime = None, 
                                completed_at: datetime = None, result: str = None):
        """Update test status in database"""
        with sqlite3.connect(self.db_path) as conn:
            update_fields = ["status = ?"]
            values = [status]
            
            if started_at:
                update_fields.append("started_at = ?")
                values.append(started_at.isoformat())
            
            if completed_at:
                update_fields.append("completed_at = ?")
                values.append(completed_at.isoformat())
            
            if result:
                update_fields.append("result = ?")
                values.append(result)
            
            values.append(test_id)
            
            query = f"UPDATE ab_tests SET {', '.join(update_fields)} WHERE test_id = ?"
            conn.execute(query, values)
    
    async def get_active_tests(self) -> List[Dict]:
        """Get list of active A/B tests"""
        active_tests = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT test_id, model_a_path, model_b_path, strategy, status, started_at
                FROM ab_tests 
                WHERE status = 'running'
                ORDER BY started_at DESC
            ''')
            
            for row in cursor.fetchall():
                active_tests.append({
                    "test_id": row[0],
                    "model_a_path": row[1],
                    "model_b_path": row[2],
                    "strategy": row[3],
                    "status": row[4],
                    "started_at": row[5]
                })
        
        return active_tests
    
    async def get_test_results(self, test_id: str) -> Optional[Dict]:
        """Get results for a completed test"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT status, result, completed_at
                FROM ab_tests 
                WHERE test_id = ?
            ''', (test_id,))
            
            row = cursor.fetchone()
            if row and row[1]:  # Has result
                return {
                    "test_id": test_id,
                    "status": row[0],
                    "result": json.loads(row[1]) if row[1] else None,
                    "completed_at": row[2]
                }
        
        return None
    
    async def stop_test(self, test_id: str, reason: str = "manual_stop"):
        """Manually stop an A/B test"""
        if test_id in self.active_tests:
            await self._rollback_test(test_id, reason)
        else:
            await self._update_test_status(test_id, "stopped", completed_at=datetime.now(), 
                                         result=f"Manually stopped: {reason}")

async def main():
    """Example usage of A/B testing framework"""
    framework = ABTestingFramework()
    
    # Create A/B test
    test_config = await framework.create_ab_test(
        model_a_path="../models/current_model.pt",
        model_b_path="../models/retrained_model.pt",
        strategy=DeploymentStrategy.CANARY,
        traffic_split={"model_a": 0.8, "model_b": 0.2},
        duration_hours=24
    )
    
    print(f"Created A/B test: {test_config.test_id}")
    
    # Start test
    success = await framework.start_ab_test(test_config.test_id)
    print(f"Test started: {success}")
    
    # Monitor for a short time
    await asyncio.sleep(10)
    
    # Get active tests
    active_tests = await framework.get_active_tests()
    print(f"Active tests: {len(active_tests)}")
    
    # Stop test
    await framework.stop_test(test_config.test_id, "demo_complete")

if __name__ == "__main__":
    asyncio.run(main())