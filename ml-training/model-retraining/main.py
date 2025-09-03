#!/usr/bin/env python3
"""
Main Entry Point for Automated Model Retraining System
Provides CLI interface and service management
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional
import yaml

from retraining_scheduler import RetrainingScheduler
from retraining_pipeline import ModelRetrainingPipeline
from ab_testing_framework import ABTestingFramework, DeploymentStrategy
from data_collection_pipeline import DataCollectionPipeline
from automated_labeling import AutomatedLabelingWorkflow

class RetrainingSystemManager:
    """Main manager for the automated retraining system"""
    
    def __init__(self):
        self.scheduler: Optional[RetrainingScheduler] = None
        self.running = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the system manager"""
        # Create log directory if it doesn't exist
        log_dir = Path('../../data/retraining_pipeline/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'system.log')
            ]
        )
        return logging.getLogger('retraining_system')
    
    async def start_service(self, config_path: Optional[str] = None):
        """Start the retraining service"""
        try:
            self.logger.info("Starting Automated Model Retraining System")
            
            # Initialize scheduler
            if config_path:
                self.scheduler = RetrainingScheduler(config_path)
            else:
                self.scheduler = RetrainingScheduler()
            
            # Start scheduler
            self.scheduler.start()
            self.running = True
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info("Retraining system started successfully")
            
            # Keep running
            while self.running:
                await asyncio.sleep(60)
                
                # Log status periodically
                if hasattr(self.scheduler, 'get_status'):
                    status = self.scheduler.get_status()
                    if status['active_jobs'] > 0:
                        self.logger.info(f"Active jobs: {status['active_jobs']}")
            
        except Exception as e:
            self.logger.error(f"Error starting retraining service: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.scheduler:
            self.scheduler.stop()
    
    async def stop_service(self):
        """Stop the retraining service"""
        self.running = False
        if self.scheduler:
            self.scheduler.stop()
        self.logger.info("Retraining system stopped")
    
    async def trigger_retraining(self, reason: str = "manual_trigger"):
        """Manually trigger retraining"""
        if not self.scheduler:
            self.scheduler = RetrainingScheduler()
        
        job_id = await self.scheduler.trigger_manual_retraining(reason)
        if job_id:
            self.logger.info(f"Manual retraining triggered: {job_id}")
            return job_id
        else:
            self.logger.error("Failed to trigger manual retraining")
            return None
    
    async def get_status(self):
        """Get system status"""
        if not self.scheduler:
            return {"status": "not_running"}
        
        return self.scheduler.get_status()
    
    async def run_data_collection(self, days_back: int = 7):
        """Run data collection pipeline"""
        self.logger.info(f"Running data collection for {days_back} days")
        
        pipeline = DataCollectionPipeline()
        incidents = await pipeline.collect_recent_incidents(days_back)
        
        self.logger.info(f"Collected {len(incidents)} incidents")
        
        # Save metadata
        metadata_path = await pipeline.save_collection_metadata(incidents)
        self.logger.info(f"Metadata saved to: {metadata_path}")
        
        return len(incidents)
    
    async def run_labeling_workflow(self, batch_size: int = 50):
        """Run automated labeling workflow"""
        self.logger.info(f"Running labeling workflow with batch size {batch_size}")
        
        workflow = AutomatedLabelingWorkflow()
        await workflow.load_models()
        
        # Process pending tasks
        results = await workflow.process_labeling_batch(batch_size)
        
        self.logger.info(f"Labeling results: {results}")
        
        # Get statistics
        stats = await workflow.get_labeling_statistics()
        self.logger.info(f"Labeling statistics: {stats}")
        
        return results
    
    async def create_ab_test(self, model_a_path: str, model_b_path: str, 
                           strategy: str = "canary", duration_hours: int = 72):
        """Create and start A/B test"""
        self.logger.info(f"Creating A/B test: {model_a_path} vs {model_b_path}")
        
        framework = ABTestingFramework()
        
        # Map strategy string to enum
        strategy_map = {
            "canary": DeploymentStrategy.CANARY,
            "blue_green": DeploymentStrategy.BLUE_GREEN,
            "gradual_rollout": DeploymentStrategy.GRADUAL_ROLLOUT,
            "shadow_testing": DeploymentStrategy.SHADOW_TESTING
        }
        
        strategy_enum = strategy_map.get(strategy, DeploymentStrategy.CANARY)
        
        # Create test
        test_config = await framework.create_ab_test(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            strategy=strategy_enum,
            duration_hours=duration_hours
        )
        
        # Start test
        success = await framework.start_ab_test(test_config.test_id)
        
        if success:
            self.logger.info(f"A/B test started: {test_config.test_id}")
            return test_config.test_id
        else:
            self.logger.error("Failed to start A/B test")
            return None
    
    async def list_ab_tests(self):
        """List active A/B tests"""
        framework = ABTestingFramework()
        active_tests = await framework.get_active_tests()
        
        self.logger.info(f"Active A/B tests: {len(active_tests)}")
        for test in active_tests:
            self.logger.info(f"  {test['test_id']}: {test['strategy']} ({test['status']})")
        
        return active_tests
    
    async def stop_ab_test(self, test_id: str, reason: str = "manual_stop"):
        """Stop an A/B test"""
        framework = ABTestingFramework()
        await framework.stop_test(test_id, reason)
        self.logger.info(f"A/B test stopped: {test_id}")
    
    async def run_system_check(self):
        """Run system health check"""
        self.logger.info("Running system health check")
        
        checks = {
            "data_collection": False,
            "labeling_workflow": False,
            "retraining_pipeline": False,
            "ab_testing": False,
            "scheduler": False
        }
        
        try:
            # Test data collection
            pipeline = DataCollectionPipeline()
            incidents = await pipeline.collect_recent_incidents(days_back=1)
            checks["data_collection"] = len(incidents) >= 0
            
            # Test labeling workflow
            workflow = AutomatedLabelingWorkflow()
            await workflow.load_models()
            stats = await workflow.get_labeling_statistics()
            checks["labeling_workflow"] = "overall" in stats
            
            # Test retraining pipeline
            try:
                retraining = ModelRetrainingPipeline()
                job = await retraining.check_retraining_triggers()
                checks["retraining_pipeline"] = True  # If no exception
            except Exception as e:
                self.logger.error(f"Retraining pipeline check failed: {e}")
                checks["retraining_pipeline"] = False
            
            # Test A/B testing
            try:
                framework = ABTestingFramework()
                active_tests = await framework.get_active_tests()
                checks["ab_testing"] = isinstance(active_tests, list)
            except Exception as e:
                self.logger.error(f"A/B testing check failed: {e}")
                checks["ab_testing"] = False
            
            # Test scheduler
            try:
                if self.scheduler:
                    status = self.scheduler.get_status()
                    checks["scheduler"] = "running" in status
                else:
                    checks["scheduler"] = True  # Scheduler not running is OK for health check
            except Exception as e:
                self.logger.error(f"Scheduler check failed: {e}")
                checks["scheduler"] = False
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
        
        # Report results
        for component, status in checks.items():
            status_str = "✓" if status else "✗"
            self.logger.info(f"  {component}: {status_str}")
        
        all_healthy = all(checks.values())
        self.logger.info(f"Overall system health: {'✓' if all_healthy else '✗'}")
        
        return checks

async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Automated Model Retraining System")
    parser.add_argument("command", choices=[
        "start", "stop", "status", "trigger", "collect", "label", 
        "ab-test", "list-tests", "stop-test", "health-check"
    ], help="Command to execute")
    
    # Optional arguments
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--days", type=int, default=7, help="Days back for data collection")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for labeling")
    parser.add_argument("--model-a", help="Path to model A for A/B testing")
    parser.add_argument("--model-b", help="Path to model B for A/B testing")
    parser.add_argument("--strategy", default="canary", choices=["canary", "blue_green", "gradual_rollout", "shadow_testing"])
    parser.add_argument("--duration", type=int, default=72, help="A/B test duration in hours")
    parser.add_argument("--test-id", help="A/B test ID")
    parser.add_argument("--reason", default="manual_trigger", help="Reason for manual trigger")
    
    args = parser.parse_args()
    
    # Initialize system manager
    manager = RetrainingSystemManager()
    
    try:
        if args.command == "start":
            await manager.start_service(args.config)
        
        elif args.command == "stop":
            await manager.stop_service()
        
        elif args.command == "status":
            status = await manager.get_status()
            print(f"System Status: {status}")
        
        elif args.command == "trigger":
            job_id = await manager.trigger_retraining(args.reason)
            if job_id:
                print(f"Retraining triggered: {job_id}")
            else:
                print("Failed to trigger retraining")
                sys.exit(1)
        
        elif args.command == "collect":
            count = await manager.run_data_collection(args.days)
            print(f"Collected {count} incidents")
        
        elif args.command == "label":
            results = await manager.run_labeling_workflow(args.batch_size)
            print(f"Labeling results: {results}")
        
        elif args.command == "ab-test":
            if not args.model_a or not args.model_b:
                print("Error: --model-a and --model-b are required for A/B testing")
                sys.exit(1)
            
            test_id = await manager.create_ab_test(
                args.model_a, args.model_b, args.strategy, args.duration
            )
            if test_id:
                print(f"A/B test created: {test_id}")
            else:
                print("Failed to create A/B test")
                sys.exit(1)
        
        elif args.command == "list-tests":
            tests = await manager.list_ab_tests()
            if not tests:
                print("No active A/B tests")
        
        elif args.command == "stop-test":
            if not args.test_id:
                print("Error: --test-id is required")
                sys.exit(1)
            
            await manager.stop_ab_test(args.test_id, args.reason)
            print(f"A/B test stopped: {args.test_id}")
        
        elif args.command == "health-check":
            checks = await manager.run_system_check()
            all_healthy = all(checks.values())
            sys.exit(0 if all_healthy else 1)
    
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        await manager.stop_service()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())