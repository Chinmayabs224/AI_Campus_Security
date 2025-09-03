"""
Automated Model Retraining Scheduler
Main orchestrator for the automated retraining system
"""

import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
from dataclasses import dataclass
import threading

from retraining_pipeline import ModelRetrainingPipeline, RetrainingJob
from ab_testing_framework import ABTestingFramework, DeploymentStrategy

@dataclass
class SchedulerConfig:
    """Configuration for retraining scheduler"""
    check_interval_hours: int = 6
    max_concurrent_jobs: int = 2
    enable_auto_deployment: bool = False
    notification_webhook: Optional[str] = None
    backup_retention_days: int = 30

class RetrainingScheduler:
    """Main scheduler for automated model retraining"""
    
    def __init__(self, config_path: str = "config/scheduler_config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.retraining_pipeline = ModelRetrainingPipeline()
        self.ab_testing_framework = ABTestingFramework()
        
        # Scheduler state
        self.running = False
        self.active_jobs: Dict[str, RetrainingJob] = {}
        self.scheduler_thread = None
        
        # Setup scheduled tasks
        self._setup_schedule()
        
    def _load_config(self, config_path: str) -> SchedulerConfig:
        """Load scheduler configuration"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                return SchedulerConfig(**config_data.get('scheduler', {}))
        except FileNotFoundError:
            print(f"Config file not found: {config_path}, using defaults")
            return SchedulerConfig()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for scheduler"""
        logger = logging.getLogger('retraining_scheduler')
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
            log_dir = Path("../../data/retraining_pipeline/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "scheduler.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_schedule(self):
        """Setup scheduled tasks"""
        # Check for retraining triggers every N hours
        schedule.every(self.config.check_interval_hours).hours.do(self._check_retraining_triggers)
        
        # Daily cleanup tasks
        schedule.every().day.at("02:00").do(self._daily_cleanup)
        
        # Weekly model performance review
        schedule.every().sunday.at("03:00").do(self._weekly_performance_review)
        
        self.logger.info("Scheduled tasks configured")
    
    def start(self):
        """Start the retraining scheduler"""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.logger.info("Starting automated retraining scheduler")
        
        # Start scheduler in separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        # Start async event loop for retraining jobs
        asyncio.create_task(self._monitor_active_jobs())
        
        self.logger.info("Retraining scheduler started successfully")
    
    def stop(self):
        """Stop the retraining scheduler"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping retraining scheduler")
        
        # Wait for active jobs to complete
        if self.active_jobs:
            self.logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete")
            # In production, implement graceful shutdown
        
        self.logger.info("Retraining scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduled tasks"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    async def _check_retraining_triggers(self):
        """Check if retraining should be triggered"""
        try:
            self.logger.info("Checking retraining triggers")
            
            # Check if we have capacity for new jobs
            if len(self.active_jobs) >= self.config.max_concurrent_jobs:
                self.logger.info("Maximum concurrent jobs reached, skipping trigger check")
                return
            
            # Check for retraining triggers
            job = await self.retraining_pipeline.check_retraining_triggers()
            
            if job:
                self.logger.info(f"Retraining triggered: {job.trigger_reason}")
                await self._start_retraining_job(job)
            else:
                self.logger.info("No retraining triggers detected")
                
        except Exception as e:
            self.logger.error(f"Error checking retraining triggers: {e}")
    
    async def _start_retraining_job(self, job: RetrainingJob):
        """Start a retraining job"""
        try:
            self.logger.info(f"Starting retraining job: {job.job_id}")
            
            # Add to active jobs
            self.active_jobs[job.job_id] = job
            
            # Execute retraining pipeline
            success = await self.retraining_pipeline.execute_retraining_job(job)
            
            if success:
                self.logger.info(f"Retraining job {job.job_id} completed successfully")
                
                # If auto-deployment is enabled and job was successful
                if self.config.enable_auto_deployment and job.deployment_approved:
                    await self._start_ab_test(job)
                else:
                    await self._notify_retraining_complete(job)
            else:
                self.logger.error(f"Retraining job {job.job_id} failed")
                await self._notify_retraining_failed(job)
            
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
                
        except Exception as e:
            self.logger.error(f"Error in retraining job {job.job_id}: {e}")
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _start_ab_test(self, job: RetrainingJob):
        """Start A/B test for newly trained model"""
        try:
            self.logger.info(f"Starting A/B test for job {job.job_id}")
            
            # Create A/B test configuration
            test_config = await self.ab_testing_framework.create_ab_test(
                model_a_path=job.current_model_path or "../models/current_model.pt",
                model_b_path=job.new_model_path,
                strategy=DeploymentStrategy.CANARY,
                traffic_split={"model_a": 0.9, "model_b": 0.1},
                duration_hours=72
            )
            
            # Start the test
            success = await self.ab_testing_framework.start_ab_test(test_config.test_id)
            
            if success:
                self.logger.info(f"A/B test {test_config.test_id} started for job {job.job_id}")
                await self._notify_ab_test_started(job, test_config.test_id)
            else:
                self.logger.error(f"Failed to start A/B test for job {job.job_id}")
                await self._notify_ab_test_failed(job)
                
        except Exception as e:
            self.logger.error(f"Error starting A/B test for job {job.job_id}: {e}")
    
    async def _monitor_active_jobs(self):
        """Monitor active retraining jobs"""
        while self.running:
            try:
                # Check job status and health
                for job_id, job in list(self.active_jobs.items()):
                    # Check if job has been running too long
                    if job.created_at:
                        runtime = datetime.now() - job.created_at
                        if runtime > timedelta(hours=24):  # 24 hour timeout
                            self.logger.warning(f"Job {job_id} has been running for {runtime}, may be stuck")
                
                # Monitor A/B tests
                active_tests = await self.ab_testing_framework.get_active_tests()
                for test in active_tests:
                    self.logger.debug(f"Monitoring A/B test: {test['test_id']}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring active jobs: {e}")
                await asyncio.sleep(60)
    
    async def _daily_cleanup(self):
        """Perform daily cleanup tasks"""
        try:
            self.logger.info("Performing daily cleanup")
            
            # Clean up old model files
            await self._cleanup_old_models()
            
            # Clean up old logs
            await self._cleanup_old_logs()
            
            # Clean up temporary files
            await self._cleanup_temp_files()
            
            # Generate daily report
            await self._generate_daily_report()
            
            self.logger.info("Daily cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error in daily cleanup: {e}")
    
    async def _weekly_performance_review(self):
        """Perform weekly performance review"""
        try:
            self.logger.info("Performing weekly performance review")
            
            # Analyze model performance trends
            performance_report = await self._analyze_weekly_performance()
            
            # Check for performance degradation patterns
            degradation_alerts = await self._check_performance_degradation()
            
            # Generate recommendations
            recommendations = await self._generate_performance_recommendations()
            
            # Send weekly report
            await self._send_weekly_report({
                'performance': performance_report,
                'alerts': degradation_alerts,
                'recommendations': recommendations
            })
            
            self.logger.info("Weekly performance review completed")
            
        except Exception as e:
            self.logger.error(f"Error in weekly performance review: {e}")
    
    async def _cleanup_old_models(self):
        """Clean up old model files"""
        models_dir = Path("../../data/retraining_pipeline/models")
        if not models_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
        
        for model_file in models_dir.glob("*.pt"):
            if model_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    model_file.unlink()
                    self.logger.info(f"Deleted old model: {model_file}")
                except Exception as e:
                    self.logger.error(f"Error deleting model {model_file}: {e}")
    
    async def _cleanup_old_logs(self):
        """Clean up old log files"""
        logs_dir = Path("../../data/retraining_pipeline/logs")
        if not logs_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for log_file in logs_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    self.logger.info(f"Deleted old log: {log_file}")
                except Exception as e:
                    self.logger.error(f"Error deleting log {log_file}: {e}")
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dirs = [
            Path("../../data/retraining"),
            Path("../../data/labeling"),
            Path("../../data/ab_testing")
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                # Clean up files older than 7 days
                cutoff_date = datetime.now() - timedelta(days=7)
                
                for temp_file in temp_dir.rglob("*"):
                    if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_date.timestamp():
                        try:
                            temp_file.unlink()
                        except Exception as e:
                            self.logger.error(f"Error deleting temp file {temp_file}: {e}")
    
    async def _generate_daily_report(self):
        """Generate daily activity report"""
        report = {
            'date': datetime.now().date().isoformat(),
            'active_jobs': len(self.active_jobs),
            'completed_jobs_today': 0,  # Would query database
            'failed_jobs_today': 0,     # Would query database
            'active_ab_tests': len(await self.ab_testing_framework.get_active_tests())
        }
        
        # Save report
        reports_dir = Path("../../data/retraining_pipeline/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Daily report generated: {report_file}")
    
    async def _analyze_weekly_performance(self) -> Dict:
        """Analyze weekly performance trends"""
        # This would analyze model performance over the past week
        return {
            'avg_detection_rate': 0.85,
            'avg_false_positive_rate': 0.08,
            'avg_inference_time_ms': 145,
            'total_inferences': 50000,
            'trend': 'stable'
        }
    
    async def _check_performance_degradation(self) -> List[Dict]:
        """Check for performance degradation patterns"""
        # This would check for concerning performance trends
        return []
    
    async def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        return [
            "Consider increasing training data for 'violence' class",
            "Monitor false positive rate in low-light conditions",
            "Evaluate model performance on edge devices"
        ]
    
    async def _send_weekly_report(self, report_data: Dict):
        """Send weekly performance report"""
        # This would send the report via email, webhook, etc.
        self.logger.info("Weekly report generated")
    
    async def _notify_retraining_complete(self, job: RetrainingJob):
        """Notify that retraining is complete"""
        message = f"Model retraining completed for job {job.job_id}"
        self.logger.info(message)
        
        # In production, send notification via webhook, email, etc.
        if self.config.notification_webhook:
            # Send webhook notification
            pass
    
    async def _notify_retraining_failed(self, job: RetrainingJob):
        """Notify that retraining failed"""
        message = f"Model retraining failed for job {job.job_id}"
        self.logger.error(message)
        
        # In production, send alert notification
        if self.config.notification_webhook:
            # Send webhook notification
            pass
    
    async def _notify_ab_test_started(self, job: RetrainingJob, test_id: str):
        """Notify that A/B test started"""
        message = f"A/B test {test_id} started for retraining job {job.job_id}"
        self.logger.info(message)
    
    async def _notify_ab_test_failed(self, job: RetrainingJob):
        """Notify that A/B test failed to start"""
        message = f"A/B test failed to start for retraining job {job.job_id}"
        self.logger.error(message)
    
    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            'running': self.running,
            'active_jobs': len(self.active_jobs),
            'config': {
                'check_interval_hours': self.config.check_interval_hours,
                'max_concurrent_jobs': self.config.max_concurrent_jobs,
                'enable_auto_deployment': self.config.enable_auto_deployment
            },
            'next_check': schedule.next_run().isoformat() if schedule.jobs else None
        }
    
    async def trigger_manual_retraining(self, reason: str = "manual_trigger") -> Optional[str]:
        """Manually trigger retraining"""
        try:
            # Create manual retraining job
            job = await self.retraining_pipeline._create_retraining_job([reason])
            
            # Start the job
            await self._start_retraining_job(job)
            
            return job.job_id
            
        except Exception as e:
            self.logger.error(f"Error triggering manual retraining: {e}")
            return None

async def main():
    """Main function for running the scheduler"""
    scheduler = RetrainingScheduler()
    
    try:
        # Start scheduler
        scheduler.start()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
            # Print status every hour
            if datetime.now().minute == 0:
                status = scheduler.get_status()
                print(f"Scheduler status: {status}")
    
    except KeyboardInterrupt:
        print("Shutting down scheduler...")
        scheduler.stop()

if __name__ == "__main__":
    asyncio.run(main())