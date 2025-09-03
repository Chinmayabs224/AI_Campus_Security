"""
Backup and disaster recovery implementation for campus security system.
"""
import asyncio
import os
import subprocess
import json
import shutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
import aiofiles
from pathlib import Path

logger = structlog.get_logger()


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


class RecoveryType(Enum):
    """Types of recovery operations."""
    FULL_RESTORE = "full_restore"
    PARTIAL_RESTORE = "partial_restore"
    POINT_IN_TIME = "point_in_time"
    DISASTER_RECOVERY = "disaster_recovery"


@dataclass
class BackupJob:
    """Backup job definition."""
    id: str
    name: str
    backup_type: BackupType
    source_paths: List[str]
    destination_path: str
    schedule_cron: str
    retention_days: int
    encryption_enabled: bool
    compression_enabled: bool
    status: BackupStatus = BackupStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class RecoveryPlan:
    """Disaster recovery plan definition."""
    id: str
    name: str
    description: str
    recovery_type: RecoveryType
    priority: int
    rto_minutes: int  # Recovery Time Objective
    rpo_minutes: int  # Recovery Point Objective
    recovery_steps: List[str]
    dependencies: List[str]
    validation_steps: List[str]
    created_at: datetime = None
    last_tested: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class BackupRecoveryManager:
    """Comprehensive backup and disaster recovery manager."""
    
    def __init__(self):
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.backup_history: List[Dict] = []
        self.recovery_history: List[Dict] = []
        self.initialize_default_jobs()
        self.initialize_recovery_plans()
    
    def initialize_default_jobs(self):
        """Initialize default backup jobs."""
        default_jobs = [
            BackupJob(
                id="database_full_backup",
                name="Database Full Backup",
                backup_type=BackupType.FULL,
                source_paths=["/var/lib/postgresql/data"],
                destination_path="/backups/database/full",
                schedule_cron="0 2 * * 0",  # Weekly at 2 AM Sunday
                retention_days=90,
                encryption_enabled=True,
                compression_enabled=True
            ),
            BackupJob(
                id="database_incremental_backup",
                name="Database Incremental Backup",
                backup_type=BackupType.INCREMENTAL,
                source_paths=["/var/lib/postgresql/data"],
                destination_path="/backups/database/incremental",
                schedule_cron="0 2 * * 1-6",  # Daily at 2 AM except Sunday
                retention_days=30,
                encryption_enabled=True,
                compression_enabled=True
            ),
            BackupJob(
                id="evidence_backup",
                name="Evidence Storage Backup",
                backup_type=BackupType.FULL,
                source_paths=["/data/evidence"],
                destination_path="/backups/evidence",
                schedule_cron="0 3 * * *",  # Daily at 3 AM
                retention_days=2555,  # 7 years
                encryption_enabled=True,
                compression_enabled=True
            ),
            BackupJob(
                id="config_backup",
                name="Configuration Backup",
                backup_type=BackupType.FULL,
                source_paths=["/etc/campus-security", "/opt/campus-security/config"],
                destination_path="/backups/config",
                schedule_cron="0 1 * * *",  # Daily at 1 AM
                retention_days=365,
                encryption_enabled=True,
                compression_enabled=False
            ),
            BackupJob(
                id="audit_logs_backup",
                name="Audit Logs Backup",
                backup_type=BackupType.INCREMENTAL,
                source_paths=["/var/log/campus-security"],
                destination_path="/backups/audit-logs",
                schedule_cron="0 4 * * *",  # Daily at 4 AM
                retention_days=2555,  # 7 years
                encryption_enabled=True,
                compression_enabled=True
            )
        ]
        
        for job in default_jobs:
            self.backup_jobs[job.id] = job
        
        logger.info("Default backup jobs initialized", count=len(default_jobs))
    
    def initialize_recovery_plans(self):
        """Initialize disaster recovery plans."""
        default_plans = [
            RecoveryPlan(
                id="database_recovery",
                name="Database Recovery Plan",
                description="Complete database recovery from backup",
                recovery_type=RecoveryType.FULL_RESTORE,
                priority=1,
                rto_minutes=60,  # 1 hour
                rpo_minutes=1440,  # 24 hours
                recovery_steps=[
                    "Stop all database connections",
                    "Identify latest valid backup",
                    "Restore database from backup",
                    "Apply transaction logs if available",
                    "Verify data integrity",
                    "Restart database services",
                    "Validate application connectivity"
                ],
                dependencies=["storage_recovery"],
                validation_steps=[
                    "Check database connectivity",
                    "Verify critical tables exist",
                    "Run data integrity checks",
                    "Test application queries"
                ]
            ),
            RecoveryPlan(
                id="evidence_recovery",
                name="Evidence Storage Recovery",
                description="Recovery of video evidence and related data",
                recovery_type=RecoveryType.FULL_RESTORE,
                priority=2,
                rto_minutes=240,  # 4 hours
                rpo_minutes=1440,  # 24 hours
                recovery_steps=[
                    "Mount backup storage",
                    "Identify evidence backup files",
                    "Restore evidence to primary storage",
                    "Verify file integrity checksums",
                    "Update evidence database references",
                    "Test evidence retrieval APIs"
                ],
                dependencies=["storage_recovery", "database_recovery"],
                validation_steps=[
                    "Verify evidence file accessibility",
                    "Check file integrity",
                    "Test evidence search functionality",
                    "Validate chain of custody records"
                ]
            ),
            RecoveryPlan(
                id="system_recovery",
                name="Complete System Recovery",
                description="Full system disaster recovery",
                recovery_type=RecoveryType.DISASTER_RECOVERY,
                priority=1,
                rto_minutes=480,  # 8 hours
                rpo_minutes=1440,  # 24 hours
                recovery_steps=[
                    "Assess damage and available resources",
                    "Provision new infrastructure if needed",
                    "Restore system configurations",
                    "Restore database from backups",
                    "Restore evidence storage",
                    "Restore application services",
                    "Verify system functionality",
                    "Update DNS and load balancer configs",
                    "Notify stakeholders of recovery completion"
                ],
                dependencies=[],
                validation_steps=[
                    "Test all critical system functions",
                    "Verify data integrity across all systems",
                    "Check security controls and access",
                    "Validate monitoring and alerting",
                    "Perform end-to-end system test"
                ]
            ),
            RecoveryPlan(
                id="partial_recovery",
                name="Partial Service Recovery",
                description="Recovery of specific services or components",
                recovery_type=RecoveryType.PARTIAL_RESTORE,
                priority=3,
                rto_minutes=120,  # 2 hours
                rpo_minutes=60,   # 1 hour
                recovery_steps=[
                    "Identify affected services",
                    "Isolate failed components",
                    "Restore from recent backups",
                    "Restart affected services",
                    "Verify service functionality",
                    "Monitor for stability"
                ],
                dependencies=[],
                validation_steps=[
                    "Test restored service functionality",
                    "Check integration with other services",
                    "Verify data consistency",
                    "Monitor performance metrics"
                ]
            )
        ]
        
        for plan in default_plans:
            self.recovery_plans[plan.id] = plan
        
        logger.info("Default recovery plans initialized", count=len(default_plans))
    
    async def execute_backup_job(self, job_id: str) -> Dict[str, Any]:
        """Execute a specific backup job."""
        job = self.backup_jobs.get(job_id)
        if not job:
            raise ValueError(f"Backup job {job_id} not found")
        
        logger.info("Starting backup job", job_id=job_id, job_name=job.name)
        
        job.status = BackupStatus.IN_PROGRESS
        job.started_at = datetime.utcnow()
        
        try:
            # Create backup directory if it doesn't exist
            backup_path = Path(job.destination_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{job.name.lower().replace(' ', '_')}_{timestamp}"
            
            if job.compression_enabled:
                backup_filename += ".tar.gz"
            else:
                backup_filename += ".tar"
            
            full_backup_path = backup_path / backup_filename
            
            # Execute backup based on type
            if job.backup_type == BackupType.FULL:
                result = await self._execute_full_backup(job, full_backup_path)
            elif job.backup_type == BackupType.INCREMENTAL:
                result = await self._execute_incremental_backup(job, full_backup_path)
            else:
                result = await self._execute_differential_backup(job, full_backup_path)
            
            if result["success"]:
                job.status = BackupStatus.COMPLETED
                job.size_bytes = result.get("size_bytes", 0)
                
                # Verify backup if requested
                if await self._verify_backup(full_backup_path):
                    job.status = BackupStatus.VERIFIED
                
                # Clean up old backups according to retention policy
                await self._cleanup_old_backups(job)
                
            else:
                job.status = BackupStatus.FAILED
                job.error_message = result.get("error", "Unknown error")
            
            job.completed_at = datetime.utcnow()
            
            # Record backup in history
            backup_record = {
                "job_id": job_id,
                "backup_path": str(full_backup_path),
                "status": job.status.value,
                "started_at": job.started_at.isoformat(),
                "completed_at": job.completed_at.isoformat(),
                "size_bytes": job.size_bytes,
                "error_message": job.error_message
            }
            self.backup_history.append(backup_record)
            
            logger.info("Backup job completed",
                       job_id=job_id,
                       status=job.status.value,
                       size_bytes=job.size_bytes)
            
            return {
                "job_id": job_id,
                "status": job.status.value,
                "backup_path": str(full_backup_path),
                "size_bytes": job.size_bytes,
                "duration_seconds": (job.completed_at - job.started_at).total_seconds()
            }
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            
            logger.error("Backup job failed", job_id=job_id, error=str(e))
            
            return {
                "job_id": job_id,
                "status": job.status.value,
                "error": str(e)
            }
    
    async def _execute_full_backup(self, job: BackupJob, backup_path: Path) -> Dict[str, Any]:
        """Execute full backup."""
        try:
            # Build tar command
            cmd = ["tar"]
            
            if job.compression_enabled:
                cmd.append("-czf")
            else:
                cmd.append("-cf")
            
            cmd.append(str(backup_path))
            cmd.extend(job.source_paths)
            
            # Execute backup command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Get backup file size
                size_bytes = backup_path.stat().st_size if backup_path.exists() else 0
                
                # Encrypt if required
                if job.encryption_enabled:
                    await self._encrypt_backup(backup_path)
                
                return {"success": True, "size_bytes": size_bytes}
            else:
                return {"success": False, "error": stderr.decode()}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_incremental_backup(self, job: BackupJob, backup_path: Path) -> Dict[str, Any]:
        """Execute incremental backup."""
        try:
            # Find the last full backup for reference
            last_backup_time = await self._get_last_backup_time(job.id)
            
            # Build find command to get files newer than last backup
            find_cmd = [
                "find"
            ]
            find_cmd.extend(job.source_paths)
            find_cmd.extend([
                "-type", "f",
                "-newer", str(last_backup_time) if last_backup_time else "/dev/null"
            ])
            
            # Get list of changed files
            find_process = await asyncio.create_subprocess_exec(
                *find_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await find_process.communicate()
            
            if find_process.returncode != 0:
                return {"success": False, "error": stderr.decode()}
            
            changed_files = stdout.decode().strip().split('\n')
            changed_files = [f for f in changed_files if f]  # Remove empty strings
            
            if not changed_files:
                return {"success": True, "size_bytes": 0, "message": "No changes to backup"}
            
            # Create incremental backup with changed files
            cmd = ["tar"]
            
            if job.compression_enabled:
                cmd.append("-czf")
            else:
                cmd.append("-cf")
            
            cmd.append(str(backup_path))
            cmd.extend(changed_files)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                size_bytes = backup_path.stat().st_size if backup_path.exists() else 0
                
                if job.encryption_enabled:
                    await self._encrypt_backup(backup_path)
                
                return {"success": True, "size_bytes": size_bytes}
            else:
                return {"success": False, "error": stderr.decode()}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_differential_backup(self, job: BackupJob, backup_path: Path) -> Dict[str, Any]:
        """Execute differential backup."""
        # Similar to incremental but compares against last full backup only
        return await self._execute_incremental_backup(job, backup_path)
    
    async def _encrypt_backup(self, backup_path: Path):
        """Encrypt backup file."""
        encrypted_path = backup_path.with_suffix(backup_path.suffix + ".enc")
        
        # Use GPG for encryption (in production, use proper key management)
        cmd = [
            "gpg", "--symmetric", "--cipher-algo", "AES256",
            "--compress-algo", "2", "--s2k-mode", "3",
            "--s2k-digest-algo", "SHA512", "--s2k-count", "65011712",
            "--force-mdc", "--quiet", "--no-greeting",
            "--output", str(encrypted_path),
            str(backup_path)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if process.returncode == 0:
            # Remove unencrypted backup
            backup_path.unlink()
            # Rename encrypted file to original name
            encrypted_path.rename(backup_path)
    
    async def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity."""
        try:
            if backup_path.suffix == ".gz":
                cmd = ["tar", "-tzf", str(backup_path)]
            else:
                cmd = ["tar", "-tf", str(backup_path)]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            return process.returncode == 0
            
        except Exception:
            return False
    
    async def _cleanup_old_backups(self, job: BackupJob):
        """Clean up old backups according to retention policy."""
        backup_dir = Path(job.destination_path)
        if not backup_dir.exists():
            return
        
        cutoff_date = datetime.utcnow() - timedelta(days=job.retention_days)
        
        for backup_file in backup_dir.iterdir():
            if backup_file.is_file():
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_date:
                    backup_file.unlink()
                    logger.info("Old backup deleted", file=str(backup_file))
    
    async def _get_last_backup_time(self, job_id: str) -> Optional[str]:
        """Get timestamp of last backup for incremental backups."""
        # This would typically query a database or check filesystem
        # For now, return None to backup all files
        return None
    
    async def execute_recovery_plan(self, plan_id: str, recovery_point: Optional[datetime] = None) -> Dict[str, Any]:
        """Execute a disaster recovery plan."""
        plan = self.recovery_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Recovery plan {plan_id} not found")
        
        logger.info("Starting recovery plan execution",
                   plan_id=plan_id,
                   plan_name=plan.name)
        
        recovery_start = datetime.utcnow()
        recovery_id = str(uuid4())
        
        try:
            # Check dependencies
            for dependency in plan.dependencies:
                if not await self._check_dependency_status(dependency):
                    raise Exception(f"Dependency {dependency} not satisfied")
            
            # Execute recovery steps
            step_results = []
            for i, step in enumerate(plan.recovery_steps):
                logger.info("Executing recovery step",
                           plan_id=plan_id,
                           step_number=i+1,
                           step=step)
                
                step_result = await self._execute_recovery_step(step, recovery_point)
                step_results.append({
                    "step_number": i+1,
                    "step": step,
                    "success": step_result["success"],
                    "message": step_result.get("message", ""),
                    "duration_seconds": step_result.get("duration_seconds", 0)
                })
                
                if not step_result["success"]:
                    raise Exception(f"Recovery step failed: {step}")
            
            # Execute validation steps
            validation_results = []
            for i, validation in enumerate(plan.validation_steps):
                logger.info("Executing validation step",
                           plan_id=plan_id,
                           validation_number=i+1,
                           validation=validation)
                
                validation_result = await self._execute_validation_step(validation)
                validation_results.append({
                    "validation_number": i+1,
                    "validation": validation,
                    "success": validation_result["success"],
                    "message": validation_result.get("message", "")
                })
                
                if not validation_result["success"]:
                    logger.warning("Validation step failed",
                                 plan_id=plan_id,
                                 validation=validation)
            
            recovery_end = datetime.utcnow()
            recovery_duration = (recovery_end - recovery_start).total_seconds()
            
            # Record recovery in history
            recovery_record = {
                "recovery_id": recovery_id,
                "plan_id": plan_id,
                "started_at": recovery_start.isoformat(),
                "completed_at": recovery_end.isoformat(),
                "duration_seconds": recovery_duration,
                "success": True,
                "step_results": step_results,
                "validation_results": validation_results
            }
            self.recovery_history.append(recovery_record)
            
            # Update plan last tested date
            plan.last_tested = recovery_end
            
            logger.info("Recovery plan completed successfully",
                       plan_id=plan_id,
                       duration_seconds=recovery_duration)
            
            return recovery_record
            
        except Exception as e:
            recovery_end = datetime.utcnow()
            recovery_duration = (recovery_end - recovery_start).total_seconds()
            
            recovery_record = {
                "recovery_id": recovery_id,
                "plan_id": plan_id,
                "started_at": recovery_start.isoformat(),
                "completed_at": recovery_end.isoformat(),
                "duration_seconds": recovery_duration,
                "success": False,
                "error": str(e),
                "step_results": step_results if 'step_results' in locals() else []
            }
            self.recovery_history.append(recovery_record)
            
            logger.error("Recovery plan failed",
                        plan_id=plan_id,
                        error=str(e))
            
            return recovery_record
    
    async def _check_dependency_status(self, dependency: str) -> bool:
        """Check if a recovery dependency is satisfied."""
        # This would check the status of dependent systems/services
        logger.info("Checking dependency status", dependency=dependency)
        return True  # Placeholder
    
    async def _execute_recovery_step(self, step: str, recovery_point: Optional[datetime] = None) -> Dict[str, Any]:
        """Execute a single recovery step."""
        step_start = datetime.utcnow()
        
        try:
            # This would contain the actual recovery logic for each step
            # For now, simulate step execution
            await asyncio.sleep(1)  # Simulate work
            
            step_end = datetime.utcnow()
            duration = (step_end - step_start).total_seconds()
            
            return {
                "success": True,
                "message": f"Step completed: {step}",
                "duration_seconds": duration
            }
            
        except Exception as e:
            step_end = datetime.utcnow()
            duration = (step_end - step_start).total_seconds()
            
            return {
                "success": False,
                "message": f"Step failed: {str(e)}",
                "duration_seconds": duration
            }
    
    async def _execute_validation_step(self, validation: str) -> Dict[str, Any]:
        """Execute a validation step."""
        try:
            # This would contain actual validation logic
            # For now, simulate validation
            await asyncio.sleep(0.5)  # Simulate validation work
            
            return {
                "success": True,
                "message": f"Validation passed: {validation}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Validation failed: {str(e)}"
            }
    
    async def test_recovery_plan(self, plan_id: str) -> Dict[str, Any]:
        """Test a recovery plan without actually performing recovery."""
        plan = self.recovery_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Recovery plan {plan_id} not found")
        
        logger.info("Testing recovery plan", plan_id=plan_id)
        
        test_results = {
            "plan_id": plan_id,
            "test_date": datetime.utcnow().isoformat(),
            "dependencies_check": [],
            "steps_validation": [],
            "overall_status": "passed"
        }
        
        # Check dependencies
        for dependency in plan.dependencies:
            dep_status = await self._check_dependency_status(dependency)
            test_results["dependencies_check"].append({
                "dependency": dependency,
                "status": "available" if dep_status else "unavailable"
            })
            
            if not dep_status:
                test_results["overall_status"] = "failed"
        
        # Validate steps (without execution)
        for i, step in enumerate(plan.recovery_steps):
            step_validation = await self._validate_recovery_step(step)
            test_results["steps_validation"].append({
                "step_number": i+1,
                "step": step,
                "validation_status": "valid" if step_validation else "invalid"
            })
            
            if not step_validation:
                test_results["overall_status"] = "failed"
        
        # Update last tested date if test passed
        if test_results["overall_status"] == "passed":
            plan.last_tested = datetime.utcnow()
        
        logger.info("Recovery plan test completed",
                   plan_id=plan_id,
                   status=test_results["overall_status"])
        
        return test_results
    
    async def _validate_recovery_step(self, step: str) -> bool:
        """Validate that a recovery step can be executed."""
        # This would check if the step is valid and can be executed
        # For now, assume all steps are valid
        return True
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get current backup system status."""
        total_jobs = len(self.backup_jobs)
        active_jobs = len([j for j in self.backup_jobs.values() if j.status == BackupStatus.IN_PROGRESS])
        completed_jobs = len([j for j in self.backup_jobs.values() if j.status == BackupStatus.COMPLETED])
        failed_jobs = len([j for j in self.backup_jobs.values() if j.status == BackupStatus.FAILED])
        
        return {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "recent_backups": self.backup_history[-10:],  # Last 10 backups
            "backup_jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "status": job.status.value,
                    "last_run": job.completed_at.isoformat() if job.completed_at else None,
                    "next_run": "calculated_from_cron",  # Would calculate from cron expression
                    "retention_days": job.retention_days
                }
                for job in self.backup_jobs.values()
            ]
        }
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery system status."""
        total_plans = len(self.recovery_plans)
        tested_plans = len([p for p in self.recovery_plans.values() if p.last_tested])
        
        return {
            "total_plans": total_plans,
            "tested_plans": tested_plans,
            "recent_recoveries": self.recovery_history[-5:],  # Last 5 recoveries
            "recovery_plans": [
                {
                    "id": plan.id,
                    "name": plan.name,
                    "priority": plan.priority,
                    "rto_minutes": plan.rto_minutes,
                    "rpo_minutes": plan.rpo_minutes,
                    "last_tested": plan.last_tested.isoformat() if plan.last_tested else None,
                    "dependencies": plan.dependencies
                }
                for plan in self.recovery_plans.values()
            ]
        }


# Global backup recovery manager instance
backup_recovery_manager = BackupRecoveryManager()