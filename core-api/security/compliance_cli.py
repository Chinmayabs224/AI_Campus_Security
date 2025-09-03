#!/usr/bin/env python3
"""
Compliance Management CLI Tool
Provides command-line interface for compliance operations.
"""
import asyncio
import click
import json
from datetime import datetime
from typing import Optional
import structlog

from .data_retention import data_retention_service
from .privacy_impact_assessment import pia_service
from .policy_enforcement import policy_enforcement_engine
from .compliance_monitor import compliance_monitor, ComplianceFramework
from .backup_recovery import backup_recovery_manager

logger = structlog.get_logger()


@click.group()
def cli():
    """Campus Security Compliance Management CLI"""
    pass


@cli.group()
def retention():
    """Data retention management commands"""
    pass


@retention.command()
def apply_policies():
    """Apply all data retention policies"""
    click.echo("Applying data retention policies...")
    
    async def run():
        results = await data_retention_service.apply_retention_policies()
        return results
    
    results = asyncio.run(run())
    
    click.echo(f"Policies applied: {results['policies_processed']}")
    click.echo(f"Records processed: {results['records_processed']}")
    click.echo(f"Records expired: {results['records_expired']}")
    click.echo(f"Records deleted: {results['records_deleted']}")
    
    if results['errors']:
        click.echo("Errors encountered:")
        for error in results['errors']:
            click.echo(f"  - {error}")


@retention.command()
def report():
    """Generate data retention report"""
    click.echo("Generating data retention report...")
    
    report = data_retention_service.generate_retention_report()
    
    click.echo(f"Total policies: {report['total_policies']}")
    click.echo(f"Total records: {report['total_records']}")
    click.echo(f"Expiring soon: {report['expiring_soon']}")
    click.echo(f"Overdue records: {report['overdue_records']}")
    click.echo(f"Deletion queue size: {report['deletion_queue_size']}")


@retention.command()
@click.argument('data_id')
def status(data_id):
    """Get retention status for specific data"""
    
    async def run():
        return await data_retention_service.get_retention_status(data_id)
    
    status = asyncio.run(run())
    
    if status:
        click.echo(f"Data ID: {status.data_id}")
        click.echo(f"Policy ID: {status.policy_id}")
        click.echo(f"Category: {status.data_category.value}")
        click.echo(f"Created: {status.created_at}")
        click.echo(f"Expires: {status.expires_at}")
        click.echo(f"Status: {status.status.value}")
    else:
        click.echo("No retention record found for this data ID")


@cli.group()
def compliance():
    """Compliance monitoring commands"""
    pass


@compliance.command()
@click.option('--framework', type=click.Choice(['gdpr', 'ferpa', 'coppa', 'ccpa']), help='Specific framework to check')
def check(framework):
    """Run compliance checks"""
    click.echo("Running compliance checks...")
    
    async def run():
        framework_enum = ComplianceFramework(framework) if framework else None
        return await compliance_monitor.run_compliance_check(framework_enum)
    
    results = asyncio.run(run())
    
    click.echo(f"Rules checked: {results['rules_checked']}")
    click.echo(f"Violations found: {results['violations_found']}")
    
    if results['violations']:
        click.echo("\nViolations:")
        for violation in results['violations']:
            click.echo(f"  - {violation['title']} ({violation['severity']})")


@compliance.command()
@click.option('--framework', type=click.Choice(['gdpr', 'ferpa', 'coppa', 'ccpa']), help='Specific framework to report on')
def report(framework):
    """Generate compliance report"""
    click.echo("Generating compliance report...")
    
    async def run():
        framework_enum = ComplianceFramework(framework) if framework else None
        return await compliance_monitor.generate_compliance_report(framework_enum)
    
    report = asyncio.run(run())
    
    click.echo(f"Framework: {report['framework']}")
    click.echo(f"Generated at: {report['generated_at']}")
    
    click.echo("\nCompliance Status:")
    for fw, status in report['compliance_status'].items():
        click.echo(f"  {fw}: {status}")
    
    click.echo(f"\nViolations Summary:")
    vs = report['violations_summary']
    click.echo(f"  Total: {vs['total']}")
    click.echo(f"  Open: {vs['open']}")
    click.echo(f"  Critical: {vs['by_severity']['critical']}")
    click.echo(f"  High: {vs['by_severity']['high']}")


@cli.group()
def policy():
    """Policy enforcement commands"""
    pass


@policy.command()
def metrics():
    """Show policy enforcement metrics"""
    metrics = policy_enforcement_engine.get_metrics()
    
    click.echo(f"Total rules: {metrics.total_rules}")
    click.echo(f"Active rules: {metrics.active_rules}")
    click.echo(f"Violations detected: {metrics.violations_detected}")
    click.echo(f"Violations resolved: {metrics.violations_resolved}")
    
    click.echo("\nViolations by severity:")
    for severity, count in metrics.violations_by_severity.items():
        click.echo(f"  {severity}: {count}")
    
    click.echo("\nViolations by type:")
    for policy_type, count in metrics.violations_by_type.items():
        click.echo(f"  {policy_type}: {count}")


@policy.command()
@click.argument('context_file', type=click.File('r'))
def evaluate(context_file):
    """Evaluate policies against context from JSON file"""
    click.echo("Evaluating policies...")
    
    try:
        context = json.load(context_file)
    except json.JSONDecodeError as e:
        click.echo(f"Error parsing JSON: {e}")
        return
    
    async def run():
        return await policy_enforcement_engine.evaluate_policies(context)
    
    violations = asyncio.run(run())
    
    click.echo(f"Violations detected: {len(violations)}")
    
    for violation in violations:
        click.echo(f"  - {violation.violation_description}")
        click.echo(f"    Severity: {violation.severity.value}")
        click.echo(f"    Action: {violation.action_taken.value}")


@policy.command()
@click.argument('violation_id')
@click.option('--resolved-by', default='cli-user', help='User resolving the violation')
def resolve(violation_id, resolved_by):
    """Resolve a policy violation"""
    
    async def run():
        return await policy_enforcement_engine.resolve_violation(violation_id, resolved_by)
    
    success = asyncio.run(run())
    
    if success:
        click.echo(f"Violation {violation_id} resolved successfully")
    else:
        click.echo(f"Violation {violation_id} not found")


@cli.group()
def backup():
    """Backup and recovery commands"""
    pass


@backup.command()
@click.argument('job_id')
def execute(job_id):
    """Execute a backup job"""
    click.echo(f"Executing backup job: {job_id}")
    
    async def run():
        return await backup_recovery_manager.execute_backup_job(job_id)
    
    try:
        result = asyncio.run(run())
        
        click.echo(f"Status: {result['status']}")
        if result['status'] == 'completed':
            click.echo(f"Backup path: {result['backup_path']}")
            click.echo(f"Size: {result.get('size_bytes', 0)} bytes")
            click.echo(f"Duration: {result.get('duration_seconds', 0)} seconds")
        elif 'error' in result:
            click.echo(f"Error: {result['error']}")
            
    except ValueError as e:
        click.echo(f"Error: {e}")


@backup.command()
def status():
    """Show backup system status"""
    status = backup_recovery_manager.get_backup_status()
    
    click.echo(f"Total jobs: {status['total_jobs']}")
    click.echo(f"Active jobs: {status['active_jobs']}")
    click.echo(f"Completed jobs: {status['completed_jobs']}")
    click.echo(f"Failed jobs: {status['failed_jobs']}")
    
    click.echo("\nBackup Jobs:")
    for job in status['backup_jobs']:
        click.echo(f"  {job['id']}: {job['name']} ({job['status']})")


@cli.group()
def recovery():
    """Disaster recovery commands"""
    pass


@recovery.command()
@click.argument('plan_id')
def execute(plan_id):
    """Execute a recovery plan"""
    click.echo(f"Executing recovery plan: {plan_id}")
    
    async def run():
        return await backup_recovery_manager.execute_recovery_plan(plan_id)
    
    try:
        result = asyncio.run(run())
        
        click.echo(f"Recovery ID: {result['recovery_id']}")
        click.echo(f"Success: {result['success']}")
        click.echo(f"Duration: {result['duration_seconds']} seconds")
        
        if result['success']:
            click.echo(f"Steps completed: {len(result['step_results'])}")
            click.echo(f"Validations completed: {len(result['validation_results'])}")
        else:
            click.echo(f"Error: {result.get('error', 'Unknown error')}")
            
    except ValueError as e:
        click.echo(f"Error: {e}")


@recovery.command()
@click.argument('plan_id')
def test(plan_id):
    """Test a recovery plan"""
    click.echo(f"Testing recovery plan: {plan_id}")
    
    async def run():
        return await backup_recovery_manager.test_recovery_plan(plan_id)
    
    try:
        result = asyncio.run(run())
        
        click.echo(f"Overall status: {result['overall_status']}")
        
        click.echo("\nDependency checks:")
        for dep in result['dependencies_check']:
            click.echo(f"  {dep['dependency']}: {dep['status']}")
        
        click.echo("\nStep validations:")
        for step in result['steps_validation']:
            click.echo(f"  Step {step['step_number']}: {step['validation_status']}")
            
    except ValueError as e:
        click.echo(f"Error: {e}")


@recovery.command()
def status():
    """Show recovery system status"""
    status = backup_recovery_manager.get_recovery_status()
    
    click.echo(f"Total plans: {status['total_plans']}")
    click.echo(f"Tested plans: {status['tested_plans']}")
    
    click.echo("\nRecovery Plans:")
    for plan in status['recovery_plans']:
        last_tested = plan['last_tested'] or 'Never'
        click.echo(f"  {plan['id']}: {plan['name']} (Priority: {plan['priority']}, Last tested: {last_tested})")


@cli.group()
def pia():
    """Privacy Impact Assessment commands"""
    pass


@pia.command()
@click.argument('title')
@click.argument('description')
@click.option('--template', help='Template name to use')
@click.option('--created-by', default='cli-user', help='User creating the PIA')
def create(title, description, template, created_by):
    """Create a new Privacy Impact Assessment"""
    click.echo("Creating Privacy Impact Assessment...")
    
    async def run():
        return await pia_service.create_assessment(title, description, created_by, template)
    
    pia = asyncio.run(run())
    
    click.echo(f"PIA created successfully!")
    click.echo(f"Assessment ID: {pia.id}")
    click.echo(f"Title: {pia.title}")
    click.echo(f"Status: {pia.status.value}")
    click.echo(f"Risk Level: {pia.overall_risk_level.value}")


@pia.command()
@click.argument('assessment_id')
def report(assessment_id):
    """Generate PIA report"""
    click.echo("Generating PIA report...")
    
    async def run():
        return await pia_service.generate_pia_report(assessment_id)
    
    try:
        report = asyncio.run(run())
        
        click.echo(f"Title: {report['title']}")
        click.echo(f"Status: {report['status']}")
        click.echo(f"Risk Level: {report['overall_risk_level']}")
        click.echo(f"Processing Activities: {len(report['processing_activities'])}")
        click.echo(f"Risk Assessments: {len(report['risk_assessments'])}")
        
    except ValueError as e:
        click.echo(f"Error: {e}")


@cli.command()
def dashboard():
    """Show compliance dashboard summary"""
    click.echo("Compliance Dashboard Summary")
    click.echo("=" * 40)
    
    # Data retention summary
    retention_report = data_retention_service.generate_retention_report()
    click.echo(f"Data Retention:")
    click.echo(f"  Total records: {retention_report['total_records']}")
    click.echo(f"  Expiring soon: {retention_report['expiring_soon']}")
    click.echo(f"  Overdue: {retention_report['overdue_records']}")
    
    # Policy enforcement summary
    policy_metrics = policy_enforcement_engine.get_metrics()
    click.echo(f"\nPolicy Enforcement:")
    click.echo(f"  Active rules: {policy_metrics.active_rules}")
    click.echo(f"  Violations detected: {policy_metrics.violations_detected}")
    click.echo(f"  Violations resolved: {policy_metrics.violations_resolved}")
    
    # Backup summary
    backup_status = backup_recovery_manager.get_backup_status()
    click.echo(f"\nBackup Status:")
    click.echo(f"  Total jobs: {backup_status['total_jobs']}")
    click.echo(f"  Failed jobs: {backup_status['failed_jobs']}")
    
    # Recovery summary
    recovery_status = backup_recovery_manager.get_recovery_status()
    click.echo(f"\nRecovery Status:")
    click.echo(f"  Total plans: {recovery_status['total_plans']}")
    click.echo(f"  Tested plans: {recovery_status['tested_plans']}")


if __name__ == '__main__':
    cli()