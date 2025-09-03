"""Add compliance and data protection tables

Revision ID: 004_compliance_tables
Revises: 003_security_tables
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_compliance_tables'
down_revision = '003_security_tables'
branch_labels = None
depends_on = None


def upgrade():
    # Data retention records table
    op.create_table('data_retention_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('policy_id', sa.String(100), nullable=False),
        sa.Column('data_id', sa.String(255), nullable=False),
        sa.Column('data_category', sa.String(50), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('expires_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('deletion_scheduled_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('deletion_method_used', sa.String(50), nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for data retention records
    op.create_index('idx_retention_records_policy_id', 'data_retention_records', ['policy_id'])
    op.create_index('idx_retention_records_data_id', 'data_retention_records', ['data_id'])
    op.create_index('idx_retention_records_expires_at', 'data_retention_records', ['expires_at'])
    op.create_index('idx_retention_records_status', 'data_retention_records', ['status'])
    
    # Policy violations table
    op.create_table('policy_violations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule_id', sa.String(100), nullable=False),
        sa.Column('policy_type', sa.String(50), nullable=False),
        sa.Column('violation_description', sa.Text, nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('detected_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('resource_id', sa.String(255), nullable=False),
        sa.Column('resource_type', sa.String(100), nullable=False),
        sa.Column('action_taken', sa.String(50), nullable=False),
        sa.Column('resolved', sa.Boolean, nullable=False, default=False),
        sa.Column('resolved_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('resolved_by', sa.String(100), nullable=True),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for policy violations
    op.create_index('idx_policy_violations_rule_id', 'policy_violations', ['rule_id'])
    op.create_index('idx_policy_violations_severity', 'policy_violations', ['severity'])
    op.create_index('idx_policy_violations_detected_at', 'policy_violations', ['detected_at'])
    op.create_index('idx_policy_violations_resolved', 'policy_violations', ['resolved'])
    
    # Data subject requests table
    op.create_table('data_subject_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('request_type', sa.String(50), nullable=False),
        sa.Column('subject_id', sa.String(255), nullable=False),
        sa.Column('subject_email', sa.String(255), nullable=False),
        sa.Column('framework', sa.String(20), nullable=False),
        sa.Column('request_details', sa.Text, nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('due_date', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('response_data', sa.JSON, nullable=True),
        sa.Column('assigned_to', sa.String(100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for data subject requests
    op.create_index('idx_dsar_subject_id', 'data_subject_requests', ['subject_id'])
    op.create_index('idx_dsar_status', 'data_subject_requests', ['status'])
    op.create_index('idx_dsar_due_date', 'data_subject_requests', ['due_date'])
    op.create_index('idx_dsar_created_at', 'data_subject_requests', ['created_at'])
    
    # Privacy impact assessments table
    op.create_table('privacy_impact_assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('overall_risk_level', sa.String(20), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('created_by', sa.String(100), nullable=False),
        sa.Column('reviewed_by', sa.String(100), nullable=True),
        sa.Column('approved_by', sa.String(100), nullable=True),
        sa.Column('approved_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('review_date', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('compliance_frameworks', sa.JSON, nullable=True),
        sa.Column('processing_activities', sa.JSON, nullable=True),
        sa.Column('risk_assessments', sa.JSON, nullable=True),
        sa.Column('mitigation_plan', sa.JSON, nullable=True),
        sa.Column('monitoring_measures', sa.JSON, nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for privacy impact assessments
    op.create_index('idx_pia_status', 'privacy_impact_assessments', ['status'])
    op.create_index('idx_pia_risk_level', 'privacy_impact_assessments', ['overall_risk_level'])
    op.create_index('idx_pia_created_by', 'privacy_impact_assessments', ['created_by'])
    op.create_index('idx_pia_review_date', 'privacy_impact_assessments', ['review_date'])
    
    # Compliance violations table (for compliance monitor)
    op.create_table('compliance_violations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule_id', sa.String(100), nullable=False),
        sa.Column('framework', sa.String(20), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('detected_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='open'),
        sa.Column('remediation_deadline', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('assigned_to', sa.String(100), nullable=True),
        sa.Column('resolution_notes', sa.Text, nullable=True),
        sa.Column('resolved_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for compliance violations
    op.create_index('idx_compliance_violations_framework', 'compliance_violations', ['framework'])
    op.create_index('idx_compliance_violations_severity', 'compliance_violations', ['severity'])
    op.create_index('idx_compliance_violations_status', 'compliance_violations', ['status'])
    op.create_index('idx_compliance_violations_detected_at', 'compliance_violations', ['detected_at'])
    
    # Backup jobs table
    op.create_table('backup_jobs',
        sa.Column('id', sa.String(100), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('backup_type', sa.String(20), nullable=False),
        sa.Column('source_paths', sa.JSON, nullable=False),
        sa.Column('destination_path', sa.String(500), nullable=False),
        sa.Column('schedule_cron', sa.String(100), nullable=False),
        sa.Column('retention_days', sa.Integer, nullable=False),
        sa.Column('encryption_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('compression_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('size_bytes', sa.BigInteger, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for backup jobs
    op.create_index('idx_backup_jobs_status', 'backup_jobs', ['status'])
    op.create_index('idx_backup_jobs_completed_at', 'backup_jobs', ['completed_at'])
    
    # Recovery plans table
    op.create_table('recovery_plans',
        sa.Column('id', sa.String(100), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('recovery_type', sa.String(50), nullable=False),
        sa.Column('priority', sa.Integer, nullable=False),
        sa.Column('rto_minutes', sa.Integer, nullable=False),
        sa.Column('rpo_minutes', sa.Integer, nullable=False),
        sa.Column('recovery_steps', sa.JSON, nullable=False),
        sa.Column('dependencies', sa.JSON, nullable=True),
        sa.Column('validation_steps', sa.JSON, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('last_tested', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for recovery plans
    op.create_index('idx_recovery_plans_priority', 'recovery_plans', ['priority'])
    op.create_index('idx_recovery_plans_last_tested', 'recovery_plans', ['last_tested'])
    
    # Recovery history table
    op.create_table('recovery_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('recovery_id', sa.String(100), nullable=False),
        sa.Column('plan_id', sa.String(100), nullable=False),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('duration_seconds', sa.Integer, nullable=False),
        sa.Column('success', sa.Boolean, nullable=False),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('step_results', sa.JSON, nullable=True),
        sa.Column('validation_results', sa.JSON, nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for recovery history
    op.create_index('idx_recovery_history_plan_id', 'recovery_history', ['plan_id'])
    op.create_index('idx_recovery_history_started_at', 'recovery_history', ['started_at'])
    op.create_index('idx_recovery_history_success', 'recovery_history', ['success'])
    
    # Privacy violations table (for privacy-specific violations)
    op.create_table('privacy_violations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('violation_type', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('data_subject_id', sa.String(255), nullable=True),
        sa.Column('data_types_affected', sa.JSON, nullable=True),
        sa.Column('detected_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('reported_by', sa.String(100), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='open'),
        sa.Column('resolution_actions', sa.JSON, nullable=True),
        sa.Column('resolved_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('breach_notification_required', sa.Boolean, nullable=False, default=False),
        sa.Column('breach_notification_sent', sa.Boolean, nullable=False, default=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for privacy violations
    op.create_index('idx_privacy_violations_type', 'privacy_violations', ['violation_type'])
    op.create_index('idx_privacy_violations_severity', 'privacy_violations', ['severity'])
    op.create_index('idx_privacy_violations_status', 'privacy_violations', ['status'])
    op.create_index('idx_privacy_violations_detected_at', 'privacy_violations', ['detected_at'])


def downgrade():
    # Drop tables in reverse order
    op.drop_table('privacy_violations')
    op.drop_table('recovery_history')
    op.drop_table('recovery_plans')
    op.drop_table('backup_jobs')
    op.drop_table('compliance_violations')
    op.drop_table('privacy_impact_assessments')
    op.drop_table('data_subject_requests')
    op.drop_table('policy_violations')
    op.drop_table('data_retention_records')