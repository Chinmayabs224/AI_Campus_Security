"""Add compliance and privacy tables

Revision ID: 010_compliance_tables
Revises: 009_analytics_tables
Create Date: 2024-01-15 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '010_compliance_tables'
down_revision = '009_analytics_tables'
branch_labels = None
depends_on = None


def upgrade():
    # Create data_subject_requests table for GDPR/FERPA compliance
    op.create_table('data_subject_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('request_type', sa.String(50), nullable=False),  # access, deletion, portability, rectification
        sa.Column('subject_email', sa.String(255), nullable=False),
        sa.Column('subject_name', sa.String(200), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),  # pending, processing, completed, rejected
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('requested_data', postgresql.JSONB, nullable=True),
        sa.Column('response_data', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('assigned_to', sa.String(100), nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create privacy_violations table for tracking privacy incidents
    op.create_table('privacy_violations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('incident_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('violation_type', sa.String(50), nullable=False),  # unauthorized_access, data_breach, improper_redaction
        sa.Column('severity', sa.String(20), nullable=False),  # low, medium, high, critical
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('affected_individuals', sa.Integer, nullable=True),
        sa.Column('data_types', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('mitigation_actions', sa.Text, nullable=True),
        sa.Column('reported_to_authority', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('authority_reference', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', sa.String(100), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ondelete='SET NULL')
    )
    
    # Create compliance_reports table for storing generated reports
    op.create_table('compliance_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('report_type', sa.String(50), nullable=False),
        sa.Column('period_start', sa.Date, nullable=False),
        sa.Column('period_end', sa.Date, nullable=False),
        sa.Column('report_data', postgresql.JSONB, nullable=False),
        sa.Column('generated_by', sa.String(100), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='generated'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create analytics_cache table for caching analytics results
    op.create_table('analytics_cache',
        sa.Column('cache_key', sa.String(255), nullable=False),
        sa.Column('cache_data', postgresql.JSONB, nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('cache_key')
    )
    
    # Create indexes for performance
    op.create_index('idx_dsr_status_created', 'data_subject_requests', ['status', 'created_at'])
    op.create_index('idx_dsr_email', 'data_subject_requests', ['subject_email'])
    op.create_index('idx_privacy_violations_severity', 'privacy_violations', ['severity', 'created_at'])
    op.create_index('idx_privacy_violations_type', 'privacy_violations', ['violation_type'])
    op.create_index('idx_compliance_reports_type_period', 'compliance_reports', ['report_type', 'period_start', 'period_end'])
    op.create_index('idx_analytics_cache_expires', 'analytics_cache', ['expires_at'])


def downgrade():
    # Drop indexes
    op.drop_index('idx_analytics_cache_expires', table_name='analytics_cache')
    op.drop_index('idx_compliance_reports_type_period', table_name='compliance_reports')
    op.drop_index('idx_privacy_violations_type', table_name='privacy_violations')
    op.drop_index('idx_privacy_violations_severity', table_name='privacy_violations')
    op.drop_index('idx_dsr_email', table_name='data_subject_requests')
    op.drop_index('idx_dsr_status_created', table_name='data_subject_requests')
    
    # Drop tables
    op.drop_table('analytics_cache')
    op.drop_table('compliance_reports')
    op.drop_table('privacy_violations')
    op.drop_table('data_subject_requests')