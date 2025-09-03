"""Add evidence management tables

Revision ID: 002
Revises: 001
Create Date: 2024-12-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create evidence table
    op.create_table('evidence',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('incident_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('checksum', sa.String(length=64), nullable=False),
        sa.Column('encryption_key_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('redaction_applied', sa.Boolean(), nullable=True),
        sa.Column('file_metadata', sa.JSON(), nullable=True),
        sa.Column('retention_until', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create evidence access log table
    op.create_table('evidence_access_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('evidence_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False),
        sa.Column('access_type', sa.String(length=50), nullable=False),
        sa.Column('purpose', sa.String(length=50), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('accessed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['evidence_id'], ['evidence.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create retention policies table
    op.create_table('retention_policies',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('retention_days', sa.Integer(), nullable=False),
        sa.Column('evidence_types', sa.JSON(), nullable=False),
        sa.Column('incident_severities', sa.JSON(), nullable=True),
        sa.Column('auto_delete', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create indexes for better performance
    op.create_index('idx_evidence_incident_id', 'evidence', ['incident_id'])
    op.create_index('idx_evidence_status', 'evidence', ['status'])
    op.create_index('idx_evidence_retention_until', 'evidence', ['retention_until'])
    op.create_index('idx_evidence_access_log_evidence_id', 'evidence_access_log', ['evidence_id'])
    op.create_index('idx_evidence_access_log_user_id', 'evidence_access_log', ['user_id'])
    op.create_index('idx_evidence_access_log_accessed_at', 'evidence_access_log', ['accessed_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_evidence_access_log_accessed_at')
    op.drop_index('idx_evidence_access_log_user_id')
    op.drop_index('idx_evidence_access_log_evidence_id')
    op.drop_index('idx_evidence_retention_until')
    op.drop_index('idx_evidence_status')
    op.drop_index('idx_evidence_incident_id')
    
    # Drop tables
    op.drop_table('retention_policies')
    op.drop_table('evidence_access_log')
    op.drop_table('evidence')