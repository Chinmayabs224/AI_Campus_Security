"""Add analytics and reporting tables

Revision ID: 009_analytics_tables
Revises: 008_audit_system
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '009_analytics_tables'
down_revision = '008_audit_system'
branch_labels = None
depends_on = None


def upgrade():
    # Create locations table for heat map data
    op.create_table('locations',
        sa.Column('id', sa.String(50), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('latitude', sa.Numeric(10, 8), nullable=True),
        sa.Column('longitude', sa.Numeric(11, 8), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create cameras table for system monitoring
    op.create_table('cameras',
        sa.Column('id', sa.String(50), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('location_id', sa.String(50), nullable=True),
        sa.Column('rtsp_url', sa.String(500), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='inactive'),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=True),
        sa.Column('performance_score', sa.Numeric(5, 2), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['location_id'], ['locations.id'], ondelete='SET NULL')
    )
    
    # Add location_id to incidents table if not exists
    try:
        op.add_column('incidents', sa.Column('location_id', sa.String(50), nullable=True))
        op.create_foreign_key('fk_incidents_location', 'incidents', 'locations', ['location_id'], ['id'], ondelete='SET NULL')
    except Exception:
        # Column might already exist
        pass
    
    # Add privacy_violation flag to incidents
    try:
        op.add_column('incidents', sa.Column('privacy_violation', sa.Boolean, nullable=False, server_default='false'))
    except Exception:
        pass
    
    # Add false_positive flag to events
    try:
        op.add_column('events', sa.Column('false_positive', sa.Boolean, nullable=False, server_default='false'))
    except Exception:
        pass
    
    # Create audit_findings table for compliance reporting
    op.create_table('audit_findings',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('finding', sa.Text, nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('resolved', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Add retention_expires to evidence table
    try:
        op.add_column('evidence', sa.Column('retention_expires', sa.DateTime(timezone=True), nullable=True))
        op.add_column('evidence', sa.Column('integrity_hash', sa.String(64), nullable=True))
    except Exception:
        pass
    
    # Create indexes for analytics performance
    op.create_index('idx_incidents_created_at', 'incidents', ['created_at'])
    op.create_index('idx_incidents_location_created', 'incidents', ['location_id', 'created_at'])
    op.create_index('idx_events_created_at', 'events', ['created_at'])
    op.create_index('idx_audit_log_timestamp', 'audit_log', ['timestamp'])
    op.create_index('idx_audit_log_resource', 'audit_log', ['resource_type', 'resource_id'])


def downgrade():
    # Drop indexes
    op.drop_index('idx_audit_log_resource', table_name='audit_log')
    op.drop_index('idx_audit_log_timestamp', table_name='audit_log')
    op.drop_index('idx_events_created_at', table_name='events')
    op.drop_index('idx_incidents_location_created', table_name='incidents')
    op.drop_index('idx_incidents_created_at', table_name='incidents')
    
    # Drop tables
    op.drop_table('audit_findings')
    op.drop_table('cameras')
    op.drop_table('locations')
    
    # Remove added columns (optional, might want to keep for data integrity)
    # op.drop_column('evidence', 'integrity_hash')
    # op.drop_column('evidence', 'retention_expires')
    # op.drop_column('events', 'false_positive')
    # op.drop_column('incidents', 'privacy_violation')
    # op.drop_column('incidents', 'location_id')