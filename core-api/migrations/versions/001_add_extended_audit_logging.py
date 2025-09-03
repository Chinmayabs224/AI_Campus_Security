"""Add extended audit logging tables

Revision ID: 001_extended_audit
Revises: 
Create Date: 2024-12-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_extended_audit'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('saml_name_id', sa.String(length=255), nullable=True),
        sa.Column('saml_attributes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_saml_name_id'), 'users', ['saml_name_id'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Create user_sessions table
    op.create_table('user_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_token', sa.String(length=255), nullable=False),
        sa.Column('refresh_token', sa.String(length=255), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_accessed', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_sessions_refresh_token'), 'user_sessions', ['refresh_token'], unique=True)
    op.create_index(op.f('ix_user_sessions_session_token'), 'user_sessions', ['session_token'], unique=True)
    op.create_index(op.f('ix_user_sessions_user_id'), 'user_sessions', ['user_id'], unique=False)

    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('key_hash', sa.String(length=255), nullable=False),
        sa.Column('prefix', sa.String(length=20), nullable=False),
        sa.Column('scopes', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('ip_whitelist', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_api_keys_key_hash'), 'api_keys', ['key_hash'], unique=True)
    op.create_index(op.f('ix_api_keys_prefix'), 'api_keys', ['prefix'], unique=False)
    op.create_index(op.f('ix_api_keys_user_id'), 'api_keys', ['user_id'], unique=False)

    # Create events table
    op.create_table('events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('camera_id', sa.String(length=50), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('threat_level', sa.String(length=20), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('bounding_boxes', sa.JSON(), nullable=True),
        sa.Column('event_metadata', sa.JSON(), nullable=True),
        sa.Column('processed', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_events_camera_id'), 'events', ['camera_id'], unique=False)
    op.create_index(op.f('ix_events_event_type'), 'events', ['event_type'], unique=False)
    op.create_index(op.f('ix_events_processed'), 'events', ['processed'], unique=False)
    op.create_index(op.f('ix_events_threat_level'), 'events', ['threat_level'], unique=False)
    op.create_index(op.f('ix_events_timestamp'), 'events', ['timestamp'], unique=False)

    # Create incidents table
    op.create_table('incidents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('priority', sa.String(length=20), nullable=False),
        sa.Column('assigned_to', sa.String(length=100), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.Column('location', sa.String(length=200), nullable=True),
        sa.Column('camera_ids', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('assigned_at', sa.DateTime(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('closed_at', sa.DateTime(), nullable=True),
        sa.Column('escalated', sa.Boolean(), nullable=True),
        sa.Column('escalated_at', sa.DateTime(), nullable=True),
        sa.Column('escalation_reason', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_incidents_assigned_to'), 'incidents', ['assigned_to'], unique=False)
    op.create_index(op.f('ix_incidents_created_at'), 'incidents', ['created_at'], unique=False)
    op.create_index(op.f('ix_incidents_escalated'), 'incidents', ['escalated'], unique=False)
    op.create_index(op.f('ix_incidents_priority'), 'incidents', ['priority'], unique=False)
    op.create_index(op.f('ix_incidents_severity'), 'incidents', ['severity'], unique=False)
    op.create_index(op.f('ix_incidents_status'), 'incidents', ['status'], unique=False)

    # Create incident_events table
    op.create_table('incident_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('incident_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['event_id'], ['events.id'], ),
        sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_incident_events_event_id'), 'incident_events', ['event_id'], unique=False)
    op.create_index(op.f('ix_incident_events_incident_id'), 'incident_events', ['incident_id'], unique=False)

    # Create incident_notes table
    op.create_table('incident_notes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('incident_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('author', sa.String(length=100), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('note_type', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_incident_notes_incident_id'), 'incident_notes', ['incident_id'], unique=False)

    # Create incident_status_history table
    op.create_table('incident_status_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('incident_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('previous_status', sa.String(length=20), nullable=True),
        sa.Column('new_status', sa.String(length=20), nullable=False),
        sa.Column('changed_by', sa.String(length=100), nullable=False),
        sa.Column('change_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_incident_status_history_incident_id'), 'incident_status_history', ['incident_id'], unique=False)

    # Create original audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('api_key_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('endpoint', sa.String(length=255), nullable=True),
        sa.Column('method', sa.String(length=10), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_ip_address'), 'audit_logs', ['ip_address'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_type'), 'audit_logs', ['resource_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)

    # Create extended audit_logs_extended table
    op.create_table('audit_logs_extended',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('username', sa.String(length=100), nullable=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('api_key_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('endpoint', sa.String(length=255), nullable=True),
        sa.Column('method', sa.String(length=10), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('compliance_tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('risk_level', sa.String(length=20), nullable=False),
        sa.Column('business_justification', sa.Text(), nullable=True),
        sa.Column('contains_pii', sa.Boolean(), nullable=False),
        sa.Column('data_classification', sa.String(length=50), nullable=True),
        sa.Column('event_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('before_state', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('after_state', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_code', sa.String(length=50), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('retention_date', sa.DateTime(), nullable=True),
        sa.Column('archived', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_extended_action'), 'audit_logs_extended', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_archived'), 'audit_logs_extended', ['archived'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_contains_pii'), 'audit_logs_extended', ['contains_pii'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_data_classification'), 'audit_logs_extended', ['data_classification'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_endpoint'), 'audit_logs_extended', ['endpoint'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_ip_address'), 'audit_logs_extended', ['ip_address'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_resource_type'), 'audit_logs_extended', ['resource_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_retention_date'), 'audit_logs_extended', ['retention_date'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_risk_level'), 'audit_logs_extended', ['risk_level'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_success'), 'audit_logs_extended', ['success'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_timestamp'), 'audit_logs_extended', ['timestamp'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_user_id'), 'audit_logs_extended', ['user_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_extended_username'), 'audit_logs_extended', ['username'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('audit_logs_extended')
    op.drop_table('audit_logs')
    op.drop_table('incident_status_history')
    op.drop_table('incident_notes')
    op.drop_table('incident_events')
    op.drop_table('incidents')
    op.drop_table('events')
    op.drop_table('api_keys')
    op.drop_table('user_sessions')
    op.drop_table('users')