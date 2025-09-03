-- Campus Security Database Initialization Script

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Events table for high-volume writes
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2),
    bounding_boxes JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Incidents table for workflow management
CREATE TABLE IF NOT EXISTS incidents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status VARCHAR(20) NOT NULL CHECK (status IN ('open', 'assigned', 'resolved')),
    assigned_to VARCHAR(100),
    location JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

-- Incident-Event relationship table
CREATE TABLE IF NOT EXISTS incident_events (
    incident_id UUID REFERENCES incidents(id) ON DELETE CASCADE,
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    PRIMARY KEY (incident_id, event_id)
);

-- Evidence table for secure file tracking
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID REFERENCES incidents(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    duration DECIMAL(10,2),
    redaction_applied BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'supervisor', 'guard', 'viewer')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

-- Audit log for compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    details JSONB
);

-- Camera configuration table
CREATE TABLE IF NOT EXISTS cameras (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location JSONB,
    rtsp_url VARCHAR(500),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance')),
    edge_node_id VARCHAR(50),
    privacy_zones JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Notification preferences table
CREATE TABLE IF NOT EXISTS notification_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    notification_type VARCHAR(50) NOT NULL,
    enabled BOOLEAN DEFAULT true,
    settings JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_cameras_status ON cameras(status);

-- Insert sample data for development
INSERT INTO users (username, email, full_name, role) VALUES
    ('admin', 'admin@campus.edu', 'System Administrator', 'admin'),
    ('supervisor1', 'supervisor@campus.edu', 'Security Supervisor', 'supervisor'),
    ('guard1', 'guard1@campus.edu', 'Security Guard 1', 'guard'),
    ('guard2', 'guard2@campus.edu', 'Security Guard 2', 'guard')
ON CONFLICT (username) DO NOTHING;

INSERT INTO cameras (id, name, location, rtsp_url, edge_node_id) VALUES
    ('CAM001', 'Main Entrance', '{"lat": 40.7128, "lng": -74.0060, "building": "Main Hall"}', 'rtsp://demo:demo@192.168.1.100:554/stream1', 'edge-node-1'),
    ('CAM002', 'Library Entrance', '{"lat": 40.7130, "lng": -74.0058, "building": "Library"}', 'rtsp://demo:demo@192.168.1.101:554/stream1', 'edge-node-1'),
    ('CAM003', 'Parking Lot A', '{"lat": 40.7125, "lng": -74.0065, "building": "Parking"}', 'rtsp://demo:demo@192.168.1.102:554/stream1', 'edge-node-2'),
    ('CAM004', 'Student Center', '{"lat": 40.7132, "lng": -74.0055, "building": "Student Center"}', 'rtsp://demo:demo@192.168.1.103:554/stream1', 'edge-node-2')
ON CONFLICT (id) DO NOTHING;