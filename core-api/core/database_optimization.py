"""
Database optimization utilities for campus security system.
"""
import asyncio
from typing import List, Dict, Any, Optional
import asyncpg
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class DatabaseOptimizer:
    """Database optimization and performance tuning utilities."""
    
    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool
    
    async def create_indexes(self):
        """Create optimized indexes for security system tables."""
        
        indexes = [
            # Events table indexes for high-volume queries
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_timestamp ON events (timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_camera_timestamp ON events (camera_id, timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_type_timestamp ON events (event_type, timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_confidence ON events (confidence_score DESC) WHERE confidence_score > 0.7",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_metadata_gin ON events USING GIN (metadata)",
            
            # Incidents table indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incidents_status ON incidents (status)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incidents_severity_created ON incidents (severity, created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incidents_assigned_to ON incidents (assigned_to) WHERE assigned_to IS NOT NULL",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incidents_created_at ON incidents (created_at DESC)",
            
            # Evidence table indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_incident_id ON evidence (incident_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_file_type ON evidence (file_type)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_created_at ON evidence (created_at DESC)",
            
            # Audit log indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_user_timestamp ON audit_log (user_id, timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_action_timestamp ON audit_log (action, timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_resource ON audit_log (resource_type, resource_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp DESC)",
            
            # User management indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users (email) WHERE email IS NOT NULL",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_role ON users (role)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active ON users (is_active) WHERE is_active = true",
            
            # Camera configuration indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_location ON cameras (location)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_active ON cameras (is_active) WHERE is_active = true",
            
            # Notification indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_notifications_user_timestamp ON notifications (user_id, created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_notifications_status ON notifications (status)",
        ]
        
        async with self.pool.acquire() as conn:
            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                    logger.info(f"Created index: {index_sql.split()[-1]}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
    
    async def create_partitions(self):
        """Create table partitions for time-series data."""
        
        # Partition events table by month
        partition_queries = [
            """
            -- Create partitioned events table if not exists
            CREATE TABLE IF NOT EXISTS events_partitioned (
                LIKE events INCLUDING ALL
            ) PARTITION BY RANGE (timestamp);
            """,
            
            # Create monthly partitions for the next 12 months
            *[f"""
            CREATE TABLE IF NOT EXISTS events_y{(datetime.now() + timedelta(days=30*i)).year}_m{(datetime.now() + timedelta(days=30*i)).month:02d}
            PARTITION OF events_partitioned
            FOR VALUES FROM ('{(datetime.now() + timedelta(days=30*i)).replace(day=1).date()}')
            TO ('{(datetime.now() + timedelta(days=30*(i+1))).replace(day=1).date()}');
            """ for i in range(12)],
            
            # Create partitioned audit_log table
            """
            CREATE TABLE IF NOT EXISTS audit_log_partitioned (
                LIKE audit_log INCLUDING ALL
            ) PARTITION BY RANGE (timestamp);
            """,
            
            # Create monthly partitions for audit log
            *[f"""
            CREATE TABLE IF NOT EXISTS audit_log_y{(datetime.now() + timedelta(days=30*i)).year}_m{(datetime.now() + timedelta(days=30*i)).month:02d}
            PARTITION OF audit_log_partitioned
            FOR VALUES FROM ('{(datetime.now() + timedelta(days=30*i)).replace(day=1).date()}')
            TO ('{(datetime.now() + timedelta(days=30*(i+1))).replace(day=1).date()}');
            """ for i in range(12)]
        ]
        
        async with self.pool.acquire() as conn:
            for query in partition_queries:
                try:
                    await conn.execute(query)
                    logger.info("Created partition table")
                except Exception as e:
                    logger.warning(f"Failed to create partition: {e}")
    
    async def optimize_queries(self):
        """Run query optimization procedures."""
        
        optimization_queries = [
            # Update table statistics
            "ANALYZE events;",
            "ANALYZE incidents;",
            "ANALYZE evidence;",
            "ANALYZE audit_log;",
            "ANALYZE users;",
            "ANALYZE cameras;",
            
            # Vacuum tables to reclaim space
            "VACUUM ANALYZE events;",
            "VACUUM ANALYZE incidents;",
            "VACUUM ANALYZE evidence;",
            "VACUUM ANALYZE audit_log;",
        ]
        
        async with self.pool.acquire() as conn:
            for query in optimization_queries:
                try:
                    await conn.execute(query)
                    logger.info(f"Executed optimization: {query}")
                except Exception as e:
                    logger.warning(f"Failed to execute optimization: {e}")
    
    async def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slow queries from pg_stat_statements."""
        
        query = """
        SELECT 
            query,
            calls,
            total_time,
            mean_time,
            rows,
            100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
        FROM pg_stat_statements
        WHERE query NOT LIKE '%pg_stat_statements%'
        ORDER BY mean_time DESC
        LIMIT $1;
        """
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, limit)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []
    
    async def get_table_sizes(self) -> List[Dict[str, Any]]:
        """Get table sizes for monitoring storage usage."""
        
        query = """
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
        """
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get table sizes: {e}")
            return []
    
    async def cleanup_old_data(self, retention_days: int = 90):
        """Clean up old data based on retention policies."""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleanup_queries = [
            # Clean up old events (keep only high-confidence ones longer)
            f"""
            DELETE FROM events 
            WHERE timestamp < '{cutoff_date}' 
            AND confidence_score < 0.8;
            """,
            
            # Clean up resolved incidents older than retention period
            f"""
            DELETE FROM incidents 
            WHERE resolved_at < '{cutoff_date}' 
            AND status = 'resolved';
            """,
            
            # Clean up old audit logs (keep critical actions longer)
            f"""
            DELETE FROM audit_log 
            WHERE timestamp < '{cutoff_date}' 
            AND action NOT IN ('login', 'evidence_access', 'incident_create');
            """,
            
            # Clean up old notifications
            f"""
            DELETE FROM notifications 
            WHERE created_at < '{cutoff_date}' 
            AND status = 'delivered';
            """
        ]
        
        deleted_counts = []
        async with self.pool.acquire() as conn:
            for query in cleanup_queries:
                try:
                    result = await conn.execute(query)
                    deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                    deleted_counts.append(deleted_count)
                    logger.info(f"Cleaned up {deleted_count} records")
                except Exception as e:
                    logger.error(f"Failed to cleanup data: {e}")
                    deleted_counts.append(0)
        
        return deleted_counts


class QueryOptimizer:
    """Query optimization utilities."""
    
    @staticmethod
    def build_incident_query(
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> tuple[str, list]:
        """Build optimized incident query with filters."""
        
        base_query = """
        SELECT i.*, 
               COUNT(e.id) as event_count,
               MAX(e.confidence_score) as max_confidence
        FROM incidents i
        LEFT JOIN events e ON e.id = ANY(
            SELECT unnest(string_to_array(i.event_ids::text, ','))::uuid
        )
        """
        
        where_conditions = []
        params = []
        param_count = 0
        
        if filters.get('status'):
            param_count += 1
            where_conditions.append(f"i.status = ${param_count}")
            params.append(filters['status'])
        
        if filters.get('severity'):
            param_count += 1
            where_conditions.append(f"i.severity = ${param_count}")
            params.append(filters['severity'])
        
        if filters.get('assigned_to'):
            param_count += 1
            where_conditions.append(f"i.assigned_to = ${param_count}")
            params.append(filters['assigned_to'])
        
        if filters.get('date_from'):
            param_count += 1
            where_conditions.append(f"i.created_at >= ${param_count}")
            params.append(filters['date_from'])
        
        if filters.get('date_to'):
            param_count += 1
            where_conditions.append(f"i.created_at <= ${param_count}")
            params.append(filters['date_to'])
        
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        base_query += """
        GROUP BY i.id
        ORDER BY i.created_at DESC
        LIMIT $%d OFFSET $%d
        """ % (param_count + 1, param_count + 2)
        
        params.extend([limit, offset])
        
        return base_query, params
    
    @staticmethod
    def build_events_query(
        camera_ids: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        time_range: Optional[tuple[datetime, datetime]] = None,
        limit: int = 1000
    ) -> tuple[str, list]:
        """Build optimized events query."""
        
        base_query = """
        SELECT id, camera_id, timestamp, event_type, confidence_score, metadata
        FROM events
        """
        
        where_conditions = []
        params = []
        param_count = 0
        
        if camera_ids:
            param_count += 1
            where_conditions.append(f"camera_id = ANY(${param_count})")
            params.append(camera_ids)
        
        if event_types:
            param_count += 1
            where_conditions.append(f"event_type = ANY(${param_count})")
            params.append(event_types)
        
        if min_confidence > 0:
            param_count += 1
            where_conditions.append(f"confidence_score >= ${param_count}")
            params.append(min_confidence)
        
        if time_range:
            param_count += 1
            where_conditions.append(f"timestamp >= ${param_count}")
            params.append(time_range[0])
            
            param_count += 1
            where_conditions.append(f"timestamp <= ${param_count}")
            params.append(time_range[1])
        
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        base_query += f" ORDER BY timestamp DESC LIMIT ${param_count + 1}"
        params.append(limit)
        
        return base_query, params