#!/usr/bin/env python3
"""
Performance optimization script for campus security system.
"""
import asyncio
import asyncpg
import redis.asyncio as redis
import argparse
import json
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class PerformanceOptimizer:
    """Automated performance optimization for the security system."""
    
    def __init__(self, config: dict):
        self.config = config
        self.db_pool = None
        self.redis_client = None
        self.optimization_results = {}
    
    async def setup(self):
        """Set up connections."""
        if 'database_url' in self.config:
            self.db_pool = await asyncpg.create_pool(
                self.config['database_url'],
                min_size=5,
                max_size=10
            )
        
        if 'redis_url' in self.config:
            self.redis_client = redis.from_url(self.config['redis_url'])
    
    async def cleanup(self):
        """Clean up connections."""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()
    
    async def optimize_database(self):
        """Optimize database performance."""
        if not self.db_pool:
            return {"error": "Database not configured"}
        
        results = {}
        
        # Create optimized indexes
        indexes = [
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_timestamp_desc ON events (timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_camera_timestamp ON events (camera_id, timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_type_confidence ON events (event_type, confidence_score DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incidents_status_created ON incidents (status, created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incidents_severity ON incidents (severity) WHERE status != 'resolved'",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_incident_created ON evidence (incident_id, created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_user_timestamp ON audit_log (user_id, timestamp DESC)",
        ]
        
        created_indexes = 0
        async with self.db_pool.acquire() as conn:
            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                    created_indexes += 1
                    logger.info(f"Created index: {index_sql.split()[-1]}")
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
        
        results['indexes_created'] = created_indexes
        
        # Update table statistics
        tables = ['events', 'incidents', 'evidence', 'audit_log', 'users', 'cameras']
        analyzed_tables = 0
        
        async with self.db_pool.acquire() as conn:
            for table in tables:
                try:
                    await conn.execute(f"ANALYZE {table}")
                    analyzed_tables += 1
                    logger.info(f"Analyzed table: {table}")
                except Exception as e:
                    logger.warning(f"Table analysis failed for {table}: {e}")
        
        results['tables_analyzed'] = analyzed_tables
        
        # Vacuum tables to reclaim space
        vacuum_results = {}
        async with self.db_pool.acquire() as conn:
            for table in tables:
                try:
                    # Get table size before vacuum
                    size_before = await conn.fetchval(
                        f"SELECT pg_total_relation_size('{table}')"
                    )
                    
                    await conn.execute(f"VACUUM ANALYZE {table}")
                    
                    # Get table size after vacuum
                    size_after = await conn.fetchval(
                        f"SELECT pg_total_relation_size('{table}')"
                    )
                    
                    space_reclaimed = size_before - size_after
                    vacuum_results[table] = {
                        'size_before': size_before,
                        'size_after': size_after,
                        'space_reclaimed': space_reclaimed
                    }
                    
                    logger.info(f"Vacuumed table {table}, reclaimed {space_reclaimed} bytes")
                    
                except Exception as e:
                    logger.warning(f"Vacuum failed for {table}: {e}")
        
        results['vacuum_results'] = vacuum_results
        
        # Optimize database configuration
        config_optimizations = [
            "SET shared_preload_libraries = 'pg_stat_statements'",
            "SET max_connections = 200",
            "SET shared_buffers = '256MB'",
            "SET effective_cache_size = '1GB'",
            "SET maintenance_work_mem = '64MB'",
            "SET checkpoint_completion_target = 0.9",
            "SET wal_buffers = '16MB'",
            "SET default_statistics_target = 100"
        ]
        
        config_applied = 0
        async with self.db_pool.acquire() as conn:
            for config_sql in config_optimizations:
                try:
                    await conn.execute(config_sql)
                    config_applied += 1
                except Exception as e:
                    logger.warning(f"Config optimization failed: {e}")
        
        results['config_optimizations_applied'] = config_applied
        
        return results
    
    async def optimize_redis(self):
        """Optimize Redis performance."""
        if not self.redis_client:
            return {"error": "Redis not configured"}
        
        results = {}
        
        # Get Redis info
        redis_info = await self.redis_client.info()
        results['redis_version'] = redis_info.get('redis_version')
        results['used_memory'] = redis_info.get('used_memory')
        results['connected_clients'] = redis_info.get('connected_clients')
        
        # Clean up expired keys
        expired_cleaned = 0
        
        # Clean up old cache entries (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        patterns_to_clean = [
            'camera_status:*',
            'analytics:*',
            'temp:*'
        ]
        
        for pattern in patterns_to_clean:
            async for key in self.redis_client.scan_iter(match=pattern):
                try:
                    # Check if key has TTL
                    ttl = await self.redis_client.ttl(key)
                    if ttl == -1:  # No expiration set
                        # Set expiration for cache keys without TTL
                        await self.redis_client.expire(key, 3600)  # 1 hour
                        expired_cleaned += 1
                except Exception as e:
                    logger.warning(f"Failed to set expiration for key {key}: {e}")
        
        results['keys_with_expiration_set'] = expired_cleaned
        
        # Optimize Redis configuration
        config_commands = [
            ('CONFIG', 'SET', 'maxmemory-policy', 'allkeys-lru'),
            ('CONFIG', 'SET', 'timeout', '300'),
            ('CONFIG', 'SET', 'tcp-keepalive', '60'),
            ('CONFIG', 'SET', 'maxclients', '1000')
        ]
        
        config_applied = 0
        for cmd in config_commands:
            try:
                await self.redis_client.execute_command(*cmd)
                config_applied += 1
                logger.info(f"Applied Redis config: {' '.join(cmd)}")
            except Exception as e:
                logger.warning(f"Redis config failed: {e}")
        
        results['redis_config_applied'] = config_applied
        
        return results
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention policies."""
        if not self.db_pool:
            return {"error": "Database not configured"}
        
        results = {}
        
        # Define retention policies (in days)
        retention_policies = {
            'events': {
                'low_confidence': 30,    # Low confidence events
                'high_confidence': 90,   # High confidence events
                'critical': 365         # Critical events
            },
            'incidents': {
                'resolved': 180,        # Resolved incidents
                'test': 7              # Test incidents
            },
            'audit_log': {
                'general': 90,         # General audit logs
                'security': 365        # Security-related logs
            },
            'notifications': {
                'delivered': 30,       # Delivered notifications
                'failed': 7           # Failed notifications
            }
        }
        
        cleanup_results = {}
        
        async with self.db_pool.acquire() as conn:
            # Clean up old events
            for event_type, days in retention_policies['events'].items():
                cutoff_date = datetime.now() - timedelta(days=days)
                
                if event_type == 'low_confidence':
                    query = """
                        DELETE FROM events 
                        WHERE timestamp < $1 AND confidence_score < 0.7
                    """
                elif event_type == 'high_confidence':
                    query = """
                        DELETE FROM events 
                        WHERE timestamp < $1 AND confidence_score >= 0.7 AND confidence_score < 0.9
                    """
                else:  # critical
                    query = """
                        DELETE FROM events 
                        WHERE timestamp < $1 AND confidence_score >= 0.9
                    """
                
                try:
                    result = await conn.execute(query, cutoff_date)
                    deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                    cleanup_results[f'events_{event_type}'] = deleted_count
                    logger.info(f"Cleaned up {deleted_count} {event_type} events")
                except Exception as e:
                    logger.error(f"Failed to cleanup {event_type} events: {e}")
            
            # Clean up old incidents
            for incident_type, days in retention_policies['incidents'].items():
                cutoff_date = datetime.now() - timedelta(days=days)
                
                if incident_type == 'resolved':
                    query = """
                        DELETE FROM incidents 
                        WHERE resolved_at < $1 AND status = 'resolved'
                    """
                else:  # test
                    query = """
                        DELETE FROM incidents 
                        WHERE created_at < $1 AND metadata->>'test' = 'true'
                    """
                
                try:
                    result = await conn.execute(query, cutoff_date)
                    deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                    cleanup_results[f'incidents_{incident_type}'] = deleted_count
                    logger.info(f"Cleaned up {deleted_count} {incident_type} incidents")
                except Exception as e:
                    logger.error(f"Failed to cleanup {incident_type} incidents: {e}")
            
            # Clean up old audit logs
            for log_type, days in retention_policies['audit_log'].items():
                cutoff_date = datetime.now() - timedelta(days=days)
                
                if log_type == 'general':
                    query = """
                        DELETE FROM audit_log 
                        WHERE timestamp < $1 
                        AND action NOT IN ('login', 'evidence_access', 'incident_create', 'user_create')
                    """
                else:  # security
                    query = """
                        DELETE FROM audit_log 
                        WHERE timestamp < $1 
                        AND action IN ('login', 'evidence_access', 'incident_create', 'user_create')
                    """
                
                try:
                    result = await conn.execute(query, cutoff_date)
                    deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                    cleanup_results[f'audit_log_{log_type}'] = deleted_count
                    logger.info(f"Cleaned up {deleted_count} {log_type} audit logs")
                except Exception as e:
                    logger.error(f"Failed to cleanup {log_type} audit logs: {e}")
        
        results['cleanup_results'] = cleanup_results
        results['total_records_cleaned'] = sum(cleanup_results.values())
        
        return results
    
    async def optimize_system_performance(self):
        """Run comprehensive system optimization."""
        logger.info("Starting system performance optimization")
        
        await self.setup()
        
        try:
            # Database optimization
            logger.info("Optimizing database performance...")
            db_results = await self.optimize_database()
            self.optimization_results['database'] = db_results
            
            # Redis optimization
            logger.info("Optimizing Redis performance...")
            redis_results = await self.optimize_redis()
            self.optimization_results['redis'] = redis_results
            
            # Data cleanup
            logger.info("Cleaning up old data...")
            cleanup_results = await self.cleanup_old_data()
            self.optimization_results['cleanup'] = cleanup_results
            
            # Summary
            self.optimization_results['timestamp'] = datetime.now().isoformat()
            self.optimization_results['optimization_summary'] = {
                'database_indexes_created': db_results.get('indexes_created', 0),
                'database_tables_analyzed': db_results.get('tables_analyzed', 0),
                'redis_configs_applied': redis_results.get('redis_config_applied', 0),
                'total_records_cleaned': cleanup_results.get('total_records_cleaned', 0)
            }
            
            logger.info("System optimization completed successfully")
            return self.optimization_results
            
        finally:
            await self.cleanup()


async def main():
    """Main function for performance optimization."""
    parser = argparse.ArgumentParser(description='Campus Security Performance Optimization')
    parser.add_argument('--database-url', required=True, help='Database connection URL')
    parser.add_argument('--redis-url', required=True, help='Redis connection URL')
    parser.add_argument('--output', default='optimization_results.json', help='Output file')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    config = {
        'database_url': args.database_url,
        'redis_url': args.redis_url,
        'dry_run': args.dry_run
    }
    
    optimizer = PerformanceOptimizer(config)
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print("Would perform the following optimizations:")
        print("- Create database indexes")
        print("- Analyze database tables")
        print("- Vacuum database tables")
        print("- Optimize Redis configuration")
        print("- Clean up old data based on retention policies")
        return
    
    print("Starting performance optimization...")
    print("This may take several minutes...")
    
    results = await optimizer.optimize_system_performance()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE OPTIMIZATION RESULTS")
    print("="*50)
    
    summary = results.get('optimization_summary', {})
    print(f"Database indexes created: {summary.get('database_indexes_created', 0)}")
    print(f"Database tables analyzed: {summary.get('database_tables_analyzed', 0)}")
    print(f"Redis configurations applied: {summary.get('redis_configs_applied', 0)}")
    print(f"Total records cleaned up: {summary.get('total_records_cleaned', 0)}")
    
    print(f"\nDetailed results saved to: {args.output}")
    print("Performance optimization completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())