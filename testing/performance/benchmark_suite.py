#!/usr/bin/env python3
"""
Performance benchmarking suite for campus security system.
"""
import asyncio
import time
import psutil
import asyncpg
import redis.asyncio as redis
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import numpy as np
import structlog

logger = structlog.get_logger()

class PerformanceBenchmark:
    """Performance benchmarking framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
    
    async def setup(self):
        """Set up benchmark environment."""
        # Database connection
        if 'database_url' in self.config:
            self.db_pool = await asyncpg.create_pool(
                self.config['database_url'],
                min_size=10,
                max_size=20
            )
        
        # Redis connection
        if 'redis_url' in self.config:
            self.redis_client = redis.from_url(self.config['redis_url'])
    
    async def cleanup(self):
        """Clean up benchmark environment."""
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def benchmark_database_operations(self) -> Dict[str, Any]:
        """Benchmark database operations."""
        if not self.db_pool:
            return {"error": "Database not configured"}
        
        results = {}
        
        # Test simple SELECT query
        start_time = time.time()
        async with self.db_pool.acquire() as conn:
            for _ in range(100):
                await conn.fetchval("SELECT 1")
        results['simple_select_100'] = time.time() - start_time
        
        # Test events table query
        start_time = time.time()
        async with self.db_pool.acquire() as conn:
            for _ in range(50):
                await conn.fetch("""
                    SELECT id, camera_id, timestamp, event_type, confidence_score
                    FROM events
                    ORDER BY timestamp DESC
                    LIMIT 10
                """)
        results['events_query_50'] = time.time() - start_time
        
        # Test incidents table query with JOIN
        start_time = time.time()
        async with self.db_pool.acquire() as conn:
            for _ in range(20):
                await conn.fetch("""
                    SELECT i.id, i.severity, i.status, COUNT(e.id) as event_count
                    FROM incidents i
                    LEFT JOIN events e ON e.id = ANY(
                        SELECT unnest(string_to_array(i.event_ids::text, ','))::uuid
                    )
                    WHERE i.created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY i.id, i.severity, i.status
                    ORDER BY i.created_at DESC
                    LIMIT 20
                """)
        results['incidents_join_query_20'] = time.time() - start_time
        
        # Test INSERT performance
        start_time = time.time()
        async with self.db_pool.acquire() as conn:
            for i in range(100):
                await conn.execute("""
                    INSERT INTO events (id, camera_id, timestamp, event_type, confidence_score, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                f"test-{i}", 
                "benchmark_cam", 
                datetime.now(), 
                "test_event", 
                0.5, 
                {"benchmark": True}
                )
        results['insert_events_100'] = time.time() - start_time
        
        # Clean up test data
        async with self.db_pool.acquire() as conn:
            await conn.execute("DELETE FROM events WHERE camera_id = 'benchmark_cam'")
        
        return results
    
    async def benchmark_redis_operations(self) -> Dict[str, Any]:
        """Benchmark Redis operations."""
        if not self.redis_client:
            return {"error": "Redis not configured"}
        
        results = {}
        
        # Test SET operations
        start_time = time.time()
        for i in range(1000):
            await self.redis_client.set(f"benchmark_key_{i}", f"value_{i}")
        results['set_operations_1000'] = time.time() - start_time
        
        # Test GET operations
        start_time = time.time()
        for i in range(1000):
            await self.redis_client.get(f"benchmark_key_{i}")
        results['get_operations_1000'] = time.time() - start_time
        
        # Test HSET operations
        start_time = time.time()
        for i in range(500):
            await self.redis_client.hset(
                f"benchmark_hash_{i}", 
                mapping={
                    "field1": f"value1_{i}",
                    "field2": f"value2_{i}",
                    "field3": f"value3_{i}"
                }
            )
        results['hset_operations_500'] = time.time() - start_time
        
        # Test LIST operations
        start_time = time.time()
        for i in range(200):
            await self.redis_client.lpush("benchmark_list", f"item_{i}")
        results['list_push_200'] = time.time() - start_time
        
        start_time = time.time()
        for _ in range(200):
            await self.redis_client.rpop("benchmark_list")
        results['list_pop_200'] = time.time() - start_time
        
        # Clean up test data
        keys_to_delete = []
        async for key in self.redis_client.scan_iter(match="benchmark_*"):
            keys_to_delete.append(key)
        
        if keys_to_delete:
            await self.redis_client.delete(*keys_to_delete)
        
        return results
    
    async def benchmark_cpu_intensive_operations(self) -> Dict[str, Any]:
        """Benchmark CPU-intensive operations."""
        results = {}
        
        # Simulate AI model inference
        start_time = time.time()
        for _ in range(100):
            # Simulate matrix operations (like neural network inference)
            matrix_a = np.random.rand(100, 100)
            matrix_b = np.random.rand(100, 100)
            result = np.dot(matrix_a, matrix_b)
            # Simulate activation function
            result = np.maximum(0, result)  # ReLU
        results['simulated_ai_inference_100'] = time.time() - start_time
        
        # Simulate image processing
        start_time = time.time()
        for _ in range(50):
            # Simulate image processing operations
            image_data = np.random.rand(640, 480, 3)  # Simulate camera frame
            # Simulate preprocessing
            gray = np.mean(image_data, axis=2)
            # Simulate edge detection
            edges = np.gradient(gray)
        results['simulated_image_processing_50'] = time.time() - start_time
        
        # Simulate privacy redaction (face detection simulation)
        start_time = time.time()
        for _ in range(20):
            # Simulate face detection algorithm
            image_data = np.random.rand(1920, 1080, 3)  # HD frame
            # Simulate feature extraction
            features = np.random.rand(1000, 128)
            # Simulate matching
            similarities = np.dot(features, features.T)
            matches = np.where(similarities > 0.8)
        results['simulated_face_detection_20'] = time.time() - start_time
        
        return results
    
    async def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent operations."""
        results = {}
        
        # Concurrent database queries
        async def db_query_task():
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.fetch("SELECT COUNT(*) FROM events")
        
        start_time = time.time()
        tasks = [db_query_task() for _ in range(50)]
        await asyncio.gather(*tasks, return_exceptions=True)
        results['concurrent_db_queries_50'] = time.time() - start_time
        
        # Concurrent Redis operations
        async def redis_task():
            if self.redis_client:
                await self.redis_client.set(f"concurrent_test_{time.time()}", "value")
        
        start_time = time.time()
        tasks = [redis_task() for _ in range(100)]
        await asyncio.gather(*tasks, return_exceptions=True)
        results['concurrent_redis_ops_100'] = time.time() - start_time
        
        # Concurrent CPU tasks
        async def cpu_task():
            await asyncio.sleep(0)  # Yield control
            matrix = np.random.rand(50, 50)
            result = np.linalg.inv(matrix)
            return result
        
        start_time = time.time()
        tasks = [cpu_task() for _ in range(20)]
        await asyncio.gather(*tasks)
        results['concurrent_cpu_tasks_20'] = time.time() - start_time
        
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_io': dict(psutil.net_io_counters()._asdict()),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting performance benchmark suite")
        
        await self.setup()
        
        try:
            # System metrics before benchmark
            system_before = self.get_system_metrics()
            
            # Run benchmarks
            db_results = await self.benchmark_database_operations()
            redis_results = await self.benchmark_redis_operations()
            cpu_results = await self.benchmark_cpu_intensive_operations()
            concurrent_results = await self.benchmark_concurrent_operations()
            
            # System metrics after benchmark
            system_after = self.get_system_metrics()
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'system_before': system_before,
                'system_after': system_after,
                'database_operations': db_results,
                'redis_operations': redis_results,
                'cpu_operations': cpu_results,
                'concurrent_operations': concurrent_results,
                'config': self.config
            }
            
            # Calculate performance scores
            results['performance_scores'] = self._calculate_performance_scores(results)
            
            return results
            
        finally:
            await self.cleanup()
    
    def _calculate_performance_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance scores based on benchmark results."""
        scores = {}
        
        # Database performance score (lower time = higher score)
        db_ops = results.get('database_operations', {})
        if 'simple_select_100' in db_ops:
            # Score based on queries per second
            qps = 100 / db_ops['simple_select_100']
            scores['database_qps'] = qps
            scores['database_score'] = min(100, qps / 10)  # Normalize to 0-100
        
        # Redis performance score
        redis_ops = results.get('redis_operations', {})
        if 'get_operations_1000' in redis_ops:
            ops_per_sec = 1000 / redis_ops['get_operations_1000']
            scores['redis_ops_per_sec'] = ops_per_sec
            scores['redis_score'] = min(100, ops_per_sec / 100)  # Normalize to 0-100
        
        # CPU performance score
        cpu_ops = results.get('cpu_operations', {})
        if 'simulated_ai_inference_100' in cpu_ops:
            inferences_per_sec = 100 / cpu_ops['simulated_ai_inference_100']
            scores['ai_inference_per_sec'] = inferences_per_sec
            scores['cpu_score'] = min(100, inferences_per_sec / 5)  # Normalize to 0-100
        
        # Overall performance score
        individual_scores = [
            scores.get('database_score', 0),
            scores.get('redis_score', 0),
            scores.get('cpu_score', 0)
        ]
        scores['overall_score'] = sum(individual_scores) / len(individual_scores)
        
        return scores


async def main():
    """Main function for running benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Campus Security Performance Benchmark')
    parser.add_argument('--database-url', help='Database connection URL')
    parser.add_argument('--redis-url', help='Redis connection URL')
    parser.add_argument('--output', default='benchmark_results.json', help='Output file')
    
    args = parser.parse_args()
    
    config = {}
    if args.database_url:
        config['database_url'] = args.database_url
    if args.redis_url:
        config['redis_url'] = args.redis_url
    
    benchmark = PerformanceBenchmark(config)
    
    print("Running performance benchmark suite...")
    print("This may take several minutes...")
    
    results = await benchmark.run_full_benchmark()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*50)
    
    scores = results.get('performance_scores', {})
    print(f"Overall Performance Score: {scores.get('overall_score', 0):.1f}/100")
    print(f"Database Score: {scores.get('database_score', 0):.1f}/100")
    print(f"Redis Score: {scores.get('redis_score', 0):.1f}/100")
    print(f"CPU Score: {scores.get('cpu_score', 0):.1f}/100")
    
    if 'database_qps' in scores:
        print(f"Database QPS: {scores['database_qps']:.1f}")
    if 'redis_ops_per_sec' in scores:
        print(f"Redis Ops/sec: {scores['redis_ops_per_sec']:.1f}")
    if 'ai_inference_per_sec' in scores:
        print(f"AI Inference/sec: {scores['ai_inference_per_sec']:.1f}")
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())