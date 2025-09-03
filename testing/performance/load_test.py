#!/usr/bin/env python3
"""
Load testing suite for campus security system.
"""
import asyncio
import aiohttp
import time
import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import argparse
import structlog

logger = structlog.get_logger()

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    base_url: str
    concurrent_users: int
    test_duration: int  # seconds
    ramp_up_time: int   # seconds
    endpoints: List[Dict[str, Any]]
    auth_token: str = None

@dataclass
class TestResult:
    """Result of a single test request."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error: str = None

class LoadTester:
    """Load testing framework for security system."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.session: aiohttp.ClientSession = None
    
    async def setup_session(self):
        """Set up HTTP session with authentication."""
        headers = {}
        if self.config.auth_token:
            headers['Authorization'] = f'Bearer {self.config.auth_token}'
        
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=timeout
        )
    
    async def cleanup_session(self):
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()
    
    async def make_request(self, endpoint_config: Dict[str, Any]) -> TestResult:
        """Make a single HTTP request and measure performance."""
        start_time = time.time()
        
        try:
            method = endpoint_config['method'].upper()
            url = f"{self.config.base_url}{endpoint_config['path']}"
            
            # Prepare request data
            kwargs = {}
            if 'data' in endpoint_config:
                if method in ['POST', 'PUT', 'PATCH']:
                    kwargs['json'] = endpoint_config['data']
            
            if 'params' in endpoint_config:
                kwargs['params'] = endpoint_config['params']
            
            # Make request
            async with self.session.request(method, url, **kwargs) as response:
                await response.text()  # Read response body
                
                response_time = time.time() - start_time
                success = 200 <= response.status < 400
                
                return TestResult(
                    endpoint=endpoint_config['name'],
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=success
                )
        
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint_config['name'],
                method=endpoint_config.get('method', 'GET'),
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def user_simulation(self, user_id: int):
        """Simulate a single user's behavior."""
        logger.info(f"Starting user simulation {user_id}")
        
        # Ramp up delay
        ramp_delay = (user_id / self.config.concurrent_users) * self.config.ramp_up_time
        await asyncio.sleep(ramp_delay)
        
        start_time = time.time()
        
        while time.time() - start_time < self.config.test_duration:
            # Select random endpoint
            endpoint = random.choice(self.config.endpoints)
            
            # Make request
            result = await self.make_request(endpoint)
            self.results.append(result)
            
            # Wait between requests (simulate user think time)
            think_time = random.uniform(0.5, 2.0)
            await asyncio.sleep(think_time)
        
        logger.info(f"User simulation {user_id} completed")
    
    async def run_load_test(self):
        """Run the complete load test."""
        logger.info(f"Starting load test with {self.config.concurrent_users} users")
        
        await self.setup_session()
        
        try:
            # Create user simulation tasks
            tasks = [
                asyncio.create_task(self.user_simulation(i))
                for i in range(self.config.concurrent_users)
            ]
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
        finally:
            await self.cleanup_session()
        
        logger.info("Load test completed")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance test report."""
        if not self.results:
            return {"error": "No test results available"}
        
        # Calculate statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        
        response_times = [r.response_time for r in self.results if r.success]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p50 = sorted_times[int(len(sorted_times) * 0.5)]
            p90 = sorted_times[int(len(sorted_times) * 0.9)]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50 = p90 = p95 = p99 = 0
        
        # Calculate throughput
        test_duration = self.config.test_duration
        throughput = successful_requests / test_duration if test_duration > 0 else 0
        
        # Endpoint statistics
        endpoint_stats = {}
        for result in self.results:
            if result.endpoint not in endpoint_stats:
                endpoint_stats[result.endpoint] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'avg_response_time': 0,
                    'response_times': []
                }
            
            stats = endpoint_stats[result.endpoint]
            stats['total'] += 1
            
            if result.success:
                stats['successful'] += 1
                stats['response_times'].append(result.response_time)
            else:
                stats['failed'] += 1
        
        # Calculate endpoint averages
        for endpoint, stats in endpoint_stats.items():
            if stats['response_times']:
                stats['avg_response_time'] = sum(stats['response_times']) / len(stats['response_times'])
            del stats['response_times']  # Remove raw data from report
        
        # Error analysis
        error_counts = {}
        for result in self.results:
            if not result.success:
                error_key = f"{result.status_code}: {result.error or 'Unknown error'}"
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
        
        return {
            'test_config': {
                'concurrent_users': self.config.concurrent_users,
                'test_duration': self.config.test_duration,
                'ramp_up_time': self.config.ramp_up_time,
                'base_url': self.config.base_url
            },
            'summary': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                'throughput_rps': throughput
            },
            'response_times': {
                'average': avg_response_time,
                'minimum': min_response_time,
                'maximum': max_response_time,
                'percentile_50': p50,
                'percentile_90': p90,
                'percentile_95': p95,
                'percentile_99': p99
            },
            'endpoint_statistics': endpoint_stats,
            'errors': error_counts,
            'timestamp': datetime.now().isoformat()
        }


def create_security_test_config(base_url: str, auth_token: str = None) -> LoadTestConfig:
    """Create load test configuration for security system."""
    
    endpoints = [
        {
            'name': 'health_check',
            'method': 'GET',
            'path': '/health'
        },
        {
            'name': 'list_incidents',
            'method': 'GET',
            'path': '/api/v1/incidents',
            'params': {'limit': 50, 'status': 'open'}
        },
        {
            'name': 'get_incident',
            'method': 'GET',
            'path': '/api/v1/incidents/123e4567-e89b-12d3-a456-426614174000'
        },
        {
            'name': 'create_event',
            'method': 'POST',
            'path': '/api/v1/events',
            'data': {
                'camera_id': 'cam_001',
                'event_type': 'intrusion',
                'confidence_score': 0.85,
                'timestamp': datetime.now().isoformat(),
                'metadata': {'location': 'Building A, Floor 1'}
            }
        },
        {
            'name': 'list_events',
            'method': 'GET',
            'path': '/api/v1/events',
            'params': {'limit': 100, 'camera_id': 'cam_001'}
        },
        {
            'name': 'get_analytics',
            'method': 'GET',
            'path': '/api/v1/analytics/incidents/summary',
            'params': {'period': '24h'}
        },
        {
            'name': 'user_auth',
            'method': 'POST',
            'path': '/api/v1/auth/login',
            'data': {
                'username': 'test_user',
                'password': 'test_password'
            }
        }
    ]
    
    return LoadTestConfig(
        base_url=base_url,
        concurrent_users=50,
        test_duration=300,  # 5 minutes
        ramp_up_time=60,    # 1 minute
        endpoints=endpoints,
        auth_token=auth_token
    )


async def main():
    """Main function for running load tests."""
    parser = argparse.ArgumentParser(description='Campus Security Load Testing')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL for testing')
    parser.add_argument('--users', type=int, default=50, help='Number of concurrent users')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--ramp-up', type=int, default=60, help='Ramp up time in seconds')
    parser.add_argument('--auth-token', help='Authentication token')
    parser.add_argument('--output', default='load_test_report.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Create test configuration
    config = create_security_test_config(args.url, args.auth_token)
    config.concurrent_users = args.users
    config.test_duration = args.duration
    config.ramp_up_time = args.ramp_up
    
    # Run load test
    tester = LoadTester(config)
    
    print(f"Starting load test with {config.concurrent_users} users for {config.test_duration} seconds")
    print(f"Target URL: {config.base_url}")
    
    start_time = time.time()
    await tester.run_load_test()
    total_time = time.time() - start_time
    
    # Generate and save report
    report = tester.generate_report()
    report['actual_test_duration'] = total_time
    
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("LOAD TEST RESULTS")
    print("="*50)
    print(f"Total Requests: {report['summary']['total_requests']}")
    print(f"Successful: {report['summary']['successful_requests']}")
    print(f"Failed: {report['summary']['failed_requests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.2f}%")
    print(f"Throughput: {report['summary']['throughput_rps']:.2f} RPS")
    print(f"Average Response Time: {report['response_times']['average']:.3f}s")
    print(f"95th Percentile: {report['response_times']['percentile_95']:.3f}s")
    print(f"99th Percentile: {report['response_times']['percentile_99']:.3f}s")
    print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())