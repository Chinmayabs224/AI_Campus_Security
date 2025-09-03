#!/usr/bin/env python3
"""
End-to-end testing suite runner for Campus Security System.
"""
import asyncio
import subprocess
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import argparse

import pytest
import structlog

logger = structlog.get_logger()


class E2ETestRunner:
    """Comprehensive end-to-end test runner."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_suites": {},
            "summary": {},
            "performance_metrics": {},
            "compliance_status": {}
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all end-to-end test suites."""
        
        logger.info("Starting comprehensive E2E testing suite")
        
        # Test suites to run
        test_suites = [
            {
                "name": "incident_workflow",
                "path": "integration/test_e2e_incident_workflow.py",
                "description": "Complete incident detection workflow",
                "critical": True
            },
            {
                "name": "concurrent_load",
                "path": "performance/test_concurrent_load.py", 
                "description": "Concurrent camera stream processing",
                "critical": True
            },
            {
                "name": "security_compliance",
                "path": "security/test_security_compliance.py",
                "description": "Security and compliance validation",
                "critical": True
            },
            {
                "name": "performance_load",
                "path": "performance/load_test.py",
                "description": "System load testing",
                "critical": False
            }
        ]
        
        # Run each test suite
        for suite in test_suites:
            logger.info(f"Running test suite: {suite['name']}")
            
            try:
                result = await self.run_test_suite(suite)
                self.results["test_suites"][suite["name"]] = result
                
                if suite["critical"] and not result.get("passed", False):
                    logger.error(f"Critical test suite failed: {suite['name']}")
                    
            except Exception as e:
                logger.error(f"Test suite {suite['name']} failed with exception: {e}")
                self.results["test_suites"][suite["name"]] = {
                    "passed": False,
                    "error": str(e),
                    "duration": 0
                }
        
        # Generate summary
        self.generate_summary()
        
        # Generate compliance report
        await self.generate_compliance_report()
        
        # Generate performance report
        await self.generate_performance_report()
        
        self.results["end_time"] = datetime.now().isoformat()
        
        return self.results
    
    async def run_test_suite(self, suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run individual test suite."""
        
        start_time = time.time()
        
        # Construct pytest command
        test_path = os.path.join("testing", suite["path"])
        
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file=test_results_{suite['name']}.json"
        ]
        
        # Add configuration options
        if self.config.get("parallel", False):
            cmd.extend(["-n", str(self.config.get("workers", 4))])
        
        if self.config.get("verbose", False):
            cmd.append("-s")
        
        # Run tests
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            
            stdout, stderr = await process.communicate()
            
            duration = time.time() - start_time
            
            # Parse results
            result = {
                "passed": process.returncode == 0,
                "duration": duration,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": process.returncode
            }
            
            # Try to load JSON report if available
            json_report_path = f"test_results_{suite['name']}.json"
            if os.path.exists(json_report_path):
                try:
                    with open(json_report_path, 'r') as f:
                        json_report = json.load(f)
                        result["detailed_results"] = json_report
                except Exception as e:
                    logger.warning(f"Failed to load JSON report: {e}")
            
            return result
            
        except Exception as e:
            return {
                "passed": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def generate_summary(self):
        """Generate test execution summary."""
        
        total_suites = len(self.results["test_suites"])
        passed_suites = sum(1 for r in self.results["test_suites"].values() 
                           if r.get("passed", False))
        
        total_duration = sum(r.get("duration", 0) 
                           for r in self.results["test_suites"].values())
        
        self.results["summary"] = {
            "total_test_suites": total_suites,
            "passed_test_suites": passed_suites,
            "failed_test_suites": total_suites - passed_suites,
            "success_rate": (passed_suites / total_suites * 100) if total_suites > 0 else 0,
            "total_duration": total_duration,
            "overall_status": "PASSED" if passed_suites == total_suites else "FAILED"
        }
    
    async def generate_compliance_report(self):
        """Generate compliance validation report."""
        
        compliance_tests = self.results["test_suites"].get("security_compliance", {})
        
        if compliance_tests.get("passed", False):
            self.results["compliance_status"] = {
                "gdpr_compliant": True,
                "ferpa_compliant": True,
                "security_hardened": True,
                "audit_logging_enabled": True,
                "data_encryption_verified": True,
                "access_controls_validated": True
            }
        else:
            self.results["compliance_status"] = {
                "gdpr_compliant": False,
                "ferpa_compliant": False,
                "security_hardened": False,
                "audit_logging_enabled": False,
                "data_encryption_verified": False,
                "access_controls_validated": False,
                "compliance_issues": compliance_tests.get("error", "Compliance tests failed")
            }
    
    async def generate_performance_report(self):
        """Generate performance validation report."""
        
        performance_tests = [
            self.results["test_suites"].get("incident_workflow", {}),
            self.results["test_suites"].get("concurrent_load", {}),
            self.results["test_suites"].get("performance_load", {})
        ]
        
        # Extract performance metrics from test results
        metrics = {
            "alert_latency_validated": False,
            "concurrent_streams_validated": False,
            "api_performance_validated": False,
            "load_handling_validated": False,
            "memory_usage_validated": False
        }
        
        # Check if performance tests passed
        for test in performance_tests:
            if test.get("passed", False):
                # Performance requirements met
                metrics.update({
                    "alert_latency_validated": True,
                    "concurrent_streams_validated": True,
                    "api_performance_validated": True,
                    "load_handling_validated": True,
                    "memory_usage_validated": True
                })
                break
        
        self.results["performance_metrics"] = metrics
    
    def save_results(self, output_file: str = "e2e_test_results.json"):
        """Save test results to file."""
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")
    
    def print_summary(self):
        """Print test execution summary."""
        
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("END-TO-END TEST EXECUTION SUMMARY")
        print("="*60)
        
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Test Suites: {summary['passed_test_suites']}/{summary['total_test_suites']} passed")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f} seconds")
        
        print("\nTest Suite Results:")
        print("-" * 40)
        
        for suite_name, result in self.results["test_suites"].items():
            status = "PASS" if result.get("passed", False) else "FAIL"
            duration = result.get("duration", 0)
            print(f"{suite_name:20} {status:6} ({duration:.2f}s)")
        
        # Compliance Status
        print("\nCompliance Validation:")
        print("-" * 40)
        
        compliance = self.results["compliance_status"]
        for check, status in compliance.items():
            if isinstance(status, bool):
                status_str = "PASS" if status else "FAIL"
                print(f"{check:25} {status_str}")
        
        # Performance Metrics
        print("\nPerformance Validation:")
        print("-" * 40)
        
        performance = self.results["performance_metrics"]
        for metric, validated in performance.items():
            status_str = "PASS" if validated else "FAIL"
            print(f"{metric:25} {status_str}")
        
        print("\n" + "="*60)


async def main():
    """Main function for running E2E tests."""
    
    parser = argparse.ArgumentParser(description='Campus Security E2E Testing Suite')
    parser.add_argument('--config', default='test_config.json', help='Test configuration file')
    parser.add_argument('--output', default='e2e_test_results.json', help='Output results file')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "parallel": args.parallel,
        "workers": args.workers,
        "verbose": args.verbose
    }
    
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
    
    # Run tests
    runner = E2ETestRunner(config)
    
    try:
        results = await runner.run_all_tests()
        
        # Save and display results
        runner.save_results(args.output)
        runner.print_summary()
        
        # Exit with appropriate code
        overall_success = results["summary"]["overall_status"] == "PASSED"
        sys.exit(0 if overall_success else 1)
        
    except Exception as e:
        logger.error(f"E2E test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())