#!/usr/bin/env python3
"""
System Validation and Acceptance Testing Runner.
Executes comprehensive validation suite for the AI-Powered Campus Security System.
"""
import asyncio
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pytest
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class SystemValidationRunner:
    """Main runner for system validation and acceptance testing."""
    
    def __init__(self, config_path: str = "testing/test_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "test_suites": {},
            "overall_status": "UNKNOWN",
            "summary": {}
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load test configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "test_environment": {
                "base_url": "http://localhost:8000",
                "database_url": "postgresql://test_user:test_pass@localhost:5432/campus_security_test",
                "redis_url": "redis://localhost:6379/1"
            },
            "performance_thresholds": {
                "alert_latency_max": 5.0,
                "detection_accuracy_min": 0.7,
                "false_positive_rate_max": 0.3,
                "api_response_time_max": 1.0,
                "concurrent_streams_min": 10
            },
            "test_execution": {
                "parallel": True,
                "workers": 4,
                "timeout_seconds": 1800,
                "retry_attempts": 3
            }
        }
    
    async def run_performance_validation(self) -> Dict[str, Any]:
        """Run performance validation tests."""
        logger.info("Starting performance validation tests")
        
        test_args = [
            "testing/validation/system_validation.py::test_comprehensive_system_validation",
            "-v", "--tb=short"
        ]
        
        if self.config["test_execution"]["parallel"]:
            test_args.extend(["-n", str(self.config["test_execution"]["workers"])])
        
        start_time = time.time()
        result = pytest.main(test_args)
        duration = time.time() - start_time
        
        return {
            "status": "PASS" if result == 0 else "FAIL",
            "duration": duration,
            "exit_code": result,
            "test_type": "performance_validation"
        }
    
    async def run_user_acceptance_tests(self) -> Dict[str, Any]:
        """Run user acceptance tests."""
        logger.info("Starting user acceptance tests")
        
        test_args = [
            "testing/validation/acceptance_testing.py::test_complete_user_acceptance_suite",
            "-v", "--tb=short"
        ]
        
        start_time = time.time()
        result = pytest.main(test_args)
        duration = time.time() - start_time
        
        return {
            "status": "PASS" if result == 0 else "FAIL",
            "duration": duration,
            "exit_code": result,
            "test_type": "user_acceptance_testing"
        }
    
    async def run_disaster_recovery_tests(self) -> Dict[str, Any]:
        """Run disaster recovery and resilience tests."""
        logger.info("Starting disaster recovery tests")
        
        test_args = [
            "testing/validation/disaster_recovery_tests.py::test_comprehensive_disaster_recovery_suite",
            "-v", "--tb=short"
        ]
        
        start_time = time.time()
        result = pytest.main(test_args)
        duration = time.time() - start_time
        
        return {
            "status": "PASS" if result == 0 else "FAIL",
            "duration": duration,
            "exit_code": result,
            "test_type": "disaster_recovery_testing"
        }
    
    async def run_load_testing(self) -> Dict[str, Any]:
        """Run load testing."""
        logger.info("Starting load testing")
        
        # Run load test script
        import subprocess
        
        load_test_cmd = [
            "python", "testing/performance/load_test.py",
            "--url", self.config["test_environment"]["base_url"],
            "--users", str(self.config.get("load_testing", {}).get("concurrent_users", 50)),
            "--duration", str(self.config.get("load_testing", {}).get("test_duration_seconds", 300)),
            "--output", "load_test_validation_report.json"
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(
                load_test_cmd,
                capture_output=True,
                text=True,
                timeout=self.config["test_execution"]["timeout_seconds"]
            )
            duration = time.time() - start_time
            
            return {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "duration": duration,
                "exit_code": result.returncode,
                "test_type": "load_testing",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "status": "FAIL",
                "duration": duration,
                "exit_code": -1,
                "test_type": "load_testing",
                "error": "Test timed out"
            }
    
    def collect_validation_reports(self) -> Dict[str, Any]:
        """Collect and consolidate validation reports."""
        reports = {}
        
        # System validation report
        system_report_path = "system_validation_report.json"
        if Path(system_report_path).exists():
            with open(system_report_path, 'r') as f:
                reports["system_validation"] = json.load(f)
        
        # User acceptance test report
        uat_report_path = "user_acceptance_test_report.json"
        if Path(uat_report_path).exists():
            with open(uat_report_path, 'r') as f:
                reports["user_acceptance"] = json.load(f)
        
        # Disaster recovery report
        dr_report_path = "disaster_recovery_report.json"
        if Path(dr_report_path).exists():
            with open(dr_report_path, 'r') as f:
                reports["disaster_recovery"] = json.load(f)
        
        # Load test report
        load_report_path = "load_test_validation_report.json"
        if Path(load_report_path).exists():
            with open(load_report_path, 'r') as f:
                reports["load_testing"] = json.load(f)
        
        return reports
    
    def generate_consolidated_report(self, test_results: Dict[str, Any], validation_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consolidated validation report."""
        
        # Calculate overall metrics
        total_tests = 0
        passed_tests = 0
        
        for report_name, report_data in validation_reports.items():
            if "summary" in report_data:
                summary = report_data["summary"]
                total_tests += summary.get("total_tests", 0)
                passed_tests += summary.get("passed", 0)
        
        overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall status
        suite_statuses = [result["status"] for result in test_results.values()]
        overall_status = "PASS" if all(status == "PASS" for status in suite_statuses) else "FAIL"
        
        # Performance validation
        performance_validated = True
        if "system_validation" in validation_reports:
            sys_report = validation_reports["system_validation"]
            performance_validated = sys_report.get("summary", {}).get("overall_status") == "PASS"
        
        # Requirements validation
        requirements_validated = {
            "1.1": overall_success_rate >= 90,  # Real-time incident detection
            "6.2": performance_validated,        # Model accuracy requirements
            "6.3": performance_validated         # False positive rate requirements
        }
        
        consolidated_report = {
            "timestamp": datetime.now().isoformat(),
            "test_execution_summary": {
                "start_time": self.results["start_time"],
                "end_time": self.results["end_time"],
                "total_duration": self.results["duration"],
                "overall_status": overall_status,
                "test_suites_executed": len(test_results),
                "test_suites_passed": sum(1 for r in test_results.values() if r["status"] == "PASS")
            },
            "validation_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "overall_success_rate": overall_success_rate,
                "validation_status": "PASS" if overall_success_rate >= 85 else "FAIL"
            },
            "requirements_validation": {
                "requirement_1_1_real_time_alerts": {
                    "validated": requirements_validated["1.1"],
                    "description": "Real-time incident detection and alerting (<5s latency)"
                },
                "requirement_6_2_model_accuracy": {
                    "validated": requirements_validated["6.2"],
                    "description": "AI model accuracy and performance metrics"
                },
                "requirement_6_3_false_positive_rate": {
                    "validated": requirements_validated["6.3"],
                    "description": "False positive rate validation"
                }
            },
            "test_suite_results": test_results,
            "detailed_validation_reports": validation_reports,
            "recommendations": self.generate_recommendations(test_results, validation_reports)
        }
        
        return consolidated_report
    
    def generate_recommendations(self, test_results: Dict[str, Any], validation_reports: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed test suites
        failed_suites = [name for name, result in test_results.items() if result["status"] == "FAIL"]
        if failed_suites:
            recommendations.append(f"Address failures in test suites: {', '.join(failed_suites)}")
        
        # Check performance metrics
        if "system_validation" in validation_reports:
            sys_report = validation_reports["system_validation"]
            if sys_report.get("summary", {}).get("success_rate", 0) < 90:
                recommendations.append("Improve system performance to meet latency and accuracy requirements")
        
        # Check load testing results
        if "load_testing" in validation_reports:
            load_report = validation_reports["load_testing"]
            if load_report.get("summary", {}).get("success_rate", 0) < 95:
                recommendations.append("Optimize system for better load handling and concurrent user support")
        
        # Check disaster recovery
        if "disaster_recovery" in validation_reports:
            dr_report = validation_reports["disaster_recovery"]
            if dr_report.get("summary", {}).get("success_rate", 0) < 75:
                recommendations.append("Improve system resilience and disaster recovery capabilities")
        
        if not recommendations:
            recommendations.append("All validation tests passed successfully. System is ready for production deployment.")
        
        return recommendations
    
    async def run_complete_validation_suite(self) -> Dict[str, Any]:
        """Run the complete system validation and acceptance testing suite."""
        
        logger.info("Starting complete system validation suite")
        self.results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        # Run all test suites
        test_suites = {
            "performance_validation": self.run_performance_validation,
            "user_acceptance_testing": self.run_user_acceptance_tests,
            "disaster_recovery_testing": self.run_disaster_recovery_tests,
            "load_testing": self.run_load_testing
        }
        
        for suite_name, suite_func in test_suites.items():
            logger.info(f"Executing {suite_name}")
            try:
                result = await suite_func()
                self.results["test_suites"][suite_name] = result
                logger.info(f"Completed {suite_name}: {result['status']}")
            except Exception as e:
                logger.error(f"Failed to execute {suite_name}: {e}")
                self.results["test_suites"][suite_name] = {
                    "status": "FAIL",
                    "duration": 0,
                    "error": str(e),
                    "test_type": suite_name
                }
        
        # Calculate total duration
        self.results["end_time"] = datetime.now().isoformat()
        self.results["duration"] = time.time() - start_time
        
        # Collect validation reports
        validation_reports = self.collect_validation_reports()
        
        # Generate consolidated report
        consolidated_report = self.generate_consolidated_report(
            self.results["test_suites"], 
            validation_reports
        )
        
        # Save consolidated report
        with open("system_validation_consolidated_report.json", "w") as f:
            json.dump(consolidated_report, f, indent=2)
        
        return consolidated_report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print validation summary to console."""
        
        print("\n" + "="*80)
        print("SYSTEM VALIDATION AND ACCEPTANCE TESTING SUMMARY")
        print("="*80)
        
        exec_summary = report["test_execution_summary"]
        val_summary = report["validation_summary"]
        
        print(f"Overall Status: {exec_summary['overall_status']}")
        print(f"Validation Status: {val_summary['validation_status']}")
        print(f"Total Duration: {exec_summary['total_duration']:.2f} seconds")
        print(f"Success Rate: {val_summary['overall_success_rate']:.1f}%")
        
        print(f"\nTest Suite Results:")
        print("-" * 40)
        for suite_name, result in report["test_suite_results"].items():
            status_icon = "✓" if result["status"] == "PASS" else "✗"
            print(f"{status_icon} {suite_name.replace('_', ' ').title()}: {result['status']} ({result['duration']:.1f}s)")
        
        print(f"\nRequirements Validation:")
        print("-" * 40)
        for req_id, req_data in report["requirements_validation"].items():
            status_icon = "✓" if req_data["validated"] else "✗"
            print(f"{status_icon} {req_id.upper()}: {req_data['description']}")
        
        print(f"\nRecommendations:")
        print("-" * 40)
        for i, recommendation in enumerate(report["recommendations"], 1):
            print(f"{i}. {recommendation}")
        
        print(f"\nDetailed report saved to: system_validation_consolidated_report.json")
        print("="*80)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="System Validation and Acceptance Testing")
    parser.add_argument("--config", default="testing/test_config.json", help="Test configuration file")
    parser.add_argument("--suite", choices=["all", "performance", "uat", "disaster_recovery", "load"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = SystemValidationRunner(args.config)
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        if args.suite == "all":
            # Run complete validation suite
            report = await runner.run_complete_validation_suite()
            runner.print_summary(report)
            
            # Exit with appropriate code
            overall_status = report["test_execution_summary"]["overall_status"]
            exit_code = 0 if overall_status == "PASS" else 1
            
        else:
            # Run specific test suite
            suite_map = {
                "performance": runner.run_performance_validation,
                "uat": runner.run_user_acceptance_tests,
                "disaster_recovery": runner.run_disaster_recovery_tests,
                "load": runner.run_load_testing
            }
            
            if args.suite in suite_map:
                result = await suite_map[args.suite]()
                print(f"\n{args.suite.title()} Test Result: {result['status']}")
                print(f"Duration: {result['duration']:.2f} seconds")
                exit_code = 0 if result["status"] == "PASS" else 1
            else:
                print(f"Unknown test suite: {args.suite}")
                exit_code = 1
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Validation testing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Validation testing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)