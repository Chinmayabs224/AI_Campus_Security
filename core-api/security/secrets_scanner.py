"""
Secrets scanning and detection for campus security system.
"""
import asyncio
import re
import os
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import structlog
import aiofiles
from pathlib import Path

logger = structlog.get_logger()


@dataclass
class SecretPattern:
    """Secret detection pattern."""
    name: str
    pattern: str
    description: str
    severity: str
    confidence: float


@dataclass
class SecretMatch:
    """Detected secret match."""
    pattern_name: str
    file_path: str
    line_number: int
    line_content: str
    matched_text: str
    severity: str
    confidence: float
    context: str = ""


class SecretsScanner:
    """Scanner for detecting secrets in code and configuration files."""
    
    def __init__(self):
        self.secret_patterns: List[SecretPattern] = []
        self.scan_results: Dict[str, List[SecretMatch]] = {}
        self.excluded_paths: Set[str] = set()
        self.excluded_extensions: Set[str] = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe'}
        self.initialize_patterns()
    
    def initialize_patterns(self):
        """Initialize secret detection patterns."""
        patterns = [
            # API Keys
            SecretPattern(
                name="aws_access_key",
                pattern=r'AKIA[0-9A-Z]{16}',
                description="AWS Access Key ID",
                severity="high",
                confidence=0.9
            ),
            SecretPattern(
                name="aws_secret_key",
                pattern=r'[A-Za-z0-9/+=]{40}',
                description="AWS Secret Access Key",
                severity="high",
                confidence=0.7
            ),
            SecretPattern(
                name="github_token",
                pattern=r'ghp_[A-Za-z0-9]{36}',
                description="GitHub Personal Access Token",
                severity="high",
                confidence=0.95
            ),
            SecretPattern(
                name="slack_token",
                pattern=r'xox[baprs]-[A-Za-z0-9-]+',
                description="Slack Token",
                severity="medium",
                confidence=0.9
            ),
            SecretPattern(
                name="stripe_key",
                pattern=r'sk_live_[A-Za-z0-9]{24}',
                description="Stripe Live Secret Key",
                severity="high",
                confidence=0.95
            ),
            SecretPattern(
                name="twilio_sid",
                pattern=r'AC[a-z0-9]{32}',
                description="Twilio Account SID",
                severity="medium",
                confidence=0.8
            ),
            SecretPattern(
                name="mailgun_key",
                pattern=r'key-[a-z0-9]{32}',
                description="Mailgun API Key",
                severity="medium",
                confidence=0.85
            ),
            
            # Database Connections
            SecretPattern(
                name="postgres_url",
                pattern=r'postgres://[^:]+:[^@]+@[^/]+/[^\\s]+',
                description="PostgreSQL Connection String",
                severity="high",
                confidence=0.9
            ),
            SecretPattern(
                name="mysql_url",
                pattern=r'mysql://[^:]+:[^@]+@[^/]+/[^\\s]+',
                description="MySQL Connection String",
                severity="high",
                confidence=0.9
            ),
            SecretPattern(
                name="mongodb_url",
                pattern=r'mongodb://[^:]+:[^@]+@[^/]+/[^\\s]+',
                description="MongoDB Connection String",
                severity="high",
                confidence=0.9
            ),
            
            # Generic Patterns
            SecretPattern(
                name="generic_password",
                pattern=r'(?i)(password|passwd|pwd)\\s*[=:]\\s*["\']?([^\\s"\']+)["\']?',
                description="Generic Password",
                severity="medium",
                confidence=0.6
            ),
            SecretPattern(
                name="generic_secret",
                pattern=r'(?i)(secret|secret_key|secretkey)\\s*[=:]\\s*["\']?([^\\s"\']+)["\']?',
                description="Generic Secret",
                severity="medium",
                confidence=0.6
            ),
            SecretPattern(
                name="generic_token",
                pattern=r'(?i)(token|access_token|auth_token)\\s*[=:]\\s*["\']?([^\\s"\']+)["\']?',
                description="Generic Token",
                severity="medium",
                confidence=0.6
            ),
            SecretPattern(
                name="generic_key",
                pattern=r'(?i)(api_key|apikey|private_key|privatekey)\\s*[=:]\\s*["\']?([^\\s"\']+)["\']?',
                description="Generic API Key",
                severity="medium",
                confidence=0.6
            ),
            
            # Cryptographic Keys
            SecretPattern(
                name="rsa_private_key",
                pattern=r'-----BEGIN RSA PRIVATE KEY-----',
                description="RSA Private Key",
                severity="critical",
                confidence=1.0
            ),
            SecretPattern(
                name="openssh_private_key",
                pattern=r'-----BEGIN OPENSSH PRIVATE KEY-----',
                description="OpenSSH Private Key",
                severity="critical",
                confidence=1.0
            ),
            SecretPattern(
                name="dsa_private_key",
                pattern=r'-----BEGIN DSA PRIVATE KEY-----',
                description="DSA Private Key",
                severity="critical",
                confidence=1.0
            ),
            SecretPattern(
                name="ec_private_key",
                pattern=r'-----BEGIN EC PRIVATE KEY-----',
                description="EC Private Key",
                severity="critical",
                confidence=1.0
            ),
            
            # JWT Tokens
            SecretPattern(
                name="jwt_token",
                pattern=r'eyJ[A-Za-z0-9_-]*\\.[A-Za-z0-9_-]*\\.[A-Za-z0-9_-]*',
                description="JWT Token",
                severity="medium",
                confidence=0.8
            ),
            
            # High Entropy Strings (potential secrets)
            SecretPattern(
                name="high_entropy_string",
                pattern=r'[A-Za-z0-9+/]{32,}={0,2}',
                description="High Entropy String (Base64)",
                severity="low",
                confidence=0.4
            )
        ]
        
        self.secret_patterns = patterns
        logger.info("Secret detection patterns initialized", count=len(patterns))
    
    def add_excluded_path(self, path: str):
        """Add path to exclusion list."""
        self.excluded_paths.add(path)
        logger.info("Path added to exclusion list", path=path)
    
    def add_excluded_extension(self, extension: str):
        """Add file extension to exclusion list."""
        self.excluded_extensions.add(extension)
        logger.info("Extension added to exclusion list", extension=extension)
    
    async def scan_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, List[SecretMatch]]:
        """Scan directory for secrets."""
        logger.info("Starting secrets scan", directory=directory_path, recursive=recursive)
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        scan_results = {}
        files_scanned = 0
        secrets_found = 0
        
        # Get all files to scan
        files_to_scan = []
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not self._is_path_excluded(os.path.join(root, d))]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    if self._should_scan_file(file_path):
                        files_to_scan.append(file_path)
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path) and self._should_scan_file(file_path):
                    files_to_scan.append(file_path)
        
        # Scan files
        for file_path in files_to_scan:
            try:
                matches = await self.scan_file(file_path)
                if matches:
                    scan_results[file_path] = matches
                    secrets_found += len(matches)
                files_scanned += 1
            except Exception as e:
                logger.warning("Failed to scan file", file=file_path, error=str(e))
        
        # Store results
        self.scan_results.update(scan_results)
        
        logger.info("Secrets scan completed",
                   directory=directory_path,
                   files_scanned=files_scanned,
                   secrets_found=secrets_found)
        
        return scan_results
    
    async def scan_file(self, file_path: str) -> List[SecretMatch]:
        """Scan single file for secrets."""
        if not self._should_scan_file(file_path):
            return []
        
        matches = []
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = await f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line_matches = self._scan_line(file_path, line_num, line)
                matches.extend(line_matches)
        
        except Exception as e:
            logger.warning("Failed to read file for scanning", file=file_path, error=str(e))
        
        return matches
    
    def _scan_line(self, file_path: str, line_num: int, line: str) -> List[SecretMatch]:
        """Scan single line for secrets."""
        matches = []
        
        for pattern in self.secret_patterns:
            regex = re.compile(pattern.pattern, re.IGNORECASE | re.MULTILINE)
            
            for match in regex.finditer(line):
                # Skip if it looks like a placeholder or example
                if self._is_likely_placeholder(match.group(0)):
                    continue
                
                # Extract context (surrounding characters)
                start = max(0, match.start() - 10)
                end = min(len(line), match.end() + 10)
                context = line[start:end].strip()
                
                secret_match = SecretMatch(
                    pattern_name=pattern.name,
                    file_path=file_path,
                    line_number=line_num,
                    line_content=line.strip(),
                    matched_text=match.group(0),
                    severity=pattern.severity,
                    confidence=pattern.confidence,
                    context=context
                )
                
                matches.append(secret_match)
        
        return matches
    
    def _should_scan_file(self, file_path: str) -> bool:
        """Check if file should be scanned."""
        # Check if path is excluded
        if self._is_path_excluded(file_path):
            return False
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() in self.excluded_extensions:
            return False
        
        # Check if file is binary
        if self._is_binary_file(file_path):
            return False
        
        # Check file size (skip very large files)
        try:
            if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB
                return False
        except OSError:
            return False
        
        return True
    
    def _is_path_excluded(self, path: str) -> bool:
        """Check if path is in exclusion list."""
        path_parts = Path(path).parts
        
        for excluded_path in self.excluded_paths:
            excluded_parts = Path(excluded_path).parts
            
            # Check if any part of the path matches excluded path
            if any(excluded_part in path_parts for excluded_part in excluded_parts):
                return True
        
        # Common directories to exclude
        excluded_dirs = {
            '.git', '.svn', '.hg', '__pycache__', 'node_modules',
            '.venv', 'venv', '.env', 'build', 'dist', '.pytest_cache'
        }
        
        return any(excluded_dir in path_parts for excluded_dir in excluded_dirs)
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Check if file is binary."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\\0' in chunk
        except Exception:
            return True
    
    def _is_likely_placeholder(self, text: str) -> bool:
        """Check if text is likely a placeholder or example."""
        placeholder_patterns = [
            r'(?i)(example|sample|test|demo|placeholder|your_|my_)',
            r'(?i)(xxx+|yyy+|zzz+)',
            r'(?i)(123+|abc+)',
            r'(?i)(changeme|replace|insert)',
            r'^[x]{8,}$',
            r'^[0]{8,}$',
            r'^[1]{8,}$'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def get_scan_summary(self) -> Dict[str, Any]:
        """Get summary of scan results."""
        total_files = len(self.scan_results)
        total_secrets = sum(len(matches) for matches in self.scan_results.values())
        
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        pattern_counts = {}
        
        for matches in self.scan_results.values():
            for match in matches:
                severity_counts[match.severity] += 1
                pattern_counts[match.pattern_name] = pattern_counts.get(match.pattern_name, 0) + 1
        
        return {
            "total_files_scanned": total_files,
            "total_secrets_found": total_secrets,
            "severity_breakdown": severity_counts,
            "pattern_breakdown": pattern_counts,
            "high_risk_files": [
                file_path for file_path, matches in self.scan_results.items()
                if any(match.severity in ["critical", "high"] for match in matches)
            ]
        }
    
    async def generate_report(self, output_file: str, format: str = "json"):
        """Generate secrets scan report."""
        summary = self.get_scan_summary()
        
        report = {
            "scan_date": datetime.utcnow().isoformat(),
            "summary": summary,
            "detailed_results": {}
        }
        
        # Add detailed results
        for file_path, matches in self.scan_results.items():
            report["detailed_results"][file_path] = [
                {
                    "pattern_name": match.pattern_name,
                    "line_number": match.line_number,
                    "severity": match.severity,
                    "confidence": match.confidence,
                    "context": match.context,
                    "description": next(
                        (p.description for p in self.secret_patterns if p.name == match.pattern_name),
                        "Unknown pattern"
                    )
                }
                for match in matches
            ]
        
        if format.lower() == "json":
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(json.dumps(report, indent=2))
        else:
            # Generate text report
            text_report = self._generate_text_report(report)
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(text_report)
        
        logger.info("Secrets scan report generated", output_file=output_file, format=format)
    
    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Generate text format report."""
        lines = []
        lines.append("SECRETS SCAN REPORT")
        lines.append("=" * 50)
        lines.append(f"Scan Date: {report['scan_date']}")
        lines.append("")
        
        # Summary
        summary = report["summary"]
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Files Scanned: {summary['total_files_scanned']}")
        lines.append(f"Secrets Found: {summary['total_secrets_found']}")
        lines.append("")
        
        # Severity breakdown
        lines.append("SEVERITY BREAKDOWN")
        lines.append("-" * 20)
        for severity, count in summary["severity_breakdown"].items():
            lines.append(f"{severity.upper()}: {count}")
        lines.append("")
        
        # High-risk files
        if summary["high_risk_files"]:
            lines.append("HIGH-RISK FILES")
            lines.append("-" * 20)
            for file_path in summary["high_risk_files"]:
                lines.append(f"- {file_path}")
            lines.append("")
        
        # Detailed results
        lines.append("DETAILED RESULTS")
        lines.append("-" * 20)
        for file_path, matches in report["detailed_results"].items():
            lines.append(f"\\nFile: {file_path}")
            for match in matches:
                lines.append(f"  Line {match['line_number']}: {match['description']} "
                           f"(Severity: {match['severity']}, Confidence: {match['confidence']})")
                lines.append(f"    Context: {match['context']}")
        
        return "\\n".join(lines)
    
    def clear_results(self):
        """Clear scan results."""
        self.scan_results.clear()
        logger.info("Scan results cleared")


# Global secrets scanner instance
secrets_scanner = SecretsScanner()