"""
HashiCorp Vault integration for secrets management in campus security system.
"""
import asyncio
import json
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import structlog
import aiohttp
import base64
from cryptography.fernet import Fernet
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class VaultConfig:
    """Vault configuration."""
    url: str
    token: Optional[str] = None
    role_id: Optional[str] = None
    secret_id: Optional[str] = None
    mount_point: str = "secret"
    namespace: Optional[str] = None
    ca_cert_path: Optional[str] = None
    verify_ssl: bool = True


class VaultClient:
    """HashiCorp Vault client for secrets management."""
    
    def __init__(self, config: VaultConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.token: Optional[str] = config.token
        self.token_expires_at: Optional[datetime] = None
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self):
        """Initialize Vault client."""
        connector = aiohttp.TCPConnector(
            verify_ssl=self.config.verify_ssl
        )
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.config.namespace:
            headers["X-Vault-Namespace"] = self.config.namespace
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        # Authenticate if no token provided
        if not self.token and self.config.role_id and self.config.secret_id:
            await self._authenticate_approle()
        
        logger.info("Vault client initialized", url=self.config.url)
    
    async def close(self):
        """Close Vault client."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Vault client closed")
    
    async def _authenticate_approle(self):
        """Authenticate using AppRole method."""
        auth_data = {
            "role_id": self.config.role_id,
            "secret_id": self.config.secret_id
        }
        
        url = f"{self.config.url}/v1/auth/approle/login"
        
        try:
            async with self.session.post(url, json=auth_data) as response:
                if response.status == 200:
                    auth_response = await response.json()
                    auth_info = auth_response.get("auth", {})
                    
                    self.token = auth_info.get("client_token")
                    lease_duration = auth_info.get("lease_duration", 3600)
                    self.token_expires_at = datetime.utcnow() + timedelta(seconds=lease_duration)
                    
                    # Update session headers with token
                    self.session.headers.update({"X-Vault-Token": self.token})
                    
                    logger.info("Successfully authenticated with Vault using AppRole")
                else:
                    error_text = await response.text()
                    raise Exception(f"AppRole authentication failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error("Vault AppRole authentication failed", error=str(e))
            raise
    
    async def _ensure_authenticated(self):
        """Ensure we have a valid token."""
        if not self.token:
            raise Exception("No Vault token available")
        
        # Check if token is expired
        if self.token_expires_at and datetime.utcnow() >= self.token_expires_at:
            if self.config.role_id and self.config.secret_id:
                await self._authenticate_approle()
            else:
                raise Exception("Vault token expired and no AppRole credentials available")
    
    async def write_secret(self, path: str, data: Dict[str, Any], encrypt_values: bool = True) -> bool:
        """Write secret to Vault."""
        await self._ensure_authenticated()
        
        # Encrypt sensitive values if requested
        if encrypt_values:
            data = self._encrypt_secret_values(data)
        
        secret_data = {
            "data": data,
            "metadata": {
                "created_by": "campus-security-system",
                "created_at": datetime.utcnow().isoformat(),
                "encrypted": encrypt_values
            }
        }
        
        url = f"{self.config.url}/v1/{self.config.mount_point}/data/{path}"
        
        try:
            async with self.session.post(url, json=secret_data) as response:
                if response.status in [200, 204]:
                    logger.info("Secret written to Vault", path=path)
                    return True
                else:
                    error_text = await response.text()
                    logger.error("Failed to write secret to Vault", 
                               path=path, status=response.status, error=error_text)
                    return False
                    
        except Exception as e:
            logger.error("Error writing secret to Vault", path=path, error=str(e))
            return False
    
    async def read_secret(self, path: str, decrypt_values: bool = True) -> Optional[Dict[str, Any]]:
        """Read secret from Vault."""
        await self._ensure_authenticated()
        
        url = f"{self.config.url}/v1/{self.config.mount_point}/data/{path}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    secret_response = await response.json()
                    secret_data = secret_response.get("data", {}).get("data", {})
                    metadata = secret_response.get("data", {}).get("metadata", {})
                    
                    # Decrypt values if they were encrypted
                    if decrypt_values and metadata.get("encrypted", False):
                        secret_data = self._decrypt_secret_values(secret_data)
                    
                    logger.info("Secret read from Vault", path=path)
                    return secret_data
                elif response.status == 404:
                    logger.warning("Secret not found in Vault", path=path)
                    return None
                else:
                    error_text = await response.text()
                    logger.error("Failed to read secret from Vault",
                               path=path, status=response.status, error=error_text)
                    return None
                    
        except Exception as e:
            logger.error("Error reading secret from Vault", path=path, error=str(e))
            return None
    
    async def delete_secret(self, path: str) -> bool:
        """Delete secret from Vault."""
        await self._ensure_authenticated()
        
        url = f"{self.config.url}/v1/{self.config.mount_point}/data/{path}"
        
        try:
            async with self.session.delete(url) as response:
                if response.status in [200, 204]:
                    logger.info("Secret deleted from Vault", path=path)
                    return True
                else:
                    error_text = await response.text()
                    logger.error("Failed to delete secret from Vault",
                               path=path, status=response.status, error=error_text)
                    return False
                    
        except Exception as e:
            logger.error("Error deleting secret from Vault", path=path, error=str(e))
            return False
    
    async def list_secrets(self, path: str = "") -> List[str]:
        """List secrets at path."""
        await self._ensure_authenticated()
        
        url = f"{self.config.url}/v1/{self.config.mount_point}/metadata/{path}"
        
        try:
            async with self.session.request("LIST", url) as response:
                if response.status == 200:
                    list_response = await response.json()
                    keys = list_response.get("data", {}).get("keys", [])
                    logger.info("Listed secrets from Vault", path=path, count=len(keys))
                    return keys
                else:
                    error_text = await response.text()
                    logger.error("Failed to list secrets from Vault",
                               path=path, status=response.status, error=error_text)
                    return []
                    
        except Exception as e:
            logger.error("Error listing secrets from Vault", path=path, error=str(e))
            return []
    
    def _encrypt_secret_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive values in secret data."""
        encrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str) and self._is_sensitive_key(key):
                # Encrypt sensitive string values
                encrypted_value = self.cipher_suite.encrypt(value.encode()).decode()
                encrypted_data[key] = f"encrypted:{encrypted_value}"
            elif isinstance(value, dict):
                # Recursively encrypt nested dictionaries
                encrypted_data[key] = self._encrypt_secret_values(value)
            else:
                # Keep non-sensitive values as-is
                encrypted_data[key] = value
        
        return encrypted_data
    
    def _decrypt_secret_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted values in secret data."""
        decrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str) and value.startswith("encrypted:"):
                # Decrypt encrypted string values
                encrypted_value = value[10:]  # Remove "encrypted:" prefix
                try:
                    decrypted_value = self.cipher_suite.decrypt(encrypted_value.encode()).decode()
                    decrypted_data[key] = decrypted_value
                except Exception as e:
                    logger.error("Failed to decrypt value", key=key, error=str(e))
                    decrypted_data[key] = value  # Keep encrypted if decryption fails
            elif isinstance(value, dict):
                # Recursively decrypt nested dictionaries
                decrypted_data[key] = self._decrypt_secret_values(value)
            else:
                # Keep non-encrypted values as-is
                decrypted_data[key] = value
        
        return decrypted_data
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key contains sensitive data that should be encrypted."""
        sensitive_keywords = [
            "password", "secret", "key", "token", "credential",
            "private", "cert", "certificate", "passphrase"
        ]
        
        key_lower = key.lower()
        return any(keyword in key_lower for keyword in sensitive_keywords)
    
    async def create_database_credentials(self, role_name: str, ttl: str = "1h") -> Optional[Dict[str, str]]:
        """Create dynamic database credentials."""
        await self._ensure_authenticated()
        
        url = f"{self.config.url}/v1/database/creds/{role_name}"
        params = {"ttl": ttl}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    creds_response = await response.json()
                    credentials = creds_response.get("data", {})
                    
                    logger.info("Dynamic database credentials created", 
                              role=role_name, ttl=ttl)
                    return credentials
                else:
                    error_text = await response.text()
                    logger.error("Failed to create database credentials",
                               role=role_name, status=response.status, error=error_text)
                    return None
                    
        except Exception as e:
            logger.error("Error creating database credentials", role=role_name, error=str(e))
            return None
    
    async def renew_token(self, increment: str = "1h") -> bool:
        """Renew the current token."""
        await self._ensure_authenticated()
        
        url = f"{self.config.url}/v1/auth/token/renew-self"
        data = {"increment": increment}
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    renew_response = await response.json()
                    auth_info = renew_response.get("auth", {})
                    
                    lease_duration = auth_info.get("lease_duration", 3600)
                    self.token_expires_at = datetime.utcnow() + timedelta(seconds=lease_duration)
                    
                    logger.info("Token renewed successfully", increment=increment)
                    return True
                else:
                    error_text = await response.text()
                    logger.error("Failed to renew token",
                               status=response.status, error=error_text)
                    return False
                    
        except Exception as e:
            logger.error("Error renewing token", error=str(e))
            return False


class SecretsManager:
    """High-level secrets management interface."""
    
    def __init__(self, vault_config: VaultConfig):
        self.vault_config = vault_config
        self.vault_client: Optional[VaultClient] = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
    
    async def initialize(self):
        """Initialize secrets manager."""
        self.vault_client = VaultClient(self.vault_config)
        await self.vault_client.initialize()
        logger.info("Secrets manager initialized")
    
    async def close(self):
        """Close secrets manager."""
        if self.vault_client:
            await self.vault_client.close()
        logger.info("Secrets manager closed")
    
    async def get_secret(self, path: str, use_cache: bool = True, cache_ttl_minutes: int = 5) -> Optional[Dict[str, Any]]:
        """Get secret with optional caching."""
        # Check cache first
        if use_cache and path in self.local_cache:
            if path in self.cache_ttl and datetime.utcnow() < self.cache_ttl[path]:
                logger.debug("Secret retrieved from cache", path=path)
                return self.local_cache[path]
        
        # Retrieve from Vault
        if not self.vault_client:
            raise Exception("Secrets manager not initialized")
        
        secret = await self.vault_client.read_secret(path)
        
        # Cache the secret
        if use_cache and secret:
            self.local_cache[path] = secret
            self.cache_ttl[path] = datetime.utcnow() + timedelta(minutes=cache_ttl_minutes)
        
        return secret
    
    async def set_secret(self, path: str, data: Dict[str, Any], invalidate_cache: bool = True) -> bool:
        """Set secret and optionally invalidate cache."""
        if not self.vault_client:
            raise Exception("Secrets manager not initialized")
        
        success = await self.vault_client.write_secret(path, data)
        
        # Invalidate cache
        if invalidate_cache and path in self.local_cache:
            del self.local_cache[path]
            if path in self.cache_ttl:
                del self.cache_ttl[path]
        
        return success
    
    async def setup_application_secrets(self):
        """Set up default application secrets."""
        secrets_to_create = {
            "database/credentials": {
                "username": "campus_security_user",
                "password": "generated_secure_password_123!",
                "host": "localhost",
                "port": "5432",
                "database": "campus_security"
            },
            "redis/credentials": {
                "host": "localhost",
                "port": "6379",
                "password": "redis_secure_password_456!"
            },
            "minio/credentials": {
                "access_key": "minio_access_key",
                "secret_key": "minio_secret_key_789!",
                "endpoint": "localhost:9000"
            },
            "jwt/secrets": {
                "secret_key": "jwt_secret_key_very_secure_012!",
                "algorithm": "HS256"
            },
            "notification/api_keys": {
                "twilio_account_sid": "twilio_account_sid",
                "twilio_auth_token": "twilio_auth_token_345!",
                "firebase_server_key": "firebase_server_key_678!"
            }
        }
        
        for path, data in secrets_to_create.items():
            success = await self.set_secret(path, data)
            if success:
                logger.info("Application secret created", path=path)
            else:
                logger.error("Failed to create application secret", path=path)


# Global secrets manager instance (will be initialized in main.py)
secrets_manager: Optional[SecretsManager] = None