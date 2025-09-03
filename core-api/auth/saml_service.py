"""
SAML authentication service for SSO integration.
"""
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import base64
import urllib.parse
import hashlib
import hmac

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography import x509

from core.config import settings

logger = structlog.get_logger()


class SAMLService:
    """SAML authentication service."""
    
    def __init__(self):
        self.entity_id = settings.SAML_ENTITY_ID
        self.sso_url = settings.SAML_SSO_URL
        self.x509_cert = settings.SAML_X509_CERT
        self._cert_obj = None
        
        if self.x509_cert:
            try:
                # Parse the X.509 certificate
                cert_data = base64.b64decode(self.x509_cert)
                self._cert_obj = x509.load_der_x509_certificate(cert_data)
            except Exception as e:
                logger.error("Failed to load SAML certificate", error=str(e))
    
    def is_configured(self) -> bool:
        """Check if SAML is properly configured."""
        return all([
            self.entity_id,
            self.sso_url,
            self.x509_cert,
            self._cert_obj
        ])
    
    def generate_auth_request(self, relay_state: Optional[str] = None) -> Dict[str, str]:
        """Generate SAML authentication request."""
        if not self.is_configured():
            raise ValueError("SAML not properly configured")
        
        # Generate request ID and timestamp
        request_id = f"_{hashlib.sha256(str(datetime.utcnow()).encode()).hexdigest()[:32]}"
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Create SAML AuthnRequest XML
        authn_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<samlp:AuthnRequest
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="{request_id}"
    Version="2.0"
    IssueInstant="{timestamp}"
    Destination="{self.sso_url}"
    AssertionConsumerServiceURL="{settings.HOST}:{settings.PORT}/auth/saml/acs"
    ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
    <saml:Issuer>{self.entity_id}</saml:Issuer>
    <samlp:NameIDPolicy
        Format="urn:oasis:names:tc:SAML:2.0:nameid-format:persistent"
        AllowCreate="true"/>
</samlp:AuthnRequest>"""
        
        # Encode the request
        encoded_request = base64.b64encode(authn_request.encode()).decode()
        
        # Create redirect URL
        params = {
            'SAMLRequest': encoded_request
        }
        
        if relay_state:
            params['RelayState'] = relay_state
        
        redirect_url = f"{self.sso_url}?{urllib.parse.urlencode(params)}"
        
        return {
            'request_id': request_id,
            'redirect_url': redirect_url,
            'relay_state': relay_state
        }
    
    def validate_response(self, saml_response: str) -> Dict[str, Any]:
        """Validate SAML response and extract user attributes."""
        if not self.is_configured():
            raise ValueError("SAML not properly configured")
        
        try:
            # Decode the SAML response
            decoded_response = base64.b64decode(saml_response)
            
            # Parse XML
            root = ET.fromstring(decoded_response)
            
            # Define namespaces
            namespaces = {
                'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol',
                'saml': 'urn:oasis:names:tc:SAML:2.0:assertion',
                'ds': 'http://www.w3.org/2000/09/xmldsig#'
            }
            
            # Validate response status
            status = root.find('.//samlp:Status/samlp:StatusCode', namespaces)
            if status is None or status.get('Value') != 'urn:oasis:names:tc:SAML:2.0:status:Success':
                raise ValueError("SAML authentication failed")
            
            # Extract assertion
            assertion = root.find('.//saml:Assertion', namespaces)
            if assertion is None:
                raise ValueError("No assertion found in SAML response")
            
            # Validate assertion conditions (timing, audience)
            conditions = assertion.find('.//saml:Conditions', namespaces)
            if conditions is not None:
                not_before = conditions.get('NotBefore')
                not_on_or_after = conditions.get('NotOnOrAfter')
                
                now = datetime.utcnow()
                
                if not_before:
                    not_before_dt = datetime.fromisoformat(not_before.replace('Z', '+00:00'))
                    if now < not_before_dt.replace(tzinfo=None):
                        raise ValueError("SAML assertion not yet valid")
                
                if not_on_or_after:
                    not_on_or_after_dt = datetime.fromisoformat(not_on_or_after.replace('Z', '+00:00'))
                    if now >= not_on_or_after_dt.replace(tzinfo=None):
                        raise ValueError("SAML assertion expired")
            
            # Extract NameID
            name_id_element = assertion.find('.//saml:Subject/saml:NameID', namespaces)
            name_id = name_id_element.text if name_id_element is not None else None
            
            # Extract attributes
            attributes = {}
            attr_statements = assertion.findall('.//saml:AttributeStatement/saml:Attribute', namespaces)
            
            for attr in attr_statements:
                attr_name = attr.get('Name')
                attr_values = [val.text for val in attr.findall('.//saml:AttributeValue', namespaces)]
                
                if attr_name and attr_values:
                    # Store single value or list based on count
                    attributes[attr_name] = attr_values[0] if len(attr_values) == 1 else attr_values
            
            return {
                'name_id': name_id,
                'attributes': attributes,
                'valid': True
            }
            
        except ET.ParseError as e:
            logger.error("Failed to parse SAML response XML", error=str(e))
            raise ValueError("Invalid SAML response format")
        except Exception as e:
            logger.error("SAML response validation failed", error=str(e))
            raise ValueError(f"SAML validation error: {str(e)}")
    
    def extract_user_info(self, saml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user information from SAML attributes."""
        attributes = saml_data.get('attributes', {})
        
        # Common SAML attribute mappings
        user_info = {
            'saml_name_id': saml_data.get('name_id'),
            'username': None,
            'email': None,
            'full_name': None,
            'role': 'viewer'  # Default role
        }
        
        # Map common attribute names
        attribute_mappings = {
            'username': [
                'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name',
                'uid',
                'sAMAccountName',
                'username'
            ],
            'email': [
                'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress',
                'mail',
                'email',
                'emailAddress'
            ],
            'full_name': [
                'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/displayname',
                'displayName',
                'cn',
                'fullName'
            ],
            'role': [
                'http://schemas.microsoft.com/ws/2008/06/identity/claims/role',
                'role',
                'memberOf',
                'groups'
            ]
        }
        
        # Extract values using mappings
        for field, possible_attrs in attribute_mappings.items():
            for attr_name in possible_attrs:
                if attr_name in attributes:
                    value = attributes[attr_name]
                    if isinstance(value, list):
                        value = value[0] if value else None
                    
                    if field == 'role':
                        # Map role values to our system roles
                        user_info[field] = self._map_saml_role(value)
                    else:
                        user_info[field] = value
                    break
        
        # Generate username from email if not provided
        if not user_info['username'] and user_info['email']:
            user_info['username'] = user_info['email'].split('@')[0]
        
        return user_info
    
    def _map_saml_role(self, saml_role: str) -> str:
        """Map SAML role to system role."""
        if not saml_role:
            return 'viewer'
        
        saml_role = saml_role.lower()
        
        # Role mapping based on common patterns
        role_mappings = {
            'admin': 'admin',
            'administrator': 'admin',
            'security_admin': 'admin',
            'security_supervisor': 'security_supervisor',
            'supervisor': 'security_supervisor',
            'security_guard': 'security_guard',
            'guard': 'security_guard',
            'security_officer': 'security_guard',
            'analyst': 'analyst',
            'security_analyst': 'analyst',
            'viewer': 'viewer',
            'user': 'viewer'
        }
        
        # Check for exact matches first
        if saml_role in role_mappings:
            return role_mappings[saml_role]
        
        # Check for partial matches
        for pattern, role in role_mappings.items():
            if pattern in saml_role:
                return role
        
        # Default to viewer
        return 'viewer'


# Global SAML service instance
saml_service = SAMLService()