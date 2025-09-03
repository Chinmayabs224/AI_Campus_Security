"""
DSAR (Data Subject Access Request) processor for GDPR compliance.
"""
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis

from models import DSARRequest, DSARStatus, DSARRequestType
from config import Config

logger = logging.getLogger(__name__)


class DSARProcessor:
    """Processor for Data Subject Access Requests."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize DSAR processor."""
        self.config = config or Config()
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            db=self.config.REDIS_DB,
            decode_responses=True
        )
    
    def process_request(self, dsar_request: DSARRequest) -> Dict[str, Any]:
        """
        Process a new DSAR request.
        
        Args:
            dsar_request: DSAR request to process
            
        Returns:
            Processing result
        """
        try:
            # Generate request ID if not provided
            if not dsar_request.request_id:
                dsar_request.request_id = str(uuid.uuid4())
            
            # Set initial status
            dsar_request.status = DSARStatus.PENDING
            dsar_request.created_at = datetime.utcnow()
            dsar_request.updated_at = datetime.utcnow()
            
            # Store request in Redis
            request_key = f"dsar_request:{dsar_request.request_id}"
            request_data = dsar_request.to_dict()
            
            # Set expiration based on retention policy
            expiration = timedelta(days=self.config.DSAR_RETENTION_DAYS)
            
            self.redis_client.setex(
                request_key,
                int(expiration.total_seconds()),
                json.dumps(request_data)
            )
            
            # Add to processing queue
            self._add_to_processing_queue(dsar_request)
            
            # Calculate estimated completion time
            estimated_completion = datetime.utcnow() + timedelta(days=30)  # GDPR requirement
            
            logger.info(f"DSAR request created: {dsar_request.request_id}")
            
            return {
                'request_id': dsar_request.request_id,
                'status': dsar_request.status.value,
                'estimated_completion': estimated_completion.isoformat(),
                'created_at': dsar_request.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process DSAR request: {str(e)}")
            raise
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get DSAR request status.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Request status information
        """
        try:
            request_key = f"dsar_request:{request_id}"
            request_data = self.redis_client.get(request_key)
            
            if not request_data:
                return None
            
            request_dict = json.loads(request_data)
            
            # Add processing progress if available
            progress_key = f"dsar_progress:{request_id}"
            progress_data = self.redis_client.get(progress_key)
            
            if progress_data:
                progress = json.loads(progress_data)
                request_dict['progress'] = progress
            
            return request_dict
            
        except Exception as e:
            logger.error(f"Failed to get DSAR status: {str(e)}")
            return None
    
    def update_request_status(
        self,
        request_id: str,
        status: DSARStatus,
        progress: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update DSAR request status.
        
        Args:
            request_id: Request identifier
            status: New status
            progress: Optional progress information
            
        Returns:
            True if successful
        """
        try:
            request_key = f"dsar_request:{request_id}"
            request_data = self.redis_client.get(request_key)
            
            if not request_data:
                return False
            
            request_dict = json.loads(request_data)
            request_dict['status'] = status.value
            request_dict['updated_at'] = datetime.utcnow().isoformat()
            
            if status in [DSARStatus.COMPLETED, DSARStatus.FAILED, DSARStatus.CANCELLED]:
                request_dict['completed_at'] = datetime.utcnow().isoformat()
            
            # Update request
            self.redis_client.setex(
                request_key,
                self.config.DSAR_RETENTION_DAYS * 24 * 3600,
                json.dumps(request_dict)
            )
            
            # Update progress if provided
            if progress:
                progress_key = f"dsar_progress:{request_id}"
                self.redis_client.setex(
                    progress_key,
                    self.config.DSAR_RETENTION_DAYS * 24 * 3600,
                    json.dumps(progress)
                )
            
            logger.info(f"DSAR request status updated: {request_id} -> {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update DSAR status: {str(e)}")
            return False
    
    def _add_to_processing_queue(self, dsar_request: DSARRequest) -> None:
        """Add DSAR request to processing queue."""
        try:
            queue_item = {
                'request_id': dsar_request.request_id,
                'request_type': dsar_request.request_type.value,
                'priority': self._calculate_priority(dsar_request),
                'created_at': dsar_request.created_at.isoformat()
            }
            
            # Add to Redis queue (using list)
            self.redis_client.lpush('dsar_processing_queue', json.dumps(queue_item))
            
            logger.info(f"Added DSAR request to processing queue: {dsar_request.request_id}")
            
        except Exception as e:
            logger.error(f"Failed to add DSAR to queue: {str(e)}")
    
    def _calculate_priority(self, dsar_request: DSARRequest) -> int:
        """
        Calculate processing priority for DSAR request.
        
        Args:
            dsar_request: DSAR request
            
        Returns:
            Priority score (higher = more urgent)
        """
        priority = 0
        
        # Erasure requests have highest priority
        if dsar_request.request_type == DSARRequestType.ERASURE:
            priority += 100
        
        # Access requests have medium priority
        elif dsar_request.request_type == DSARRequestType.ACCESS:
            priority += 50
        
        # Other requests have lower priority
        else:
            priority += 25
        
        # Increase priority based on age
        if dsar_request.created_at:
            age_days = (datetime.utcnow() - dsar_request.created_at).days
            priority += age_days * 5
        
        return priority
    
    def get_pending_requests(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pending DSAR requests for processing.
        
        Args:
            limit: Maximum number of requests to return
            
        Returns:
            List of pending requests
        """
        try:
            pending_requests = []
            
            # Get items from processing queue
            for _ in range(limit):
                queue_item = self.redis_client.rpop('dsar_processing_queue')
                if not queue_item:
                    break
                
                item_data = json.loads(queue_item)
                request_id = item_data['request_id']
                
                # Get full request data
                request_status = self.get_request_status(request_id)
                if request_status and request_status['status'] == DSARStatus.PENDING.value:
                    pending_requests.append(request_status)
                    
                    # Update status to processing
                    self.update_request_status(request_id, DSARStatus.PROCESSING)
            
            return pending_requests
            
        except Exception as e:
            logger.error(f"Failed to get pending requests: {str(e)}")
            return []
    
    def process_access_request(self, request_id: str) -> Dict[str, Any]:
        """
        Process a data access request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Processing result
        """
        try:
            # Get request details
            request_data = self.get_request_status(request_id)
            if not request_data:
                return {'error': 'Request not found'}
            
            dsar_request = DSARRequest.from_dict(request_data)
            
            # Simulate data collection process
            collected_data = {
                'personal_data': {
                    'name': dsar_request.subject_name,
                    'email': dsar_request.subject_email,
                    'subject_id': dsar_request.subject_id
                },
                'evidence_records': [],
                'access_logs': [],
                'processing_activities': []
            }
            
            # Update progress
            progress = {
                'stage': 'data_collection',
                'completion_percentage': 50,
                'message': 'Collecting personal data records'
            }
            
            self.update_request_status(request_id, DSARStatus.PROCESSING, progress)
            
            # Simulate completion
            progress['stage'] = 'completed'
            progress['completion_percentage'] = 100
            progress['message'] = 'Data collection completed'
            progress['data_package'] = collected_data
            
            self.update_request_status(request_id, DSARStatus.COMPLETED, progress)
            
            return {
                'success': True,
                'request_id': request_id,
                'data_package': collected_data
            }
            
        except Exception as e:
            logger.error(f"Failed to process access request: {str(e)}")
            self.update_request_status(request_id, DSARStatus.FAILED)
            return {'error': str(e)}
    
    def process_erasure_request(self, request_id: str) -> Dict[str, Any]:
        """
        Process a data erasure request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Processing result
        """
        try:
            # Get request details
            request_data = self.get_request_status(request_id)
            if not request_data:
                return {'error': 'Request not found'}
            
            dsar_request = DSARRequest.from_dict(request_data)
            
            # Simulate erasure process
            erasure_results = {
                'evidence_records_deleted': 0,
                'access_logs_anonymized': 0,
                'personal_data_removed': 0
            }
            
            # Update progress
            progress = {
                'stage': 'data_erasure',
                'completion_percentage': 50,
                'message': 'Erasing personal data records'
            }
            
            self.update_request_status(request_id, DSARStatus.PROCESSING, progress)
            
            # Simulate completion
            progress['stage'] = 'completed'
            progress['completion_percentage'] = 100
            progress['message'] = 'Data erasure completed'
            progress['erasure_results'] = erasure_results
            
            self.update_request_status(request_id, DSARStatus.COMPLETED, progress)
            
            return {
                'success': True,
                'request_id': request_id,
                'erasure_results': erasure_results
            }
            
        except Exception as e:
            logger.error(f"Failed to process erasure request: {str(e)}")
            self.update_request_status(request_id, DSARStatus.FAILED)
            return {'error': str(e)}
    
    def list_requests(
        self,
        status: Optional[DSARStatus] = None,
        request_type: Optional[DSARRequestType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List DSAR requests with optional filtering.
        
        Args:
            status: Filter by status
            request_type: Filter by request type
            limit: Maximum number of requests to return
            
        Returns:
            List of DSAR requests
        """
        try:
            # Get all DSAR request keys
            request_keys = self.redis_client.keys("dsar_request:*")
            requests = []
            
            for key in request_keys[:limit]:
                request_data = self.redis_client.get(key)
                if request_data:
                    request_dict = json.loads(request_data)
                    
                    # Apply filters
                    if status and request_dict.get('status') != status.value:
                        continue
                    
                    if request_type and request_dict.get('request_type') != request_type.value:
                        continue
                    
                    requests.append(request_dict)
            
            # Sort by creation date (newest first)
            requests.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return requests
            
        except Exception as e:
            logger.error(f"Failed to list DSAR requests: {str(e)}")
            return []