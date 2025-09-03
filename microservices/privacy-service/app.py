"""
Privacy and Redaction Service
Flask microservice for automatic face detection, blurring, and privacy zone enforcement.
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json

from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import redis
from werkzeug.utils import secure_filename
import tempfile
import io
import base64

from config import Config
from models import PrivacyZone, RedactionRequest, DSARRequest
from face_detector import FaceDetector
from video_processor import VideoProcessor
from dsar_processor import DSARProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize Redis connection
redis_client = redis.Redis(
    host=app.config['REDIS_HOST'],
    port=app.config['REDIS_PORT'],
    db=app.config['REDIS_DB'],
    decode_responses=True
)

# Initialize AI models
face_detector = FaceDetector()
video_processor = VideoProcessor(face_detector)
dsar_processor = DSARProcessor()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connection
        redis_client.ping()
        
        # Test AI models
        face_detector_status = face_detector.is_ready()
        
        return jsonify({
            'status': 'healthy',
            'service': 'privacy-service',
            'redis': 'connected',
            'face_detector': 'ready' if face_detector_status else 'not_ready',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503


@app.route('/redact/image', methods=['POST'])
def redact_image():
    """
    Redact faces in an image.
    
    Expected form data:
    - file: Image file
    - privacy_zones: JSON string of privacy zones (optional)
    - blur_strength: Blur strength (1-100, default 50)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Get optional parameters
        privacy_zones_json = request.form.get('privacy_zones', '[]')
        blur_strength = int(request.form.get('blur_strength', 50))
        
        try:
            privacy_zones_data = json.loads(privacy_zones_json)
            privacy_zones = [PrivacyZone(**zone) for zone in privacy_zones_data]
        except (json.JSONDecodeError, TypeError) as e:
            return jsonify({'error': f'Invalid privacy_zones format: {str(e)}'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        # Apply face blurring
        redacted_image = face_detector.blur_faces(image, faces, blur_strength)
        
        # Apply privacy zones
        if privacy_zones:
            redacted_image = face_detector.apply_privacy_zones(redacted_image, privacy_zones)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', redacted_image)
        result_bytes = buffer.tobytes()
        
        # Create response
        result_b64 = base64.b64encode(result_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'privacy_zones_applied': len(privacy_zones),
            'redacted_image': result_b64,
            'content_type': 'image/jpeg'
        }), 200
        
    except Exception as e:
        logger.error(f"Image redaction failed: {str(e)}")
        return jsonify({'error': f'Redaction failed: {str(e)}'}), 500


@app.route('/redact/video', methods=['POST'])
def redact_video():
    """
    Redact faces in a video.
    
    Expected form data:
    - file: Video file
    - privacy_zones: JSON string of privacy zones (optional)
    - blur_strength: Blur strength (1-100, default 50)
    - frame_skip: Process every Nth frame (default 1)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Get optional parameters
        privacy_zones_json = request.form.get('privacy_zones', '[]')
        blur_strength = int(request.form.get('blur_strength', 50))
        frame_skip = int(request.form.get('frame_skip', 1))
        
        try:
            privacy_zones_data = json.loads(privacy_zones_json)
            privacy_zones = [PrivacyZone(**zone) for zone in privacy_zones_data]
        except (json.JSONDecodeError, TypeError) as e:
            return jsonify({'error': f'Invalid privacy_zones format: {str(e)}'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
            file.save(temp_input.name)
            input_path = temp_input.name
        
        # Process video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
            output_path = temp_output.name
        
        try:
            result = video_processor.process_video(
                input_path=input_path,
                output_path=output_path,
                privacy_zones=privacy_zones,
                blur_strength=blur_strength,
                frame_skip=frame_skip
            )
            
            # Read processed video
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            
            # Encode result
            result_b64 = base64.b64encode(video_bytes).decode('utf-8')
            
            return jsonify({
                'success': True,
                'frames_processed': result['frames_processed'],
                'faces_detected': result['total_faces'],
                'privacy_zones_applied': len(privacy_zones),
                'processing_time': result['processing_time'],
                'redacted_video': result_b64,
                'content_type': 'video/mp4'
            }), 200
            
        finally:
            # Cleanup temporary files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except OSError:
                pass
        
    except Exception as e:
        logger.error(f"Video redaction failed: {str(e)}")
        return jsonify({'error': f'Redaction failed: {str(e)}'}), 500


@app.route('/privacy-zones', methods=['POST'])
def create_privacy_zone():
    """Create a new privacy zone configuration."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate privacy zone data
        try:
            privacy_zone = PrivacyZone(**data)
        except TypeError as e:
            return jsonify({'error': f'Invalid privacy zone data: {str(e)}'}), 400
        
        # Store in Redis with expiration
        zone_key = f"privacy_zone:{privacy_zone.zone_id}"
        zone_data = privacy_zone.to_dict()
        
        redis_client.setex(
            zone_key,
            app.config['PRIVACY_ZONE_TTL'],
            json.dumps(zone_data)
        )
        
        logger.info(f"Privacy zone created: {privacy_zone.zone_id}")
        
        return jsonify({
            'success': True,
            'zone_id': privacy_zone.zone_id,
            'message': 'Privacy zone created successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Privacy zone creation failed: {str(e)}")
        return jsonify({'error': f'Failed to create privacy zone: {str(e)}'}), 500


@app.route('/privacy-zones/<zone_id>', methods=['GET'])
def get_privacy_zone(zone_id: str):
    """Get privacy zone configuration by ID."""
    try:
        zone_key = f"privacy_zone:{zone_id}"
        zone_data = redis_client.get(zone_key)
        
        if not zone_data:
            return jsonify({'error': 'Privacy zone not found'}), 404
        
        zone_dict = json.loads(zone_data)
        
        return jsonify({
            'success': True,
            'privacy_zone': zone_dict
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get privacy zone: {str(e)}")
        return jsonify({'error': f'Failed to get privacy zone: {str(e)}'}), 500


@app.route('/privacy-zones', methods=['GET'])
def list_privacy_zones():
    """List all privacy zones."""
    try:
        # Get all privacy zone keys
        zone_keys = redis_client.keys("privacy_zone:*")
        zones = []
        
        for key in zone_keys:
            zone_data = redis_client.get(key)
            if zone_data:
                zones.append(json.loads(zone_data))
        
        return jsonify({
            'success': True,
            'privacy_zones': zones,
            'count': len(zones)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list privacy zones: {str(e)}")
        return jsonify({'error': f'Failed to list privacy zones: {str(e)}'}), 500


@app.route('/privacy-zones/<zone_id>', methods=['DELETE'])
def delete_privacy_zone(zone_id: str):
    """Delete privacy zone configuration."""
    try:
        zone_key = f"privacy_zone:{zone_id}"
        deleted = redis_client.delete(zone_key)
        
        if deleted:
            logger.info(f"Privacy zone deleted: {zone_id}")
            return jsonify({
                'success': True,
                'message': 'Privacy zone deleted successfully'
            }), 200
        else:
            return jsonify({'error': 'Privacy zone not found'}), 404
        
    except Exception as e:
        logger.error(f"Failed to delete privacy zone: {str(e)}")
        return jsonify({'error': f'Failed to delete privacy zone: {str(e)}'}), 500


@app.route('/dsar/request', methods=['POST'])
def create_dsar_request():
    """Create a Data Subject Access Request (DSAR)."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate DSAR request data
        try:
            dsar_request = DSARRequest(**data)
        except TypeError as e:
            return jsonify({'error': f'Invalid DSAR request data: {str(e)}'}), 400
        
        # Process DSAR request
        result = dsar_processor.process_request(dsar_request)
        
        return jsonify({
            'success': True,
            'request_id': result['request_id'],
            'status': result['status'],
            'estimated_completion': result['estimated_completion'],
            'message': 'DSAR request created successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"DSAR request creation failed: {str(e)}")
        return jsonify({'error': f'Failed to create DSAR request: {str(e)}'}), 500


@app.route('/dsar/request/<request_id>', methods=['GET'])
def get_dsar_status(request_id: str):
    """Get DSAR request status."""
    try:
        status = dsar_processor.get_request_status(request_id)
        
        if not status:
            return jsonify({'error': 'DSAR request not found'}), 404
        
        return jsonify({
            'success': True,
            'request_status': status
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get DSAR status: {str(e)}")
        return jsonify({'error': f'Failed to get DSAR status: {str(e)}'}), 500


@app.route('/batch/redact', methods=['POST'])
def batch_redact():
    """
    Batch redaction endpoint for multiple files.
    
    Expected JSON data:
    - files: List of file URLs or base64 encoded files
    - privacy_zones: List of privacy zones
    - blur_strength: Blur strength (1-100, default 50)
    """
    try:
        data = request.get_json()
        
        if not data or 'files' not in data:
            return jsonify({'error': 'No files provided'}), 400
        
        files = data['files']
        privacy_zones_data = data.get('privacy_zones', [])
        blur_strength = data.get('blur_strength', 50)
        
        # Parse privacy zones
        privacy_zones = [PrivacyZone(**zone) for zone in privacy_zones_data]
        
        results = []
        
        for i, file_data in enumerate(files):
            try:
                # Process each file
                if file_data.get('type') == 'image':
                    result = face_detector.process_base64_image(
                        file_data['data'],
                        privacy_zones,
                        blur_strength
                    )
                elif file_data.get('type') == 'video':
                    result = video_processor.process_base64_video(
                        file_data['data'],
                        privacy_zones,
                        blur_strength
                    )
                else:
                    result = {'error': 'Unsupported file type'}
                
                results.append({
                    'file_index': i,
                    'filename': file_data.get('filename', f'file_{i}'),
                    'result': result
                })
                
            except Exception as e:
                results.append({
                    'file_index': i,
                    'filename': file_data.get('filename', f'file_{i}'),
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'processed_count': len([r for r in results if 'error' not in r]),
            'error_count': len([r for r in results if 'error' in r])
        }), 200
        
    except Exception as e:
        logger.error(f"Batch redaction failed: {str(e)}")
        return jsonify({'error': f'Batch redaction failed: {str(e)}'}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )