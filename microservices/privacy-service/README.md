# Privacy and Redaction Service

Flask microservice for automatic face detection, blurring, and privacy zone enforcement in the AI-Powered Campus Security system.

## Features

- **Automatic Face Detection**: Uses FaceNet (MTCNN) for accurate face detection
- **Multiple Redaction Types**: Blur, pixelation, black boxes, and custom masks
- **Privacy Zones**: Configurable zones for automatic redaction
- **Video Processing**: Frame-by-frame processing for video files
- **DSAR Compliance**: Data Subject Access Request processing for GDPR compliance
- **Batch Processing**: Handle multiple files simultaneously
- **Real-time Processing**: Fast processing for live camera feeds

## API Endpoints

### Health Check
```
GET /health
```

### Image Redaction
```
POST /redact/image
Content-Type: multipart/form-data

Parameters:
- file: Image file
- privacy_zones: JSON string of privacy zones (optional)
- blur_strength: Blur strength 1-100 (default: 50)
```

### Video Redaction
```
POST /redact/video
Content-Type: multipart/form-data

Parameters:
- file: Video file
- privacy_zones: JSON string of privacy zones (optional)
- blur_strength: Blur strength 1-100 (default: 50)
- frame_skip: Process every Nth frame (default: 1)
```

### Privacy Zone Management
```
POST /privacy-zones          # Create privacy zone
GET /privacy-zones           # List all privacy zones
GET /privacy-zones/{id}      # Get specific privacy zone
DELETE /privacy-zones/{id}   # Delete privacy zone
```

### DSAR (Data Subject Access Requests)
```
POST /dsar/request           # Create DSAR request
GET /dsar/request/{id}       # Get DSAR status
```

### Batch Processing
```
POST /batch/redact           # Process multiple files
```

## Configuration

Environment variables:

- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 1)
- `FACE_DETECTION_CONFIDENCE`: Face detection confidence threshold (default: 0.7)
- `FACE_DETECTION_DEVICE`: Processing device 'cpu' or 'cuda' (default: cpu)
- `MAX_CONTENT_LENGTH`: Maximum file size in bytes (default: 500MB)
- `PRIVACY_ZONE_TTL`: Privacy zone cache TTL in seconds (default: 86400)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export FACE_DETECTION_DEVICE=cpu
```

3. Run the service:
```bash
python app.py
```

Or using Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## Docker Deployment

Build and run with Docker:

```bash
docker build -t privacy-service .
docker run -p 5000:5000 -e REDIS_HOST=redis privacy-service
```

## Privacy Zone Format

Privacy zones are defined as polygons with the following structure:

```json
{
  "zone_id": "zone_1",
  "name": "Sensitive Area",
  "coordinates": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "redaction_type": "blur",
  "blur_strength": 50,
  "active": true
}
```

Supported redaction types:
- `blur`: Gaussian blur
- `pixelate`: Pixelation effect
- `black_box`: Solid black rectangle
- `mask`: Custom mask overlay

## DSAR Request Format

Data Subject Access Requests follow GDPR requirements:

```json
{
  "request_type": "access",
  "subject_email": "user@example.com",
  "subject_name": "John Doe",
  "description": "Request for personal data access",
  "date_range_start": "2024-01-01T00:00:00Z",
  "date_range_end": "2024-12-31T23:59:59Z"
}
```

Request types:
- `access`: Data access request
- `rectification`: Data correction request
- `erasure`: Data deletion request
- `portability`: Data portability request
- `restriction`: Processing restriction request

## Performance Considerations

- **GPU Acceleration**: Use CUDA-enabled device for faster processing
- **Frame Skipping**: For videos, process every Nth frame to improve speed
- **Batch Processing**: Process multiple files together for efficiency
- **Caching**: Privacy zones and models are cached in Redis
- **Async Processing**: Long-running tasks are processed asynchronously

## Security Features

- **Encryption**: All processed data can be encrypted at rest
- **Access Logging**: Complete audit trail of all data access
- **Rate Limiting**: Prevents abuse of the service
- **Input Validation**: Comprehensive validation of all inputs
- **Secure Headers**: Security headers for web requests

## Testing

Run the test suite:

```bash
python test_service.py
```

This will test:
- Health check endpoint
- Image redaction functionality
- Privacy zone management
- DSAR request processing

## Integration

The privacy service integrates with:

- **Core API**: Evidence processing and storage
- **Edge Services**: Real-time camera feed processing
- **Notification Service**: DSAR completion notifications
- **Audit Service**: Complete audit logging

## Compliance

This service helps ensure compliance with:

- **GDPR**: Data Subject Access Rights
- **FERPA**: Educational privacy requirements
- **COPPA**: Children's privacy protection
- **Local Privacy Laws**: Configurable privacy controls

## Monitoring

Key metrics to monitor:

- Processing time per image/video
- Face detection accuracy
- Privacy zone application success rate
- DSAR request processing time
- Error rates and types
- Resource utilization (CPU, memory, GPU)