# Event Generator Service

The Event Generator service is responsible for converting AI detection results into structured security events, extracting incident video clips, and providing local buffering capabilities for network outage scenarios.

## Features

### 1. Security Event Generation
- **Event Aggregation**: Combines multiple detections into coherent security events
- **Event Classification**: Maps AI detections to security event types (intrusion, loitering, etc.)
- **Severity Assessment**: Calculates event severity based on confidence and detection patterns
- **Metadata Enrichment**: Adds contextual information and processing timestamps

### 2. Event Deduplication
- **Temporal Deduplication**: Prevents duplicate events within configurable time windows
- **Spatial Deduplication**: Uses bounding box IoU to identify spatially similar events
- **Hash-based Tracking**: Efficient duplicate detection using event fingerprints

### 3. Incident Clip Extraction
- **Configurable Duration**: Pre/post-event clip extraction with customizable timing
- **Frame Buffering**: Circular buffer maintains recent frames for clip extraction
- **Video Encoding**: Outputs standard MP4 clips with configurable quality settings
- **Memory Management**: Automatic cleanup and memory limit enforcement

### 4. Local Buffering
- **Network Outage Resilience**: Continues operation during connectivity loss
- **SQLite Storage**: Reliable local database for events and clip metadata
- **Automatic Synchronization**: Resumes sync when network connectivity returns
- **Data Compression**: Optional compression to reduce storage requirements

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Detection  │───▶│  Event Generator │───▶│ Security Events │
│     Results     │    │    & Aggregator  │    │   & Metadata    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Video Frames   │───▶│  Clip Extractor  │───▶│ Incident Clips  │
│    (Buffered)   │    │   & Processor    │    │   (MP4 Files)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Network Outage  │───▶│  Local Buffer    │───▶│ Sync to Central │
│   Scenarios     │    │   & Database     │    │    Services     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Data Models

### SecurityEvent
Core event structure containing:
- Event identification and timing
- Classification (type, severity, status)
- Detection data (bounding boxes, confidence)
- Location and context information
- Processing metadata

### IncidentClip
Video clip metadata including:
- Associated event information
- Timing and duration
- File path and encoding details
- Processing status flags

### EventBuffer
Local storage container for:
- Buffered events and clips
- Synchronization status
- Storage management metadata

## Configuration

The service uses a hierarchical configuration system:

```json
{
  "event_config": {
    "min_confidence": 0.5,
    "dedup_time_window": 10.0,
    "aggregation_window": 5.0
  },
  "clip_config": {
    "pre_event_duration": 10.0,
    "post_event_duration": 15.0,
    "output_fps": 15.0
  },
  "buffer_config": {
    "max_buffer_size_mb": 1000,
    "sync_retry_interval": 30
  }
}
```

## Usage

### Basic Service Setup

```python
from edge_services.event_generator import EdgeEventService, EdgeEventConfig

# Create configuration
config = EdgeEventConfig()
config.enable_clip_extraction = True
config.enable_local_buffering = True

# Create and start service
service = EdgeEventService(config)
service.start()

# Add cameras for frame buffering
service.add_camera("camera_01")
service.add_camera("camera_02")

# Process detection results
service.process_detection_result(inference_result)

# Process video frames
service.process_frame("camera_01", frame, timestamp)
```

### Event Callbacks

```python
def handle_security_event(event):
    print(f"New event: {event.event_type.value} at {event.camera_id}")
    if event.severity == SeverityLevel.CRITICAL:
        # Trigger immediate alert
        send_emergency_alert(event)

def handle_incident_clip(clip):
    print(f"Clip extracted: {clip.file_path}")
    # Upload to evidence storage
    upload_to_evidence_system(clip)

service.add_event_callback(handle_security_event)
service.add_clip_callback(handle_incident_clip)
```

### Synchronization Setup

```python
def sync_with_central_api(events, clips):
    try:
        # Upload events to central API
        for event in events:
            response = api_client.post("/events", event.to_dict())
            if response.status_code != 200:
                return False
        
        # Upload clips
        for clip in clips:
            if clip.file_path and Path(clip.file_path).exists():
                response = api_client.upload("/clips", clip.file_path)
                if response.status_code != 200:
                    return False
        
        return True
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return False

service.set_sync_callback(sync_with_central_api)
```

## Event Types and Mapping

The service maps AI detection classes to security event types:

| Detection Class | Event Type | Description |
|----------------|------------|-------------|
| person | intrusion | Unauthorized person detected |
| car, truck, motorcycle | suspicious_activity | Vehicle in restricted area |
| handbag, suitcase, backpack | abandoned_object | Unattended object detected |

## Performance Considerations

### Memory Management
- Frame buffers are limited by configurable memory thresholds
- Automatic cleanup removes old processed data
- Compression reduces storage requirements

### Processing Efficiency
- Asynchronous processing prevents blocking
- Batch operations for database interactions
- Configurable queue sizes prevent memory overflow

### Storage Optimization
- SQLite database with proper indexing
- Configurable retention policies
- Automatic cleanup of old files

## Monitoring and Diagnostics

### Service Status
```python
status = service.get_comprehensive_status()
print(f"Events generated: {status['service']['events_generated']}")
print(f"Clips extracted: {status['service']['clips_extracted']}")
print(f"Buffer status: {status['buffer_service']}")
```

### Performance Metrics
- Event generation rate and deduplication efficiency
- Clip extraction success rate and processing time
- Buffer utilization and sync success rate
- Network connectivity and sync attempt statistics

## Testing

Run the test suite to verify functionality:

```bash
python -m edge_services.event_generator.test_event_generation
```

The test suite covers:
- Event generation from detection results
- Clip extraction and video processing
- Local buffering and synchronization
- Event deduplication logic

## Requirements

### Dependencies
- OpenCV (cv2) for video processing
- SQLite3 for local database
- NumPy for array operations
- Standard library modules (threading, queue, json, etc.)

### System Requirements
- Sufficient disk space for video clips and database
- Memory for frame buffering (configurable)
- Network connectivity for synchronization (optional)

## Error Handling

The service implements comprehensive error handling:
- Graceful degradation during component failures
- Automatic retry mechanisms for transient errors
- Detailed logging for debugging and monitoring
- Fallback modes for network connectivity issues

## Security Considerations

- Event data validation and sanitization
- Secure file handling for video clips
- Database integrity and transaction safety
- Configurable retention policies for compliance