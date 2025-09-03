# Edge Services

Edge computing services for real-time AI inference on camera streams.

## Components

- **inference-engine/**: YOLO-based object detection service
- **stream-processor/**: RTSP stream ingestion and processing
- **event-generator/**: Security event creation and local buffering
- **edge-coordinator/**: Edge node management and coordination

## Requirements

- NVIDIA GPU support for inference acceleration
- RTSP camera stream compatibility
- Local storage for incident clip buffering
- Network resilience for offline operation