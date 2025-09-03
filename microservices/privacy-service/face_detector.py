"""
Face detection and blurring using FaceNet and OpenCV.
"""
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import logging
from typing import List, Tuple, Optional
import base64
import io

from models import Face, BoundingBox, PrivacyZone, RedactionType
from config import Config

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection and redaction using MTCNN and OpenCV."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize face detector."""
        self.config = config or Config()
        
        # Determine device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config.FACE_DETECTION_DEVICE == 'cuda' 
            else 'cpu'
        )
        
        # Initialize MTCNN
        try:
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=self.device,
                keep_all=True
            )
            self._model_ready = True
            logger.info(f"Face detector initialized on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}")
            self.mtcnn = None
            self._model_ready = False
    
    def is_ready(self) -> bool:
        """Check if face detector is ready."""
        return self._model_ready and self.mtcnn is not None
    
    def detect_faces(self, image: np.ndarray) -> List[Face]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected faces
        """
        if not self.is_ready():
            logger.warning("Face detector not ready")
            return []
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Detect faces
            boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
            
            faces = []
            
            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    if prob >= self.config.FACE_DETECTION_CONFIDENCE:
                        # Convert box coordinates
                        x1, y1, x2, y2 = box.astype(int)
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Create bounding box
                        bbox = BoundingBox(
                            x=max(0, x1),
                            y=max(0, y1),
                            width=max(0, width),
                            height=max(0, height),
                            confidence=float(prob)
                        )
                        
                        # Convert landmarks if available
                        face_landmarks = None
                        if landmark is not None:
                            face_landmarks = [(int(x), int(y)) for x, y in landmark]
                        
                        # Create face object
                        face = Face(
                            bounding_box=bbox,
                            landmarks=face_landmarks,
                            confidence=float(prob),
                            face_id=f"face_{i}"
                        )
                        
                        faces.append(face)
            
            logger.info(f"Detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []
    
    def blur_faces(
        self, 
        image: np.ndarray, 
        faces: List[Face], 
        blur_strength: int = 50
    ) -> np.ndarray:
        """
        Apply blur to detected faces.
        
        Args:
            image: Input image
            faces: List of detected faces
            blur_strength: Blur intensity (1-100)
            
        Returns:
            Image with blurred faces
        """
        if not faces:
            return image.copy()
        
        result_image = image.copy()
        
        # Convert blur strength to kernel size
        kernel_size = max(3, int(blur_strength * 0.5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        for face in faces:
            bbox = face.bounding_box
            
            # Extract face region
            x1 = max(0, bbox.x)
            y1 = max(0, bbox.y)
            x2 = min(image.shape[1], bbox.x + bbox.width)
            y2 = min(image.shape[0], bbox.y + bbox.height)
            
            if x2 > x1 and y2 > y1:
                # Extract face region
                face_region = result_image[y1:y2, x1:x2]
                
                # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
                
                # Replace face region with blurred version
                result_image[y1:y2, x1:x2] = blurred_face
        
        return result_image
    
    def pixelate_faces(
        self, 
        image: np.ndarray, 
        faces: List[Face], 
        pixel_size: int = 20
    ) -> np.ndarray:
        """
        Apply pixelation to detected faces.
        
        Args:
            image: Input image
            faces: List of detected faces
            pixel_size: Size of pixels for pixelation
            
        Returns:
            Image with pixelated faces
        """
        if not faces:
            return image.copy()
        
        result_image = image.copy()
        
        for face in faces:
            bbox = face.bounding_box
            
            # Extract face region
            x1 = max(0, bbox.x)
            y1 = max(0, bbox.y)
            x2 = min(image.shape[1], bbox.x + bbox.width)
            y2 = min(image.shape[0], bbox.y + bbox.height)
            
            if x2 > x1 and y2 > y1:
                # Extract face region
                face_region = result_image[y1:y2, x1:x2]
                
                # Resize down and up to create pixelation effect
                height, width = face_region.shape[:2]
                
                # Calculate new dimensions
                new_width = max(1, width // pixel_size)
                new_height = max(1, height // pixel_size)
                
                # Resize down
                small = cv2.resize(face_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Resize back up
                pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Replace face region
                result_image[y1:y2, x1:x2] = pixelated
        
        return result_image
    
    def black_box_faces(self, image: np.ndarray, faces: List[Face]) -> np.ndarray:
        """
        Apply black boxes to detected faces.
        
        Args:
            image: Input image
            faces: List of detected faces
            
        Returns:
            Image with black boxes over faces
        """
        if not faces:
            return image.copy()
        
        result_image = image.copy()
        
        for face in faces:
            bbox = face.bounding_box
            
            # Draw black rectangle
            x1 = max(0, bbox.x)
            y1 = max(0, bbox.y)
            x2 = min(image.shape[1], bbox.x + bbox.width)
            y2 = min(image.shape[0], bbox.y + bbox.height)
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return result_image
    
    def apply_privacy_zones(
        self, 
        image: np.ndarray, 
        privacy_zones: List[PrivacyZone]
    ) -> np.ndarray:
        """
        Apply privacy zone redaction to image.
        
        Args:
            image: Input image
            privacy_zones: List of privacy zones to apply
            
        Returns:
            Image with privacy zones applied
        """
        if not privacy_zones:
            return image.copy()
        
        result_image = image.copy()
        
        for zone in privacy_zones:
            if not zone.active:
                continue
            
            try:
                # Create mask from polygon coordinates
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                points = np.array(zone.coordinates, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
                
                # Apply redaction based on type
                if zone.redaction_type == RedactionType.BLUR:
                    # Apply blur to the zone
                    kernel_size = max(3, int(zone.blur_strength * 0.5))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    blurred = cv2.GaussianBlur(result_image, (kernel_size, kernel_size), 0)
                    result_image = np.where(mask[..., None] == 255, blurred, result_image)
                    
                elif zone.redaction_type == RedactionType.BLACK_BOX:
                    # Apply black box to the zone
                    result_image[mask == 255] = [0, 0, 0]
                    
                elif zone.redaction_type == RedactionType.PIXELATE:
                    # Apply pixelation to the zone
                    pixel_size = max(5, zone.blur_strength // 5)
                    
                    # Get bounding rectangle of the zone
                    x, y, w, h = cv2.boundingRect(points)
                    
                    if w > 0 and h > 0:
                        # Extract zone region
                        zone_region = result_image[y:y+h, x:x+w]
                        zone_mask = mask[y:y+h, x:x+w]
                        
                        # Apply pixelation
                        new_width = max(1, w // pixel_size)
                        new_height = max(1, h // pixel_size)
                        
                        small = cv2.resize(zone_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        # Apply only to masked area
                        result_image[y:y+h, x:x+w] = np.where(
                            zone_mask[..., None] == 255, 
                            pixelated, 
                            result_image[y:y+h, x:x+w]
                        )
                
            except Exception as e:
                logger.error(f"Failed to apply privacy zone {zone.zone_id}: {str(e)}")
                continue
        
        return result_image
    
    def process_base64_image(
        self, 
        base64_data: str, 
        privacy_zones: List[PrivacyZone], 
        blur_strength: int = 50
    ) -> dict:
        """
        Process base64 encoded image.
        
        Args:
            base64_data: Base64 encoded image
            privacy_zones: Privacy zones to apply
            blur_strength: Blur strength for faces
            
        Returns:
            Processing result dictionary
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Invalid image data'}
            
            # Detect faces
            faces = self.detect_faces(image)
            
            # Apply face blurring
            result_image = self.blur_faces(image, faces, blur_strength)
            
            # Apply privacy zones
            result_image = self.apply_privacy_zones(result_image, privacy_zones)
            
            # Encode result
            _, buffer = cv2.imencode('.jpg', result_image)
            result_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            return {
                'success': True,
                'faces_detected': len(faces),
                'privacy_zones_applied': len([z for z in privacy_zones if z.active]),
                'redacted_image': result_b64
            }
            
        except Exception as e:
            logger.error(f"Base64 image processing failed: {str(e)}")
            return {'error': str(e)}