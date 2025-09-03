"""
Test script for Privacy Service functionality.
"""
import requests
import json
import base64
import cv2
import numpy as np
from PIL import Image
import io

def create_test_image():
    """Create a simple test image."""
    # Create a simple test image with a face-like pattern
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Draw a simple face
    cv2.circle(img, (200, 200), 100, (255, 255, 255), -1)  # Face
    cv2.circle(img, (170, 170), 15, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (230, 170), 15, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (200, 230), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    return img

def test_health_endpoint():
    """Test health check endpoint."""
    try:
        response = requests.get('http://localhost:5000/health')
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_image_redaction():
    """Test image redaction endpoint."""
    try:
        # Create test image
        test_img = create_test_image()
        
        # Encode image
        _, buffer = cv2.imencode('.jpg', test_img)
        img_bytes = buffer.tobytes()
        
        # Prepare form data
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        data = {
            'blur_strength': 50,
            'privacy_zones': json.dumps([])
        }
        
        # Send request
        response = requests.post('http://localhost:5000/redact/image', files=files, data=data)
        print(f"Image redaction status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Faces detected: {result.get('faces_detected', 0)}")
            print(f"Privacy zones applied: {result.get('privacy_zones_applied', 0)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Image redaction test failed: {e}")
        return False

def test_privacy_zone_creation():
    """Test privacy zone creation."""
    try:
        privacy_zone = {
            'zone_id': 'test_zone_1',
            'name': 'Test Privacy Zone',
            'coordinates': [[50, 50], [150, 50], [150, 150], [50, 150]],
            'redaction_type': 'blur',
            'blur_strength': 75,
            'active': True
        }
        
        response = requests.post(
            'http://localhost:5000/privacy-zones',
            json=privacy_zone,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Privacy zone creation status: {response.status_code}")
        
        if response.status_code == 201:
            result = response.json()
            print(f"Created zone ID: {result.get('zone_id')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Privacy zone creation test failed: {e}")
        return False

def test_dsar_request():
    """Test DSAR request creation."""
    try:
        dsar_request = {
            'request_id': 'test_dsar_1',
            'request_type': 'access',
            'subject_email': 'test@example.com',
            'subject_name': 'Test User',
            'description': 'Test DSAR request for access to personal data'
        }
        
        response = requests.post(
            'http://localhost:5000/dsar/request',
            json=dsar_request,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"DSAR request status: {response.status_code}")
        
        if response.status_code == 201:
            result = response.json()
            print(f"DSAR request ID: {result.get('request_id')}")
            print(f"Status: {result.get('status')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"DSAR request test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("=== Privacy Service Tests ===\n")
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Image Redaction", test_image_redaction),
        ("Privacy Zone Creation", test_privacy_zone_creation),
        ("DSAR Request", test_dsar_request)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}\n")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name}: FAILED - {e}\n")
    
    # Summary
    print("=== Test Results ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == '__main__':
    run_all_tests()