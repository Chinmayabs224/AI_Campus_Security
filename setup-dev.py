#!/usr/bin/env python3
"""
Development Environment Setup Script for AI Campus Security System
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_prerequisites():
    """Check if required tools are installed."""
    print("Checking prerequisites...")
    
    required_tools = {
        'python': 'python --version',
        'docker': 'docker --version',
        'docker-compose': 'docker-compose --version',
        'node': 'node --version',
        'npm': 'npm --version'
    }
    
    missing_tools = []
    for tool, command in required_tools.items():
        result = run_command(command, check=False)
        if result.returncode != 0:
            missing_tools.append(tool)
        else:
            print(f"✓ {tool} is installed")
    
    if missing_tools:
        print(f"\n❌ Missing required tools: {', '.join(missing_tools)}")
        print("Please install the missing tools and run this script again.")
        sys.exit(1)
    
    print("✓ All prerequisites are installed\n")

def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    print("Setting up Python environment...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('venv'):
        print("Creating Python virtual environment...")
        run_command(f"{sys.executable} -m venv venv")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = 'venv\\Scripts\\activate'
        pip_command = 'venv\\Scripts\\pip'
    else:  # Unix/Linux/macOS
        activate_script = 'venv/bin/activate'
        pip_command = 'venv/bin/pip'
    
    # Install requirements
    print("Installing Python dependencies...")
    run_command(f"{pip_command} install --upgrade pip")
    run_command(f"{pip_command} install -r requirements.txt")
    
    print("✓ Python environment setup complete\n")

def setup_frontend():
    """Set up React frontend."""
    print("Setting up React frontend...")
    
    frontend_dir = Path('frontend')
    if frontend_dir.exists():
        print("Installing frontend dependencies...")
        run_command('npm install', cwd='frontend')
        print("✓ Frontend setup complete\n")
    else:
        print("❌ Frontend directory not found. Please ensure React app was created successfully.")

def setup_docker_environment():
    """Set up Docker development environment."""
    print("Setting up Docker environment...")
    
    # Copy environment file if it doesn't exist
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("✓ Created .env file from .env.example")
        else:
            print("❌ .env.example file not found")
    
    # Start Docker services
    print("Starting Docker services...")
    run_command('docker-compose up -d postgres redis minio')
    
    print("✓ Docker environment setup complete\n")

def download_yolo_model():
    """Download YOLO model for development."""
    print("Setting up YOLO model...")
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'yolov8n.pt'
    if not model_path.exists():
        print("Downloading YOLOv8 nano model...")
        try:
            # This will be handled by ultralytics when first used
            print("✓ YOLO model will be downloaded automatically on first use")
        except Exception as e:
            print(f"Note: YOLO model will be downloaded when first needed: {e}")
    else:
        print("✓ YOLO model already exists")
    
    print("✓ Model setup complete\n")

def create_directories():
    """Create necessary directories."""
    print("Creating project directories...")
    
    directories = [
        'logs',
        'data',
        'models',
        'uploads',
        'evidence'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created {directory}/ directory")
    
    print("✓ Directory setup complete\n")

def setup_ml_environment():
    """Set up ML training environment."""
    print("Setting up ML training environment...")
    
    try:
        # Run ML setup script
        ml_setup_script = Path('ml-training/setup_ml_env.py')
        if ml_setup_script.exists():
            run_command(f"{sys.executable} {ml_setup_script}")
            print("✓ ML environment setup complete")
        else:
            print("⚠ ML setup script not found, skipping ML setup")
    except Exception as e:
        print(f"⚠ ML setup encountered issues: {e}")
        print("You can run ML setup manually later:")
        print("python ml-training/setup_ml_env.py")

def print_next_steps():
    """Print next steps for the user."""
    print("🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("1. Start all services:")
    print("   docker-compose up -d")
    print("\n2. Start the frontend development server:")
    print("   cd frontend && npm start")
    print("\n3. Set up ML training (if not done automatically):")
    print("   python ml-training/setup_ml_env.py")
    print("   python ml-training/data-processing/download_dataset.py")
    print("\n4. Access the services:")
    print("   - Dashboard: http://localhost:3000")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - MinIO Console: http://localhost:9001")
    print("\n5. Check service logs:")
    print("   docker-compose logs -f [service-name]")
    print("\n6. To stop all services:")
    print("   docker-compose down")

def main():
    """Main setup function."""
    print("🚀 AI Campus Security System - Development Setup")
    print("=" * 50)
    
    try:
        check_prerequisites()
        create_directories()
        setup_python_environment()
        setup_frontend()
        setup_docker_environment()
        download_yolo_model()
        setup_ml_environment()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()