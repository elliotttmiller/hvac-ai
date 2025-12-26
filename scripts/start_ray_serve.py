"""
Ray Serve Launcher - Start the distributed inference graph
Launches Ray Serve with the inference graph deployment.

Usage:
    python scripts/start_ray_serve.py

Environment Variables:
    YOLO_MODEL_PATH: Path to YOLO model weights (default: ai_model/best.pt)
    CONF_THRESHOLD: Confidence threshold (default: 0.5)
    RAY_ADDRESS: Ray cluster address (default: local mode)
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [AI-ENGINE] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RayServe")


def check_ray_installed():
    """Check if Ray Serve is installed."""
    try:
        import ray
        from ray import serve
        logger.info(f"✅ Ray {ray.__version__} detected")
        return True
    except ImportError:
        logger.error("❌ Ray Serve not installed")
        logger.error("   Install with: pip install ray[serve]")
        return False


def check_model_exists(model_path: str) -> bool:
    """Check if model file exists."""
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at: {model_path}")
        logger.error("   Set YOLO_MODEL_PATH environment variable to point to your model")
        return False
    logger.info(f"✅ Model found: {model_path}")
    return True


def start_ray_serve():
    """Start Ray Serve with the inference graph."""
    logger.info("🚀 Starting Ray Serve Inference Graph")
    logger.info("=" * 60)
    
    # Check dependencies
    if not check_ray_installed():
        sys.exit(1)
    
    # Get configuration
    repo_root = Path(__file__).parent.parent
    model_path = os.getenv(
        'YOLO_MODEL_PATH',
        str(repo_root / 'ai_model' / 'best.pt')
    )
    conf_threshold = os.getenv('CONF_THRESHOLD', '0.5')
    
    # Check model
    if not check_model_exists(model_path):
        sys.exit(1)
    
    # Set environment variables for the inference graph
    os.environ['YOLO_MODEL_PATH'] = model_path
    os.environ['CONF_THRESHOLD'] = conf_threshold
    
    # Change to python-services directory for proper imports
    python_services_dir = repo_root / 'python-services'
    os.chdir(python_services_dir)
    
    # Add python-services to Python path
    if str(python_services_dir) not in sys.path:
        sys.path.insert(0, str(python_services_dir))
    
    logger.info(f"   Model path: {model_path}")
    logger.info(f"   Confidence threshold: {conf_threshold}")
    logger.info(f"   Working directory: {os.getcwd()}")
    logger.info("")
    
    # Start Ray Serve
    logger.info("⚡ Launching Ray Serve...")
    logger.info("   This will start the inference graph with:")
    logger.info("   - ObjectDetector (40% GPU)")
    logger.info("   - TextExtractor (30% GPU)")
    logger.info("   - Ingress (API Gateway)")
    logger.info("")
    
    try:
        # Use serve run command
        cmd = [
            sys.executable,
            '-m', 'ray.serve',
            'run',
            'core.inference_graph:entrypoint',
            '--host', '0.0.0.0',
            '--port', '8000'
        ]
        
        logger.info(f"   Command: {' '.join(cmd)}")
        logger.info("")
        
        # Run Ray Serve (this blocks)
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down Ray Serve...")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ray Serve failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"❌ Failed to start Ray Serve: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    start_ray_serve()
