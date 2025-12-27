"""
Start Script for Ray Serve (HVAC Cortex)
Launches the distributed inference graph programmatically.
This bypasses CLI issues on Windows.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SERVICES_DIR = REPO_ROOT / "services" / "hvac-ai"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [AI-ENGINE] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RayServe")

def start_ray_serve():
    # 1. Validate Environment
    model_path = os.environ.get("MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"MODEL_PATH not set or invalid: {model_path}")
        logger.warning("Inference may fail. Set MODEL_PATH in .env.local")
    
    # 2. Set PYTHONPATH
    # Add both the hvac-ai service directory AND the parent services directory
    # This allows inference_graph to import from hvac-ai AND from hvac-domain
    sys.path.insert(0, str(SERVICES_DIR))
    sys.path.insert(0, str(SERVICES_DIR.parent))
    logger.info(f"Added to sys.path: {SERVICES_DIR}")
    logger.info(f"Added to sys.path: {SERVICES_DIR.parent}")
    
    try:
        import ray
        from ray import serve
        
        # 3. Initialize Ray
        # We start a local Ray instance. 'ignore_reinit_error' allows restarts.
        # num_gpus=1 ensures Ray sees your GPU.
        if ray.is_initialized():
            ray.shutdown()
            
        logger.info("Initializing Ray cluster...")
        # Force num_gpus=1 if we know we have a GPU, otherwise let Ray auto-detect
        ray.init(dashboard_host="127.0.0.1", num_gpus=1, ignore_reinit_error=True)
        
        logger.info("Starting Serve instance...")
        # Bind to 0.0.0.0 so it's accessible
        serve.start(detached=False, http_options={"host": "0.0.0.0", "port": 8000})
        
        # 4. Deploy the Graph
        logger.info("Deploying Inference Graph...")
        
        # Import the entrypoint dynamically to register it with Serve
        try:
            from inference_graph import entrypoint
        except ImportError as e:
            logger.error(f"Failed to import 'inference_graph': {e}")
            logger.error(f"Current sys.path: {sys.path}")
            sys.exit(1)
        
        # Deploy the application
        serve.run(entrypoint)
        
        logger.info("[AI-ENGINE] Inference Graph Deployed Successfully!")
        logger.info("[AI-ENGINE]   - ObjectDetector (GPU)")
        logger.info("[AI-ENGINE]   - TextExtractor (GPU)")
        logger.info("[AI-ENGINE]   - Ingress (HTTP :8000)")
        
        # Keep the script running to keep the serve instance alive
        # The parent process (start_unified.py) will kill this script on exit
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Stopping Ray Serve...")
            ray.shutdown()
            
    except ImportError as e:
        logger.error(f"❌ Failed to import Ray: {e}")
        logger.error("Please run: pip install 'ray[serve]'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start Ray Serve: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_ray_serve()