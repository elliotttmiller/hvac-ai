"""
Start Script for Ray Serve (HVAC AI)
Launches the distributed inference graph with FastAPI routing.
Reads all configuration from environment variables.
"""

import sys
import os
import logging
import time
from pathlib import Path

# --- Path & Environment Setup ---
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
# The directory where our AI code lives and runs from
SERVICES_AI_DIR = REPO_ROOT / "services" / "hvac-ai"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [AI-ENGINE] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RayServe")

def start_ray_serve():
    """Initializes Ray and deploys the FastAPI application via Ray Serve."""
    
    # --- CRITICAL FIX: Change Working Directory ---
    # This ensures that 'from inference_graph import app' works as you expect.
    os.chdir(SERVICES_AI_DIR)
    # Add paths for imports
    sys.path.insert(0, str(SERVICES_AI_DIR))  # Current dir for inference_graph.py
    sys.path.insert(0, str(SERVICES_AI_DIR.parent))  # Parent 'services' dir for hvac-domain
    logger.info(f"Changed CWD to: {os.getcwd()}")
    logger.info(f"Python path updated with: {SERVICES_AI_DIR}")
    
    # --- Read Config from Environment ---
    BACKEND_PORT = int(os.environ.get('BACKEND_PORT', '8000'))
    RAY_ADDRESS = os.environ.get('RAY_ADDRESS', '')
    USE_GPU = os.environ.get('RAY_USE_GPU', '1') == '1'
    
    try:
        import ray
        from ray import serve
        
        # 1. Initialize Ray Cluster
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("Initializing Ray cluster...")
        
        ray.init(
            dashboard_host="127.0.0.1",
            num_gpus=1 if USE_GPU else 0,
            ignore_reinit_error=True
        )
        
        logger.info("✅ Ray cluster initialized.")
        
        # 2. Import the FastAPI App (Now works due to CWD change)
        logger.info("Importing FastAPI app from inference_graph...")
        try:
            from inference_graph import app as fastapi_app
        except ImportError as e:
            logger.error(f"Failed to import 'inference_graph.app': {e}")
            logger.error("Ensure 'inference_graph.py' is in the current directory.")
            sys.exit(1)
        
        # 3. Deploy the FastAPI ASGI application directly with uvicorn
        # Skip Ray Serve for now and just run FastAPI directly on the backend port
        logger.info(f"Deploying FastAPI application (Port: {BACKEND_PORT})...")
        logger.info("=" * 60)
        logger.info("✅ RAY SERVE DEPLOYMENT STARTING")
        logger.info(f"   - Backend: http://0.0.0.0:{BACKEND_PORT}")
        logger.info(f"   - Health:  http://0.0.0.0:{BACKEND_PORT}/health")
        logger.info(f"   - Dashboard: http://127.0.0.1:8265")
        logger.info("=" * 60)
        
        # Start Ray Serve (without blocking)
        serve.start()
        
        # For now, we'll run FastAPI directly with uvicorn
        # This allows the health checks to work while inference happens in Ray
        import uvicorn
        logger.info("Starting FastAPI with uvicorn...")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=BACKEND_PORT, log_level="info")
        
    except ImportError as e:
        logger.error(f"[ERROR] Failed to import Ray. Please run: pip install 'ray[serve]'", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Failed to start Ray Serve.", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    start_ray_serve()