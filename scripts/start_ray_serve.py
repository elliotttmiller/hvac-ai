"""
Start Script for Ray Serve (HVAC AI)
Launches the distributed inference graph with FastAPI routing.
Reads all configuration from environment variables from .env.local.
"""

import sys
import os
import logging
import time
from pathlib import Path

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SERVICES_DIR = REPO_ROOT / "services"
HVAC_AI_DIR = SERVICES_DIR / "hvac-ai"

# Load environment from .env.local BEFORE any other imports
def load_env_file(env_file_path):
    """Load environment variables from .env.local file."""
    if not env_file_path.exists():
        return
    try:
        with open(env_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        print(f"Warning: Failed to load .env.local: {e}")

# Load environment variables early
load_env_file(REPO_ROOT / ".env.local")

# Inject path for cross-service imports
sys.path.insert(0, str(SERVICES_DIR))
sys.path.insert(0, str(HVAC_AI_DIR))

# --- Configuration (Read from Environment) ---
BACKEND_PORT = int(os.environ.get('BACKEND_PORT', '8000'))
RAY_ADDRESS = os.environ.get('RAY_ADDRESS', '')
USE_GPU = os.environ.get('RAY_USE_GPU', '1') == '1'

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [AI-ENGINE] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RayServe")

def start_ray_serve():
    """Initializes Ray and deploys the FastAPI application via Ray Serve."""
    
    try:
        import ray
        from ray import serve
        
        # 1. Initialize Ray Cluster
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("Initializing Ray cluster...")
        
        if RAY_ADDRESS:
            logger.info(f"Connecting to existing Ray cluster at {RAY_ADDRESS}...")
            ray.init(address=RAY_ADDRESS)
        else:
            ray.init(
                dashboard_host="127.0.0.1",
                num_gpus=1 if USE_GPU else 0,
                ignore_reinit_error=True
            )
        
        logger.info("✅ Ray cluster initialized.")
        
        # 2. Import the FastAPI Application Entrypoint
        logger.info("Importing FastAPI app from inference_graph...")
        try:
            from inference_graph import app as fastapi_app
        except ImportError as e:
            logger.error(f"Failed to import 'inference_graph.app': {e}")
            sys.exit(1)
        
        # 3. Start Ray Serve
        logger.info(f"Starting Ray Serve on 0.0.0.0:{BACKEND_PORT}...")
        serve.start(
            http_options={"host": "0.0.0.0", "port": BACKEND_PORT},
            detached=False
        )
        
        # 4. Create a Ray Serve Deployment from the FastAPI App
        @serve.deployment
        class APIServer:
            def __init__(self):
                self.app = fastapi_app
            
            async def __call__(self, scope, receive, send):
                """ASGI callable that delegates to FastAPI."""
                await self.app(scope, receive, send)
        
        # 5. Deploy the application
        APIServer.deploy()
        
        logger.info("=" * 60)
        logger.info("✅ APPLICATION IS LIVE!")
        logger.info(f"   - Backend: http://0.0.0.0:{BACKEND_PORT}")
        logger.info(f"   - Health:  http://0.0.0.0:{BACKEND_PORT}/health")
        logger.info("=" * 60)
        
        # Keep the script alive
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Shutting down Ray Serve...")
            ray.shutdown()
            
    except ImportError as e:
        logger.error(f"[ERROR] Failed to import Ray. Please run: pip install 'ray[serve]'", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Failed to start Ray Serve.", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    start_ray_serve()