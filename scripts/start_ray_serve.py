"""
Start Script for Ray Serve (HVAC Cortex)
Launches the distributed inference graph with FastAPI routing.
Loads configuration from .env.local
"""

import sys
import os
import logging
import time
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SERVICES_DIR = REPO_ROOT / "services"
HVAC_AI_DIR = SERVICES_DIR / "hvac-ai"

# Load Environment Variables from .env.local
def load_env_file(env_file_path):
    """Load environment variables from .env.local file."""
    if not env_file_path.exists():
        return
    
    try:
        with open(env_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        logger.warning(f"Failed to load .env.local: {e}")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [AI-ENGINE] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RayServe")

# Load .env.local BEFORE any other logic
env_file = REPO_ROOT / ".env.local"
load_env_file(env_file)

# Extract key environment variables with defaults
BACKEND_PORT = int(os.environ.get('BACKEND_PORT', os.environ.get('PORT', '8000')))
RAY_PORT = int(os.environ.get('RAY_PORT', '10001'))
RAY_ADDRESS = os.environ.get('RAY_ADDRESS', '')
RAY_HEAD = os.environ.get('RAY_HEAD', '1') == '1'
RAY_USE_GPU = os.environ.get('RAY_USE_GPU', '1') == '1'
SKIP_MODEL = os.environ.get('SKIP_MODEL', '0') == '1'
FORCE_CPU = os.environ.get('FORCE_CPU', '0') == '1'
MODEL_PATH = os.environ.get('MODEL_PATH')



def start_ray_serve():
    """Start Ray Serve with FastAPI integration."""
    
    # Log configuration loaded from .env.local
    logger.info("=" * 60)
    logger.info("[CONFIG] Ray Serve Configuration:")
    logger.info(f"  Backend Port (Ray Serve HTTP): {BACKEND_PORT}")
    logger.info(f"  Ray Cluster Port (Reference): {RAY_PORT}")
    logger.info(f"  RAY_ADDRESS: {RAY_ADDRESS or '(none - local head node)'}")
    logger.info(f"  RAY_HEAD: {RAY_HEAD}")
    logger.info(f"  RAY_USE_GPU: {RAY_USE_GPU}")
    logger.info(f"  SKIP_MODEL: {SKIP_MODEL}")
    logger.info(f"  FORCE_CPU: {FORCE_CPU}")
    logger.info(f"  MODEL_PATH: {MODEL_PATH or '(not set)'}")
    logger.info("=" * 60)
    
    # 1. Path Injection (CRITICAL)
    sys.path.insert(0, str(SERVICES_DIR))
    sys.path.insert(0, str(HVAC_AI_DIR))
    
    # Set PYTHONPATH for Ray worker processes
    pythonpath_parts = [str(HVAC_AI_DIR), str(SERVICES_DIR)]
    os.environ['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
    
    logger.info(f"Added to sys.path: {HVAC_AI_DIR}")
    logger.info(f"Added to sys.path: {SERVICES_DIR}")
    logger.info(f"Exported PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    try:
        import ray
        from ray import serve
        
        # 2. Initialize Ray with configuration from .env.local
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("Initializing Ray cluster...")
        
        # Build Ray init arguments
        ray_init_kwargs = {
            "dashboard_host": "127.0.0.1",
            "ignore_reinit_error": True,
        }
        
        # Add GPU resources if enabled in .env.local
        if RAY_USE_GPU:
            ray_init_kwargs["num_gpus"] = 1
            logger.info("[CONFIG] GPU support enabled")
        else:
            logger.info("[CONFIG] GPU support disabled")
        
        # Use existing Ray cluster if RAY_ADDRESS is set
        if RAY_ADDRESS:
            logger.info(f"Connecting to existing Ray cluster at {RAY_ADDRESS}...")
            ray.init(address=RAY_ADDRESS)
        else:
            logger.info(f"Starting local Ray head node...")
            ray.init(**ray_init_kwargs)
        
        logger.info("✅ Ray cluster initialized")
        
        # 3. Import and prepare the FastAPI app
        logger.info("Importing FastAPI app...")
        try:
            from inference_graph import app
        except ImportError as e:
            logger.error(f"Failed to import 'inference_graph.app': {e}")
            sys.exit(1)
        
        # 4. Start Ray Serve with HTTP configuration from .env.local
        logger.info(f"Starting Ray Serve on 0.0.0.0:{BACKEND_PORT}...")
        serve.start(
            http_options={"host": "0.0.0.0", "port": BACKEND_PORT},
            detached=False
        )
        logger.info(f"✅ Ray Serve started at http://0.0.0.0:{BACKEND_PORT}")
        
        # 5. Deploy the FastAPI app with a Deployment wrapper
        logger.info("Deploying FastAPI application on Ray Serve...")
        logger.info("FastAPI will handle routing for /api/hvac/analyze")
        
        # Create an inline deployment for the app
        @serve.deployment
        class APIServer:
            def __init__(self):
                self.app = app
            
            async def __call__(self, scope, receive, send):
                """ASGI callable that delegates to FastAPI."""
                await self.app(scope, receive, send)
        
        # Build the deployment graph
        graph = APIServer.bind()
        serve.run(graph)
        
        logger.info("=" * 60)
        logger.info("[ENDPOINTS] Ray Serve + FastAPI Routing:")
        logger.info(f"  Ray Cluster Dashboard: http://127.0.0.1:8265")
        logger.info(f"  Ray Serve HTTP (BACKEND_PORT): http://0.0.0.0:{BACKEND_PORT}")
        logger.info(f"  ├─ Health check: http://0.0.0.0:{BACKEND_PORT}/health")
        logger.info(f"  └─ Analysis endpoint: http://0.0.0.0:{BACKEND_PORT}/api/hvac/analyze")
        logger.info(f"  Ray Cluster Port (RAY_PORT): {RAY_PORT}")
        logger.info("=" * 60)
        
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Stopping Ray Serve...")
            ray.shutdown()
            
    except ImportError as e:
        logger.error(f"Failed to import Ray or dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start Ray Serve: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    start_ray_serve()
