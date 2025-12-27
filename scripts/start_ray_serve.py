"""
Start Script for Ray Serve (HVAC AI)
Launches the distributed inference graph using the Application Builder pattern.
"""

# --- OPTIMIZATION: Disable Phone-Home Checks Globally ---
# Must be set before ANY other imports
import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import sys
import logging
from pathlib import Path
import time

# --- Path & Environment Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.resolve()
SERVICES_DIR = (REPO_ROOT / "services").resolve()
SERVICES_AI_DIR = (SERVICES_DIR / "hvac-ai").resolve()
SERVICES_ROOT = SERVICES_DIR

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [RayServe] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy logs
logging.getLogger("ray.serve").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def main():
    logger.info("=" * 70)
    logger.info("HVAC AI Platform - Ray Serve Launcher")
    logger.info("=" * 70)
    
    # --- Step 1: Path Setup ---
    os.chdir(SERVICES_AI_DIR)
    sys.path.insert(0, str(SERVICES_AI_DIR))
    sys.path.insert(0, str(SERVICES_ROOT))
    
    logger.info("[1/4] Path Setup:")
    logger.info(f"      ✅ CWD set to: {os.getcwd()}")
    
    # --- Step 2: Environment Validation ---
    BACKEND_PORT = int(os.getenv('BACKEND_PORT', '8000'))
    USE_GPU = os.getenv('RAY_USE_GPU', '1') == '1'
    MODEL_PATH = os.getenv('MODEL_PATH')
    
    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        abs_model_path = REPO_ROOT / (MODEL_PATH or "")
        if not abs_model_path.exists():
            logger.error(f"❌ MODEL_PATH is invalid: {MODEL_PATH}")
            sys.exit(1)
        else:
            os.environ['MODEL_PATH'] = str(abs_model_path)
    
    # --- Step 3: Ray Initialization ---
    logger.info("[2/4] Initializing Ray cluster...")
    try:
        import ray
        from ray import serve
        
        if ray.is_initialized():
            ray.shutdown()
        
        # Inject config into workers
        runtime_env = {
            "env_vars": {
                "DISABLE_MODEL_SOURCE_CHECK": "True",
                "PYTHONPATH": os.pathsep.join(sys.path) 
            }
        }

        ray.init(
            num_gpus=1 if USE_GPU else 0,
            ignore_reinit_error=True,
            runtime_env=runtime_env
        )
        logger.info("      ✅ Ray cluster initialized")
        
    except ImportError:
        logger.error("❌ Ray not installed. Run: pip install 'ray[serve]'")
        sys.exit(1)
    
    # --- Step 4: Import Application Builder ---
    logger.info("[3/4] Importing application builder...")
    try:
        from inference_graph import build_app
    except ImportError as e:
        logger.error(f"❌ Failed to import build_app: {e}", exc_info=True)
        sys.exit(1)
    
# --- Step 5: Deploy Application ---
    try:
        logger.info(f"      🚀 Deploying application on port {BACKEND_PORT}...")
        
        serve.start(
            detached=False,
            http_options={"host": "0.0.0.0", "port": BACKEND_PORT}
        )
        
        serve.run(
            build_app(),
            name="default",
        )
        
        logger.info("      ✅ Deployment request sent. Waiting for workers...")
        
        # Keep script alive - CRITICAL FIX
        # We use a loop that handles signals properly
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n✅ Shutdown signal received.")
    except Exception as e:
        logger.error(f"❌ Application deployment failed: {e}", exc_info=True)
    finally:
        # Only shutdown if we actually started Ray
        if 'ray' in locals() and ray.is_initialized():
            logger.info("Shutting down Ray Serve...")
            serve.shutdown()
            ray.shutdown()
            logger.info("✅ Ray cluster shut down cleanly.")
        sys.exit(1)

if __name__ == "__main__":
    main()