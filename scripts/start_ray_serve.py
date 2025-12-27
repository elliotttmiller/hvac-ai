"""
Start Script for Ray Serve (HVAC AI)
Launches the distributed inference graph using the Application Builder pattern.
"""

# Set global environment variables before any imports
import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # Bypass PaddleOCR connectivity checks globally

import sys
import logging
from pathlib import Path
import time
from typing import Any, cast

# --- Path & Environment Setup ---
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.resolve()
SERVICES_DIR = (REPO_ROOT / "services").resolve()
SERVICES_AI_DIR = (SERVICES_DIR / "hvac-ai").resolve()
SERVICES_ROOT = SERVICES_DIR

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [RayServe] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("HVAC AI Platform - Ray Serve Launcher")
    logger.info("=" * 70)
    
    # --- Step 1: Path Setup ---
    os.chdir(SERVICES_AI_DIR)
    sys.path.insert(0, str(SERVICES_AI_DIR))
    sys.path.insert(0, str(SERVICES_ROOT))
    
    # --- Step 2: Environment Validation ---
    BACKEND_PORT = int(os.getenv('BACKEND_PORT', '8000'))
    USE_GPU = os.getenv('RAY_USE_GPU', '1') == '1'
    MODEL_PATH = os.getenv('MODEL_PATH')
    
    # --- CRITICAL FIX: Validate Model Path ---
    if not MODEL_PATH:
        # Fallback to default location if env var is missing
        default_model = REPO_ROOT / "ai_model/models/hvac_obb_l_20251224_214011/weights/best.pt"
        if default_model.exists():
             MODEL_PATH = str(default_model)
             os.environ['MODEL_PATH'] = MODEL_PATH
             logger.info(f"      ⚠️ MODEL_PATH not set, using default: {MODEL_PATH}")
    
    if not MODEL_PATH or not Path(MODEL_PATH).exists() or not Path(MODEL_PATH).is_file():
        logger.error(f"❌ MODEL_PATH is invalid or is a directory: {MODEL_PATH}")
        logger.error(f"   Must point to a .pt file.")
        sys.exit(1)
        
    logger.info(f"      ✅ MODEL_PATH: {MODEL_PATH}")

    # --- Step 3: Ray Initialization ---
    try:
        import ray
        from ray import serve
        
        if ray.is_initialized():
            ray.shutdown()
        
        # --- Workaround: run sync handlers in threadpool to avoid proxy async-generator GeneratorExit logs.
        # This reduces noisy "async generator ignored GeneratorExit" messages from Ray Serve proxy
        # on certain client disconnects. It's a safe opt-in for development environments.
        os.environ.setdefault("RAY_SERVE_RUN_SYNC_IN_THREADPOOL", "1")

        # Raise the log level for Ray Serve proxy internals to reduce noisy stack traces.
        # We'll silence the specific proxy logger and the broader ray.serve logger.
        try:
            logging.getLogger("ray.serve").setLevel(logging.WARNING)
            logging.getLogger("ray.serve._private.proxy").setLevel(logging.WARNING)
            logging.getLogger("proxy").setLevel(logging.WARNING)
        except Exception:
            # Best-effort; don't fail startup if loggers aren't present
            pass
        
        # Set environment variables for Ray workers before initialization
        os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # Bypass PaddleOCR connectivity checks
        
        ray.init(
            num_gpus=1 if USE_GPU else 0, 
            ignore_reinit_error=True,
            runtime_env={
                "env_vars": {
                    "DISABLE_MODEL_SOURCE_CHECK": "True"
                }
            }
        )
        
    except ImportError:
        logger.error("❌ Ray not installed. Run: pip install 'ray[serve]'")
        sys.exit(1)
    
    # --- Step 4: Import Application Builder ---
    try:
        from inference_graph import build_app
    except ImportError as e:
        logger.error(f"❌ Failed to import build_app from inference_graph: {e}", exc_info=True)
        sys.exit(1)
    
    # --- Step 5: Deploy Application ---
    try:
        # Ensure Serve listens on the configured host/port via env vars
        os.environ['RAY_SERVE_HTTP_HOST'] = '0.0.0.0'
        os.environ['RAY_SERVE_HTTP_PORT'] = str(BACKEND_PORT)

        # Cast serve to Any to avoid Pylance errors on .run()
        cast(Any, serve).run(
            build_app(),
            blocking=True
        )
        
    except KeyboardInterrupt:
        logger.info("\n✅ Shutdown signal received.")
    except Exception as e:
        logger.error(f"❌ Application deployment failed: {e}", exc_info=True)
    finally:
        if 'ray' in locals() and ray.is_initialized():
            serve.shutdown()
            ray.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()