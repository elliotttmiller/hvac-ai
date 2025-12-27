"""
Ray Serve Application Launcher - HVAC AI Platform
Implements the official Ray Serve "Application Builder" pattern.

This script:
1. Sets up paths and environment
2. Initializes Ray cluster with GPU support
3. Imports and builds the application graph
4. Runs the application using serve.run()

This is the clean, official pattern recommended by the Ray team.
"""

import sys
import os
import logging
from pathlib import Path

# --- Configuration & Paths ---
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SERVICES_ROOT = REPO_ROOT / "services"
SERVICES_AI_DIR = SERVICES_ROOT / "hvac-ai"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [RayServe] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for Ray Serve launcher."""
    
    logger.info("=" * 70)
    logger.info("HVAC AI Platform - Ray Serve Launcher")
    logger.info("=" * 70)
    
    # --- Step 1: Path Setup ---
    logger.info("[1/5] Setting up Python paths...")
    sys.path.insert(0, str(SERVICES_AI_DIR))  # hvac-ai module
    sys.path.insert(0, str(SERVICES_ROOT))     # For hvac-domain imports
    logger.info(f"      ✅ Added {SERVICES_AI_DIR}")
    logger.info(f"      ✅ Added {SERVICES_ROOT}")
    
    # --- Step 2: Environment Validation ---
    logger.info("[2/5] Loading environment variables...")
    BACKEND_PORT = int(os.getenv('BACKEND_PORT', '8000'))
    RAY_ADDRESS = os.getenv('RAY_ADDRESS', '')
    USE_GPU = os.getenv('RAY_USE_GPU', '1') == '1'
    MODEL_PATH = os.getenv('MODEL_PATH')
    
    if not MODEL_PATH:
        logger.error("❌ MODEL_PATH environment variable not set")
        sys.exit(1)
    
    logger.info(f"      ✅ BACKEND_PORT: {BACKEND_PORT}")
    logger.info(f"      ✅ RAY_USE_GPU: {USE_GPU}")
    logger.info(f"      ✅ MODEL_PATH: {MODEL_PATH}")
    
    # --- Step 3: Ray Initialization ---
    logger.info("[3/5] Initializing Ray cluster...")
    try:
        import ray
        from ray import serve
        
        # Shutdown any existing Ray instance
        if ray.is_initialized():
            ray.shutdown()
        
        # Initialize Ray with GPU support
        ray_init_kwargs = {
            'dashboard_host': '127.0.0.1',
            'ignore_reinit_error': True,
        }
        
        if USE_GPU:
            ray_init_kwargs['num_gpus'] = 1
        
        ray.init(**ray_init_kwargs)
        logger.info("      ✅ Ray cluster initialized")
        logger.info(f"         Dashboard: http://127.0.0.1:8265")
        
    except ImportError:
        logger.error("❌ Ray not installed. Run: pip install 'ray[serve]'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to initialize Ray: {e}", exc_info=True)
        sys.exit(1)
    
    # --- Step 4: Import Application Builder ---
    logger.info("[4/5] Importing application builder...")
    try:
        from inference_graph import build_app
        logger.info("      ✅ Successfully imported build_app from inference_graph")
    except ImportError as e:
        logger.error(f"❌ Failed to import inference_graph.build_app: {e}", exc_info=True)
        sys.exit(1)
    
    # --- Step 5: Deploy Application ---
    logger.info("[5/5] Deploying application with Ray Serve...")
    try:
        logger.info("=" * 70)
        logger.info("✅ APPLICATION DEPLOYMENT DETAILS")
        logger.info("=" * 70)
        logger.info(f"Backend URL:     http://0.0.0.0:{BACKEND_PORT}")
        logger.info(f"Health Endpoint: http://127.0.0.1:{BACKEND_PORT}/health")
        logger.info(f"Analyze Endpoint: http://127.0.0.1:{BACKEND_PORT}/api/hvac/analyze")
        logger.info(f"Ray Dashboard:   http://127.0.0.1:8265")
        logger.info("=" * 70)
        logger.info("Starting Ray Serve application... (Press Ctrl+C to stop)")
        logger.info("=" * 70)
        
        # Run the application using the official serve.run() API
        serve.run(
            build_app(),
            blocking=True  # This keeps the process alive
        )
        
    except KeyboardInterrupt:
        logger.info("\n✅ Shutdown signal received, cleaning up...")
        if ray.is_initialized():
            ray.shutdown()
        logger.info("✅ Ray cluster shut down cleanly")
    except Exception as e:
        logger.error(f"❌ Application deployment failed: {e}", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)