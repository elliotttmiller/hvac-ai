"""
Unified Start Script - HVAC Cortex Infrastructure
Launches the complete AI platform with distributed inference.

Modes:
    --ray-serve: Use Ray Serve for distributed inference (recommended)
    --legacy: Use legacy FastAPI backend (default for backward compatibility)

Usage:
    python scripts/start_unified.py --ray-serve
    python scripts/start_unified.py --legacy
"""

import subprocess
import os
import sys
import time
import threading
from datetime import datetime
import json
import argparse
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PYTHON_SERVICES = REPO_ROOT / "python-services"
BACKEND_SCRIPT = SCRIPT_DIR / "backend_start.py"
RAY_SERVE_SCRIPT = SCRIPT_DIR / "start_ray_serve.py"
FRONTEND_CMD = "npm run dev"
PORT = 8000


def _ts():
    """Get formatted timestamp."""
    return datetime.now().strftime("%H:%M:%S")


def start_process(command, name, color_code, env=None, cwd=None):
    """
    Start a subprocess and return the Popen object.
    
    Args:
        command: Command string to execute
        name: Display name for the process
        color_code: ANSI color code for output
        env: Environment variables
        cwd: Working directory
    """
    if env is None:
        env = os.environ.copy()
    
    # Force unbuffered output for Python to ensure logs appear immediately
    if "python" in command:
        env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd or os.getcwd(),
        shell=True,
        env=env,
        bufsize=1,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace'
    )
    print(f"{color_code}[{name}] Process started (PID: {process.pid})\033[0m")
    return process


def stream_output(process, name, color_code):
    """
    Stream output from a process with colored prefix.
    Runs in a separate thread.
    """
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            clean_line = line.rstrip()
            if clean_line:
                print(f"{color_code}[{_ts()} {name}] {clean_line}\033[0m")
    except Exception as e:
        print(f"Error streaming {name}: {e}")
    finally:
        process.stdout.close()


def wait_for_backend_health(url: str, timeout: float = 60.0) -> bool:
    """
    Wait for backend to become healthy.
    
    Args:
        url: Health check URL
        timeout: Timeout in seconds
        
    Returns:
        True if backend is healthy, False otherwise
    """
    import urllib.request
    import urllib.error
    
    deadline = time.time() + timeout
    
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = resp.read().decode('utf-8')
                try:
                    j = json.loads(data)
                    status = j.get('status')
                    if status in ('healthy', 'model_not_loaded'):
                        print(f"{_ts()} [STARTUP] Backend health OK: {status}")
                        return True
                except Exception:
                    # Non-JSON response but server responded
                    print(f"{_ts()} [STARTUP] Backend responded (non-JSON), assuming up")
                    return True
        except (urllib.error.URLError, Exception):
            pass
        time.sleep(0.5)
    
    return False


def start_legacy_backend(env):
    """Start the legacy FastAPI backend."""
    print(f"\n{_ts()} [STARTUP] Starting Legacy FastAPI Backend...")
    backend_proc = start_process(
        f"{sys.executable} {BACKEND_SCRIPT}",
        "BACKEND",
        "\033[36m",  # Cyan
        env,
        cwd=SCRIPT_DIR
    )
    return backend_proc


def start_ray_serve_backend(env):
    """Start Ray Serve distributed inference backend."""
    print(f"\n{_ts()} [STARTUP] Starting Ray Serve (Distributed Inference)...")
    print(f"{_ts()} [STARTUP] This will launch:")
    print(f"{_ts()} [STARTUP]   - ObjectDetector (40% GPU)")
    print(f"{_ts()} [STARTUP]   - TextExtractor (30% GPU)")
    print(f"{_ts()} [STARTUP]   - Ingress (API Gateway)")
    
    backend_proc = start_process(
        f"{sys.executable} {RAY_SERVE_SCRIPT}",
        "AI-ENGINE",
        "\033[35m",  # Magenta
        env,
        cwd=REPO_ROOT
    )
    return backend_proc


def start_frontend(env):
    """Start the Next.js frontend."""
    print(f"\n{_ts()} [STARTUP] Starting Next.js Frontend...")
    frontend_env = env.copy()
    frontend_env["NODE_OPTIONS"] = "--trace-warnings"
    
    frontend_proc = start_process(
        FRONTEND_CMD,
        "UI-CLIENT",
        "\033[32m",  # Green
        frontend_env,
        cwd=REPO_ROOT
    )
    return frontend_proc


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HVAC AI Platform - Unified Launcher"
    )
    parser.add_argument(
        '--mode',
        choices=['ray-serve', 'legacy'],
        default='legacy',
        help='Backend mode: ray-serve (distributed) or legacy (FastAPI)'
    )
    parser.add_argument(
        '--no-frontend',
        action='store_true',
        help='Skip frontend startup (backend only)'
    )
    args = parser.parse_args()
    
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print("\033[1;36m🚀 HVAC Cortex - AI Infrastructure\033[0m")
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print(f"Mode: {args.mode.upper()}")
    print(f"Frontend: {'Disabled' if args.no_frontend else 'Enabled'}")
    print("\033[1;36m" + "=" * 60 + "\033[0m\n")
    
    # Load environment
    backend_env = os.environ.copy()
    env_file = REPO_ROOT / ".env.local"
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key not in backend_env:
                        backend_env[key] = val
        except Exception as e:
            print(f"{_ts()} [STARTUP] Failed to read .env.local: {e}")
    
    # Start backend
    if args.mode == 'ray-serve':
        backend_proc = start_ray_serve_backend(backend_env)
        backend_name = "AI-ENGINE"
        backend_color = "\033[35m"
    else:
        backend_proc = start_legacy_backend(backend_env)
        backend_name = "BACKEND"
        backend_color = "\033[36m"
    
    # Wait for backend health
    health_url = f"http://127.0.0.1:{PORT}/health"
    print(f"\n{_ts()} [STARTUP] Waiting for backend at {health_url}...")
    if not wait_for_backend_health(health_url, timeout=90.0):
        print(f"{_ts()} [STARTUP] Backend failed health check; terminating.")
        try:
            if backend_proc.poll() is None:
                backend_proc.terminate()
        except Exception:
            pass
        sys.exit(1)
    
    # Start frontend if requested
    frontend_proc = None
    if not args.no_frontend:
        frontend_proc = start_frontend(backend_env)
    
    # Stream outputs
    t_backend = threading.Thread(
        target=stream_output,
        args=(backend_proc, backend_name, backend_color)
    )
    t_backend.daemon = True
    t_backend.start()
    
    if frontend_proc:
        t_frontend = threading.Thread(
            target=stream_output,
            args=(frontend_proc, "UI-CLIENT", "\033[32m")
        )
        t_frontend.daemon = True
        t_frontend.start()
    
    print("\n\033[1;33m⚡ Platform running. Press Ctrl+C to stop all services.\033[0m\n")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print(f"\n❌ Backend exited with code {backend_proc.returncode}")
                break
            if frontend_proc and frontend_proc.poll() is not None:
                print(f"\n❌ Frontend exited with code {frontend_proc.returncode}")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping platform...")
    finally:
        print("Terminating processes...")
        
        if frontend_proc and frontend_proc.poll() is None:
            if os.name == 'nt':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(frontend_proc.pid)])
            else:
                frontend_proc.terminate()
        
        if backend_proc.poll() is None:
            backend_proc.terminate()
        
        print("✅ Shutdown complete.")


if __name__ == "__main__":
    main()
