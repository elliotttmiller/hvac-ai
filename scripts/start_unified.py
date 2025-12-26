"""
Unified Start Script - HVAC Cortex Infrastructure
Launches the complete AI platform in Ray Serve mode (distributed inference).

Usage:
    python scripts/start_unified.py
    python scripts/start_unified.py --no-frontend
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
PYTHON_SERVICES = REPO_ROOT / "services" / "hvac-ai"
BACKEND_SCRIPT = SCRIPT_DIR / "backend_start.py"
RAY_SERVE_SCRIPT = SCRIPT_DIR / "start_ray_serve.py"
FRONTEND_CMD = "npm run dev"
# Default HTTP port for backend health checks. Can be overridden by .env.local or
# environment variable PORT or BACKEND_PORT.
PORT = 8000

# Prefer a repository-root virtualenv python if available so devs can create a
# single .venv at the repo root and the launcher will use it automatically.
venv_python_win = REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'
venv_python_posix = REPO_ROOT / '.venv' / 'bin' / 'python'
if venv_python_win.exists():
    PREFERRED_PYTHON = str(venv_python_win)
elif venv_python_posix.exists():
    PREFERRED_PYTHON = str(venv_python_posix)
else:
    PREFERRED_PYTHON = str(sys.executable)


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
        if not process.stdout:
            return
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            clean_line = line.rstrip()
            if clean_line:
                print(f"{color_code}[{_ts()} {name}] {clean_line}\033[0m")
    except Exception as e:
        print(f"Error streaming {name}: {e}")
    finally:
        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass


def start_and_stream(command, name, color_code, env=None, cwd=None):
    """Start a process and immediately stream its output in a background thread.

    This helps capture early startup logs (important for processes that fail fast).
    It also performs a quick early-exit check and prints the tail of stdout if
    the process exits within a short window.
    """
    proc = start_process(command, name, color_code, env=env, cwd=cwd)

    t = threading.Thread(target=stream_output, args=(proc, name, color_code))
    t.daemon = True
    t.start()

    # Quick early-exit detection: if the process exits within 2s, dump any
    # remaining output to help debugging start failures.
    time.sleep(1.5)
    if proc.poll() is not None:
        try:
            # Try to read any remaining output (non-blocking since process closed)
            if proc.stdout:
                remaining = proc.stdout.read()
                if remaining:
                    for l in remaining.splitlines():
                        print(f"{color_code}[{_ts()} {name}] {l}\033[0m")
        except Exception:
            pass
        print(f"{color_code}[{_ts()} {name}] Process exited early with code {proc.returncode}\033[0m")

    return proc


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
    backend_proc = start_and_stream(
        f'"{PREFERRED_PYTHON}" "{BACKEND_SCRIPT}"',
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
    
    backend_proc = start_and_stream(
        f'"{PREFERRED_PYTHON}" "{RAY_SERVE_SCRIPT}"',
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
    # Ensure the Next.js dev server picks up the intended port (default 3000)
    frontend_port = str(frontend_env.get('FRONTEND_PORT', frontend_env.get('PORT', '3000')))
    frontend_env['PORT'] = frontend_port
    print(f"{_ts()} [STARTUP] Frontend PORT={frontend_env['PORT']}")
    
    frontend_proc = start_and_stream(
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
        '--no-frontend',
        action='store_true',
        help='Skip frontend startup (backend only)'
    )
    args = parser.parse_args()
    
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print("\033[1;36m🚀 HVAC Cortex - AI Infrastructure (Ray Serve)\033[0m")
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print(f"Mode: RAY-SERVE")
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
                    # Prefer values from .env.local for local development so
                    # project-specific overrides are applied consistently.
                    backend_env[key] = val
        except Exception as e:
            print(f"{_ts()} [STARTUP] Failed to read .env.local: {e}")

    # Print effective paths and key env vars to help debugging startup issues
    print(f"{_ts()} [STARTUP] SCRIPT_DIR={SCRIPT_DIR}")
    print(f"{_ts()} [STARTUP] REPO_ROOT={REPO_ROOT}")
    print(f"{_ts()} [STARTUP] .env.local present={env_file.exists()} ({env_file})")
    print(f"{_ts()} [STARTUP] MODEL_PATH={backend_env.get('MODEL_PATH')}")

    # Audit important env vars and apply sane defaults for local dev
    important = [
        'MODEL_PATH', 'SKIP_MODEL', 'FORCE_CPU',
        'RAY_ADDRESS', 'RAY_PORT', 'RAY_HEAD', 'RAY_USE_GPU'
    ]
    print(f"{_ts()} [STARTUP] Environment audit:")
    for k in important:
        print(f"{_ts()} [STARTUP]   {k}={backend_env.get(k)}")

    # Apply sensible defaults for missing flags in local dev
    backend_env.setdefault('SKIP_MODEL', '0')
    backend_env.setdefault('FORCE_CPU', '0')
    # Ensure Python services path is available and export PYTHONPATH for child
    if not PYTHON_SERVICES.exists():
        print(f"{_ts()} [STARTUP] WARNING: expected Python services path not found: {PYTHON_SERVICES}")
        # fall back to generic services/ directory
        alt = REPO_ROOT / 'services'
        if alt.exists():
            print(f"{_ts()} [STARTUP] Falling back to: {alt}")
            backend_env.setdefault('PYTHONPATH', str(alt))
        else:
            print(f"{_ts()} [STARTUP] ERROR: No services directory found; imports may fail.")
    else:
        backend_env.setdefault('PYTHONPATH', str(PYTHON_SERVICES.parent))

    # Allow PORT override from .env.local or environment
    base_port = PORT
    try:
        effective_port = int(backend_env.get('PORT', backend_env.get('BACKEND_PORT', base_port)))
    except Exception:
        effective_port = base_port

    # Validate MODEL_PATH presence if SKIP_MODEL is not enabled
    try:
        skip_model_flag = str(backend_env.get('SKIP_MODEL', '0')).strip() in ('1', 'true', 'True')
    except Exception:
        skip_model_flag = False

    model_path = backend_env.get('MODEL_PATH')
    if model_path:
        # Normalize Windows paths from .env (they may be quoted)
        model_path_unquoted = model_path.strip('"').strip("'")
        if not Path(model_path_unquoted).exists():
            print(f"{_ts()} [STARTUP] WARNING: MODEL_PATH does not exist: {model_path_unquoted}")
            if not skip_model_flag:
                print(f"{_ts()} [STARTUP] If you intended to skip model loading for dev, set SKIP_MODEL=1 in .env.local")
        else:
            # store normalized path back into env so child processes see a clean path
            backend_env['MODEL_PATH'] = str(Path(model_path_unquoted))
    else:
        if not skip_model_flag:
            print(f"{_ts()} [STARTUP] WARNING: MODEL_PATH is not set and SKIP_MODEL is false. Backend may fail to load model.")

    # Ensure Ray-related variables have reasonable defaults or guidance
    if not backend_env.get('RAY_ADDRESS') and not backend_env.get('RAY_HEAD'):
        print(f"{_ts()} [STARTUP] NOTE: No RAY_ADDRESS or RAY_HEAD provided; start_ray_serve.py will attempt to start a head node locally.")

        # Ensure child Python processes encode output safely on Windows consoles
        # Use replace mode so non-encodable characters don't raise UnicodeEncodeError
        backend_env.setdefault('PYTHONIOENCODING', 'utf-8:replace')
        backend_env.setdefault('PYTHONUTF8', '1')

    # Configure useful frontend/env fallbacks so Next.js dev server has correct API urls
    api_base = backend_env.get('LOCAL_API_BASE_URL') or backend_env.get('NEXT_PUBLIC_API_BASE_URL') or f"http://127.0.0.1:{effective_port}"
    backend_env.setdefault('NEXT_PUBLIC_API_BASE_URL', api_base)
    backend_env.setdefault('NEXT_PUBLIC_AI_SERVICE_URL', api_base)
    backend_env.setdefault('LOCAL_API_BASE_URL', api_base)
    backend_env.setdefault('NEXTAUTH_URL', backend_env.get('NEXTAUTH_URL', 'http://localhost:3000'))
    backend_env.setdefault('NGROK_AUTHTOKEN', backend_env.get('NGROK_AUTHTOKEN', ''))
    # If RAY_ADDRESS isn't provided, we leave it unset; start_ray_serve should
    # decide whether to launch a head node or connect to an existing cluster.

    # Validate critical script paths
    if not RAY_SERVE_SCRIPT.exists():
        print(f"{_ts()} [STARTUP] ERROR: Ray Serve script not found at: {RAY_SERVE_SCRIPT}")
        print(f"{_ts()} [STARTUP] Please ensure {RAY_SERVE_SCRIPT} exists and is executable.")
        sys.exit(2)
    
    # Start backend (Ray Serve only)
    backend_proc = start_ray_serve_backend(backend_env)
    backend_name = "AI-ENGINE"
    backend_color = "\033[35m"
    
    # Wait for backend health
    health_url = f"http://127.0.0.1:{effective_port}/health"
    print(f"\n{_ts()} [STARTUP] Waiting for backend at {health_url}...")
    if not wait_for_backend_health(health_url, timeout=90.0):
        print(f"{_ts()} [STARTUP] Backend failed health check; terminating.")
        print(f"{_ts()} [STARTUP] Tips: check the AI-ENGINE / BACKEND process output above for errors.")
        print(f"{_ts()} [STARTUP] If the backend exposes diagnostics, try: http://127.0.0.1:{effective_port}/api/v1/diagnostics")
        print(f"{_ts()} [STARTUP] Also inspect logs at: {REPO_ROOT / 'logs' / 'backend.log'} if present.")
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
    
    # Output streaming started at process launch (start_and_stream)
    
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
