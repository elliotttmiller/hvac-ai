"""
Unified Start Script - HVAC AI Platform
Launches the Ray Serve backend and Next.js frontend in parallel.
"""

import subprocess
import os
import sys
import time
import threading
from datetime import datetime
import argparse
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RAY_SERVE_SCRIPT = SCRIPT_DIR / "start_ray_serve.py"
FRONTEND_CMD = "npm run dev"

# Load environment from .env.local to get ports
shared_env = os.environ.copy()
env_file = REPO_ROOT / ".env.local"
if env_file.exists():
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    k, v = line.strip().split('=', 1)
                    shared_env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass

BACKEND_PORT = int(shared_env.get('BACKEND_PORT', '8000'))
FRONTEND_PORT = int(shared_env.get('FRONTEND_PORT', '3000'))

def _ts():
    """Formatted timestamp for logs."""
    return datetime.now().strftime("%H:%M:%S")

def get_venv_python():
    """Locates the Python executable inside the .venv directory."""
    venv_dir = REPO_ROOT / ".venv"
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists():
        print(f"[{_ts()}] [STARTUP] ✅ Using Virtual Environment: {venv_python}")
        return str(venv_python)
    else:
        print(f"[{_ts()}] [STARTUP] ⚠️  .venv not found. Using system Python.")
        return sys.executable

def start_process(command_args, name, color_code, env, cwd):
    """Starts a subprocess and returns the Popen object."""
    is_shell_cmd = isinstance(command_args, str)
    process = subprocess.Popen(
        command_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        shell=is_shell_cmd,
        env=env,
        bufsize=1,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace'
    )
    print(f"{color_code}[{name}] Process started (PID: {process.pid})\033[0m")
    return process

def stream_output(process, name, color_code):
    """Reads and prints process output line by line in a thread."""
    try:
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if line:
                    print(f"{color_code}[{_ts()} {name}] {line.rstrip()}\033[0m")
    except Exception:
        pass

def start_and_stream(command_args, name, color_code, env, cwd):
    """Starts a process and streams its output."""
    proc = start_process(command_args, name, color_code, env=env, cwd=cwd)
    thread = threading.Thread(target=stream_output, args=(proc, name, color_code))
    thread.daemon = True
    thread.start()
    return proc

def wait_for_backend_health(url: str, timeout: float = 120.0) -> bool:
    """Polls a URL until it returns a 200 OK."""
    import urllib.request
    
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[{_ts()}] [STARTUP] ✅ Backend health check passed.")
                    return True
        except Exception:
            pass
        time.sleep(2.0)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-frontend', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HVAC AI - Full Stack Launcher")
    print("=" * 60)
    
    python_exe = get_venv_python()

    # Start Backend
    backend_cmd = [python_exe, str(RAY_SERVE_SCRIPT)]
    backend_proc = start_and_stream(backend_cmd, "AI-ENGINE", "\033[35m", shared_env, REPO_ROOT)
    
    # Wait for Backend
    health_url = f"http://127.0.0.1:{BACKEND_PORT}/health"
    print(f"[INFO] Waiting for backend at {health_url}...")
    
    if not wait_for_backend_health(health_url):
        print("[ERROR] Backend failed to start. Terminating.")
        backend_proc.terminate()
        sys.exit(1)
    
    # Start Frontend
    frontend_proc = None
    if not args.no_frontend:
        frontend_env = shared_env.copy()
        frontend_env['PORT'] = str(FRONTEND_PORT)
        frontend_proc = start_and_stream(FRONTEND_CMD, "UI-CLIENT", "\033[32m", frontend_env, REPO_ROOT)
    
    print("\n[INFO] Platform Running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            if backend_proc.poll() is not None:
                print(f"[ERROR] Backend exited unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping platform...")
    finally:
        for proc in [backend_proc, frontend_proc]:
            if proc and proc.poll() is None:
                if os.name == 'nt':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(proc.pid)])
                else:
                    proc.terminate()
        print("[INFO] Shutdown complete.")

if __name__ == "__main__":
    main()