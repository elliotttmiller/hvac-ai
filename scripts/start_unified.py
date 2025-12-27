"""
Unified Start Script - HVAC AI Platform
Launches the Ray Serve backend and Next.js frontend in parallel.
"""

# Set global environment variables before any imports
import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # Bypass PaddleOCR connectivity checks globally

import subprocess
import sys
import time
import threading
from datetime import datetime
import argparse
from pathlib import Path
import urllib.request
import urllib.error
import json
import itertools
import shutil

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

def wait_for_backend_health(url: str, timeout: float = 300.0) -> bool:
    """Polls a URL until it returns a 200 OK with status 'healthy' and all deployments ready.
    Shows a real-time progress bar with status updates.
    """
    deadline = time.time() + timeout
    start_time = time.time()
    
    # Progress bar characters
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    bar_width = min(40, shutil.get_terminal_size().columns - 50)  # Responsive width
    
    print(f"[{_ts()}] [STARTUP] 🚀 Starting backend initialization...")
    print(f"[{_ts()}] [STARTUP] 📡 Monitoring health at {url}")
    
    last_status = ""
    last_deployments = {}
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        progress = min(elapsed / timeout, 1.0)
        
        # Create progress bar
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Spinner
        spin = next(spinner)
        
        # Time display
        elapsed_str = f"{int(elapsed)}s"
        
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    try:
                        data = json.loads(resp.read().decode('utf-8'))
                        status = data.get('status', 'unknown')
                        deployments = data.get('deployments', {})
                        
                        # Check if status changed
                        status_changed = (status != last_status or deployments != last_deployments)
                        last_status = status
                        last_deployments = deployments
                        
                        if status == 'healthy':
                            # Check that all deployments are ready
                            all_ready = all(
                                dep_status == 'ready'
                                for dep_status in deployments.values()
                            )
                            if all_ready:
                                # Clear the progress line and show success
                                print(f"\r\033[K[{_ts()}] [STARTUP] ✅ Backend fully ready - all deployments initialized in {elapsed_str}")
                                print()
                                return True
                            else:
                                # Show deployment status
                                ready_count = sum(1 for s in deployments.values() if s == 'ready')
                                total_count = len(deployments)
                                status_msg = f"Deployments: {ready_count}/{total_count} ready"
                        elif status == 'initializing':
                            status_msg = f"Backend initializing..."
                        else:
                            status_msg = f"Status: {status}"
                            
                        # Show progress if status changed or first time
                        if status_changed or elapsed % 1 < 0.1:  # Update every ~1 second
                            print(f"\r{spin} [{bar}] {elapsed_str} | {status_msg}", end="", flush=True)
                            
                    except json.JSONDecodeError:
                        status_msg = "Invalid health response"
                        print(f"\r{spin} [{bar}] {elapsed_str} | {status_msg}", end="", flush=True)
                else:
                    status_msg = f"HTTP {resp.status}"
                    print(f"\r{spin} [{bar}] {elapsed_str} | {status_msg}", end="", flush=True)
                    
        except urllib.error.URLError:
            status_msg = "Connecting..."
            print(f"\r{spin} [{bar}] {elapsed_str} | {status_msg}", end="", flush=True)
        except Exception as e:
            status_msg = f"Error: {str(e)[:20]}..."
            print(f"\r{spin} [{bar}] {elapsed_str} | {status_msg}", end="", flush=True)

        time.sleep(0.5)  # Update twice per second for smooth animation
    
    # Timeout reached
    print(f"\r\033[K[{_ts()}] [STARTUP] ❌ Backend initialization timed out after {int(elapsed)}s")
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
    
    if not wait_for_backend_health(health_url, timeout=600.0):  # 10 minutes timeout
        print("[ERROR] Backend failed to fully initialize. Terminating.")
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