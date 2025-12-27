"""
Unified Start Script - HVAC AI Platform
Launches the Ray Serve backend and Next.js frontend in parallel.
Features aggressive process cleanup and real-time status monitoring.
"""

# --- CRITICAL OPTIMIZATION: Disable Phone-Home Checks Globally ---
import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import subprocess
import sys
import time
import threading
from datetime import datetime
import argparse
from pathlib import Path
import itertools
import urllib.request
import urllib.error
import json
import shutil

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RAY_SERVE_SCRIPT = SCRIPT_DIR / "start_ray_serve.py"
FRONTEND_CMD = "npm run dev"

# Load environment
shared_env = os.environ.copy()
# Ensure optimization flags are passed to all children
shared_env['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
shared_env['PYTHONUNBUFFERED'] = '1'

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
    return datetime.now().strftime("%H:%M:%S")

def kill_existing_processes():
    """Aggressively kills any conflicting processes to ensure a clean start."""
    print(f"[{_ts()}] [CLEANUP] 🧹 Scanning for conflicting processes...")
    
    if os.name == 'nt':  # Windows
        # 1. Kill by Port (most precise)
        for port in [BACKEND_PORT, FRONTEND_PORT, 8265]: # 8265 is Ray Dashboard
            try:
                # Find PID using port
                cmd = f'netstat -ano | findstr :{port}'
                output = subprocess.check_output(cmd, shell=True).decode()
                if output:
                    for line in output.splitlines():
                        parts = line.strip().split()
                        if len(parts) > 4:
                            pid = parts[-1]
                            subprocess.run(['taskkill', '/F', '/PID', pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass # No process found on port

        # 2. Kill by Name (cleanup zombies)
        # Be careful not to kill OURSELF (this script)
        # We skip this for 'python.exe' to avoid killing the launcher itself if run via python
        targets = ['node.exe', 'ray.exe', 'raylet.exe', 'gcs_server.exe']
        for target in targets:
            subprocess.run(['taskkill', '/F', '/IM', target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
    else:  # Linux/Mac
        subprocess.run(['pkill', '-f', 'ray'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['fuser', '-k', f'{BACKEND_PORT}/tcp'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['fuser', '-k', f'{FRONTEND_PORT}/tcp'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[{_ts()}] [CLEANUP] ✅ Environment clean.")

def get_venv_python():
    venv_dir = REPO_ROOT / ".venv"
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable

def start_process(command_args, env, cwd):
    is_shell = isinstance(command_args, str)
    return subprocess.Popen(
        command_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        shell=is_shell,
        env=env,
        bufsize=1,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace'
    )

def stream_output(process, name, color_code):
    try:
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if line:
                    clean_line = line.rstrip()
                    # Filter out noisy logs
                    if "GET /health" not in clean_line and "checking connectivity" not in clean_line.lower():
                        print(f"{color_code}[{_ts()} {name}] {clean_line}\033[0m")
    except Exception:
        pass

def wait_for_backend_health(url: str, timeout: float = 300.0) -> bool:
    start_time = time.time()
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    
    print(f"[{_ts()}] [STARTUP] 🚀 Starting backend initialization...")
    
    while (time.time() - start_time) < timeout:
        elapsed = int(time.time() - start_time)
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    if data.get("status") == "healthy":
                        sys.stdout.write(f"\r\033[K[{_ts()}] [STARTUP] ✅ Backend ready in {elapsed}s\n")
                        sys.stdout.flush()
                        return True
                    else:
                        status_msg = "Initializing..."
                else:
                    status_msg = f"HTTP {resp.status}"
        except Exception:
            status_msg = "Connecting..."

        sys.stdout.write(f"\r{next(spinner)} [{elapsed}s] {status_msg}")
        sys.stdout.flush()
        time.sleep(1.0)
    
    print(f"\n[{_ts()}] [STARTUP] ❌ Timeout waiting for backend.")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-frontend', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HVAC AI - Full Stack Launcher")
    print("=" * 60)
    
    # 1. Clean up first
    kill_existing_processes()
    
    python_exe = get_venv_python()
    print(f"[{_ts()}] [STARTUP] ✅ Using Python: {python_exe}")

    # 2. Start Backend
    backend_cmd = [python_exe, str(RAY_SERVE_SCRIPT)]
    backend_proc = start_process(backend_cmd, env=shared_env, cwd=REPO_ROOT)
    
    t_back = threading.Thread(target=stream_output, args=(backend_proc, "AI-ENGINE", "\033[35m"))
    t_back.daemon = True
    t_back.start()
    
    # 3. Wait for Health
    health_url = f"http://127.0.0.1:{BACKEND_PORT}/health"
    if not wait_for_backend_health(health_url, timeout=300.0):
        backend_proc.terminate()
        sys.exit(1)
    
    # 4. Start Frontend
    frontend_proc = None
    if not args.no_frontend:
        print(f"\n[{_ts()}] [INFO] Starting Frontend...")
        frontend_env = shared_env.copy()
        frontend_env['PORT'] = str(FRONTEND_PORT)
        frontend_proc = start_process(FRONTEND_CMD, env=frontend_env, cwd=REPO_ROOT)
        
        t_front = threading.Thread(target=stream_output, args=(frontend_proc, "UI-CLIENT", "\033[32m"))
        t_front.daemon = True
        t_front.start()
    
    print(f"\n[{_ts()}] [INFO] Platform Running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("\n[ERROR] Backend exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] [INFO] Stopping platform...")
    finally:
        # Cleanup again on exit
        if frontend_proc:
            if os.name == 'nt':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(frontend_proc.pid)])
            else:
                frontend_proc.terminate()
        
        if backend_proc:
            if os.name == 'nt':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(backend_proc.pid)])
            else:
                backend_proc.terminate()
        
        # Final sweep
        kill_existing_processes()
        print("[INFO] Shutdown complete.")

if __name__ == "__main__":
    main()