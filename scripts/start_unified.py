"""
Unified Start Script - HVAC Cortex Infrastructure
Launches the complete AI platform in Ray Serve mode (distributed inference).
Automatically detects and uses the local .venv environment.
"""

import subprocess
import os
import sys
import time
import threading
from datetime import datetime
import argparse
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PYTHON_SERVICES = REPO_ROOT / "services" / "hvac-ai"
RAY_SERVE_SCRIPT = SCRIPT_DIR / "start_ray_serve.py"
FRONTEND_CMD = "npm run dev"
PORT = 8000

def _ts():
    """Get formatted timestamp."""
    return datetime.now().strftime("%H:%M:%S")

def get_venv_python():
    """
    Locate the Python executable inside the .venv directory.
    Returns the path to the venv python if found, else returns system python.
    """
    venv_dir = REPO_ROOT / ".venv"
    
    if os.name == "nt":  # Windows
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:  # Linux / Mac
        venv_python = venv_dir / "bin" / "python"

    if venv_python.exists():
        print(f"[{_ts()}] [STARTUP] ✅ Using Virtual Environment: {venv_python}")
        return str(venv_python)
    else:
        print(f"[{_ts()}] [STARTUP] ⚠️  .venv not found. Using system Python: {sys.executable}")
        return sys.executable

def start_process(command_args, name, color_code, env=None, cwd=None):
    """
    Start a subprocess and return the Popen object.
    Accepts command as a list of arguments (safer/cleaner).
    """
    if env is None:
        env = os.environ.copy()
    
    # Force unbuffered output for Python
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    
    # If command is a string, shell=True. If list, shell=False (better for Windows)
    use_shell = isinstance(command_args, str)
    
    process = subprocess.Popen(
        command_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd or os.getcwd(),
        shell=use_shell,
        env=env,
        bufsize=1,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace'
    )
    print(f"{color_code}[{name}] Process started (PID: {process.pid})\033[0m")
    return process

def stream_output(process, name, color_code):
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

def start_and_stream(command_args, name, color_code, env=None, cwd=None):
    proc = start_process(command_args, name, color_code, env=env, cwd=cwd)
    t = threading.Thread(target=stream_output, args=(proc, name, color_code))
    t.daemon = True
    t.start()
    return proc

def wait_for_backend_health(url: str, timeout: float = 60.0) -> bool:
    import urllib.request
    import urllib.error
    
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[{_ts()}] [STARTUP] Backend health OK")
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-frontend', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HVAC Cortex - AI Infrastructure Launcher")
    print("=" * 60)
    
    # 0. Clean up any leftover processes on ports 8000 and 3000
    print(f"[{_ts()}] [STARTUP] Cleaning up any leftover processes...")
    if os.name == 'nt':  # Windows
        # Kill any process using port 8000 (Ray Serve)
        try:
            subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                timeout=5
            )
            # Note: On Windows, we'd need to parse netstat output to kill specific ports
            # For now, Ray will handle port cleanup on restart
        except Exception:
            pass
    else:  # Linux/Mac
        try:
            subprocess.run(['fuser', '-k', '8000/tcp'], capture_output=True, timeout=5)
            subprocess.run(['fuser', '-k', '3000/tcp'], capture_output=True, timeout=5)
        except Exception:
            pass
    
    # 1. Determine Python Interpreter (Venv or System)
    python_exe = get_venv_python()

    # Load Environment
    backend_env = os.environ.copy()
    env_file = REPO_ROOT / ".env.local"
    
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        backend_env[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass

    backend_env["PYTHONPATH"] = str(PYTHON_SERVICES)
    
    # Check Model Path
    model_path = backend_env.get('MODEL_PATH')
    if model_path:
        print(f"[INFO] Model Path: {model_path}")
    else:
        print("[WARN] MODEL_PATH not set in .env.local")

    # Start Backend (Using Venv Python)
    # We pass a LIST of arguments to avoid shell parsing issues on Windows
    backend_cmd = [python_exe, str(RAY_SERVE_SCRIPT)]
    
    backend_proc = start_and_stream(
        backend_cmd,
        "AI-ENGINE",
        "\033[35m", # Magenta
        backend_env,
        cwd=REPO_ROOT
    )
    
    # Wait for Health
    health_url = f"http://127.0.0.1:{PORT}/health"
    print(f"[INFO] Waiting for backend at {health_url}...")
    
    if not wait_for_backend_health(health_url, timeout=90.0):
        print("[ERROR] Backend failed to start. Terminating.")
        backend_proc.terminate()
        sys.exit(1)
    
    # Start Frontend
    frontend_proc = None
    if not args.no_frontend:
        print("[INFO] Starting Frontend...")
        frontend_proc = start_and_stream(
            FRONTEND_CMD, # Keep as string for shell execution (npm needs shell)
            "UI-CLIENT",
            "\033[32m", # Green
            backend_env,
            cwd=REPO_ROOT
        )
    
    print("\n[INFO] Platform Running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print(f"[ERROR] Backend exited with code {backend_proc.returncode}")
                break
    except KeyboardInterrupt:
        print("\n[INFO] Stopping platform...")
    finally:
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
        print("[INFO] Shutdown complete.")

if __name__ == "__main__":
    main()