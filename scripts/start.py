import subprocess
import os
import sys
import time
import threading
from datetime import datetime
import json

# Configuration
# Resolve paths relative to this script so the launcher works when invoked
# from the project root (python scripts/start.py) or from inside scripts/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_SCRIPT = os.path.join(SCRIPT_DIR, "backend_start.py")
FRONTEND_CMD = "npm run dev"
# Backend port (should match backend_start.py)
PORT = 8000

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def start_process(command, name, color_code, env=None, cwd=None):
    """
    Starts a subprocess and returns the Popen object.
    """
    if env is None:
        env = os.environ.copy()
        
    # Force unbuffered output for Python backend to ensure logs appear immediately
    if "python" in command:
        env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge stderr into stdout
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
    Reads from a process stdout line by line and prints with a prefix.
    Runs in a separate thread.
    """
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            # Clean up line
            clean_line = line.rstrip()
            if clean_line:
                print(f"{color_code}[{_ts()} {name}] {clean_line}\033[0m")
    except Exception as e:
        print(f"Error streaming {name}: {e}")
    finally:
        process.stdout.close()

def main():
    print("\033[1;36m🚀 Starting HVAC AI Platform (Full Stack)\033[0m")
    print("=========================================")

    # 1. Start Backend (Python/FastAPI)
    # Cyan Color for Backend
    backend_env = os.environ.copy()
    # Start backend from the scripts directory so relative imports / file paths
    # inside the backend script behave as expected.
    backend_proc = start_process(
        f"{sys.executable} {BACKEND_SCRIPT}", 
        "BACKEND", 
        "\033[36m", 
        backend_env,
        cwd=SCRIPT_DIR,
    )

    # 2. Start Frontend (Next.js)
    # Green Color for Frontend
    frontend_env = os.environ.copy()
    frontend_env["NODE_OPTIONS"] = "--trace-warnings"
    # Run frontend from the repository root (not the scripts/ folder)
    frontend_env_cwd = REPO_ROOT
    frontend_proc = start_process(
        FRONTEND_CMD, 
        "FRONTEND", 
        "\033[32m", 
        frontend_env,
        cwd=REPO_ROOT,
    )

    # Wait for backend health before starting the frontend
    def wait_for_backend_health(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
        import time
        import urllib.request
        import urllib.error

        deadline = time.time() + timeout
        # If SKIP_MODEL is set in env, accept either 'healthy' or 'model_not_loaded' as up
        accept_model_not_loaded = os.environ.get('SKIP_MODEL', '0') == '1'

        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    data = resp.read().decode('utf-8')
                    try:
                        j = json.loads(data)
                        status = j.get('status')
                        if status == 'healthy' or (accept_model_not_loaded and status == 'model_not_loaded'):
                            print(f"{_ts()} [STARTUP] Backend health OK: {status}")
                            return True
                        else:
                            print(f"{_ts()} [STARTUP] Backend reported status={status}; waiting...")
                    except Exception:
                        # non-json response, but server responded
                        print(f"{_ts()} [STARTUP] Backend responded (non-json), assuming up")
                        return True
            except urllib.error.URLError:
                pass
            except Exception as e:
                print(f"{_ts()} [STARTUP] Health check error: {e}")
            time.sleep(interval)
        return False

    health_url = f"http://127.0.0.1:{PORT}/health"
    print(f"{_ts()} [STARTUP] Waiting up to 30s for backend at {health_url}...")
    ok = wait_for_backend_health(health_url, timeout=30.0, interval=0.5)
    if not ok:
        print(f"{_ts()} [STARTUP] Backend failed health check; terminating.")
        # terminate backend and exit
        try:
            if backend_proc.poll() is None:
                backend_proc.terminate()
        except Exception:
            pass
        return

    # 3. Create Threads to stream output simultaneously
    t_backend = threading.Thread(target=stream_output, args=(backend_proc, "BACKEND", "\033[36m"))
    t_frontend = threading.Thread(target=stream_output, args=(frontend_proc, "FRONTEND", "\033[32m"))

    t_backend.daemon = True
    t_frontend.daemon = True

    t_backend.start()
    t_frontend.start()

    print("\033[1;33m⚡ Platform running. Press Ctrl+C to stop all services.\033[0m\n")

    try:
        # Keep main thread alive while child processes run
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if backend_proc.poll() is not None:
                print(f"\n❌ Backend exited unexpectedly with code {backend_proc.returncode}")
                break
            if frontend_proc.poll() is not None:
                print(f"\n❌ Frontend exited unexpectedly with code {frontend_proc.returncode}")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping platform...")
    finally:
        # Cleanup
        print("Terminating processes...")
        
        # Kill Frontend
        if frontend_proc.poll() is None:
            # On Windows, taskkill is often more effective for node trees
            if os.name == 'nt':
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(frontend_proc.pid)])
            else:
                frontend_proc.terminate()
        
        # Kill Backend
        if backend_proc.poll() is None:
            backend_proc.terminate()
            
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
import subprocess
import os
import sys
import time
from datetime import datetime


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def start_frontend():
    """Start the Next.js frontend server with verbose logging and stream output to this console.

    This spawns `npm run dev` with NODE_OPTIONS set for extra traces and forwards
    stdout/stderr line-by-line to the parent process so you'll see Next.js "compiling"/"ready" logs.
    """
    frontend_command = "npm run dev"
    env = os.environ.copy()
    # Add helpful Node options (no harm if already set)
    env.setdefault("NODE_OPTIONS", "")
    env["NODE_OPTIONS"] = (env.get("NODE_OPTIONS", "") + " --trace-warnings --trace-deprecation").strip()

    # Use text mode and line buffering so we can stream logs live
    process = subprocess.Popen(
        frontend_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        shell=True,
        env=env,
        bufsize=1,
        universal_newlines=True,
    )

    print(f"⚛️  Frontend process started (pid={process.pid}) — streaming logs below")
    return process


def stream_process_output(process):
    """Read lines from process.stdout and print them with timestamps.

    Returns when the process exits and all output has been drained.
    """
    assert process.stdout is not None
    try:
        for line in iter(process.stdout.readline, ""):
            # Protect against None/empty lines
            if line is None:
                break
            # Print raw line with timestamp (strip trailing newlines to avoid double spacing)
            sys.stdout.write(f"[{_ts()}] {line.rstrip()}\n")
            sys.stdout.flush()
        # drain any remaining output
        remaining = process.stdout.read()
        if remaining:
            for l in remaining.splitlines():
                sys.stdout.write(f"[{_ts()}] {l}\n")
                sys.stdout.flush()
    except Exception as e:
        print(f"{_ts()} - Error while streaming output: {e}")


def main():
    print("🚀 Starting HVAC AI Frontend")
    print("============================")

    proc = start_frontend()

    try:
        # Stream output until the process finishes or user interrupts
        stream_process_output(proc)

        # Wait for exit code
        rc = proc.wait()
        print(f"\n{_ts()} - Frontend process exited with code {rc}")
    except KeyboardInterrupt:
        print(f"\n{_ts()} - KeyboardInterrupt received, terminating frontend (pid={proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print(f"{_ts()} - Frontend stopped.")


if __name__ == "__main__":
    main()