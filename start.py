import subprocess
import os

def start_frontend():
    """Start the Next.js frontend server with verbose logging."""
    frontend_command = "npm run dev"
    env = os.environ.copy()
    env["NODE_OPTIONS"] = "--trace-warnings --trace-deprecation"
    frontend_process = subprocess.Popen(frontend_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd(), shell=True, env=env)
    print("⚛️  Frontend server started on http://localhost:3000 with verbose logging enabled.")
    return frontend_process

def main():
    print("🚀 Starting HVAC AI Frontend")
    print("============================")
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