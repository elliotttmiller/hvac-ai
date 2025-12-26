# HVAC AI Service

This directory contains the Python service for the HVAC AI inference pipeline.

Recommended local development workflow (single repo-level virtualenv):

1. Create a repo-level virtual environment (from the repo root):

```powershell
# Create a .venv at the project root
C:/Users/AMD/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv

# Activate it in PowerShell
. .\.venv\Scripts\Activate.ps1

# Upgrade pip and install runtime deps (example)
python -m pip install --upgrade pip setuptools wheel
python -m pip install "ray[serve]" torch torchvision torchaudio numpy opencv-python-headless aiohttp

# Optional: freeze installed deps for reproducibility
python -m pip freeze > services/hvac-ai/requirements.txt
```

2. Run the unified starter (from repo root) — the launcher prefers the repo `.venv` automatically:

```powershell
# From repository root
.\.venv\Scripts\python.exe scripts\start_unified.py
```

Notes
- Do not commit the `.venv/` directory. It is included in `.gitignore`.
- Pin versions in `services/hvac-ai/requirements.txt` for reproducible development and CI.
- If you prefer per-service venvs, adjust the workflow accordingly; the launcher will still pick up `MODEL_PATH` from `.env.local`.
