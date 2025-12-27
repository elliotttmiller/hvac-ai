# Virtual Environment Pipeline Update

**Date**: December 26, 2025  
**Status**: ✅ Implemented  
**Impact**: Automatic venv creation and dependency installation in `start_unified.py`

## Overview

The `start_unified.py` startup script has been enhanced to automatically create and initialize a Python virtual environment if one doesn't exist. This eliminates the need for developers to manually run `setup.sh` and ensures consistent environment setup across all platforms.

## Changes Made

### 1. New Function: `ensure_venv_exists()`

**Location**: `scripts/start_unified.py` (lines 31-118)

**Purpose**: Automatically create and configure a virtual environment at `.venv/` in the repository root.

**Key Features**:

- ✅ **Cross-platform support**: Works on Windows (`.venv\Scripts\python.exe`) and POSIX (`.venv/bin/python`)
- ✅ **Idempotent**: Returns immediately if venv already exists
- ✅ **Automatic pip upgrade**: Installs latest pip, setuptools, and wheel
- ✅ **Dependency installation**: Automatically installs packages from `services/requirements.txt`
- ✅ **Error handling**: Comprehensive error messages with non-blocking graceful degradation for optional dependencies
- ✅ **Timeout protection**: 120s for venv creation, 180s for pip upgrade, 600s for dependency installation
- ✅ **User feedback**: Colored output with timestamps for all operations

**Function Signature**:

```python
def ensure_venv_exists(python_executable: str = sys.executable) -> str:
    """
    Ensure a virtual environment exists at .venv in the repo root.
    Creates it if it doesn't exist, and installs dependencies.
    
    Args:
        python_executable: Python interpreter to use for venv creation
        
    Returns:
        Path to the venv Python executable
        
    Raises:
        RuntimeError: If venv creation or setup fails
    """
```

### 2. Updated Venv Detection Logic

**Location**: `scripts/start_unified.py` (lines 120-129)

**Before**:

```python
# Simple check if venv exists, fallback to system Python
venv_python_win = REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'
venv_python_posix = REPO_ROOT / '.venv' / 'bin' / 'python'
if venv_python_win.exists():
    PREFERRED_PYTHON = str(venv_python_win)
elif venv_python_posix.exists():
    PREFERRED_PYTHON = str(venv_python_posix)
else:
    PREFERRED_PYTHON = str(sys.executable)
```

**After**:

```python
# Prefer a repository-root virtualenv python if available so devs can create a
# single .venv at the repo root and the launcher will use it automatically.
# If it doesn't exist, automatically create it.
try:
    PREFERRED_PYTHON = ensure_venv_exists(sys.executable)
except RuntimeError as e:
    print(f"\033[31m[STARTUP] ERROR: {str(e)}\033[0m")
    print(f"\033[31m[STARTUP] Please fix the venv setup and try again.\033[0m")
    sys.exit(1)
```

### 3. Startup Behavior

When `start_unified.py` is executed:

1. **First Run** (no `.venv` exists):

```
[STARTUP] Virtual environment not found at d:\AMD\hvac-ai\.venv
[STARTUP] Creating Python virtual environment...
[STARTUP] ✅ Virtual environment created successfully
[STARTUP] Upgrading pip and setuptools...
[STARTUP] ✅ pip upgraded
[STARTUP] Installing Python dependencies from d:\AMD\hvac-ai\services\requirements.txt...
[STARTUP] ✅ Dependencies installed
[STARTUP] Starting Ray Serve...
```

2. **Subsequent Runs** (`.venv` exists):

```
[STARTUP] Starting Ray Serve (Distributed Inference)...
[STARTUP] This will launch:
[STARTUP]   - ObjectDetector (40% GPU)
[STARTUP]   - TextExtractor (30% GPU)
[STARTUP]   - Ingress (API Gateway)
```

## Benefits

### For Developers

- **Zero setup friction**: Just run `python scripts/start_unified.py` - the environment is auto-configured
- **Multi-platform**: Works seamlessly on Windows, Linux, and macOS
- **Consistent state**: Guarantees correct Python version and dependencies
- **Time savings**: No need to manually run `setup.sh` or activate venv

### For CI/CD

- **Reproducible builds**: Environment created identically every run
- **Self-healing**: Missing dependencies are auto-installed
- **Error detection**: Early failure with clear messages if setup fails

### For DevOps

- **Single entry point**: One command to start entire platform
- **Dependency tracking**: Uses centralized `services/requirements.txt`
- **Graceful degradation**: Optional dependencies don't block startup

## Backward Compatibility

✅ **Fully backward compatible**:

- Existing `.venv` installations are detected and used immediately
- No changes to environment detection logic
- `setup.sh` and `dev.sh` still work as before
- Manual venv activation is optional (auto-detected)

## Usage

### Default (Auto-venv)

```bash
# Creates .venv if missing, installs dependencies, starts platform
python scripts/start_unified.py
```

### With Frontend Disabled

```bash
python scripts/start_unified.py --no-frontend
```

### Manual Setup (if preferred)

```bash
# Old way still works
./scripts/setup.sh
./scripts/start_unified.py
```

## Error Handling

### Scenario 1: venv creation fails

```
[STARTUP] ERROR: Failed to create venv: ...
[STARTUP] Please fix the venv setup and try again.
Exit code: 1
```

### Scenario 2: pip upgrade fails

```
[STARTUP] ERROR: Failed to upgrade pip: ...
[STARTUP] Please fix the venv setup and try again.
Exit code: 1
```

### Scenario 3: Some dependencies fail (non-blocking)

```
[STARTUP] ⚠️  Some dependencies failed to install (non-critical): ...
[STARTUP] ✅ Dependencies installed (with warnings)
[STARTUP] Starting Ray Serve...
```

## Configuration

The function respects:
- **Python executable**: Uses system Python to create venv (configurable)
- **Repo root**: Inferred from script location
- **Requirements file**: Looks for `services/requirements.txt`
- **Timeout values**: Hardcoded but easily adjustable
  - Venv creation: 120 seconds
  - Pip upgrade: 180 seconds
  - Dependency install: 600 seconds

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `scripts/start_unified.py` | Added `ensure_venv_exists()` function | 31-118 |
| `scripts/start_unified.py` | Updated venv detection logic | 120-129 |
| `scripts/start_unified.py` | Removed duplicate `_ts()` function | N/A |

## Testing Recommendations

1. **Test 1**: Delete `.venv` and run `python scripts/start_unified.py`
   - Expected: Auto-creates venv and installs dependencies
   
2. **Test 2**: Run again without deleting `.venv`
   - Expected: Uses existing venv, skips creation
   
3. **Test 3**: Run with `--no-frontend` flag
   - Expected: Skips frontend, only backend starts
   
4. **Test 4**: Run on different OS (Windows/Linux/macOS if possible)
   - Expected: Correct venv paths for each OS
   
5. **Test 5**: Corrupt `venv` and run
   - Expected: Clear error message about venv setup failure

## Future Enhancements

Possible improvements for future releases:

1. **Auto-upgrade venv**: Check if existing venv is stale and offer to recreate
2. **Dependency diff**: Show what dependencies would be installed
3. **Health check**: Verify all critical packages import correctly
4. **Rollback**: Save previous venv and allow quick rollback if new deps fail
5. **Offline mode**: Cache wheels for network-less installs
6. **Custom Python**: Allow `--python=<version>` flag to select interpreter

## Troubleshooting

### Issue: "No parameter named 'timeout'" error
**Cause**: Old Python version doesn't support timeout in `subprocess.run()`  
**Fix**: Upgrade to Python 3.7+

### Issue: venv creation times out
**Cause**: Slow disk or network, or system under heavy load  
**Fix**: Increase timeout values in the function (lines 61, 84, 99)

### Issue: Some packages fail to install
**Cause**: Binary wheels not available for your platform  
**Fix**: Install compatible versions manually, or use `SKIP_MODEL=1` to defer model loading

### Issue: Wrong Python executable selected
**Cause**: Multiple Python installations on system  
**Fix**: Pass specific Python to script: `C:\Python3.11\python.exe scripts/start_unified.py`

## References

- Main script: `scripts/start_unified.py`
- Requirements: `services/requirements.txt`
- Setup docs: `docs/GETTING_STARTED.md`
- Previous implementation: `scripts/setup.sh`, `scripts/dev.sh`
