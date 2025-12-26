"""
Shim package for `services.hvac_domain` mapping to `services/hvac-domain` directory.
This allows Python imports to work with the hyphenated directory name.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_TARGET = _ROOT / 'hvac-domain'

# Add the target directory to Python path
if _TARGET.is_dir() and str(_TARGET) not in sys.path:
    sys.path.insert(0, str(_TARGET))

__all__ = []
