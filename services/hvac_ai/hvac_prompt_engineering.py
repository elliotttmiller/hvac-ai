"""
Wrapper shim for `services.hvac_ai.hvac_prompt_engineering` re-exporting
the implementation from `services/hvac-ai/hvac_prompt_engineering.py`.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(__file__)
_SRC = os.path.abspath(os.path.join(_HERE, '..', 'hvac-ai', 'hvac_prompt_engineering.py'))

if os.path.isfile(_SRC):
    spec = importlib.util.spec_from_file_location(__name__ + "_impl", _SRC)
    _mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = _mod
    spec.loader.exec_module(_mod)

    for _name in dir(_mod):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_mod, _name)
else:
    raise ImportError(f"Could not locate hvac_prompt_engineering implementation at {_SRC}")
