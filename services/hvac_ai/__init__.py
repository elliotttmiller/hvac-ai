"""
Shim package for `services.hvac_ai` mapping to `services/hvac-ai` directory.
"""
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_TARGET = os.path.join(_ROOT, 'hvac-ai')

if os.path.isdir(_TARGET):
    __path__.append(_TARGET)

__all__ = []
