"""
Shim package for `services.hvac_domain` mapping to `services/hvac-domain` directory.
"""
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_TARGET = os.path.join(_ROOT, 'hvac-domain')

if os.path.isdir(_TARGET):
    __path__.append(_TARGET)

__all__ = []
