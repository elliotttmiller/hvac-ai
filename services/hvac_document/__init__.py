"""
Shim package for `services.hvac_document` mapping to `services/hvac-document` directory.
"""
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_TARGET = os.path.join(_ROOT, 'hvac-document')

if os.path.isdir(_TARGET):
    __path__.append(_TARGET)

__all__ = []
