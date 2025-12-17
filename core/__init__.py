"""
Compatibility shim package `core` that exposes the `python-services/core` folder
as a top-level `core` package to simplify imports like `from core.ai import ...`.

This is a small helper for local development and static analysis.
"""
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_TARGET = os.path.join(_ROOT, 'python-services', 'core')

if os.path.isdir(_TARGET):
    __path__.append(_TARGET)

__all__ = []
