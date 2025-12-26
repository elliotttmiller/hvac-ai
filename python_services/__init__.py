"""
Compatibility shim package `python_services` to mirror the existing
`python-services` directory (dash) for imports and static analysis.

Some tests and examples import `python_services.*` (underscore). The real
package folder is `python-services` (with a dash), which is not a valid
Python package name. This shim exposes the `python-services` contents under
the `python_services` package by extending this package's __path__.

This is intentionally lightweight and only used to aid imports and tools.
"""
import os

# Compute path to the real `python-services` directory (sibling of this shim)
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_ALT = os.path.join(_ROOT, 'services', 'hvac-analysis')

if os.path.isdir(_ALT):
    # Allow subpackages (e.g., python_services.core) to be resolved from
    # the python-services directory which contains 'core/', 'tests/', etc.
    __path__.append(_ALT)

__all__ = []
