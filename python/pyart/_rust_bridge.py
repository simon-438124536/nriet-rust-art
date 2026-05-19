"""Helpers for optional access to private Rust kernels."""

from importlib import import_module


def get_rust_module():
    """Import and return ``pyart._rust``."""
    return import_module("pyart._rust")


def has_rust_module():
    """Return True when ``pyart._rust`` is importable."""
    try:
        get_rust_module()
    except ImportError:
        return False
    return True
