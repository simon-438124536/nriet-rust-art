"""Build-check shim for the Rust-backed Py-ART package."""


def check_build():
    """Return whether the private Rust extension can be imported."""
    try:
        import pyart._rust  # noqa: F401
    except ImportError:
        return False
    return True
