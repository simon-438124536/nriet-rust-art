"""Compatibility shim for Py-ART's bundled SciPy ``cKDTree`` copy."""

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - exercised only without SciPy.
    class cKDTree:
        """Placeholder used when SciPy is unavailable."""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "pyart.map.ckdtree.cKDTree requires scipy.spatial.cKDTree "
                "until the Rust/Python replacement is implemented."
            ) from exc


__all__ = ["cKDTree"]
