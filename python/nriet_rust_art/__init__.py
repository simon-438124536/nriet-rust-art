"""Python shell for the NRIET Rust ART backend."""

try:
    from ._core import __version__, rust_backend_ready, sum_f64, version
except ImportError:  # pragma: no cover - used before the native extension is built
    __version__ = "0.1.0"

    def rust_backend_ready() -> bool:
        return False

    def version() -> str:
        return __version__

    def sum_f64(values) -> float:
        return float(sum(values))

__all__ = ["__version__", "rust_backend_ready", "sum_f64", "version"]
