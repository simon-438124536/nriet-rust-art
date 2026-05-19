"""Private Rust-backed shim for the 3D phase unwrap extension."""

import numpy as np

from .._rust_bridge import get_rust_module


def _rust_kernel():
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, "unwrap_3d", None)


def unwrap_3d(image, mask, unwrapped_image, wrap_around):
    """3D phase unwrapping using the private Rust backend."""
    image = np.asarray(image)
    mask = np.asarray(mask)
    unwrapped_image = np.asarray(unwrapped_image)

    kernel = _rust_kernel()
    if kernel is None:
        raise NotImplementedError(
            "pyart.correct._unwrap_3d.unwrap_3d requires the C/Rust phase "
            "unwrapping backend and is not implemented by the bootstrap shim."
        )
    kernel(image, mask, unwrapped_image, wrap_around)
    return None
