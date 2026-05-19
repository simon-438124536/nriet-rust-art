"""Pure Python bootstrap shim for the Cython ``_unwrap_1d`` extension."""

import numpy as np

from .._rust_bridge import get_rust_module


def _rust_kernel():
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, "unwrap_1d", None)


def _can_use_rust(image, unwrapped_image):
    return (
        image.ndim == 1
        and unwrapped_image.ndim == 1
        and image.dtype == np.float64
        and unwrapped_image.dtype == np.float64
        and image.shape == unwrapped_image.shape
        and image.size != 0
        and image.flags.c_contiguous
        and unwrapped_image.flags.c_contiguous
        and unwrapped_image.flags.writeable
        and not np.may_share_memory(image, unwrapped_image)
    )


def unwrap_1d(image, unwrapped_image):
    """Phase unwrapping using the naive approach."""
    image = np.asarray(image)
    unwrapped_image = np.asarray(unwrapped_image)

    kernel = _rust_kernel()
    if kernel is not None and _can_use_rust(image, unwrapped_image):
        kernel(image, unwrapped_image)
        return None

    periods = 0
    unwrapped_image[0] = image[0]
    for i in range(1, image.shape[0]):
        difference = image[i] - image[i - 1]
        if difference > np.pi:
            periods -= 1
        elif difference < -np.pi:
            periods += 1
        unwrapped_image[i] = image[i] + 2 * np.pi * periods
    return None
