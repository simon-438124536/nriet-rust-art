"""Low-level KDP retrieval kernels.

This module mirrors the small Cython ``_kdp_proc`` surface used by
``kdp_proc.py`` while allowing the implementation to dispatch to Rust kernels
when they are registered.
"""

import numpy as np

from .._rust_bridge import get_rust_module


_INVALID_FINITE_ORDER = "Invalid finite_order"
_FLOAT64_MAX = np.finfo(np.float64).max


def _rust_kernel(name):
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, name, None)


def _validate_finite_order(finite_order):
    if finite_order != "low":
        raise ValueError(_INVALID_FINITE_ORDER)


def _float64_2d_array(name, value, *, writable=False):
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D float64 array")
    if array.dtype != np.float64:
        raise ValueError(f"{name} must be a 2D float64 array")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writeable")
    return array


def _validate_output_shape(input_array, output_array):
    if input_array.shape != output_array.shape:
        raise ValueError("output array shape must match input array shape")


def _validate_term_shape(k):
    nr, ng = k.shape
    if nr != 0 and ng != 0 and ng < 3:
        raise ValueError("lowpass_maesaka_term requires at least 3 range gates")


def _validate_jac_shape(d2kdr2):
    nr, ng = d2kdr2.shape
    if nr != 0 and ng in (1, 2, 3):
        raise ValueError("lowpass_maesaka_jac received an unsupported range gate count")


def _can_use_rust(input_array, output_array):
    return not np.may_share_memory(input_array, output_array)


def _can_use_rust_forward_reverse_phidp(k, phi_near, phi_far):
    if not (
        type(k) is np.ndarray
        and type(phi_near) is np.ndarray
        and type(phi_far) is np.ndarray
        and k.ndim == 2
        and phi_near.ndim == 1
        and phi_far.ndim == 1
        and k.dtype == np.float64
        and phi_near.dtype == np.float64
        and phi_far.dtype == np.float64
        and k.flags.c_contiguous
        and phi_near.flags.c_contiguous
        and phi_far.flags.c_contiguous
        and phi_near.shape == (k.shape[0],)
        and phi_far.shape == (k.shape[0],)
    ):
        return False

    # Keep NumPy's overflow-warning behavior for extreme finite inputs.
    ng = max(k.shape[1], 1)
    k_limit = np.sqrt(_FLOAT64_MAX / (4.0 * ng))
    finite_k = k[np.isfinite(k)]
    if finite_k.size and np.max(np.abs(finite_k)) > k_limit:
        return False

    boundary_limit = _FLOAT64_MAX / 4.0
    for boundary in (phi_near, phi_far):
        finite_boundary = boundary[np.isfinite(boundary)]
        if finite_boundary.size and np.max(np.abs(finite_boundary)) > boundary_limit:
            return False

    return True


def lowpass_maesaka_term(k, dr, finite_order, d2kdr2):
    """
    Compute the Maesaka low-pass filter term in place.

    ``d2kdr2`` is updated in place and ``None`` is returned, matching the Py-ART
    Cython oracle.
    """

    _validate_finite_order(finite_order)
    k = _float64_2d_array("k", k)
    d2kdr2 = _float64_2d_array("d2kdr2", d2kdr2, writable=True)
    _validate_output_shape(k, d2kdr2)
    _validate_term_shape(k)

    kernel = _rust_kernel("lowpass_maesaka_term")
    if kernel is not None and _can_use_rust(k, d2kdr2):
        kernel(k, float(dr), finite_order, d2kdr2)
        return None

    dr2 = float(dr) ** 2.0
    nr, ng = k.shape
    for r in range(nr):
        for g in range(ng):
            if g > 0 and g < ng - 1:
                d2kdr2[r, g] = (k[r, g + 1] - 2.0 * k[r, g] + k[r, g - 1]) / dr2
            elif g == 0:
                d2kdr2[r, g] = (k[r, g] - 2.0 * k[r, g + 1] + k[r, g + 2]) / dr2
            else:
                d2kdr2[r, g] = (k[r, g] - 2.0 * k[r, g - 1] + k[r, g - 2]) / dr2
    return None


def forward_reverse_phidp(k, bcs):
    """
    Compute Maesaka forward and reverse propagation differential phases.

    This mirrors the NumPy helper in ``kdp_proc.py`` and dispatches only dense
    float64 arrays to Rust. All other inputs keep the Python oracle path.
    """

    _, ng = k.shape
    phi_near, phi_far = bcs

    kernel = _rust_kernel("forward_reverse_phidp")
    if kernel is not None and _can_use_rust_forward_reverse_phidp(
        k, phi_near, phi_far
    ):
        return kernel(k, phi_near, phi_far)

    phi_f = np.zeros_like(k, subok=False)
    phi_f[:, 1:] = np.cumsum(k[:, :-1] ** 2, axis=1)
    phidp_f = phi_f + phi_near[:, np.newaxis].repeat(ng, axis=1)

    phi_r = np.zeros_like(k, subok=False)
    phi_r[:, :-1] = np.cumsum(k[:, :0:-1] ** 2, axis=1)[:, ::-1]
    phidp_r = phi_far[:, np.newaxis].repeat(ng, axis=1) - phi_r

    return phidp_f, phidp_r


def lowpass_maesaka_jac(d2kdr2, dr, Clpf, finite_order, dJlpfdk):
    """
    Compute the Maesaka low-pass filter Jacobian in place.

    ``dJlpfdk`` is updated in place and ``None`` is returned, matching the
    Py-ART Cython oracle.
    """

    _validate_finite_order(finite_order)
    d2kdr2 = _float64_2d_array("d2kdr2", d2kdr2)
    dJlpfdk = _float64_2d_array("dJlpfdk", dJlpfdk, writable=True)
    _validate_output_shape(d2kdr2, dJlpfdk)
    _validate_jac_shape(d2kdr2)

    kernel = _rust_kernel("lowpass_maesaka_jac")
    if kernel is not None and _can_use_rust(d2kdr2, dJlpfdk):
        kernel(d2kdr2, float(dr), float(Clpf), finite_order, dJlpfdk)
        return None

    dr2 = float(dr) ** 2.0
    scale = float(Clpf) / dr2
    nr, ng = d2kdr2.shape
    for r in range(nr):
        for g in range(ng):
            if g > 2 and g < ng - 3:
                dJlpfdk[r, g] = scale * (
                    d2kdr2[r, g - 1] - 2.0 * d2kdr2[r, g] + d2kdr2[r, g + 1]
                )
            elif g == 2:
                dJlpfdk[r, g] = scale * (
                    d2kdr2[r, g - 2]
                    + d2kdr2[r, g - 1]
                    - 2.0 * d2kdr2[r, g]
                    + d2kdr2[r, g + 1]
                )
            elif g == 1:
                dJlpfdk[r, g] = scale * (
                    d2kdr2[r, g + 1]
                    - 2.0 * d2kdr2[r, g]
                    - 2.0 * d2kdr2[r, g - 1]
                )
            elif g == 0:
                dJlpfdk[r, g] = scale * (d2kdr2[r, g] + d2kdr2[r, g + 1])
            elif g == ng - 3:
                dJlpfdk[r, g] = scale * (
                    d2kdr2[r, g + 2]
                    + d2kdr2[r, g + 1]
                    - 2.0 * d2kdr2[r, g]
                    + d2kdr2[r, g - 1]
                )
            elif g == ng - 2:
                dJlpfdk[r, g] = scale * (
                    d2kdr2[r, g - 1]
                    - 2.0 * d2kdr2[r, g]
                    - 2.0 * d2kdr2[r, g + 1]
                )
            else:
                dJlpfdk[r, g] = scale * (d2kdr2[r, g] + d2kdr2[r, g - 1])
    return None
