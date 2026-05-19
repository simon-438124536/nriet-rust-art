"""Pure Python bootstrap shim for connected-region edge collection."""

import numpy as np

from .._rust_bridge import get_rust_module


def _rust_kernel():
    try:
        rust = get_rust_module()
    except ImportError:
        return None
    return getattr(rust, "_fast_edge_finder", None)


def _can_use_rust(labels, data):
    return (
        labels.ndim == 2
        and data.ndim == 2
        and labels.dtype == np.int32
        and data.dtype == np.float32
        and labels.shape == data.shape
        and labels.flags.c_contiguous
        and data.flags.c_contiguous
    )


def _fast_edge_finder(
    labels, data, rays_wrap_around, max_gap_x, max_gap_y, total_nodes
):
    """
    Return gate indices and velocities for all edges between labeled regions.

    The implementation follows the original Cython loops so early bootstrap
    behavior is correct for small and moderate inputs, while Rust can replace
    this hot path later.
    """
    labels = np.asarray(labels)
    data = np.asarray(data)
    rays_wrap_around = bool(rays_wrap_around)
    max_gap_x = int(max_gap_x)
    max_gap_y = int(max_gap_y)

    kernel = _rust_kernel()
    if kernel is not None and _can_use_rust(labels, data):
        return kernel(
            labels,
            data,
            int(rays_wrap_around),
            max_gap_x,
            max_gap_y,
            int(total_nodes),
        )

    l_index = []
    n_index = []
    l_velo = []
    n_velo = []

    right = labels.shape[0] - 1
    bottom = labels.shape[1] - 1

    def add_edge(label, neighbor, vel, nvel):
        if neighbor == label or neighbor == 0:
            return
        l_index.append(label)
        n_index.append(neighbor)
        l_velo.append(vel)
        n_velo.append(nvel)

    for x_index in range(labels.shape[0]):
        for y_index in range(labels.shape[1]):
            label = int(labels[x_index, y_index])
            if label == 0:
                continue

            vel = float(data[x_index, y_index])

            x_check = x_index - 1
            if x_check == -1 and rays_wrap_around:
                x_check = right
            if x_check != -1:
                neighbor = int(labels[x_check, y_index])
                nvel = float(data[x_check, y_index])
                if neighbor == 0:
                    for _ in range(max_gap_x):
                        x_check -= 1
                        if x_check == -1:
                            if rays_wrap_around:
                                x_check = right
                            else:
                                break
                        neighbor = int(labels[x_check, y_index])
                        nvel = float(data[x_check, y_index])
                        if neighbor != 0:
                            break
                add_edge(label, neighbor, vel, nvel)

            x_check = x_index + 1
            if x_check == right + 1 and rays_wrap_around:
                x_check = 0
            if x_check != right + 1:
                neighbor = int(labels[x_check, y_index])
                nvel = float(data[x_check, y_index])
                if neighbor == 0:
                    for _ in range(max_gap_x):
                        x_check += 1
                        if x_check == right + 1:
                            if rays_wrap_around:
                                x_check = 0
                            else:
                                break
                        neighbor = int(labels[x_check, y_index])
                        nvel = float(data[x_check, y_index])
                        if neighbor != 0:
                            break
                add_edge(label, neighbor, vel, nvel)

            y_check = y_index - 1
            if y_check != -1:
                neighbor = int(labels[x_index, y_check])
                nvel = float(data[x_index, y_check])
                if neighbor == 0:
                    for _ in range(max_gap_y):
                        y_check -= 1
                        if y_check == -1:
                            break
                        neighbor = int(labels[x_index, y_check])
                        nvel = float(data[x_index, y_check])
                        if neighbor != 0:
                            break
                add_edge(label, neighbor, vel, nvel)

            y_check = y_index + 1
            if y_check != bottom + 1:
                neighbor = int(labels[x_index, y_check])
                nvel = float(data[x_index, y_check])
                if neighbor == 0:
                    for _ in range(max_gap_y):
                        y_check += 1
                        if y_check == bottom + 1:
                            break
                        neighbor = int(labels[x_index, y_check])
                        nvel = float(data[x_index, y_check])
                        if neighbor != 0:
                            break
                add_edge(label, neighbor, vel, nvel)

    indices = (
        np.asarray(l_index, dtype=np.int32),
        np.asarray(n_index, dtype=np.int32),
    )
    velocities = (
        np.asarray(l_velo, dtype=np.float64),
        np.asarray(n_velo, dtype=np.float64),
    )
    return indices, velocities
