import os

import numpy as np

os.environ.setdefault("PYART_QUIET", "1")

from pyart.correct import _fast_edge_finder, _unwrap_1d  # noqa: E402


def _expected_unwrap_1d(image):
    out = np.empty_like(image, dtype=np.float64)
    periods = 0
    out[0] = image[0]
    for i in range(1, image.shape[0]):
        difference = image[i] - image[i - 1]
        if difference > np.pi:
            periods -= 1
        elif difference < -np.pi:
            periods += 1
        out[i] = image[i] + 2 * np.pi * periods
    return out


def _expected_unwrap_1d_aliased(image):
    periods = 0
    image[0] = image[0]
    for i in range(1, image.shape[0]):
        difference = image[i] - image[i - 1]
        if difference > np.pi:
            periods -= 1
        elif difference < -np.pi:
            periods += 1
        image[i] = image[i] + 2 * np.pi * periods
    return image


def test_unwrap_1d_python_fallback_matches_oracle_formula(monkeypatch):
    monkeypatch.setattr(_unwrap_1d, "_rust_kernel", lambda: None)
    image = np.array([0.2, 1.1, 3.4, -2.7, -3.0, 2.9, 0.1], dtype=np.float64)
    out = np.full_like(image, -999.0)

    result = _unwrap_1d.unwrap_1d(image, out)

    assert result is None
    np.testing.assert_array_equal(out, _expected_unwrap_1d(image))


def test_unwrap_1d_uses_rust_kernel_for_oracle_compatible_arrays(monkeypatch):
    calls = []

    def rust_unwrap(image, out):
        calls.append((image.dtype, out.dtype, image.shape))
        out[:] = _expected_unwrap_1d(image)

    monkeypatch.setattr(_unwrap_1d, "_rust_kernel", lambda: rust_unwrap)
    image = np.array([0.0, 3.5, -2.9, 2.8], dtype=np.float64)
    out = np.zeros_like(image)

    _unwrap_1d.unwrap_1d(image, out)

    assert calls == [(np.dtype("float64"), np.dtype("float64"), (4,))]
    np.testing.assert_array_equal(out, _expected_unwrap_1d(image))


def test_unwrap_1d_keeps_python_path_for_aliasing(monkeypatch):
    def rust_unwrap(image, out):
        raise AssertionError("rust kernel must not run for aliased arrays")

    monkeypatch.setattr(_unwrap_1d, "_rust_kernel", lambda: rust_unwrap)
    image = np.array([0.0, 3.5, -2.9, 2.8], dtype=np.float64)
    expected = _expected_unwrap_1d_aliased(image.copy())

    _unwrap_1d.unwrap_1d(image, image)

    np.testing.assert_array_equal(image, expected)


def test_fast_edge_finder_python_fallback_matches_oracle_order(monkeypatch):
    monkeypatch.setattr(_fast_edge_finder, "_rust_kernel", lambda: None)
    labels = np.array(
        [[1, 0, 2, 0], [0, 0, 0, 3], [4, 0, 5, 0]],
        dtype=np.int32,
    )
    data = np.array(
        [[1.5, 0.0, 2.5, 0.0], [0.0, 0.0, 0.0, 3.5], [4.5, 0.0, 5.5, 0.0]],
        dtype=np.float32,
    )

    indices, velocities = _fast_edge_finder._fast_edge_finder(
        labels, data, True, 1, 1, int(np.count_nonzero(labels))
    )

    np.testing.assert_array_equal(
        indices[0], np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 5, 5, 5], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        indices[1], np.array([4, 4, 2, 5, 5, 1, 1, 1, 5, 2, 2, 4], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        velocities[0],
        np.array([1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 5.5, 5.5, 5.5]),
    )
    np.testing.assert_array_equal(
        velocities[1],
        np.array([4.5, 4.5, 2.5, 5.5, 5.5, 1.5, 1.5, 1.5, 5.5, 2.5, 2.5, 4.5]),
    )


def test_fast_edge_finder_uses_rust_kernel_for_oracle_compatible_arrays(monkeypatch):
    calls = []
    expected_indices = (
        np.array([1, 2], dtype=np.int32),
        np.array([2, 1], dtype=np.int32),
    )
    expected_velocities = (
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([2.0, 1.0], dtype=np.float64),
    )

    def rust_fast_edge(labels, data, rays_wrap_around, max_gap_x, max_gap_y, total_nodes):
        calls.append(
            (
                labels.dtype,
                data.dtype,
                bool(rays_wrap_around),
                max_gap_x,
                max_gap_y,
                total_nodes,
            )
        )
        return expected_indices, expected_velocities

    monkeypatch.setattr(_fast_edge_finder, "_rust_kernel", lambda: rust_fast_edge)
    labels = np.array([[1, 2]], dtype=np.int32)
    data = np.array([[1.0, 2.0]], dtype=np.float32)

    indices, velocities = _fast_edge_finder._fast_edge_finder(labels, data, True, 0, 0, 2)

    assert calls == [(np.dtype("int32"), np.dtype("float32"), True, 0, 0, 2)]
    np.testing.assert_array_equal(indices[0], expected_indices[0])
    np.testing.assert_array_equal(indices[1], expected_indices[1])
    np.testing.assert_array_equal(velocities[0], expected_velocities[0])
    np.testing.assert_array_equal(velocities[1], expected_velocities[1])


def test_fast_edge_finder_keeps_python_path_for_non_oracle_dtype(monkeypatch):
    def rust_fast_edge(*args):
        raise AssertionError("rust kernel must not run for non-float32 data")

    monkeypatch.setattr(_fast_edge_finder, "_rust_kernel", lambda: rust_fast_edge)
    labels = np.array([[1, 2]], dtype=np.int32)
    data = np.array([[1.0, 2.0]], dtype=np.float64)

    indices, velocities = _fast_edge_finder._fast_edge_finder(labels, data, False, 0, 0, 2)

    np.testing.assert_array_equal(indices[0], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(indices[1], np.array([2, 1], dtype=np.int32))
    np.testing.assert_array_equal(velocities[0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(velocities[1], np.array([2.0, 1.0]))
