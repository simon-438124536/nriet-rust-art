import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import qvp  # noqa: E402


def _fallback_project(data_in, data_height, grid_height, monkeypatch, **kwargs):
    monkeypatch.setattr(qvp, "_rust_kernel", lambda _name: None)
    return qvp.project_to_vertical(data_in, data_height, grid_height, **kwargs)


def _assert_projected_equal(actual, expected):
    assert np.ma.isMaskedArray(actual)
    assert np.ma.isMaskedArray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    actual_mask = np.ma.getmaskarray(actual)
    expected_mask = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    np.testing.assert_allclose(
        actual.data[~expected_mask],
        expected.data[~expected_mask],
        rtol=0.0,
        atol=0.0,
    )


def _dense_inputs():
    data_in = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    data_height = np.array([0.0, 100.0, 200.0], dtype=np.float64)
    grid_height = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0], dtype=np.float64)
    return data_in, data_height, grid_height


def test_project_to_vertical_none_python_fallback_reference(monkeypatch):
    inputs = _dense_inputs()

    actual = _fallback_project(*inputs, monkeypatch, interp_kind="none")

    assert actual.dtype == np.float64
    assert actual.fill_value == np.ma.masked_all(1).fill_value
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.array([False, True, False, True, False, True]),
    )
    np.testing.assert_array_equal(actual.data[~actual.mask], np.array([10.0, 20.0, 30.0]))


def test_project_to_vertical_none_dispatches_to_private_rust_kernel(monkeypatch):
    data_in, data_height, grid_height = _dense_inputs()
    calls = []

    def rust_kernel(data_arg, height_arg, grid_arg):
        calls.append((data_arg.dtype, data_arg.shape, height_arg.shape, grid_arg.shape))
        return (
            np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=np.float64),
            np.array([False, True, False, True, False, True], dtype=bool),
        )

    monkeypatch.setattr(
        qvp,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_qvp_project_to_vertical_none_dense_f64" else None,
    )

    actual = qvp.project_to_vertical(
        data_in, data_height, grid_height, interp_kind="none"
    )

    assert calls == [(np.dtype("float64"), (3,), (3,), (6,))]
    expected = np.ma.masked_all(grid_height.size)
    expected[[0, 2, 4]] = [1.0, 2.0, 3.0]
    _assert_projected_equal(actual, expected)


@pytest.mark.parametrize(
    ("data_in", "data_height", "grid_height"),
    [
        (
            np.ma.array([10.0, 20.0], mask=[False, True], dtype=np.float64),
            np.array([0.0, 100.0], dtype=np.float64),
            np.array([0.0, 100.0], dtype=np.float64),
        ),
        (
            np.array([10.0, 20.0, 30.0], dtype=np.float64)[::2],
            np.array([0.0, 200.0], dtype=np.float64),
            np.array([0.0, 100.0, 200.0], dtype=np.float64),
        ),
        (
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([0.0, 100.0], dtype=np.float64),
            np.array([0.0, 100.0], dtype=np.float64),
        ),
        (
            np.array([10.0, np.nan], dtype=np.float64),
            np.array([0.0, 100.0], dtype=np.float64),
            np.array([0.0, 100.0], dtype=np.float64),
        ),
        (
            np.array([10.0, 20.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0, 100.0], dtype=np.float64),
        ),
        (
            np.array([10.0, 20.0], dtype=np.float64),
            np.array([0.0, 100.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ),
    ],
)
def test_project_to_vertical_none_keeps_python_path_for_unsupported_inputs(
    monkeypatch, data_in, data_height, grid_height
):
    try:
        expected = _fallback_project(
            data_in, data_height, grid_height, monkeypatch, interp_kind="none"
        )
    except Exception as expected_error:
        expected_error_type = type(expected_error)
        expected_error_args = expected_error.args
    else:
        expected_error_type = None
        expected_error_args = None

    def fail_if_called(name):
        if name != "_qvp_project_to_vertical_none_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported project_to_vertical input used Rust")

        return kernel

    monkeypatch.setattr(qvp, "_rust_kernel", fail_if_called)
    if expected_error_type is not None:
        with pytest.raises(expected_error_type) as actual_error:
            qvp.project_to_vertical(
                data_in, data_height, grid_height, interp_kind="none"
            )
        assert actual_error.value.args == expected_error_args
    else:
        actual = qvp.project_to_vertical(
            data_in, data_height, grid_height, interp_kind="none"
        )
        _assert_projected_equal(actual, expected)


def test_project_to_vertical_nearest_keeps_python_path(monkeypatch):
    data_in, data_height, grid_height = _dense_inputs()

    def fail_if_called(_name):
        def kernel(*_args):
            raise AssertionError("non-none interpolation should stay Python-owned")

        return kernel

    monkeypatch.setattr(qvp, "_rust_kernel", fail_if_called)
    actual = qvp.project_to_vertical(
        np.ma.array(data_in), data_height, grid_height, interp_kind="nearest"
    )
    expected = _fallback_project(
        np.ma.array(data_in), data_height, grid_height, monkeypatch, interp_kind="nearest"
    )

    _assert_projected_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_project_to_vertical_none_matches_python_fallback(monkeypatch):
    data_in, data_height, grid_height = _dense_inputs()
    expected = _fallback_project(
        data_in.copy(), data_height.copy(), grid_height.copy(), monkeypatch, interp_kind="none"
    )

    import pyart._rust as rust

    calls = []

    def rust_kernel(name):
        if name == "_qvp_project_to_vertical_none_dense_f64":

            def project(data_arg, height_arg, grid_arg):
                calls.append((data_arg.shape, height_arg.shape, grid_arg.shape))
                return rust._qvp_project_to_vertical_none_dense_f64(
                    data_arg, height_arg, grid_arg
                )

            return project
        return getattr(rust, name, None)

    monkeypatch.setattr(qvp, "_rust_kernel", rust_kernel)
    actual = qvp.project_to_vertical(
        data_in.copy(), data_height.copy(), grid_height.copy(), interp_kind="none"
    )

    assert calls == [((3,), (3,), (6,))]
    _assert_projected_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_project_to_vertical_none_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    data_in, data_height, grid_height = _dense_inputs()
    data, mask = rust._qvp_project_to_vertical_none_dense_f64(
        data_in, data_height, grid_height
    )
    np.testing.assert_array_equal(data, np.array([10.0, 0.0, 20.0, 0.0, 30.0, 0.0]))
    np.testing.assert_array_equal(mask, np.array([False, True, False, True, False, True]))

    with pytest.raises(ValueError, match="non-empty"):
        rust._qvp_project_to_vertical_none_dense_f64(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            grid_height,
        )
    with pytest.raises(ValueError, match="same length"):
        rust._qvp_project_to_vertical_none_dense_f64(
            data_in,
            np.array([0.0], dtype=np.float64),
            grid_height,
        )
    with pytest.raises(ValueError, match="at least two"):
        rust._qvp_project_to_vertical_none_dense_f64(
            data_in,
            data_height,
            np.array([0.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._qvp_project_to_vertical_none_dense_f64(
            np.arange(6.0, dtype=np.float64)[::2],
            data_height,
            grid_height,
        )
    with pytest.raises(ValueError, match="finite"):
        rust._qvp_project_to_vertical_none_dense_f64(
            np.array([10.0, np.nan, 30.0], dtype=np.float64),
            data_height,
            grid_height,
        )
    with pytest.raises(ValueError, match="mask-free"):
        rust._qvp_project_to_vertical_none_dense_f64(
            np.ma.array(data_in, mask=[False, True, False]),
            data_height,
            grid_height,
        )
    with pytest.raises(ValueError, match="float64"):
        rust._qvp_project_to_vertical_none_dense_f64(
            data_in.astype(np.float32),
            data_height,
            grid_height,
        )
    with pytest.raises(ValueError, match="1D float64"):
        rust._qvp_project_to_vertical_none_dense_f64(
            data_in.reshape(1, 3),
            data_height,
            grid_height,
        )
    with pytest.raises(ValueError, match="1D float64"):
        rust._qvp_project_to_vertical_none_dense_f64(
            np.array([object(), object(), object()], dtype=object),
            data_height,
            grid_height,
        )
    readonly = data_in.copy()
    readonly.flags.writeable = False
    data, mask = rust._qvp_project_to_vertical_none_dense_f64(
        readonly, data_height, grid_height
    )
    np.testing.assert_array_equal(data, np.array([10.0, 0.0, 20.0, 0.0, 30.0, 0.0]))
    np.testing.assert_array_equal(mask, np.array([False, True, False, True, False, True]))
