import os

import numpy as np
import pytest

import pyart.io.sigmet as sigmet


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    for name in (
        "_sigmet_time_ordered_by_reversal_i32",
        "_sigmet_time_ordered_by_roll_i32",
        "_sigmet_time_ordered_by_reverse_roll_i32",
        "_sigmet_time_order_roll_index_i32",
        "_sigmet_time_order_reverse_index_i32",
        "_sigmet_time_order_full_index_i32",
    ):
        if not hasattr(rust, name):
            pytest.skip(f"pyart._rust has no {name} kernel")
    return rust


def _metadata(times):
    return {"DBZ": {"time": np.asarray(times, dtype=np.int32)}}


def _xhdr_data(times):
    return {"XHDR": np.asarray(times, dtype=np.int32).reshape(-1, 1)}


def _fallback(func, data, metadata, rays_per_sweep, monkeypatch):
    monkeypatch.setattr(sigmet, "_rust_kernel", lambda _name: None)
    return func(data, metadata, rays_per_sweep)


def _orderable_volume(times):
    times = np.asarray(times, dtype=np.int32)
    data = {
        "DBZ": np.arange(times.size * 2, dtype=np.int16).reshape(times.size, 2),
    }
    metadata = {
        "DBZ": {
            "time": times.copy(),
            "azimuth": np.arange(times.size, dtype=np.float64),
        }
    }
    return data, metadata


def _copy_volume(data, metadata):
    return (
        {key: value.copy() for key, value in data.items()},
        {
            field: {key: value.copy() for key, value in field_metadata.items()}
            for field, field_metadata in metadata.items()
        },
    )


def _assert_volume_equal(actual_data, actual_metadata, expected_data, expected_metadata):
    assert actual_data.keys() == expected_data.keys()
    assert actual_metadata.keys() == expected_metadata.keys()
    for field in actual_data:
        np.testing.assert_array_equal(actual_data[field], expected_data[field])
    for field in actual_metadata:
        assert actual_metadata[field].keys() == expected_metadata[field].keys()
        for key in actual_metadata[field]:
            np.testing.assert_array_equal(
                actual_metadata[field][key], expected_metadata[field][key]
            )


@pytest.mark.parametrize(
    ("func", "times", "rays_per_sweep", "expected"),
    [
        (sigmet._is_time_ordered_by_reversal, [3, 2, 1, 10], [1, 3], True),
        (sigmet._is_time_ordered_by_reversal, [0, 2, 1, 3], [4], False),
        (sigmet._is_time_ordered_by_roll, [2, 3, 1, 10], [1, 3], True),
        (sigmet._is_time_ordered_by_roll, [0, 3, 1, 2], [4], False),
        (sigmet._is_time_ordered_by_reverse_roll, [3, 2, 1, 10], [1, 3], True),
        (sigmet._is_time_ordered_by_reverse_roll, [0, 3, 1, 2], [4], False),
        (sigmet._is_time_ordered_by_roll, [np.iinfo(np.int32).min, np.iinfo(np.int32).max], [2], True),
    ],
)
def test_sigmet_time_order_python_fallback_reference(
    monkeypatch, func, times, rays_per_sweep, expected
):
    with np.errstate(over="ignore"):
        actual = _fallback(
            func,
            {},
            _metadata(times),
            np.asarray(rays_per_sweep, dtype=np.int64),
            monkeypatch,
        )

    assert actual is expected


@pytest.mark.parametrize(
    ("func", "kernel_name"),
    [
        (sigmet._is_time_ordered_by_reversal, "_sigmet_time_ordered_by_reversal_i32"),
        (sigmet._is_time_ordered_by_roll, "_sigmet_time_ordered_by_roll_i32"),
        (
            sigmet._is_time_ordered_by_reverse_roll,
            "_sigmet_time_ordered_by_reverse_roll_i32",
        ),
    ],
)
def test_sigmet_time_order_dispatches_dense_metadata_to_private_rust(
    monkeypatch, func, kernel_name
):
    calls = []

    def kernel(ref_time, rays_per_sweep):
        calls.append(
            (
                ref_time.dtype,
                ref_time.shape,
                rays_per_sweep.dtype,
                rays_per_sweep.shape,
            )
        )
        return False

    monkeypatch.setattr(
        sigmet,
        "_rust_kernel",
        lambda name: kernel if name == kernel_name else None,
    )

    actual = func({}, _metadata([0, 1, 2]), np.array([3], dtype=np.int64))

    assert actual is False
    assert calls == [(np.dtype(np.int32), (3,), np.dtype(np.int64), (1,))]


def test_sigmet_time_order_dispatches_dense_xhdr_to_private_rust(monkeypatch):
    calls = []

    def kernel(ref_time, rays_per_sweep):
        calls.append((ref_time.dtype, ref_time.shape, rays_per_sweep.dtype))
        return True

    monkeypatch.setattr(
        sigmet,
        "_rust_kernel",
        lambda name: kernel
        if name == "_sigmet_time_ordered_by_reversal_i32"
        else None,
    )

    actual = sigmet._is_time_ordered_by_reversal(
        _xhdr_data([0, 1, 2]),
        _metadata([10, 11, 12]),
        np.array([3], dtype=np.int64),
    )

    assert actual is True
    assert calls == [(np.dtype(np.int32), (3,), np.dtype(np.int64))]


@pytest.mark.parametrize(
    ("func", "rays_per_sweep"),
    [
        (sigmet._is_time_ordered_by_reversal, [3]),
        (sigmet._is_time_ordered_by_roll, np.array([3], dtype=np.int32)),
        (sigmet._is_time_ordered_by_reverse_roll, np.array([-1], dtype=np.int64)),
    ],
)
def test_sigmet_time_order_unsupported_inputs_keep_python_path(
    monkeypatch, func, rays_per_sweep
):
    def kernel(_ref_time, _rays_per_sweep):
        return False

    monkeypatch.setattr(
        sigmet,
        "_rust_kernel",
        lambda name: kernel if name.startswith("_sigmet_time_ordered") else None,
    )

    actual = func({}, _metadata([0, 1, 2]), rays_per_sweep)
    expected = _fallback(
        func,
        {},
        _metadata([0, 1, 2]),
        rays_per_sweep,
        monkeypatch,
    )

    assert actual == expected


@pytest.mark.parametrize(
    ("func", "times", "rays_per_sweep"),
    [
        (sigmet._is_time_ordered_by_reversal, [0, 1], [3]),
        (sigmet._is_time_ordered_by_roll, [0, 1], [3]),
        (sigmet._is_time_ordered_by_reverse_roll, [0, 1], [3]),
    ],
)
def test_sigmet_time_order_rust_error_keeps_python_path(
    monkeypatch, func, times, rays_per_sweep
):
    def kernel(_ref_time, _rays_per_sweep):
        raise ValueError("native failure")

    monkeypatch.setattr(
        sigmet,
        "_rust_kernel",
        lambda name: kernel if name.startswith("_sigmet_time_ordered") else None,
    )

    with np.errstate(over="ignore"):
        try:
            actual = func(
                {},
                _metadata(times),
                np.asarray(rays_per_sweep, dtype=np.int64),
            )
        except Exception as actual_error:
            with pytest.raises(type(actual_error)) as expected_error:
                _fallback(
                    func,
                    {},
                    _metadata(times),
                    np.asarray(rays_per_sweep, dtype=np.int64),
                    monkeypatch,
                )
            assert actual_error.args == expected_error.value.args
        else:
            expected = _fallback(
                func,
                {},
                _metadata(times),
                np.asarray(rays_per_sweep, dtype=np.int64),
                monkeypatch,
            )
            assert actual == expected


def test_sigmet_time_order_oversized_input_keeps_python_path(monkeypatch):
    def kernel(_ref_time, _rays_per_sweep):
        return False

    monkeypatch.setattr(
        sigmet,
        "_rust_kernel",
        lambda name: kernel if name == "_sigmet_time_ordered_by_reversal_i32" else None,
    )
    monkeypatch.setattr(sigmet, "SIGMET_TIME_ORDER_RUST_MAX_RAYS", 2)

    actual = sigmet._is_time_ordered_by_reversal(
        {},
        _metadata([0, 1, 2]),
        np.array([3], dtype=np.int64),
    )
    expected = _fallback(
        sigmet._is_time_ordered_by_reversal,
        {},
        _metadata([0, 1, 2]),
        np.array([3], dtype=np.int64),
        monkeypatch,
    )

    assert actual == expected


@pytest.mark.parametrize(
    ("func", "times", "rays_per_sweep", "expected_time"),
    [
        (
            sigmet._time_order_data_and_metadata_roll,
            [2, 3, 1, 10, 11, 12],
            [3, 3],
            [1, 2, 3, 10, 11, 12],
        ),
        (
            sigmet._time_order_data_and_metadata_reverse,
            [3, 2, 1, 10, 11, 12],
            [3, 3],
            [1, 2, 3, 10, 11, 12],
        ),
        (
            sigmet._time_order_data_and_metadata_full,
            [2, 1, 1, 5, 10, 11],
            [4, 2],
            [1, 1, 2, 5, 10, 11],
        ),
        (
            sigmet._time_order_data_and_metadata_roll,
            [2, 3, 1, 10],
            [1, 3],
            [1, 2, 3, 10],
        ),
    ],
)
def test_sigmet_time_order_index_python_fallback_reference(
    monkeypatch, func, times, rays_per_sweep, expected_time
):
    data, metadata = _orderable_volume(times)
    _fallback(
        func,
        data,
        metadata,
        np.asarray(rays_per_sweep, dtype=np.int64),
        monkeypatch,
    )

    np.testing.assert_array_equal(
        metadata["DBZ"]["time"], np.asarray(expected_time, dtype=np.int32)
    )


def test_sigmet_time_order_index_dispatches_private_rust_and_applies_order(
    monkeypatch,
):
    data, metadata = _orderable_volume([2, 3, 1])
    calls = []

    def kernel(ref_time, rays_per_sweep):
        calls.append((ref_time.dtype, ref_time.shape, rays_per_sweep.dtype))
        return np.array([2, 0, 1], dtype=np.int64)

    monkeypatch.setattr(
        sigmet,
        "_rust_kernel",
        lambda name: kernel if name == "_sigmet_time_order_roll_index_i32" else None,
    )

    sigmet._time_order_data_and_metadata_roll(
        data, metadata, np.array([3], dtype=np.int64)
    )

    assert calls == [(np.dtype(np.int32), (3,), np.dtype(np.int64))]
    np.testing.assert_array_equal(data["DBZ"], np.array([[4, 5], [0, 1], [2, 3]], dtype=np.int16))
    np.testing.assert_array_equal(metadata["DBZ"]["time"], np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(metadata["DBZ"]["azimuth"], np.array([2.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    ("bad_order", "expected_order"),
    [
        (np.array([2, 0, 1], dtype=np.int32), np.array([2, 0, 1], dtype=np.int64)),
        (np.array([2, 0, -1], dtype=np.int64), np.array([2, 0, 1], dtype=np.int64)),
        (np.array([2, 0, 0], dtype=np.int64), np.array([2, 0, 1], dtype=np.int64)),
        (np.array([2, 0], dtype=np.int64), np.array([2, 0, 1], dtype=np.int64)),
    ],
)
def test_sigmet_time_order_index_bad_rust_order_keeps_python_path(
    monkeypatch, bad_order, expected_order
):
    data, metadata = _orderable_volume([2, 3, 1])

    def kernel(_ref_time, _rays_per_sweep):
        return bad_order

    monkeypatch.setattr(
        sigmet,
        "_rust_kernel",
        lambda name: kernel if name == "_sigmet_time_order_roll_index_i32" else None,
    )

    sigmet._time_order_data_and_metadata_roll(
        data, metadata, np.array([3], dtype=np.int64)
    )

    np.testing.assert_array_equal(metadata["DBZ"]["time"], np.array([2, 3, 1], dtype=np.int32)[expected_order])


@pytest.mark.parametrize(
    ("func", "kernel_name"),
    [
        (sigmet._time_order_data_and_metadata_roll, "_sigmet_time_order_roll_index_i32"),
        (
            sigmet._time_order_data_and_metadata_reverse,
            "_sigmet_time_order_reverse_index_i32",
        ),
        (sigmet._time_order_data_and_metadata_full, "_sigmet_time_order_full_index_i32"),
    ],
)
def test_sigmet_time_order_index_unsupported_targets_keep_python_path(
    monkeypatch, func, kernel_name
):
    data, metadata = _orderable_volume([3, 2, 1])
    data["DBZ"] = data["DBZ"][::2]

    def kernel(_ref_time, _rays_per_sweep):
        return np.array([2, 1, 0], dtype=np.int64)

    monkeypatch.setattr(
        sigmet,
        "_rust_kernel",
        lambda name: kernel if name == kernel_name else None,
    )

    try:
        func(data, metadata, np.array([3], dtype=np.int64))
    except Exception as actual_error:
        expected_data, expected_metadata = _orderable_volume([3, 2, 1])
        expected_data["DBZ"] = expected_data["DBZ"][::2]
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback(
                func,
                expected_data,
                expected_metadata,
                np.array([3], dtype=np.int64),
                monkeypatch,
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected_data, expected_metadata = _orderable_volume([3, 2, 1])
        expected_data["DBZ"] = expected_data["DBZ"][::2]
        _fallback(
            func,
            expected_data,
            expected_metadata,
            np.array([3], dtype=np.int64),
            monkeypatch,
        )
        _assert_volume_equal(data, metadata, expected_data, expected_metadata)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust sigmet time-order parity",
)
@pytest.mark.parametrize(
    ("func", "data", "metadata", "rays_per_sweep"),
    [
        (
            sigmet._is_time_ordered_by_reversal,
            {},
            _metadata([0, 1, 2, 5, 4, 3]),
            np.array([3, 3], dtype=np.int64),
        ),
        (
            sigmet._is_time_ordered_by_roll,
            _xhdr_data([2, 3, 1, 10]),
            _metadata([99, 98, 97, 96]),
            np.array([1, 3], dtype=np.int64),
        ),
        (
            sigmet._is_time_ordered_by_reverse_roll,
            {},
            _metadata([3, 2, 1, 10]),
            np.array([1, 3], dtype=np.int64),
        ),
    ],
)
def test_sigmet_time_order_real_rust_matches_python_fallback(
    monkeypatch, func, data, metadata, rays_per_sweep
):
    with np.errstate(over="ignore"):
        expected = _fallback(func, data, metadata, rays_per_sweep, monkeypatch)
    monkeypatch.undo()

    with np.errstate(over="ignore"):
        actual = func(data, metadata, rays_per_sweep)

    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust sigmet time-order index parity",
)
@pytest.mark.parametrize(
    ("func", "times", "rays_per_sweep"),
    [
        (sigmet._time_order_data_and_metadata_roll, [2, 3, 1, 10, 11, 12], [3, 3]),
        (sigmet._time_order_data_and_metadata_reverse, [3, 2, 1, 10, 11, 12], [3, 3]),
        (sigmet._time_order_data_and_metadata_full, [2, 1, 1, 5, 10, 11], [4, 2]),
        (sigmet._time_order_data_and_metadata_roll, [2, 3, 1, 10], [1, 3]),
    ],
)
def test_sigmet_time_order_index_real_rust_matches_python_fallback(
    monkeypatch, func, times, rays_per_sweep
):
    data, metadata = _orderable_volume(times)
    expected_data, expected_metadata = _copy_volume(data, metadata)
    _fallback(
        func,
        expected_data,
        expected_metadata,
        np.asarray(rays_per_sweep, dtype=np.int64),
        monkeypatch,
    )
    monkeypatch.undo()

    func(data, metadata, np.asarray(rays_per_sweep, dtype=np.int64))

    _assert_volume_equal(data, metadata, expected_data, expected_metadata)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust sigmet time-order checks",
)
def test_sigmet_time_order_direct_rust_helpers():
    rust = _rust_or_skip()
    ref_time = np.array([3, 2, 1, 10], dtype=np.int32)
    rays = np.array([1, 3], dtype=np.int64)

    assert rust._sigmet_time_ordered_by_reversal_i32(ref_time, rays) is True
    assert rust._sigmet_time_ordered_by_reverse_roll_i32(ref_time, rays) is True
    assert rust._sigmet_time_ordered_by_roll_i32(
        np.array([0, 3, 1, 2], dtype=np.int32),
        np.array([4], dtype=np.int64),
    ) is False

    with pytest.raises(ValueError):
        rust._sigmet_time_ordered_by_reversal_i32(ref_time[::2], rays)
    with pytest.raises(ValueError):
        rust._sigmet_time_ordered_by_roll_i32(ref_time, np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError):
        rust._sigmet_time_ordered_by_reverse_roll_i32(
            ref_time,
            np.array([ref_time.size + 1], dtype=np.int64),
        )
    with pytest.raises(ValueError):
        rust._sigmet_time_ordered_by_reversal_i32(
            np.zeros(sigmet.SIGMET_TIME_ORDER_RUST_MAX_RAYS + 1, dtype=np.int32),
            np.array([], dtype=np.int64),
        )

    np.testing.assert_array_equal(
        rust._sigmet_time_order_roll_index_i32(
            np.array([2, 3, 1, 10], dtype=np.int32),
            np.array([1, 3], dtype=np.int64),
        ),
        np.array([2, 0, 1, 3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        rust._sigmet_time_order_reverse_index_i32(
            np.array([3, 2, 1], dtype=np.int32),
            np.array([3], dtype=np.int64),
        ),
        np.array([2, 1, 0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        rust._sigmet_time_order_full_index_i32(
            np.array([2, 1, 1, 5], dtype=np.int32),
            np.array([4], dtype=np.int64),
        ),
        np.array([1, 2, 0, 3], dtype=np.int64),
    )
