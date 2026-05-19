import io
import os

import numpy as np
import pytest

import pyart.io.nexrad_level2 as nexrad_level2


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_nexrad_level2_scan_msgs_i64"):
        pytest.skip("pyart._rust has no NEXRAD Level II scan-msg kernel")
    return rust


def _fallback_scan_msgs(elev_nums, monkeypatch):
    monkeypatch.setattr(nexrad_level2, "_rust_kernel", lambda _name: None)
    return nexrad_level2._nexrad_scan_msgs(elev_nums)


def _assert_scan_msgs_equal(actual, expected):
    assert type(actual) is list
    assert len(actual) == len(expected)
    for actual_scan, expected_scan in zip(actual, expected):
        assert actual_scan.dtype == expected_scan.dtype
        assert actual_scan.shape == expected_scan.shape
        np.testing.assert_array_equal(actual_scan, expected_scan)


def test_nexrad_level2_constructor_closes_owned_handle_on_failure(monkeypatch):
    handle = io.BytesIO(b"")

    def fake_open(filename, mode):
        assert filename == "broken-level2"
        assert mode == "rb"
        return handle

    monkeypatch.setattr("builtins.open", fake_open)

    with pytest.raises(Exception):
        nexrad_level2.NEXRADLevel2File("broken-level2")

    assert handle.closed


def test_nexrad_level2_constructor_keeps_external_handle_open_on_failure():
    handle = io.BytesIO(b"")

    with pytest.raises(Exception):
        nexrad_level2.NEXRADLevel2File(handle)

    assert not handle.closed


def test_nexrad_level2_scan_info_does_not_mutate_nscans_for_short_scans():
    obj = nexrad_level2.NEXRADLevel2File.__new__(nexrad_level2.NEXRADLevel2File)
    obj.nscans = 3
    obj.scan_msgs = [
        np.array([0], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.array([3], dtype=np.int64),
    ]
    obj.radial_records = [
        {},
        {"REF": {"ngates": 10, "gate_spacing": 250, "first_gate": 0}},
        {},
        {},
    ]

    expected = [
        {
            "nrays": 2,
            "ngates": [10],
            "gate_spacing": [250],
            "first_gate": [0],
            "moments": ["REF"],
        }
    ]

    assert obj.scan_info() == expected
    assert obj.nscans == 3
    assert obj.scan_info() == expected
    assert obj.nscans == 3


def test_nexrad_level2_scan_msgs_python_fallback_reference(monkeypatch):
    elev_nums = np.array([2, 1, 2, 3], dtype=np.int64)

    actual = _fallback_scan_msgs(elev_nums, monkeypatch)

    _assert_scan_msgs_equal(
        actual,
        [
            np.array([1], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ],
    )


def test_nexrad_level2_scan_msgs_python_fallback_preserves_empty_scans(monkeypatch):
    elev_nums = np.array([3, 3], dtype=np.int64)

    actual = _fallback_scan_msgs(elev_nums, monkeypatch)

    _assert_scan_msgs_equal(
        actual,
        [
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        ],
    )


def test_nexrad_level2_scan_msgs_dispatches_dense_int64_to_private_rust(monkeypatch):
    calls = []

    def kernel(elev_nums):
        calls.append((elev_nums.dtype, elev_nums.shape))
        return [
            np.array([10], dtype=np.int64),
            np.array([], dtype=np.int64),
        ]

    monkeypatch.setattr(
        nexrad_level2,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_level2_scan_msgs_i64" else None,
    )

    actual = nexrad_level2._nexrad_scan_msgs(np.array([1, 2], dtype=np.int64))

    assert calls == [(np.dtype(np.int64), (2,))]
    _assert_scan_msgs_equal(
        actual,
        [np.array([10], dtype=np.int64), np.array([], dtype=np.int64)],
    )


@pytest.mark.parametrize(
    "case",
    [
        lambda: np.array([2, 1, 2], dtype=np.int64)[::2],
        lambda: np.array([2, 1, 2], dtype=np.int32),
        lambda: np.array([[1, 2], [1, 3]], dtype=np.int64),
        lambda: np.array([], dtype=np.int64),
        lambda: np.array([1.0, 2.0], dtype=np.float64),
        lambda: np.array([True, False], dtype=bool),
        lambda: np.array(["1", "2"], dtype=object),
    ],
)
def test_nexrad_level2_scan_msgs_unsupported_inputs_keep_python_path(
    monkeypatch, case
):
    def fail_if_called(name):
        if name == "_nexrad_level2_scan_msgs_i64":
            raise AssertionError("unsupported NEXRAD Level II scan input used Rust")
        return None

    elev_nums = case()
    monkeypatch.setattr(nexrad_level2, "_rust_kernel", fail_if_called)

    try:
        actual = nexrad_level2._nexrad_scan_msgs(elev_nums)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_scan_msgs(case(), monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_scan_msgs(case(), monkeypatch)
        _assert_scan_msgs_equal(actual, expected)


@pytest.mark.parametrize(
    "kernel_result",
    [
        (np.array([0], dtype=np.int64),),
        [np.array([0], dtype=np.int32)],
        [np.array([[0]], dtype=np.int64)],
        [np.array([0], dtype=np.int64)],
    ],
)
def test_nexrad_level2_scan_msgs_bad_rust_output_keeps_python_path(
    monkeypatch, kernel_result
):
    def kernel(_elev_nums):
        return kernel_result

    monkeypatch.setattr(
        nexrad_level2,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_level2_scan_msgs_i64" else None,
    )
    elev_nums = np.array([2, 1, 2], dtype=np.int64)

    actual = nexrad_level2._nexrad_scan_msgs(elev_nums)
    expected = _fallback_scan_msgs(elev_nums, monkeypatch)

    _assert_scan_msgs_equal(actual, expected)


def test_nexrad_level2_scan_msgs_nonpositive_elevations_match_python(monkeypatch):
    elev_nums = np.array([0, 2, -1], dtype=np.int64)
    expected = _fallback_scan_msgs(elev_nums, monkeypatch)
    monkeypatch.undo()

    actual = nexrad_level2._nexrad_scan_msgs(elev_nums)

    _assert_scan_msgs_equal(actual, expected)


def test_nexrad_level2_msg_nums_preserves_python_concatenate_semantics():
    obj = nexrad_level2.NEXRADLevel2File.__new__(nexrad_level2.NEXRADLevel2File)
    obj.scan_msgs = [
        np.array([0, 2], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([1], dtype=np.int64),
    ]

    np.testing.assert_array_equal(
        obj._msg_nums([2, 0, 2]), np.array([1, 0, 2, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(obj._msg_nums([1]), np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="need at least one array"):
        obj._msg_nums([])
    with pytest.raises(TypeError):
        obj._msg_nums(None)


def test_nexrad_level2_msg_nums_dispatches_dense_scan_msgs_to_private_rust(monkeypatch):
    obj = nexrad_level2.NEXRADLevel2File.__new__(nexrad_level2.NEXRADLevel2File)
    obj.scan_msgs = [
        np.array([0, 2], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([1], dtype=np.int64),
    ]
    calls = []

    def kernel(scan_msgs, scans):
        calls.append(([item.copy() for item in scan_msgs], scans.copy()))
        return np.array([9, 8, 7, 6], dtype=np.int64)

    monkeypatch.setattr(
        nexrad_level2,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_level2_msg_nums_i64" else None,
    )

    actual = obj._msg_nums([2, -3, 2])

    assert len(calls) == 1
    assert len(calls[0][0]) == 3
    np.testing.assert_array_equal(calls[0][0][0], np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(calls[0][0][1], np.array([], dtype=np.int64))
    np.testing.assert_array_equal(calls[0][0][2], np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(calls[0][1], np.array([2, 0, 2], dtype=np.int64))
    np.testing.assert_array_equal(actual, np.array([9, 8, 7, 6], dtype=np.int64))


@pytest.mark.parametrize(
    "scan_factory",
    [
        lambda: [],
        lambda: None,
        lambda: (scan for scan in [0, 1]),
        lambda: np.array([0, 1], dtype=np.int64),
        lambda: [True],
        lambda: [99],
        lambda: [-99],
    ],
)
def test_nexrad_level2_msg_nums_unsupported_scans_keep_python_path(
    monkeypatch, scan_factory
):
    obj = nexrad_level2.NEXRADLevel2File.__new__(nexrad_level2.NEXRADLevel2File)
    obj.scan_msgs = [
        np.array([0, 2], dtype=np.int64),
        np.array([1], dtype=np.int64),
    ]

    def fail_if_called(name):
        if name == "_nexrad_level2_msg_nums_i64":
            raise AssertionError("unsupported scans used Rust")
        return None

    monkeypatch.setattr(nexrad_level2, "_rust_kernel", fail_if_called)

    scans = scan_factory()
    try:
        actual = obj._msg_nums(scans)
    except Exception as actual_error:
        expected_obj = nexrad_level2.NEXRADLevel2File.__new__(
            nexrad_level2.NEXRADLevel2File
        )
        expected_obj.scan_msgs = [
            np.array([0, 2], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ]
        with pytest.raises(type(actual_error)) as expected_error:
            expected_obj._msg_nums(scan_factory())
        assert actual_error.args == expected_error.value.args
    else:
        expected_obj = nexrad_level2.NEXRADLevel2File.__new__(
            nexrad_level2.NEXRADLevel2File
        )
        expected_obj.scan_msgs = [
            np.array([0, 2], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ]
        expected = expected_obj._msg_nums(scan_factory())
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "scan_msgs",
    [
        [np.array([0, 2], dtype=np.int32), np.array([1], dtype=np.int64)],
        [np.array([[0, 2]], dtype=np.int64), np.array([1], dtype=np.int64)],
        [np.array([0, 2], dtype=np.int64)[::2], np.array([1], dtype=np.int64)],
        [[0, 2], np.array([1], dtype=np.int64)],
    ],
)
def test_nexrad_level2_msg_nums_unsupported_scan_msgs_keep_python_path(
    monkeypatch, scan_msgs
):
    obj = nexrad_level2.NEXRADLevel2File.__new__(nexrad_level2.NEXRADLevel2File)
    obj.scan_msgs = scan_msgs

    def fail_if_called(name):
        if name == "_nexrad_level2_msg_nums_i64":
            raise AssertionError("unsupported scan_msgs used Rust")
        return None

    monkeypatch.setattr(nexrad_level2, "_rust_kernel", fail_if_called)

    try:
        actual = obj._msg_nums([0, 1])
    except Exception as actual_error:
        expected_obj = nexrad_level2.NEXRADLevel2File.__new__(
            nexrad_level2.NEXRADLevel2File
        )
        expected_obj.scan_msgs = scan_msgs
        with pytest.raises(type(actual_error)) as expected_error:
            expected_obj._msg_nums([0, 1])
        assert actual_error.args == expected_error.value.args
    else:
        expected_obj = nexrad_level2.NEXRADLevel2File.__new__(
            nexrad_level2.NEXRADLevel2File
        )
        expected_obj.scan_msgs = scan_msgs
        expected = expected_obj._msg_nums([0, 1])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kernel_result",
    [
        np.array([0, 1], dtype=np.int32),
        np.array([[0, 1]], dtype=np.int64),
        np.array([0], dtype=np.int64),
    ],
)
def test_nexrad_level2_msg_nums_bad_rust_output_keeps_python_path(
    monkeypatch, kernel_result
):
    obj = nexrad_level2.NEXRADLevel2File.__new__(nexrad_level2.NEXRADLevel2File)
    obj.scan_msgs = [
        np.array([0, 2], dtype=np.int64),
        np.array([1], dtype=np.int64),
    ]

    def kernel(_scan_msgs, _scans):
        return kernel_result

    monkeypatch.setattr(
        nexrad_level2,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_level2_msg_nums_i64" else None,
    )

    actual = obj._msg_nums([0, 1])

    np.testing.assert_array_equal(actual, np.array([0, 2, 1], dtype=np.int64))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust NEXRAD Level II scan-msg parity",
)
def test_nexrad_level2_scan_msgs_real_rust_matches_python_fallback(monkeypatch):
    elev_nums = np.array([2, 1, 2, 3], dtype=np.int64)
    expected = _fallback_scan_msgs(elev_nums, monkeypatch)
    monkeypatch.undo()

    actual = nexrad_level2._nexrad_scan_msgs(elev_nums)

    _assert_scan_msgs_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust NEXRAD Level II scan-msg checks",
)
def test_nexrad_level2_scan_msgs_direct_rust_helper():
    rust = _rust_or_skip()

    actual = rust._nexrad_level2_scan_msgs_i64(
        np.array([2, 1, 2, 3], dtype=np.int64)
    )
    _assert_scan_msgs_equal(
        actual,
        [
            np.array([1], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ],
    )

    actual = rust._nexrad_level2_scan_msgs_i64(np.array([0, -1], dtype=np.int64))
    assert actual == []

    with pytest.raises(ValueError, match="non-empty"):
        rust._nexrad_level2_scan_msgs_i64(np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._nexrad_level2_scan_msgs_i64(np.arange(6, dtype=np.int64)[::2])
    with pytest.raises(ValueError, match="size limit"):
        rust._nexrad_level2_scan_msgs_i64(
            np.zeros(
                nexrad_level2.NEXRAD_SCAN_MSGS_RUST_MAX_RECORDS + 1,
                dtype=np.int64,
            )
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust NEXRAD Level II msg-num parity",
)
def test_nexrad_level2_msg_nums_real_rust_matches_python_fallback(monkeypatch):
    obj = nexrad_level2.NEXRADLevel2File.__new__(nexrad_level2.NEXRADLevel2File)
    obj.scan_msgs = [
        np.array([0, 2], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([1], dtype=np.int64),
    ]
    monkeypatch.setattr(nexrad_level2, "_rust_kernel", lambda _name: None)
    expected = obj._msg_nums([2, 0, 2])
    monkeypatch.undo()

    actual = obj._msg_nums([2, 0, 2])

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust NEXRAD Level II msg-num checks",
)
def test_nexrad_level2_msg_nums_direct_rust_helper():
    rust = _rust_or_skip()
    scan_msgs = [
        np.array([0, 2], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([1], dtype=np.int64),
    ]

    actual = rust._nexrad_level2_msg_nums_i64(
        scan_msgs, np.array([2, 0, 2], dtype=np.int64)
    )
    np.testing.assert_array_equal(actual, np.array([1, 0, 2, 1], dtype=np.int64))

    actual = rust._nexrad_level2_msg_nums_i64(
        scan_msgs, np.array([1], dtype=np.int64)
    )
    assert actual.dtype == np.int64
    assert actual.shape == (0,)

    with pytest.raises(ValueError, match="non-negative"):
        rust._nexrad_level2_msg_nums_i64(scan_msgs, np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError, match="out of range"):
        rust._nexrad_level2_msg_nums_i64(scan_msgs, np.array([3], dtype=np.int64))
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._nexrad_level2_msg_nums_i64(
            scan_msgs, np.arange(4, dtype=np.int64)[::2]
        )
    with pytest.raises(ValueError, match="non-empty"):
        rust._nexrad_level2_msg_nums_i64(
            scan_msgs, np.array([], dtype=np.int64)
        )
