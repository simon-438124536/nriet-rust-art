import importlib
import importlib.util
import os

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

gatefilter_module = importlib.import_module("pyart.filters.gatefilter")
GateFilter = gatefilter_module.GateFilter


class DummyRadar:
    nrays = 2
    ngates = 3
    antenna_transition = None
    fields = {}

    def check_field_exists(self, field):
        if field not in self.fields:
            raise KeyError(field)


def _gatefilter(initial=None):
    gatefilter = GateFilter(DummyRadar())
    if initial is not None:
        gatefilter._gate_excluded = np.array(initial, dtype=np.bool_)
    return gatefilter


def _gatefilter_with_field(data, initial=None):
    radar = DummyRadar()
    radar.fields = {"reflectivity": {"data": data}}
    gatefilter = GateFilter(radar)
    if initial is not None:
        gatefilter._gate_excluded = np.array(initial, dtype=np.bool_)
    return gatefilter


def _gatefilter_with_gate_altitude(data, initial=None):
    radar = DummyRadar()
    radar.gate_altitude = {"data": data}
    gatefilter = GateFilter(radar)
    if initial is not None:
        gatefilter._gate_excluded = np.array(initial, dtype=np.bool_)
    return gatefilter


def _gatefilter_with_transition(data, initial=None, ngates=3):
    radar = DummyRadar()
    radar.nrays = len(data) if data is not None else 2
    radar.ngates = ngates
    radar.antenna_transition = None if data is None else {"data": data}
    gatefilter = GateFilter(radar)
    if initial is not None:
        gatefilter._gate_excluded = np.array(initial, dtype=np.bool_)
    return gatefilter


def test_merge_python_fallback_fills_masked_values_with_exclude_masked(monkeypatch):
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    gatefilter = _gatefilter([[False, True, False], [False, False, True]])
    marked = np.ma.array(
        [[False, True, False], [True, False, True]],
        mask=[[True, False, False], [False, True, False]],
    )

    gatefilter._merge(marked, "or", True)

    expected_marked = np.ma.filled(marked, True)
    expected = np.logical_or(
        np.array([[False, True, False], [False, False, True]], dtype=np.bool_),
        expected_marked,
    )
    np.testing.assert_array_equal(gatefilter._gate_excluded, expected)
    assert gatefilter._gate_excluded.dtype == np.bool_


def test_merge_dispatches_to_private_rust_kernel_after_mask_fill(monkeypatch):
    calls = []
    gatefilter = _gatefilter([[False, True, False], [True, False, True]])
    marked = np.ma.array(
        [[True, False, False], [False, True, False]],
        mask=[[False, True, False], [False, False, True]],
    )

    def rust_merge(gate_excluded, marked_arg, op):
        calls.append((gate_excluded.copy(), marked_arg.copy(), op))
        return np.full(gate_excluded.shape, True, dtype=np.bool_)

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_merge if name == "_gatefilter_merge" else None,
    )

    gatefilter._merge(marked, "and", False)

    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0][0],
        np.array([[False, True, False], [True, False, True]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(calls[0][1], np.ma.filled(marked, False))
    assert calls[0][2] == "and"
    np.testing.assert_array_equal(
        gatefilter._gate_excluded, np.ones((2, 3), dtype=np.bool_)
    )


def test_merge_falls_back_for_non_bool_new_and_preserves_dtype(monkeypatch):
    def rust_merge(*_args):
        raise AssertionError("non-bool marked arrays must use Python fallback")

    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: rust_merge)
    gatefilter = _gatefilter()
    marked = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16)

    gatefilter._merge(marked, "new", False)

    np.testing.assert_array_equal(gatefilter._gate_excluded, marked)
    assert gatefilter._gate_excluded.dtype == np.int16


def test_merge_falls_back_for_broadcast_shape(monkeypatch):
    def rust_merge(*_args):
        raise AssertionError("broadcast-shaped marked arrays must use Python fallback")

    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: rust_merge)
    gatefilter = _gatefilter([[False, True, False], [False, False, True]])

    gatefilter._merge(np.array([True, False, True], dtype=np.bool_), "or", False)

    expected = np.logical_or(
        np.array([[False, True, False], [False, False, True]], dtype=np.bool_),
        np.array([True, False, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(gatefilter._gate_excluded, expected)


def test_merge_value_errors_match_oracle_order_and_args(monkeypatch):
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    gatefilter = _gatefilter()
    marked = np.ma.array(
        [[False, False, False], [False, False, False]],
        mask=[[True, False, False], [False, False, False]],
    )

    with pytest.raises(ValueError) as exclude_error:
        gatefilter._merge(marked, "or", "bad")
    assert exclude_error.value.args == ("exclude_masked must be 'True' or 'False'",)

    with pytest.raises(ValueError) as op_error:
        gatefilter._merge(marked, "xor", True)
    assert op_error.value.args == ("invalid 'op' parameter: ", "xor")


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("op", ["or", "and", "new"])
@pytest.mark.parametrize("exclude_masked", [True, False])
def test_real_rust_gatefilter_merge_matches_python_fallback(
    monkeypatch, op, exclude_masked
):
    if importlib.util.find_spec("pyart._rust") is None:
        pytest.fail("pyart._rust is required for installed-package validation")

    import pyart._rust as rust

    rust_kernel = getattr(rust, "_gatefilter_merge", None)
    if rust_kernel is None:
        pytest.fail("pyart._rust has not registered _gatefilter_merge")

    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)
    marked = np.ma.array(
        [[True, False, False], [False, True, False]],
        mask=[[False, True, False], [True, False, True]],
    )

    expected_filter = _gatefilter(initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter._merge(marked, op, exclude_masked)

    actual_filter = _gatefilter(initial)
    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_gatefilter_merge" else None,
    )
    actual_filter._merge(marked, op, exclude_masked)

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


@pytest.mark.parametrize(
    ("method", "op", "inclusive", "exclude_masked"),
    [
        ("exclude_below", "or", False, True),
        ("exclude_below", "and", True, False),
        ("exclude_above", "new", False, True),
        ("exclude_above", "or", True, False),
        ("include_below", "and", False, True),
        ("include_below", "or", True, False),
        ("include_above", "and", False, True),
        ("include_above", "new", True, False),
    ],
)
def test_threshold_methods_match_python_fallback(
    monkeypatch, method, op, inclusive, exclude_masked
):
    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [np.nan, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        "reflectivity", 2.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    actual_filter = _gatefilter_with_field(data, initial)
    actual_filter_method = getattr(actual_filter, method)
    actual_filter_method(
        "reflectivity", 2.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


def test_threshold_method_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_compare_merge(
        gate_excluded,
        data,
        data_mask,
        value,
        comparator,
        invert_marked,
        exclude_masked,
        op,
    ):
        calls.append(
            (
                gate_excluded.copy(),
                data.copy(),
                data_mask.copy(),
                value,
                comparator,
                invert_marked,
                exclude_masked,
                op,
            )
        )
        return np.full(gate_excluded.shape, True, dtype=np.bool_)

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_compare_merge
        if name == "_gatefilter_compare_merge"
        else None,
    )
    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [3.0, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.zeros((2, 3), dtype=np.bool_)
    gatefilter = _gatefilter_with_field(data, initial)

    gatefilter.include_above(
        "reflectivity", 2.0, exclude_masked=False, op="and", inclusive=True
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], initial)
    np.testing.assert_array_equal(calls[0][1], np.ma.getdata(data))
    np.testing.assert_array_equal(calls[0][2], np.ma.getmaskarray(data))
    assert calls[0][3:] == (2.0, "ge", True, False, "and")
    np.testing.assert_array_equal(
        gatefilter._gate_excluded, np.ones((2, 3), dtype=np.bool_)
    )


@pytest.mark.parametrize(
    ("op", "inclusive", "exclude_masked"),
    [
        ("or", False, True),
        ("and", True, False),
        ("new", False, False),
    ],
)
def test_exclude_above_toa_matches_python_fallback(
    monkeypatch, op, inclusive, exclude_masked
):
    gate_altitude = np.ma.array(
        [[500.0, 1000.0, 1500.0], [np.nan, 2000.0, 2500.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    expected_filter = _gatefilter_with_gate_altitude(gate_altitude, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter.exclude_above_toa(
        1500.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    actual_filter = _gatefilter_with_gate_altitude(gate_altitude, initial)
    actual_filter.exclude_above_toa(
        1500.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


@pytest.mark.parametrize(
    ("inclusive", "expected_comparator"),
    [
        (False, "gt"),
        (True, "ge"),
    ],
)
def test_exclude_above_toa_dispatches_to_private_rust_kernel(
    monkeypatch, inclusive, expected_comparator
):
    calls = []

    def rust_compare_merge(
        gate_excluded,
        data,
        data_mask,
        value,
        comparator,
        invert_marked,
        exclude_masked,
        op,
    ):
        calls.append(
            (
                data.copy(),
                data_mask.copy(),
                value,
                comparator,
                invert_marked,
                exclude_masked,
                op,
            )
        )
        return np.full(gate_excluded.shape, True, dtype=np.bool_)

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_compare_merge
        if name == "_gatefilter_compare_merge"
        else None,
    )
    gate_altitude = np.ma.array(
        [[500.0, 1000.0, 1500.0], [np.nan, 2000.0, 2500.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    gatefilter = _gatefilter_with_gate_altitude(
        gate_altitude, np.zeros((2, 3), dtype=np.bool_)
    )

    gatefilter.exclude_above_toa(
        1500.0, exclude_masked=False, op="or", inclusive=inclusive
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.ma.getdata(gate_altitude))
    np.testing.assert_array_equal(calls[0][1], np.ma.getmaskarray(gate_altitude))
    assert calls[0][2:] == (1500.0, expected_comparator, False, False, "or")
    np.testing.assert_array_equal(
        gatefilter._gate_excluded, np.ones((2, 3), dtype=np.bool_)
    )


@pytest.mark.parametrize("value", ["1500", b"1500"])
def test_exclude_above_toa_text_threshold_keeps_python_exception(monkeypatch, value):
    gate_altitude = np.ma.array(
        [[500.0, 1000.0, 1500.0], [np.nan, 2000.0, 2500.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )

    expected_filter = _gatefilter_with_gate_altitude(gate_altitude)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    with pytest.raises(Exception) as expected_error:
        expected_filter.exclude_above_toa(value)

    def fail_if_called(_name):
        raise AssertionError("text thresholds must stay on the Python path")

    actual_filter = _gatefilter_with_gate_altitude(gate_altitude)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", fail_if_called)
    with pytest.raises(Exception) as actual_error:
        actual_filter.exclude_above_toa(value)

    assert type(actual_error.value) is type(expected_error.value)
    assert actual_error.value.args == expected_error.value.args


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("op", "inclusive", "exclude_masked"),
    [
        ("or", False, True),
        ("and", True, False),
        ("new", False, False),
    ],
)
def test_real_rust_gatefilter_exclude_above_toa_matches_python_fallback(
    monkeypatch, op, inclusive, exclude_masked
):
    import pyart._rust as rust

    kernel = getattr(rust, "_gatefilter_compare_merge", None)
    assert kernel is not None

    gate_altitude = np.ma.array(
        [[500.0, 1000.0, 1500.0], [np.nan, 2000.0, 2500.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    expected_filter = _gatefilter_with_gate_altitude(gate_altitude, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter.exclude_above_toa(
        1500.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    actual_filter = _gatefilter_with_gate_altitude(gate_altitude, initial)

    calls = []

    def rust_kernel(name):
        calls.append(name)
        return kernel if name == "_gatefilter_compare_merge" else None

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        rust_kernel,
    )
    actual_filter.exclude_above_toa(
        1500.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    assert calls == ["_gatefilter_compare_merge"]
    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


@pytest.mark.parametrize(
    ("method", "trans_value", "op", "exclude_masked"),
    [
        ("exclude_transition", 1.0, "or", True),
        ("exclude_transition", np.nan, "new", False),
        ("include_not_transition", 0.0, "and", True),
        ("include_not_transition", np.nan, "new", False),
    ],
)
def test_transition_methods_match_python_fallback(
    monkeypatch, method, trans_value, op, exclude_masked
):
    transition = np.array([0.0, 1.0, np.nan], dtype=np.float64)
    initial = np.array(
        [[False, True, False], [True, False, True], [False, False, True]],
        dtype=np.bool_,
    )

    expected_filter = _gatefilter_with_transition(transition, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        trans_value=trans_value, exclude_masked=exclude_masked, op=op
    )

    actual_filter = _gatefilter_with_transition(transition, initial)
    getattr(actual_filter, method)(
        trans_value=trans_value, exclude_masked=exclude_masked, op=op
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


def test_transition_methods_without_transition_match_python_fallback(monkeypatch):
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    for method in ("exclude_transition", "include_not_transition"):
        expected_filter = _gatefilter_with_transition(None, initial)
        monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
        getattr(expected_filter, method)(op="new")

        actual_filter = _gatefilter_with_transition(None, initial)
        getattr(actual_filter, method)(op="new")

        np.testing.assert_array_equal(
            actual_filter._gate_excluded, expected_filter._gate_excluded
        )


@pytest.mark.parametrize(
    ("method", "trans_value", "op"),
    [
        ("exclude_transition", 1.0, "new"),
        ("include_not_transition", 0.0, "new"),
    ],
)
def test_transition_methods_masked_transition_preserves_oracle_exclude_masked_behavior(
    monkeypatch, method, trans_value, op
):
    transition = np.ma.array(
        [0.0, 1.0, 0.0],
        mask=[False, True, False],
        dtype=np.float64,
    )

    def fail_transition_kernel(name):
        if name == "_gatefilter_transition_merge":
            raise AssertionError("masked antenna_transition must use Python fallback")
        return None

    outputs = []
    for exclude_masked in (False, True):
        expected_filter = _gatefilter_with_transition(transition)
        monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
        getattr(expected_filter, method)(
            trans_value=trans_value, exclude_masked=exclude_masked, op=op
        )

        actual_filter = _gatefilter_with_transition(transition)
        monkeypatch.setattr(gatefilter_module, "_rust_kernel", fail_transition_kernel)
        getattr(actual_filter, method)(
            trans_value=trans_value, exclude_masked=exclude_masked, op=op
        )

        np.testing.assert_array_equal(
            actual_filter._gate_excluded, expected_filter._gate_excluded
        )
        outputs.append(actual_filter._gate_excluded.copy())

    np.testing.assert_array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize(
    ("method", "expected_invert"),
    [
        ("exclude_transition", False),
        ("include_not_transition", True),
    ],
)
def test_transition_methods_dispatch_to_private_rust_kernel(
    monkeypatch, method, expected_invert
):
    calls = []

    def rust_transition_merge(gate_excluded, transitions, trans_value, invert, op):
        calls.append(
            (gate_excluded.copy(), transitions.copy(), trans_value, invert, op)
        )
        return np.full(gate_excluded.shape, True, dtype=np.bool_)

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_transition_merge
        if name == "_gatefilter_transition_merge"
        else None,
    )
    transition = np.array([0.0, 1.0, np.nan], dtype=np.float64)
    initial = np.zeros((3, 3), dtype=np.bool_)
    gatefilter = _gatefilter_with_transition(transition, initial)

    getattr(gatefilter, method)(trans_value=np.float64(1.0), exclude_masked=False, op="or")

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], initial)
    np.testing.assert_array_equal(calls[0][1], transition)
    assert calls[0][2:] == (1.0, expected_invert, "or")
    np.testing.assert_array_equal(
        gatefilter._gate_excluded, np.ones((3, 3), dtype=np.bool_)
    )


@pytest.mark.parametrize(
    ("transition", "trans_value", "op", "exclude_masked"),
    [
        (np.array([0, 1, 0], dtype=np.int32), 1, "or", True),
        (np.array([0.0, 1.0, 2.0], dtype=np.float64)[::2], 1.0, "or", True),
        (np.ma.array([0.0, 1.0, 0.0], mask=[False, True, False]), 1.0, "or", True),
        ([0.0, 1.0, 0.0], 1.0, "or", True),
        (np.array([[0.0, 1.0, 0.0]], dtype=np.float64), 1.0, "or", True),
        (np.array([0.0, 1.0, 0.0], dtype=np.float64), "1", "or", True),
        (np.array([0.0, 1.0, 0.0], dtype=np.float64), 1.0, "xor", True),
        (np.array([0.0, 1.0, 0.0], dtype=np.float64), 1.0, "or", "bad"),
    ],
)
@pytest.mark.parametrize("method", ["exclude_transition", "include_not_transition"])
def test_transition_methods_keep_python_path_for_unsupported_inputs(
    monkeypatch, method, transition, trans_value, op, exclude_masked
):
    def fail_transition_kernel(name):
        if name == "_gatefilter_transition_merge":
            raise AssertionError("unsupported transition input should use fallback")
        return None

    actual_filter = _gatefilter_with_transition(transition)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", fail_transition_kernel)

    try:
        getattr(actual_filter, method)(
            trans_value=trans_value, exclude_masked=exclude_masked, op=op
        )
    except Exception as actual_error:
        expected_filter = _gatefilter_with_transition(transition)
        monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            getattr(expected_filter, method)(
                trans_value=trans_value, exclude_masked=exclude_masked, op=op
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected_filter = _gatefilter_with_transition(transition)
        monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
        getattr(expected_filter, method)(
            trans_value=trans_value, exclude_masked=exclude_masked, op=op
        )
        np.testing.assert_array_equal(
            actual_filter._gate_excluded, expected_filter._gate_excluded
        )
        assert type(actual_filter._gate_excluded) is type(expected_filter._gate_excluded)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("method", ["exclude_transition", "include_not_transition"])
@pytest.mark.parametrize("trans_value", [0.0, 1.0, np.nan])
@pytest.mark.parametrize("op", ["or", "and", "new"])
def test_real_rust_gatefilter_transition_methods_match_python_fallback(
    monkeypatch, method, trans_value, op
):
    import pyart._rust as rust

    kernel = getattr(rust, "_gatefilter_transition_merge", None)
    assert kernel is not None

    transition = np.array([0.0, 1.0, np.nan], dtype=np.float64)
    initial = np.array(
        [[False, True, False], [True, False, True], [False, False, True]],
        dtype=np.bool_,
    )

    expected_filter = _gatefilter_with_transition(transition, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(trans_value=trans_value, op=op)

    actual_filter = _gatefilter_with_transition(transition, initial)
    calls = []

    def rust_kernel(name):
        calls.append(name)
        return kernel if name == "_gatefilter_transition_merge" else None

    monkeypatch.setattr(gatefilter_module, "_rust_kernel", rust_kernel)
    getattr(actual_filter, method)(trans_value=trans_value, op=op)

    assert calls == ["_gatefilter_transition_merge"]
    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gatefilter_transition_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    gate_excluded = np.zeros((3, 2), dtype=np.bool_)

    with pytest.raises(ValueError, match="length"):
        rust._gatefilter_transition_merge(
            gate_excluded, np.array([0.0, 1.0], dtype=np.float64), 1.0, False, "or"
        )

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._gatefilter_transition_merge(
            gate_excluded,
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)[::2],
            1.0,
            False,
            "or",
        )

    with pytest.raises(ValueError) as exc_info:
        rust._gatefilter_transition_merge(
            gate_excluded, np.array([0.0, 1.0, 2.0], dtype=np.float64), 1.0, False, "xor"
        )
    assert exc_info.value.args == ("invalid 'op' parameter: ", "xor")


@pytest.mark.parametrize("n_gates", [1, 2, 10, 0, -1, -2, -3, -4])
@pytest.mark.parametrize(
    ("op", "exclude_masked"),
    [
        ("or", True),
        ("and", False),
        ("new", True),
    ],
)
def test_exclude_last_gates_matches_python_fallback(
    monkeypatch, n_gates, op, exclude_masked
):
    data = np.ma.array(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter.exclude_last_gates(
        "reflectivity", n_gates=n_gates, exclude_masked=exclude_masked, op=op
    )

    actual_filter = _gatefilter_with_field(data, initial)
    actual_filter.exclude_last_gates(
        "reflectivity", n_gates=n_gates, exclude_masked=exclude_masked, op=op
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


def test_exclude_last_gates_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_last_gates_merge(gate_excluded, n_gates, op):
        calls.append((gate_excluded.copy(), n_gates, op))
        return np.full(gate_excluded.shape, True, dtype=np.bool_)

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_last_gates_merge
        if name == "_gatefilter_last_gates_merge"
        else None,
    )
    data = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)
    gatefilter = _gatefilter_with_field(data, initial)

    gatefilter.exclude_last_gates(
        "reflectivity", n_gates=np.int64(2), exclude_masked=False, op="and"
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], initial)
    assert calls[0][1:] == (2, "and")
    np.testing.assert_array_equal(
        gatefilter._gate_excluded, np.ones((2, 3), dtype=np.bool_)
    )


@pytest.mark.parametrize(
    ("data", "n_gates", "op", "exclude_masked"),
    [
        (np.arange(6.0, dtype=np.float64).reshape(2, 3), 1.5, "or", True),
        (np.arange(6.0, dtype=np.float64).reshape(2, 3), "2", "or", True),
        (np.arange(3.0, dtype=np.float64), 1, "or", True),
        (np.arange(6.0, dtype=np.float64).reshape(2, 3), 1, "xor", True),
        (np.arange(6.0, dtype=np.float64).reshape(2, 3), 1, "or", "bad"),
    ],
)
def test_exclude_last_gates_keeps_python_path_for_unsupported_inputs(
    monkeypatch, data, n_gates, op, exclude_masked
):
    def fail_last_gates_kernel(name):
        if name == "_gatefilter_last_gates_merge":
            raise AssertionError("unsupported last-gates input should use fallback")
        return None

    actual_filter = _gatefilter_with_field(data)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", fail_last_gates_kernel)

    try:
        actual_filter.exclude_last_gates(
            "reflectivity", n_gates=n_gates, exclude_masked=exclude_masked, op=op
        )
    except Exception as actual_error:
        expected_filter = _gatefilter_with_field(data)
        monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            expected_filter.exclude_last_gates(
                "reflectivity",
                n_gates=n_gates,
                exclude_masked=exclude_masked,
                op=op,
            )
        assert actual_error.args == expected_error.value.args
    else:
        expected_filter = _gatefilter_with_field(data)
        monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
        expected_filter.exclude_last_gates(
            "reflectivity", n_gates=n_gates, exclude_masked=exclude_masked, op=op
        )
        np.testing.assert_array_equal(
            actual_filter._gate_excluded, expected_filter._gate_excluded
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("n_gates", [0, 1, -1, 10])
@pytest.mark.parametrize("op", ["or", "and", "new"])
def test_real_rust_gatefilter_exclude_last_gates_matches_python_fallback(
    monkeypatch, n_gates, op
):
    import pyart._rust as rust

    kernel = getattr(rust, "_gatefilter_last_gates_merge", None)
    assert kernel is not None

    data = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter.exclude_last_gates(
        "reflectivity", n_gates=n_gates, exclude_masked=True, op=op
    )

    actual_filter = _gatefilter_with_field(data, initial)
    calls = []

    def rust_kernel(name):
        calls.append(name)
        return kernel if name == "_gatefilter_last_gates_merge" else None

    monkeypatch.setattr(gatefilter_module, "_rust_kernel", rust_kernel)
    actual_filter.exclude_last_gates(
        "reflectivity", n_gates=n_gates, exclude_masked=True, op=op
    )

    assert calls == ["_gatefilter_last_gates_merge"]
    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gatefilter_last_gates_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._gatefilter_last_gates_merge(
            np.zeros((2, 3), dtype=np.bool_)[:, ::-1], 1, "or"
        )

    with pytest.raises(ValueError) as exc_info:
        rust._gatefilter_last_gates_merge(
            np.zeros((2, 3), dtype=np.bool_), 1, "xor"
        )
    assert exc_info.value.args == ("invalid 'op' parameter: ", "xor")


@pytest.mark.parametrize("exclude_masked", [False, True])
def test_include_threshold_masked_gates_use_exclude_masked_after_inversion(
    monkeypatch, exclude_masked
):
    data = np.ma.array(
        [[0.0, 2.0, 4.0]],
        mask=[[False, True, False]],
        dtype=np.float64,
    )
    initial = np.ones((1, 3), dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter.include_below(
        "reflectivity", 3.0, exclude_masked=exclude_masked, op="and"
    )

    actual_filter = _gatefilter_with_field(data, initial)
    actual_filter.include_below(
        "reflectivity", 3.0, exclude_masked=exclude_masked, op="and"
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded[0, 1] == exclude_masked


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("exclude_below", {"value": "bad"}),
        ("exclude_below", {"op": "xor"}),
        ("exclude_below", {"exclude_masked": "bad"}),
        ("include_above", {"value": True}),
    ],
)
def test_threshold_methods_keep_unsupported_inputs_on_python_path(
    monkeypatch, method, kwargs
):
    kwargs = dict(kwargs)
    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [3.0, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    value = kwargs.pop("value", 2.0)
    op = kwargs.pop("op", "or" if method.startswith("exclude") else "and")
    exclude_masked = kwargs.pop("exclude_masked", True)

    def fail_if_called(name):
        if name != "_gatefilter_compare_merge":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported threshold input used Rust")

        return kernel

    monkeypatch.setattr(gatefilter_module, "_rust_kernel", fail_if_called)
    gatefilter = _gatefilter_with_field(data)

    if value == "bad":
        with pytest.raises(TypeError):
            getattr(gatefilter, method)(
                "reflectivity", value, exclude_masked=exclude_masked, op=op
            )
    elif op == "xor":
        with pytest.raises(ValueError) as exc_info:
            getattr(gatefilter, method)(
                "reflectivity", value, exclude_masked=exclude_masked, op=op
            )
        assert exc_info.value.args == ("invalid 'op' parameter: ", "xor")
    elif exclude_masked == "bad":
        with pytest.raises(ValueError) as exc_info:
            getattr(gatefilter, method)(
                "reflectivity", value, exclude_masked=exclude_masked, op=op
            )
        assert exc_info.value.args == ("exclude_masked must be 'True' or 'False'",)
    else:
        getattr(gatefilter, method)("reflectivity", value, exclude_masked=exclude_masked, op=op)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("method", ["exclude_below", "exclude_above", "include_below", "include_above"])
@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("exclude_masked", [False, True])
def test_real_rust_gatefilter_threshold_methods_match_python_fallback(
    monkeypatch, method, inclusive, exclude_masked
):
    import pyart._rust as rust

    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [np.nan, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)
    op = "or" if method.startswith("exclude") else "and"

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        "reflectivity", 2.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    actual_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    getattr(actual_filter, method)(
        "reflectivity", 2.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("gate_excluded", "data", "data_mask", "comparator", "match"),
    [
        (
            np.zeros((2, 3), dtype=np.bool_),
            np.ones((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.bool_),
            "lt",
            "same shape",
        ),
        (
            np.zeros((2, 3), dtype=np.bool_)[:, ::-1],
            np.ones((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.bool_),
            "lt",
            "C-contiguous",
        ),
        (
            np.zeros((2, 3), dtype=np.bool_),
            np.ones((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.bool_),
            "bad",
            "invalid comparator",
        ),
    ],
)
def test_real_rust_gatefilter_threshold_rejects_unsafe_direct_inputs(
    gate_excluded, data, data_mask, comparator, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._gatefilter_compare_merge(
            gate_excluded, data, data_mask, 1.0, comparator, False, True, "or"
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gatefilter_threshold_rejects_invalid_op_with_python_args():
    import pyart._rust as rust

    gate_excluded = np.zeros((1, 2), dtype=np.bool_)
    data = np.ones((1, 2), dtype=np.float64)
    data_mask = np.zeros((1, 2), dtype=np.bool_)

    with pytest.raises(ValueError) as exc_info:
        rust._gatefilter_compare_merge(
            gate_excluded, data, data_mask, 1.0, "eq", False, True, "xor"
        )

    assert exc_info.value.args == ("invalid 'op' parameter: ", "xor")


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gatefilter_threshold_rejects_non_bool_exclude_masked():
    import pyart._rust as rust

    gate_excluded = np.zeros((1, 2), dtype=np.bool_)
    data = np.ones((1, 2), dtype=np.float64)
    data_mask = np.zeros((1, 2), dtype=np.bool_)

    with pytest.raises(TypeError, match="exclude_masked.*bool"):
        rust._gatefilter_compare_merge(
            gate_excluded, data, data_mask, 1.0, "eq", False, "bad", "or"
        )


@pytest.mark.parametrize(
    ("method", "op", "exclude_masked", "value"),
    [
        ("exclude_equal", "or", True, 2.0),
        ("exclude_equal", "and", False, np.nan),
        ("exclude_not_equal", "new", True, 2.0),
        ("exclude_not_equal", "or", False, np.nan),
        ("include_equal", "and", True, 2.0),
        ("include_equal", "or", False, np.nan),
        ("include_not_equal", "and", True, 2.0),
        ("include_not_equal", "new", False, np.nan),
    ],
)
def test_equality_methods_match_python_fallback(
    monkeypatch, method, op, exclude_masked, value
):
    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [np.nan, 5.0, 2.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        "reflectivity", value, exclude_masked=exclude_masked, op=op
    )

    actual_filter = _gatefilter_with_field(data, initial)
    getattr(actual_filter, method)(
        "reflectivity", value, exclude_masked=exclude_masked, op=op
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


def test_equality_method_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_compare_merge(
        gate_excluded,
        data,
        data_mask,
        value,
        comparator,
        invert_marked,
        exclude_masked,
        op,
    ):
        calls.append((value, comparator, invert_marked, exclude_masked, op))
        return np.full(gate_excluded.shape, True, dtype=np.bool_)

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_compare_merge
        if name == "_gatefilter_compare_merge"
        else None,
    )
    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [3.0, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    gatefilter = _gatefilter_with_field(data, np.zeros((2, 3), dtype=np.bool_))

    gatefilter.include_not_equal("reflectivity", 2.0, exclude_masked=False, op="and")

    assert calls == [(2.0, "ne", True, False, "and")]
    np.testing.assert_array_equal(
        gatefilter._gate_excluded, np.ones((2, 3), dtype=np.bool_)
    )


@pytest.mark.parametrize("exclude_masked", [False, True])
def test_include_equality_masked_gates_use_exclude_masked_after_inversion(
    monkeypatch, exclude_masked
):
    data = np.ma.array(
        [[0.0, 2.0, 4.0]],
        mask=[[False, True, False]],
        dtype=np.float64,
    )
    initial = np.ones((1, 3), dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter.include_equal(
        "reflectivity", 2.0, exclude_masked=exclude_masked, op="and"
    )

    actual_filter = _gatefilter_with_field(data, initial)
    actual_filter.include_equal(
        "reflectivity", 2.0, exclude_masked=exclude_masked, op="and"
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded[0, 1] == exclude_masked


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    "method", ["exclude_equal", "exclude_not_equal", "include_equal", "include_not_equal"]
)
@pytest.mark.parametrize("exclude_masked", [False, True])
@pytest.mark.parametrize("value", [2.0, np.nan])
def test_real_rust_gatefilter_equality_methods_match_python_fallback(
    monkeypatch, method, exclude_masked, value
):
    import pyart._rust as rust

    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [np.nan, 5.0, 2.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)
    op = "or" if method.startswith("exclude") else "and"

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        "reflectivity", value, exclude_masked=exclude_masked, op=op
    )

    actual_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    getattr(actual_filter, method)(
        "reflectivity", value, exclude_masked=exclude_masked, op=op
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )


@pytest.mark.parametrize(
    ("method", "op", "exclude_masked"),
    [
        ("exclude_invalid", "or", True),
        ("exclude_invalid", "new", False),
        ("include_valid", "and", True),
        ("include_valid", "or", False),
    ],
)
def test_finite_methods_match_python_fallback(monkeypatch, method, op, exclude_masked):
    data = np.ma.array(
        [[-1.0, np.nan, np.inf], [-np.inf, 5.0, 2.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        "reflectivity", exclude_masked=exclude_masked, op=op
    )

    actual_filter = _gatefilter_with_field(data, initial)
    getattr(actual_filter, method)("reflectivity", exclude_masked=exclude_masked, op=op)

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


def test_finite_method_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_finite_merge(gate_excluded, data, data_mask, exclude_masked, op):
        calls.append((data.copy(), data_mask.copy(), exclude_masked, op))
        return np.full(gate_excluded.shape, True, dtype=np.bool_)

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_finite_merge if name == "_gatefilter_finite_merge" else None,
    )
    data = np.ma.array(
        [[-1.0, np.nan, 2.0], [np.inf, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    gatefilter = _gatefilter_with_field(data, np.zeros((2, 3), dtype=np.bool_))

    gatefilter.include_valid("reflectivity", exclude_masked=False, op="and")

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.ma.getdata(data))
    np.testing.assert_array_equal(calls[0][1], np.ma.getmaskarray(data))
    assert calls[0][2:] == (False, "and")
    np.testing.assert_array_equal(
        gatefilter._gate_excluded, np.ones((2, 3), dtype=np.bool_)
    )


@pytest.mark.parametrize("exclude_masked", [False, True])
def test_include_valid_masked_gates_use_exclude_masked(monkeypatch, exclude_masked):
    data = np.ma.array(
        [[1.0, 2.0, np.inf]],
        mask=[[False, True, False]],
        dtype=np.float64,
    )
    initial = np.ones((1, 3), dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter.include_valid(
        "reflectivity", exclude_masked=exclude_masked, op="and"
    )

    actual_filter = _gatefilter_with_field(data, initial)
    actual_filter.include_valid(
        "reflectivity", exclude_masked=exclude_masked, op="and"
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded[0, 1] == exclude_masked


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("method", ["exclude_invalid", "include_valid"])
@pytest.mark.parametrize("exclude_masked", [False, True])
def test_real_rust_gatefilter_finite_methods_match_python_fallback(
    monkeypatch, method, exclude_masked
):
    import pyart._rust as rust

    data = np.ma.array(
        [[-1.0, np.nan, np.inf], [-np.inf, 5.0, 2.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)
    op = "or" if method == "exclude_invalid" else "and"

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        "reflectivity", exclude_masked=exclude_masked, op=op
    )

    actual_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    getattr(actual_filter, method)("reflectivity", exclude_masked=exclude_masked, op=op)

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("gate_excluded", "data", "data_mask", "match"),
    [
        (
            np.zeros((2, 3), dtype=np.bool_),
            np.ones((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.bool_),
            "same shape",
        ),
        (
            np.zeros((2, 3), dtype=np.bool_)[:, ::-1],
            np.ones((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.bool_),
            "C-contiguous",
        ),
    ],
)
def test_real_rust_gatefilter_finite_rejects_unsafe_direct_inputs(
    gate_excluded, data, data_mask, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._gatefilter_finite_merge(gate_excluded, data, data_mask, True, "or")


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gatefilter_finite_rejects_invalid_op_with_python_args():
    import pyart._rust as rust

    gate_excluded = np.zeros((1, 2), dtype=np.bool_)
    data = np.ones((1, 2), dtype=np.float64)
    data_mask = np.zeros((1, 2), dtype=np.bool_)

    with pytest.raises(ValueError) as exc_info:
        rust._gatefilter_finite_merge(gate_excluded, data, data_mask, True, "xor")

    assert exc_info.value.args == ("invalid 'op' parameter: ", "xor")


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gatefilter_finite_rejects_non_bool_exclude_masked():
    import pyart._rust as rust

    gate_excluded = np.zeros((1, 2), dtype=np.bool_)
    data = np.ones((1, 2), dtype=np.float64)
    data_mask = np.zeros((1, 2), dtype=np.bool_)

    with pytest.raises(TypeError, match="exclude_masked.*bool"):
        rust._gatefilter_finite_merge(gate_excluded, data, data_mask, "bad", "or")


@pytest.mark.parametrize(
    ("method", "op", "inclusive", "exclude_masked", "v1", "v2"),
    [
        ("exclude_inside", "or", False, True, 4.0, 1.0),
        ("exclude_inside", "and", True, False, 1.0, 4.0),
        ("exclude_outside", "new", False, True, 1.0, 4.0),
        ("exclude_outside", "or", True, False, 4.0, 1.0),
        ("include_inside", "and", False, True, 1.0, 4.0),
        ("include_inside", "or", True, False, 4.0, 1.0),
        ("include_outside", "and", False, True, 1.0, 4.0),
        ("include_outside", "new", True, False, 4.0, 1.0),
    ],
)
def test_interval_methods_match_python_fallback(
    monkeypatch, method, op, inclusive, exclude_masked, v1, v2
):
    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [np.nan, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        "reflectivity", v1, v2, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    actual_filter = _gatefilter_with_field(data, initial)
    getattr(actual_filter, method)(
        "reflectivity", v1, v2, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded.dtype == expected_filter._gate_excluded.dtype


def test_interval_method_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_interval_merge(
        gate_excluded,
        data,
        data_mask,
        v1,
        v2,
        mode,
        inclusive,
        invert_marked,
        exclude_masked,
        op,
    ):
        calls.append(
            (
                gate_excluded.copy(),
                data.copy(),
                data_mask.copy(),
                v1,
                v2,
                mode,
                inclusive,
                invert_marked,
                exclude_masked,
                op,
            )
        )
        return np.full(gate_excluded.shape, True, dtype=np.bool_)

    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: rust_interval_merge
        if name == "_gatefilter_interval_merge"
        else None,
    )
    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [3.0, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.zeros((2, 3), dtype=np.bool_)
    gatefilter = _gatefilter_with_field(data, initial)

    gatefilter.include_outside(
        "reflectivity", 5.0, 1.0, exclude_masked=False, op="and", inclusive=True
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], initial)
    np.testing.assert_array_equal(calls[0][1], np.ma.getdata(data))
    np.testing.assert_array_equal(calls[0][2], np.ma.getmaskarray(data))
    assert calls[0][3:] == (1.0, 5.0, "outside", True, True, False, "and")
    np.testing.assert_array_equal(
        gatefilter._gate_excluded, np.ones((2, 3), dtype=np.bool_)
    )


@pytest.mark.parametrize("exclude_masked", [False, True])
def test_include_interval_masked_gates_use_exclude_masked_after_inversion(
    monkeypatch, exclude_masked
):
    data = np.ma.array(
        [[0.0, 2.0, 4.0]],
        mask=[[False, True, False]],
        dtype=np.float64,
    )
    initial = np.ones((1, 3), dtype=np.bool_)

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    expected_filter.include_inside(
        "reflectivity", 1.0, 3.0, exclude_masked=exclude_masked, op="and"
    )

    actual_filter = _gatefilter_with_field(data, initial)
    actual_filter.include_inside(
        "reflectivity", 1.0, 3.0, exclude_masked=exclude_masked, op="and"
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )
    assert actual_filter._gate_excluded[0, 1] == exclude_masked


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("exclude_inside", {"v1": "bad", "v2": 2.0}),
        ("exclude_inside", {"op": "xor"}),
        ("exclude_inside", {"exclude_masked": "bad"}),
        ("include_outside", {"v1": True, "v2": 2.0}),
    ],
)
def test_interval_methods_keep_unsupported_inputs_on_python_path(
    monkeypatch, method, kwargs
):
    kwargs = dict(kwargs)
    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [3.0, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    v1 = kwargs.pop("v1", 1.0)
    v2 = kwargs.pop("v2", 4.0)
    op = kwargs.pop("op", "or" if method.startswith("exclude") else "and")
    exclude_masked = kwargs.pop("exclude_masked", True)

    def fail_if_called(name):
        if name != "_gatefilter_interval_merge":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported interval input used Rust")

        return kernel

    monkeypatch.setattr(gatefilter_module, "_rust_kernel", fail_if_called)
    gatefilter = _gatefilter_with_field(data)

    if v1 == "bad":
        with pytest.raises(TypeError):
            getattr(gatefilter, method)(
                "reflectivity", v1, v2, exclude_masked=exclude_masked, op=op
            )
    elif op == "xor":
        with pytest.raises(ValueError) as exc_info:
            getattr(gatefilter, method)(
                "reflectivity", v1, v2, exclude_masked=exclude_masked, op=op
            )
        assert exc_info.value.args == ("invalid 'op' parameter: ", "xor")
    elif exclude_masked == "bad":
        with pytest.raises(ValueError) as exc_info:
            getattr(gatefilter, method)(
                "reflectivity", v1, v2, exclude_masked=exclude_masked, op=op
            )
        assert exc_info.value.args == ("exclude_masked must be 'True' or 'False'",)
    else:
        getattr(gatefilter, method)(
            "reflectivity", v1, v2, exclude_masked=exclude_masked, op=op
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    "method", ["exclude_inside", "exclude_outside", "include_inside", "include_outside"]
)
@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("exclude_masked", [False, True])
def test_real_rust_gatefilter_interval_methods_match_python_fallback(
    monkeypatch, method, inclusive, exclude_masked
):
    import pyart._rust as rust

    data = np.ma.array(
        [[-1.0, 0.0, 2.0], [np.nan, 5.0, 9.0]],
        mask=[[False, True, False], [False, False, True]],
        dtype=np.float64,
    )
    initial = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)
    op = "or" if method.startswith("exclude") else "and"

    expected_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(gatefilter_module, "_rust_kernel", lambda _name: None)
    getattr(expected_filter, method)(
        "reflectivity", 4.0, 1.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    actual_filter = _gatefilter_with_field(data, initial)
    monkeypatch.setattr(
        gatefilter_module,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    getattr(actual_filter, method)(
        "reflectivity", 4.0, 1.0, exclude_masked=exclude_masked, op=op, inclusive=inclusive
    )

    np.testing.assert_array_equal(
        actual_filter._gate_excluded, expected_filter._gate_excluded
    )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("gate_excluded", "data", "data_mask", "mode", "match"),
    [
        (
            np.zeros((2, 3), dtype=np.bool_),
            np.ones((2, 2), dtype=np.float64),
            np.zeros((2, 3), dtype=np.bool_),
            "inside",
            "same shape",
        ),
        (
            np.zeros((2, 3), dtype=np.bool_)[:, ::-1],
            np.ones((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.bool_),
            "inside",
            "C-contiguous",
        ),
        (
            np.zeros((2, 3), dtype=np.bool_),
            np.ones((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.bool_),
            "bad",
            "invalid interval mode",
        ),
    ],
)
def test_real_rust_gatefilter_interval_rejects_unsafe_direct_inputs(
    gate_excluded, data, data_mask, mode, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._gatefilter_interval_merge(
            gate_excluded, data, data_mask, 1.0, 4.0, mode, True, False, True, "or"
        )


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gatefilter_interval_rejects_invalid_op_with_python_args():
    import pyart._rust as rust

    gate_excluded = np.zeros((1, 2), dtype=np.bool_)
    data = np.ones((1, 2), dtype=np.float64)
    data_mask = np.zeros((1, 2), dtype=np.bool_)

    with pytest.raises(ValueError) as exc_info:
        rust._gatefilter_interval_merge(
            gate_excluded,
            data,
            data_mask,
            1.0,
            4.0,
            "inside",
            True,
            False,
            True,
            "xor",
        )

    assert exc_info.value.args == ("invalid 'op' parameter: ", "xor")


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gatefilter_interval_rejects_non_bool_exclude_masked():
    import pyart._rust as rust

    gate_excluded = np.zeros((1, 2), dtype=np.bool_)
    data = np.ones((1, 2), dtype=np.float64)
    data_mask = np.zeros((1, 2), dtype=np.bool_)

    with pytest.raises(TypeError, match="exclude_masked.*bool"):
        rust._gatefilter_interval_merge(
            gate_excluded,
            data,
            data_mask,
            1.0,
            4.0,
            "inside",
            True,
            False,
            "bad",
            "or",
        )
