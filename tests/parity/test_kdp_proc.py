import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import _kdp_proc, kdp_proc  # noqa: E402


def _reference_term(k, dr):
    out = np.empty_like(k)
    dr2 = dr**2.0
    nr, ng = k.shape
    for r in range(nr):
        for g in range(ng):
            if g > 0 and g < ng - 1:
                out[r, g] = (k[r, g + 1] - 2.0 * k[r, g] + k[r, g - 1]) / dr2
            elif g == 0:
                out[r, g] = (k[r, g] - 2.0 * k[r, g + 1] + k[r, g + 2]) / dr2
            else:
                out[r, g] = (k[r, g] - 2.0 * k[r, g - 1] + k[r, g - 2]) / dr2
    return out


def _reference_jac(d2kdr2, dr, clpf):
    out = np.empty_like(d2kdr2)
    scale = clpf / dr**2.0
    nr, ng = d2kdr2.shape
    for r in range(nr):
        for g in range(ng):
            if g > 2 and g < ng - 3:
                out[r, g] = scale * (
                    d2kdr2[r, g - 1] - 2.0 * d2kdr2[r, g] + d2kdr2[r, g + 1]
                )
            elif g == 2:
                out[r, g] = scale * (
                    d2kdr2[r, g - 2]
                    + d2kdr2[r, g - 1]
                    - 2.0 * d2kdr2[r, g]
                    + d2kdr2[r, g + 1]
                )
            elif g == 1:
                out[r, g] = scale * (
                    d2kdr2[r, g + 1]
                    - 2.0 * d2kdr2[r, g]
                    - 2.0 * d2kdr2[r, g - 1]
                )
            elif g == 0:
                out[r, g] = scale * (d2kdr2[r, g] + d2kdr2[r, g + 1])
            elif g == ng - 3:
                out[r, g] = scale * (
                    d2kdr2[r, g + 2]
                    + d2kdr2[r, g + 1]
                    - 2.0 * d2kdr2[r, g]
                    + d2kdr2[r, g - 1]
                )
            elif g == ng - 2:
                out[r, g] = scale * (
                    d2kdr2[r, g - 1]
                    - 2.0 * d2kdr2[r, g]
                    - 2.0 * d2kdr2[r, g + 1]
                )
            else:
                out[r, g] = scale * (d2kdr2[r, g] + d2kdr2[r, g - 1])
    return out


def _reference_forward_reverse_phidp(k, bcs):
    _, ng = k.shape
    phi_near, phi_far = bcs

    phi_f = np.zeros_like(k, subok=False)
    phi_f[:, 1:] = np.cumsum(k[:, :-1] ** 2, axis=1)
    phidp_f = phi_f + phi_near[:, np.newaxis].repeat(ng, axis=1)

    phi_r = np.zeros_like(k, subok=False)
    phi_r[:, :-1] = np.cumsum(k[:, :0:-1] ** 2, axis=1)[:, ::-1]
    phidp_r = phi_far[:, np.newaxis].repeat(ng, axis=1) - phi_r

    return phidp_f, phidp_r


def _fallback_forward_reverse_phidp(k, bcs, monkeypatch):
    monkeypatch.setattr(_kdp_proc, "_rust_kernel", lambda _name: None)
    return _kdp_proc.forward_reverse_phidp(k, bcs)


def _range_radar(ranges):
    return SimpleNamespace(range={"data": ranges})


def _fallback_parse_range_resolution(
    ranges, monkeypatch, check_uniform=True, atol=1.0, verbose=False
):
    monkeypatch.setattr(_kdp_proc, "_rust_kernel", lambda _name: None)
    return kdp_proc._parse_range_resolution(
        _range_radar(ranges),
        check_uniform=check_uniform,
        atol=atol,
        verbose=verbose,
    )


def _forward_reverse_case(ng):
    if ng == 0:
        k = np.empty((2, 0), dtype=np.float64)
    else:
        k = np.linspace(-2.5, 3.5, 2 * ng, dtype=np.float64).reshape(2, ng)
        if ng > 2:
            k[0, 2] = np.nan
    phi_near = np.array([1.25, -0.0], dtype=np.float64)
    phi_far = np.array([9.5, -0.0], dtype=np.float64)
    return k, [phi_near, phi_far]


def _assert_forward_reverse_equal(actual, expected):
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_array_equal(actual_part, expected_part)
        np.testing.assert_array_equal(np.signbit(actual_part), np.signbit(expected_part))


def test_lowpass_maesaka_term_matches_oracle_formula_in_place():
    k = np.linspace(-2.5, 5.5, 24, dtype=np.float64).reshape(3, 8)
    out = np.full_like(k, -999.0)

    result = _kdp_proc.lowpass_maesaka_term(k, 250.0, "low", out)

    assert result is None
    np.testing.assert_array_equal(out, _reference_term(k, 250.0))


def test_lowpass_maesaka_jac_matches_oracle_formula_in_place():
    d2kdr2 = np.linspace(-1.25, 2.75, 24, dtype=np.float64).reshape(3, 8)
    out = np.full_like(d2kdr2, -999.0)

    result = _kdp_proc.lowpass_maesaka_jac(d2kdr2, 125.0, 0.6, "low", out)

    assert result is None
    np.testing.assert_array_equal(out, _reference_jac(d2kdr2, 125.0, 0.6))


@pytest.mark.parametrize("ngates", [0, 1, 2, 3, 4])
def test_forward_reverse_phidp_python_fallback_matches_oracle(monkeypatch, ngates):
    k, bcs = _forward_reverse_case(ngates)

    actual = _fallback_forward_reverse_phidp(k, bcs, monkeypatch)
    expected = _reference_forward_reverse_phidp(k, bcs)

    _assert_forward_reverse_equal(actual, expected)


def test_forward_reverse_phidp_dispatches_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(k, phi_near, phi_far):
        calls.append((k.dtype, k.shape, phi_near.copy(), phi_far.copy()))
        return (
            np.full(k.shape, 1.0, dtype=np.float64),
            np.full(k.shape, 2.0, dtype=np.float64),
        )

    monkeypatch.setattr(
        _kdp_proc,
        "_rust_kernel",
        lambda name: rust_kernel if name == "forward_reverse_phidp" else None,
    )
    k, bcs = _forward_reverse_case(4)

    actual = _kdp_proc.forward_reverse_phidp(k, bcs)

    assert calls[0][0:2] == (np.float64, (2, 4))
    np.testing.assert_array_equal(calls[0][2], bcs[0])
    np.testing.assert_array_equal(calls[0][3], bcs[1])
    _assert_forward_reverse_equal(
        actual,
        (
            np.full(k.shape, 1.0, dtype=np.float64),
            np.full(k.shape, 2.0, dtype=np.float64),
        ),
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda k, bcs: (k.astype(np.float32), bcs),
        lambda k, bcs: (np.asfortranarray(k), bcs),
        lambda k, bcs: (k, [bcs[0].astype(np.float32), bcs[1]]),
        lambda k, bcs: (
            k,
            [
                bcs[0],
                np.array([bcs[1][0], 99.0, bcs[1][1], 100.0], dtype=np.float64)[
                    ::2
                ],
            ],
        ),
    ],
)
def test_forward_reverse_phidp_keeps_python_path_for_unsupported_inputs(
    monkeypatch, mutate
):
    k, bcs = _forward_reverse_case(4)
    k, bcs = mutate(k, bcs)

    expected = _fallback_forward_reverse_phidp(k, bcs, monkeypatch)

    def fail_if_called(name):
        if name != "forward_reverse_phidp":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported forward-reverse input used Rust")

        return kernel

    monkeypatch.setattr(_kdp_proc, "_rust_kernel", fail_if_called)
    actual = _kdp_proc.forward_reverse_phidp(k, bcs)

    _assert_forward_reverse_equal(actual, expected)


def test_forward_reverse_phidp_keeps_python_overflow_warning_path(monkeypatch):
    k = np.array([[np.finfo(np.float64).max, 1.0]], dtype=np.float64)
    bcs = [
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    ]

    def fail_if_called(name):
        if name != "forward_reverse_phidp":
            return None

        def kernel(*_args):
            raise AssertionError("overflow-warning input used Rust")

        return kernel

    monkeypatch.setattr(_kdp_proc, "_rust_kernel", fail_if_called)
    with pytest.warns(RuntimeWarning, match="overflow"):
        actual = _kdp_proc.forward_reverse_phidp(k, bcs)

    monkeypatch.setattr(_kdp_proc, "_rust_kernel", lambda _name: None)
    with pytest.warns(RuntimeWarning, match="overflow"):
        expected = _kdp_proc.forward_reverse_phidp(k, bcs)

    _assert_forward_reverse_equal(actual, expected)


def test_forward_reverse_phidp_high_level_verbose_preserves_output(capsys):
    k, bcs = _forward_reverse_case(4)
    expected = _reference_forward_reverse_phidp(k, bcs)

    actual = kdp_proc._forward_reverse_phidp(k, bcs, verbose=True)

    _assert_forward_reverse_equal(actual, expected)
    captured = capsys.readouterr().out
    assert "Forward-reverse PHIDP MBE:" in captured
    assert "Forward-reverse PHIDP MAE:" in captured


@pytest.mark.parametrize(
    ("ranges", "atol", "expected"),
    [
        (np.array([0.0, 100.0], dtype=np.float64), 0.0, np.float64(100.0)),
        (np.array([0.0, 100.0, 200.0], dtype=np.float64), 0.0, np.float64(100.0)),
        (np.array([0.0, 100.0, 201.0], dtype=np.float64), 1.0, np.float64(100.0)),
    ],
)
def test_parse_range_resolution_python_fallback_uniform_cases(
    monkeypatch, ranges, atol, expected
):
    actual = _fallback_parse_range_resolution(ranges, monkeypatch, atol=atol)

    assert type(actual) is type(expected)
    assert actual == expected


def test_parse_range_resolution_dispatches_to_private_rust_kernel(monkeypatch, capsys):
    calls = []

    def rust_kernel(ranges, atol):
        calls.append((ranges.dtype, ranges.shape, atol))
        return 125.0

    monkeypatch.setattr(
        _kdp_proc,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_kdp_range_resolution_uniform" else None,
    )
    ranges = np.array([0.0, 125.0, 250.0], dtype=np.float64)

    actual = kdp_proc._parse_range_resolution(
        _range_radar(ranges), check_uniform=True, atol=np.float64(1.0), verbose=True
    )

    assert calls == [(np.float64, (3,), 1.0)]
    assert type(actual) is np.float64
    assert actual == np.float64(125.0)
    assert capsys.readouterr().out == "Range resolution: 125.00 m\n"


def test_parse_range_resolution_rust_none_preserves_python_value_error(monkeypatch):
    def rust_kernel(_ranges, _atol):
        return None

    monkeypatch.setattr(
        _kdp_proc,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_kdp_range_resolution_uniform" else None,
    )

    with pytest.raises(ValueError) as exc_info:
        kdp_proc._parse_range_resolution(
            _range_radar(np.array([0.0, 100.0, 205.0], dtype=np.float64)),
            check_uniform=True,
            atol=1.0,
        )

    assert exc_info.value.args == ("Radar gate spacing is not uniform",)


@pytest.mark.parametrize(
    ("ranges", "kwargs"),
    [
        (np.array([], dtype=np.float64), {}),
        (np.array([0.0], dtype=np.float64), {}),
        (np.array([0, 100, 200], dtype=np.int32), {}),
        (np.array([0.0, 100.0, 200.0], dtype=np.float32), {}),
        ([0.0, 100.0, 200.0], {}),
        (np.ma.array([0.0, 100.0, 200.0], mask=[False, False, False]), {}),
        (np.array([0.0, 50.0, 100.0, 150.0, 200.0], dtype=np.float64)[::2], {}),
        (np.array([0.0, np.nan, 200.0], dtype=np.float64), {}),
        (np.array([0.0, np.inf], dtype=np.float64), {}),
        (np.array([0.0, 100.0, 200.0], dtype=np.float64), {"check_uniform": False}),
        (np.array([0.0, 100.0, 200.0], dtype=np.float64), {"atol": -1.0}),
    ],
)
def test_parse_range_resolution_keeps_python_path_for_unsupported_inputs(
    monkeypatch, ranges, kwargs
):
    def fail_if_called(name):
        if name != "_kdp_range_resolution_uniform":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported range-resolution input used Rust")

        return kernel

    monkeypatch.setattr(_kdp_proc, "_rust_kernel", fail_if_called)
    radar = _range_radar(ranges)
    try:
        actual = kdp_proc._parse_range_resolution(radar, **kwargs)
    except Exception as actual_error:
        expected_kwargs = {"check_uniform": True, "atol": 1.0, "verbose": False}
        expected_kwargs.update(kwargs)
        monkeypatch.setattr(_kdp_proc, "_rust_kernel", lambda _name: None)
        with pytest.raises(type(actual_error)) as expected_error:
            kdp_proc._parse_range_resolution(radar, **expected_kwargs)
        assert actual_error.args == expected_error.value.args
    else:
        expected_kwargs = {"check_uniform": True, "atol": 1.0, "verbose": False}
        expected_kwargs.update(kwargs)
        expected = _fallback_parse_range_resolution(ranges, monkeypatch, **expected_kwargs)
        assert type(actual) is type(expected)
        assert actual == expected


def test_parse_range_resolution_verbose_has_no_output_on_failure(monkeypatch, capsys):
    with pytest.raises(ValueError):
        _fallback_parse_range_resolution(
            np.array([0.0, 100.0, 205.0], dtype=np.float64),
            monkeypatch,
            atol=1.0,
            verbose=True,
        )

    assert capsys.readouterr().out == ""


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("ranges", "atol"),
    [
        (np.array([0.0, 100.0], dtype=np.float64), 0.0),
        (np.array([0.0, 100.0, 201.0], dtype=np.float64), 1.0),
    ],
)
def test_real_rust_parse_range_resolution_matches_python_fallback(
    monkeypatch, ranges, atol
):
    import pyart._rust as rust

    kernel = getattr(rust, "_kdp_range_resolution_uniform")
    expected = _fallback_parse_range_resolution(ranges, monkeypatch, atol=atol)
    calls = []

    def rust_kernel(name):
        if name == "_kdp_range_resolution_uniform":
            calls.append(name)
            return kernel
        return None

    monkeypatch.setattr(_kdp_proc, "_rust_kernel", rust_kernel)
    actual = kdp_proc._parse_range_resolution(_range_radar(ranges), atol=atol)

    assert calls == ["_kdp_range_resolution_uniform"]
    assert type(actual) is type(expected) is np.float64
    assert actual == expected


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("ranges", "atol", "match"),
    [
        (np.array([0.0], dtype=np.float64), 1.0, "at least two"),
        (np.array([0.0, 100.0, 200.0], dtype=np.float64)[::2], 1.0, "C-contiguous"),
        (np.array([0.0, np.nan], dtype=np.float64), 1.0, "finite"),
        (np.array([0.0, 100.0], dtype=np.float64), np.nan, "finite"),
        (np.array([0.0, 100.0], dtype=np.float64), -1.0, "non-negative"),
    ],
)
def test_real_rust_range_resolution_rejects_unsafe_direct_inputs(
    ranges, atol, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust._kdp_range_resolution_uniform(ranges, atol)


@pytest.mark.parametrize("ngates", [1, 2, 3])
def test_lowpass_maesaka_jac_rejects_unsafe_small_gate_counts_before_dispatch(
    monkeypatch, ngates
):
    def rust_kernel(*_args):
        raise AssertionError("unsafe small gate counts must not dispatch to Rust")

    monkeypatch.setattr(
        _kdp_proc,
        "_rust_kernel",
        lambda name: rust_kernel if name == "lowpass_maesaka_jac" else None,
    )
    d2kdr2 = np.ones((1, ngates), dtype=np.float64)
    out = np.full_like(d2kdr2, -1.0)

    with pytest.raises(ValueError) as exc_info:
        _kdp_proc.lowpass_maesaka_jac(d2kdr2, 125.0, 0.6, "low", out)

    assert exc_info.value.args == (
        "lowpass_maesaka_jac received an unsupported range gate count",
    )
    np.testing.assert_array_equal(out, np.full_like(d2kdr2, -1.0))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("ngates", [0, 1, 2, 3, 4])
def test_real_rust_forward_reverse_phidp_matches_python_fallback(
    monkeypatch, ngates
):
    k, bcs = _forward_reverse_case(ngates)
    expected = _fallback_forward_reverse_phidp(k, bcs, monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(
        _kdp_proc,
        "_rust_kernel",
        lambda name: getattr(rust, name, None),
    )
    actual = _kdp_proc.forward_reverse_phidp(k, bcs)

    assert actual[0].dtype == expected[0].dtype == np.float64
    assert actual[1].dtype == expected[1].dtype == np.float64
    _assert_forward_reverse_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize(
    ("k", "phi_near", "phi_far", "match"),
    [
        (
            np.ones((2, 4), dtype=np.float64),
            np.ones(1, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            "boundary condition arrays",
        ),
        (
            np.ones((2, 4), dtype=np.float64)[:, ::-1],
            np.ones(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            "C-contiguous",
        ),
    ],
)
def test_real_rust_forward_reverse_phidp_rejects_unsafe_direct_inputs(
    k, phi_near, phi_far, match
):
    import pyart._rust as rust

    with pytest.raises(ValueError, match=match):
        rust.forward_reverse_phidp(k, phi_near, phi_far)


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (
            _kdp_proc.lowpass_maesaka_term,
            (
                np.ones((2, 4), dtype=np.float64),
                100.0,
                "high",
                np.full((2, 4), -1.0, dtype=np.float64),
            ),
        ),
        (
            _kdp_proc.lowpass_maesaka_jac,
            (
                np.ones((2, 4), dtype=np.float64),
                100.0,
                1.0,
                "high",
                np.full((2, 4), -1.0, dtype=np.float64),
            ),
        ),
    ],
)
def test_invalid_finite_order_is_exact_value_error_and_does_not_mutate(func, args):
    out = args[-1]
    before = out.copy()

    with pytest.raises(ValueError) as exc_info:
        func(*args)

    assert exc_info.value.args == ("Invalid finite_order",)
    np.testing.assert_array_equal(out, before)


def test_dtype_validation_happens_before_output_mutation():
    k = np.ones((2, 4), dtype=np.float32)
    out = np.full((2, 4), -1.0, dtype=np.float64)

    with pytest.raises(ValueError):
        _kdp_proc.lowpass_maesaka_term(k, 100.0, "low", out)

    np.testing.assert_array_equal(out, np.full((2, 4), -1.0, dtype=np.float64))


def test_shape_validation_happens_before_output_mutation():
    k = np.ones((2, 4), dtype=np.float64)
    out = np.full((2, 5), -1.0, dtype=np.float64)

    with pytest.raises(ValueError):
        _kdp_proc.lowpass_maesaka_term(k, 100.0, "low", out)

    np.testing.assert_array_equal(out, np.full((2, 5), -1.0, dtype=np.float64))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
@pytest.mark.parametrize("ngates", [1, 2, 3])
def test_real_rust_lowpass_maesaka_jac_rejects_small_gate_inputs_without_panic(
    ngates,
):
    import pyart._rust as rust

    d2kdr2 = np.ones((1, ngates), dtype=np.float64)
    out = np.full_like(d2kdr2, -1.0)

    with pytest.raises(ValueError) as exc_info:
        rust.lowpass_maesaka_jac(d2kdr2, 125.0, 0.6, "low", out)

    assert exc_info.value.args == (
        "lowpass_maesaka_jac received an unsupported range gate count",
    )
    np.testing.assert_array_equal(out, np.full_like(d2kdr2, -1.0))
