import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.retrieve import qpe  # noqa: E402


class _Radar(SimpleNamespace):
    def check_field_exists(self, name):
        if name not in self.fields:
            raise KeyError("Field not available: " + name)


def _radar(refl=None, kdp=None, att=None):
    fields = {}
    if refl is not None:
        fields["refl"] = {"data": refl}
    if kdp is not None:
        fields["kdp"] = {"data": kdp}
    if att is not None:
        fields["att"] = {"data": att}
    return _Radar(fields=fields, instrument_parameters={})


def _compute_zkdp(
    refl,
    kdp,
    *,
    main_field="refl",
    thresh=1.0,
    thresh_max=True,
):
    return qpe.est_rain_rate_zkdp(
        _radar(refl=refl, kdp=kdp),
        alphaz=0.0376,
        betaz=0.6112,
        alphakdp=29.70,
        betakdp=0.85,
        refl_field="refl",
        kdp_field="kdp",
        rr_field="rr",
        main_field=main_field,
        thresh=thresh,
        thresh_max=thresh_max,
    )["data"]


def _compute_za(
    refl,
    att,
    *,
    main_field="att",
    thresh=2.0,
    thresh_max=False,
):
    return qpe.est_rain_rate_za(
        _radar(refl=refl, att=att),
        alphaz=0.0376,
        betaz=0.6112,
        alphaa=250.0,
        betaa=0.91,
        refl_field="refl",
        a_field="att",
        rr_field="rr",
        main_field=main_field,
        thresh=thresh,
        thresh_max=thresh_max,
    )["data"]


def _fallback_zkdp(refl, kdp, monkeypatch, **kwargs):
    monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)
    return _compute_zkdp(refl, kdp, **kwargs)


def _fallback_za(refl, att, monkeypatch, **kwargs):
    monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)
    return _compute_za(refl, att, **kwargs)


def _assert_rain_rate_close(actual, expected):
    assert np.ma.isMaskedArray(actual)
    assert np.ma.isMaskedArray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    mask = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)
    np.testing.assert_array_equal(actual.data[mask], expected.data[mask])
    np.testing.assert_allclose(
        actual.data[~mask],
        expected.data[~mask],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(np.signbit(actual.data), np.signbit(expected.data))


def _dense_inputs():
    refl = np.array([[0.0, 15.0, 30.0], [40.0, 5.0, 25.0]], dtype=np.float64)
    kdp = np.array([[-0.5, 0.0, 0.4], [1.0, 2.5, 4.0]], dtype=np.float64)
    att = np.array([[0.0, 0.01, 0.03], [0.08, 0.4, 1.0]], dtype=np.float64)
    return refl, kdp, att


def test_zkdp_threshold_blend_python_fallback_reference(monkeypatch):
    refl, kdp, _att = _dense_inputs()

    actual = _fallback_zkdp(
        refl.copy(),
        kdp.copy(),
        monkeypatch,
        main_field="refl",
        thresh=1.0,
        thresh_max=True,
    )

    assert actual.dtype == np.float64
    assert np.ma.isMaskedArray(actual)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual), np.zeros(refl.shape, dtype=bool)
    )


@pytest.mark.parametrize(
    ("name", "compute", "args", "kwargs", "expected_thresh", "expected_thresh_max"),
    [
        (
            "zkdp",
            _compute_zkdp,
            lambda: (np.array([[5.0, 25.0]], dtype=np.float64), np.array([[0.0, 2.0]], dtype=np.float64)),
            {"main_field": "refl", "thresh": 1.0, "thresh_max": True},
            1.0,
            True,
        ),
        (
            "za",
            _compute_za,
            lambda: (np.array([[5.0, 25.0]], dtype=np.float64), np.array([[0.01, 0.5]], dtype=np.float64)),
            {"main_field": "att", "thresh": 2.0, "thresh_max": False},
            2.0,
            False,
        ),
    ],
)
def test_blend_dispatches_to_private_rust_kernel(
    monkeypatch, name, compute, args, kwargs, expected_thresh, expected_thresh_max
):
    calls = []

    def rust_kernel(rain_main, rain_secondary, thresh, thresh_max):
        calls.append(
            (
                rain_main.dtype,
                rain_secondary.dtype,
                rain_main.shape,
                rain_secondary.shape,
                thresh,
                thresh_max,
            )
        )
        rain_main[...] = 7.0 if name == "zkdp" else 11.0

    monkeypatch.setattr(
        qpe,
        "_rust_kernel",
        lambda kernel_name: rust_kernel
        if kernel_name == "_qpe_threshold_blend_dense_f64"
        else None,
    )

    actual = compute(*args(), **kwargs)

    assert calls == [
        (
            np.dtype("float64"),
            np.dtype("float64"),
            (1, 2),
            (1, 2),
            expected_thresh,
            expected_thresh_max,
        )
    ]
    np.testing.assert_array_equal(
        actual.data, np.full((1, 2), 7.0 if name == "zkdp" else 11.0)
    )


@pytest.mark.parametrize(
    ("rain_main", "rain_secondary", "thresh", "thresh_max"),
    [
        (
            np.ma.array([[1.0, 2.0]], mask=[[False, True]], dtype=np.float64),
            np.ma.array([[10.0, 20.0]], dtype=np.float64),
            1.5,
            True,
        ),
        (
            np.ma.array([[1.0, 2.0]], dtype=np.float64),
            np.ma.array([[10.0, 20.0]], mask=[[False, True]], dtype=np.float64),
            1.5,
            True,
        ),
        (
            np.ma.array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)[:, ::2]),
            np.ma.array([[10.0, 20.0]], dtype=np.float64),
            1.5,
            True,
        ),
        (
            np.ma.array([[1.0, 2.0]], dtype=np.float32),
            np.ma.array([[10.0, 20.0]], dtype=np.float32),
            1.5,
            True,
        ),
        (
            np.ma.array([[1.0, 2.0]], dtype=np.float64),
            np.ma.array([[10.0]], dtype=np.float64),
            1.5,
            True,
        ),
        (
            np.ma.array([[np.nan, 2.0]], dtype=np.float64),
            np.ma.array([[10.0, 20.0]], dtype=np.float64),
            1.5,
            True,
        ),
        (
            np.ma.array([[1.0, 2.0]], dtype=np.float64),
            np.ma.array([[10.0, np.inf]], dtype=np.float64),
            1.5,
            True,
        ),
        (
            np.ma.array([[1.0, 2.0]], dtype=np.float64),
            np.ma.array([[10.0, 20.0]], dtype=np.float64),
            None,
            True,
        ),
        (
            np.ma.array([[1.0, 2.0]], dtype=np.float64),
            np.ma.array([[10.0, 20.0]], dtype=np.float64),
            1.5,
            1,
        ),
    ],
)
def test_blend_helper_keeps_unsupported_inputs_python_owned(
    monkeypatch, rain_main, rain_secondary, thresh, thresh_max
):
    def fail_if_called(name):
        if name != "_qpe_threshold_blend_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported threshold blend input used Rust")

        return kernel

    monkeypatch.setattr(qpe, "_rust_kernel", fail_if_called)

    assert (
        qpe._blend_rain_rate_threshold_rust(
            rain_main, rain_secondary, thresh, thresh_max
        )
        is None
    )


def test_zkdp_masked_input_keeps_python_assignment_semantics(monkeypatch):
    refl = np.ma.array([[0.0, 20.0], [40.0, 10.0]], mask=[[False, True], [False, False]])
    kdp = np.array([[0.0, 0.5], [1.5, 3.0]], dtype=np.float64)

    def fail_blend(name):
        if name != "_qpe_threshold_blend_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("masked blend input used Rust")

        return kernel

    monkeypatch.setattr(qpe, "_rust_kernel", fail_blend)
    actual = _compute_zkdp(refl.copy(), kdp.copy(), thresh=1.0, thresh_max=True)
    expected = _fallback_zkdp(
        refl.copy(), kdp.copy(), monkeypatch, thresh=1.0, thresh_max=True
    )

    _assert_rain_rate_close(actual, expected)


def test_zkdp_thresh_none_preserves_python_exception(monkeypatch):
    refl, kdp, _att = _dense_inputs()

    def fail_blend(name):
        if name != "_qpe_threshold_blend_dense_f64":
            return None

        def kernel(*_args):
            raise AssertionError("thresh=None blend input used Rust")

        return kernel

    monkeypatch.setattr(qpe, "_rust_kernel", fail_blend)
    with pytest.raises(TypeError) as actual_error:
        _compute_zkdp(refl.copy(), kdp.copy(), thresh=None)

    monkeypatch.setattr(qpe, "_rust_kernel", lambda _name: None)
    with pytest.raises(TypeError) as expected_error:
        _compute_zkdp(refl.copy(), kdp.copy(), thresh=None)

    assert actual_error.value.args == expected_error.value.args


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_zkdp_blend_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    refl, kdp, _att = _dense_inputs()
    expected_kdp = kdp.copy()
    expected = _fallback_zkdp(
        refl.copy(),
        expected_kdp,
        monkeypatch,
        main_field="refl",
        thresh=1.0,
        thresh_max=True,
    )
    actual_kdp = kdp.copy()
    calls = []

    def counted_kernel(name):
        if name == "_qpe_threshold_blend_dense_f64":

            def blend(rain_main, rain_secondary, thresh, thresh_max):
                calls.append((rain_main.shape, rain_secondary.shape, thresh, thresh_max))
                return rust._qpe_threshold_blend_dense_f64(
                    rain_main, rain_secondary, thresh, thresh_max
                )

            return blend
        return getattr(rust, name, None)

    monkeypatch.setattr(qpe, "_rust_kernel", counted_kernel)
    actual = _compute_zkdp(
        refl.copy(),
        actual_kdp,
        main_field="refl",
        thresh=1.0,
        thresh_max=True,
    )

    assert calls == [((2, 3), (2, 3), 1.0, True)]
    np.testing.assert_array_equal(actual_kdp, expected_kdp)
    _assert_rain_rate_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_za_blend_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    refl, _kdp, att = _dense_inputs()
    expected = _fallback_za(
        refl.copy(),
        att.copy(),
        monkeypatch,
        main_field="att",
        thresh=2.0,
        thresh_max=False,
    )
    calls = []

    def counted_kernel(name):
        if name == "_qpe_threshold_blend_dense_f64":

            def blend(rain_main, rain_secondary, thresh, thresh_max):
                calls.append((rain_main.shape, rain_secondary.shape, thresh, thresh_max))
                return rust._qpe_threshold_blend_dense_f64(
                    rain_main, rain_secondary, thresh, thresh_max
                )

            return blend
        return getattr(rust, name, None)

    monkeypatch.setattr(qpe, "_rust_kernel", counted_kernel)
    actual = _compute_za(
        refl.copy(),
        att.copy(),
        main_field="att",
        thresh=2.0,
        thresh_max=False,
    )

    assert calls == [((2, 3), (2, 3), 2.0, False)]
    _assert_rain_rate_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_threshold_blend_mutates_main_and_rejects_unsafe_inputs():
    import pyart._rust as rust

    main = np.array([[0.5, 2.0, 3.0]], dtype=np.float64)
    secondary = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
    rust._qpe_threshold_blend_dense_f64(main, secondary, 1.0, True)
    np.testing.assert_array_equal(main, np.array([[0.5, 20.0, 30.0]]))
    main = np.array([[0.5, 2.0, 3.0]], dtype=np.float64)
    rust._qpe_threshold_blend_dense_f64(main, secondary, 1.0, False)
    np.testing.assert_array_equal(main, np.array([[10.0, 2.0, 3.0]]))

    with pytest.raises(ValueError, match="same shape"):
        rust._qpe_threshold_blend_dense_f64(
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            1.0,
            True,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        rust._qpe_threshold_blend_dense_f64(
            np.arange(8.0, dtype=np.float64).reshape(2, 4)[:, ::2],
            np.ones((2, 2), dtype=np.float64),
            1.0,
            True,
        )
    with pytest.raises(ValueError, match="finite"):
        rust._qpe_threshold_blend_dense_f64(
            np.array([[np.nan]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            1.0,
            True,
        )
    with pytest.raises(ValueError, match="finite"):
        rust._qpe_threshold_blend_dense_f64(
            np.array([[1.0]], dtype=np.float64),
            np.array([[np.inf]], dtype=np.float64),
            1.0,
            True,
        )
    readonly = np.array([[1.0]], dtype=np.float64)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        rust._qpe_threshold_blend_dense_f64(
            readonly,
            np.array([[2.0]], dtype=np.float64),
            1.0,
            True,
        )
    with pytest.raises(ValueError, match="non-boolean"):
        rust._qpe_threshold_blend_dense_f64(
            np.array([[1.0]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            True,
            True,
        )
    with pytest.raises(ValueError, match="numeric scalar"):
        rust._qpe_threshold_blend_dense_f64(
            np.array([[1.0]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            "1.0",
            True,
        )
