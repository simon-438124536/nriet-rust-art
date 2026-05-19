import os

import numpy as np
import pytest

from pyart.io import nexrad_cdm


class _FakeMomentVar:
    def __init__(self, raw, attrs=None):
        self._raw = raw
        self._attrs = dict(attrs or {})
        self.auto_maskandscale = None
        for key, value in self._attrs.items():
            setattr(self, key, value)

    def set_auto_maskandscale(self, value):
        self.auto_maskandscale = value

    def ncattrs(self):
        return list(self._attrs.keys())

    def __getitem__(self, index):
        return self._raw[index]


def _rust_or_skip():
    try:
        import pyart._rust as rust
    except ImportError:
        pytest.skip("pyart._rust is not importable in this test mode")
    if not hasattr(rust, "_nexrad_cdm_moment_u8"):
        pytest.skip("pyart._rust has no NEXRAD CDM moment kernels")
    return rust


def _fallback_moment(moment_var, index, ngates, monkeypatch):
    monkeypatch.setattr(nexrad_cdm, "_rust_kernel", lambda _name: None)
    return nexrad_cdm._get_moment_data(moment_var, index, ngates)


def _assert_masked_equal(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.fill_value == expected.fill_value
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize(
    "moment_var",
    [
        _FakeMomentVar(np.array([[[0, 1, 2, 3]]], dtype=np.uint8)),
        _FakeMomentVar(
            np.array([[[0, 1, 2, -1]]], dtype=np.int8),
            {"_Unsigned": "true"},
        ),
        _FakeMomentVar(
            np.array([[[0, 1, 2, -1]]], dtype=np.int16),
            {"_Unsigned": "true", "scale_factor": 0.5, "add_offset": -1.0},
        ),
        _FakeMomentVar(
            np.array([[[0, 1, 2, 65535]]], dtype=np.uint16),
            {"scale_factor": -0.25, "add_offset": 2.0},
        ),
        _FakeMomentVar(
            np.array([[[-2, 0, 1, 2]]], dtype=np.int16),
            {"scale_factor": 0.5, "add_offset": -1.0},
        ),
        _FakeMomentVar(
            np.array([[[0.0, 1.0, 2.0, np.nan]]], dtype=np.float32),
            {"scale_factor": 0.5, "add_offset": -1.0},
        ),
    ],
)
def test_nexrad_cdm_moment_python_fallback_reference_cases(monkeypatch, moment_var):
    actual = _fallback_moment(moment_var, 0, 4, monkeypatch)

    assert moment_var.auto_maskandscale is False
    assert type(actual) is np.ma.MaskedArray
    assert actual.dtype == np.float64
    assert actual.fill_value == 1e20


def test_nexrad_cdm_moment_dispatches_dense_u8_to_private_rust(monkeypatch):
    raw = np.array([[[0, 1, 2, 255]]], dtype=np.uint8)
    moment_var = _FakeMomentVar(raw, {"scale_factor": 0.5, "add_offset": -1.0})
    calls = []
    out = np.array([[0.0, 1.0, 0.0, 126.5]], dtype=np.float64)
    mask = np.array([[True, True, False, False]], dtype=bool)

    def kernel(raw_arg, scale, add_offset):
        calls.append((raw_arg.dtype, raw_arg.shape, scale, add_offset))
        return out.copy(), mask.copy()

    monkeypatch.setattr(
        nexrad_cdm,
        "_rust_kernel",
        lambda name: kernel if name == "_nexrad_cdm_moment_u8" else None,
    )

    actual = nexrad_cdm._get_moment_data(moment_var, 0, 4)

    assert calls == [(np.dtype(np.uint8), (1, 4), 0.5, -1.0)]
    assert actual.dtype == np.float64
    assert actual.fill_value == 1e20
    np.testing.assert_array_equal(actual.data, out)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)


@pytest.mark.parametrize(
    "moment_var",
    [
        _FakeMomentVar(
            np.array([[[0, 1, 2, 3]]], dtype=np.uint8),
            {"scale_factor": np.float32(0.5), "add_offset": np.float32(-1.0)},
        ),
        _FakeMomentVar(
            np.array([[[0, 1, 2, 3]]], dtype=np.uint8),
            {"scale_factor": np.nan},
        ),
        _FakeMomentVar(
            np.array([[[0, 1, 2, 3]]], dtype=np.uint8),
            {"add_offset": np.array([0.0, 1.0])},
        ),
    ],
)
def test_nexrad_cdm_moment_unsupported_inputs_keep_python_path(
    monkeypatch, moment_var
):
    def rust_kernel(name):
        if name not in {
            "_nexrad_cdm_moment_u8",
            "_nexrad_cdm_moment_u16",
            "_nexrad_cdm_moment_i8",
            "_nexrad_cdm_moment_i16",
            "_nexrad_cdm_moment_f32",
            "_nexrad_cdm_moment_f64",
        }:
            return None

        def fail(*_args):
            raise AssertionError(f"unsupported input used Rust kernel {name}")

        return fail

    monkeypatch.setattr(nexrad_cdm, "_rust_kernel", rust_kernel)
    try:
        actual = nexrad_cdm._get_moment_data(moment_var, 0, 4)
    except Exception as actual_error:
        with pytest.raises(type(actual_error)) as expected_error:
            _fallback_moment(moment_var, 0, 4, monkeypatch)
        assert actual_error.args == expected_error.value.args
    else:
        expected = _fallback_moment(moment_var, 0, 4, monkeypatch)
        _assert_masked_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for real Rust NEXRAD CDM parity",
)
@pytest.mark.parametrize(
    "moment_var",
    [
        _FakeMomentVar(np.array([[[0, 1, 2, 3]]], dtype=np.uint8)),
        _FakeMomentVar(
            np.array([[[0, 1, 2, -1]]], dtype=np.int8),
            {"_Unsigned": "true", "scale_factor": 0.5, "add_offset": -1.0},
        ),
        _FakeMomentVar(
            np.array([[[0, 1, 2, -1]]], dtype=np.int16),
            {"_Unsigned": "true", "scale_factor": 0.5, "add_offset": -1.0},
        ),
        _FakeMomentVar(
            np.array([[[-2, 0, 1, 2]]], dtype=np.int16),
            {"scale_factor": 0.5, "add_offset": -1.0},
        ),
        _FakeMomentVar(
            np.array([[[0.0, 1.0, 2.0, np.nan]]], dtype=np.float32),
            {"scale_factor": 0.5, "add_offset": -1.0},
        ),
        _FakeMomentVar(
            np.array([[[0.0, 1.0, 2.0, np.nan]]], dtype=np.float64),
        ),
    ],
)
def test_real_rust_nexrad_cdm_moment_matches_python_fallback(
    monkeypatch, moment_var
):
    rust = _rust_or_skip()
    expected = _fallback_moment(moment_var, 0, 4, monkeypatch)
    monkeypatch.setattr(nexrad_cdm, "_rust_kernel", lambda name: getattr(rust, name, None))

    actual = nexrad_cdm._get_moment_data(moment_var, 0, 4)

    _assert_masked_equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust NEXRAD CDM checks",
)
def test_real_rust_nexrad_cdm_moment_direct_preserves_masked_payloads():
    rust = _rust_or_skip()

    data, mask = rust._nexrad_cdm_moment_u8(
        np.array([[0, 1, 2, 255]], dtype=np.uint8), 0.5, -1.0
    )

    np.testing.assert_array_equal(
        data, np.array([[0.0, 1.0, 0.0, 126.5]], dtype=np.float64)
    )
    np.testing.assert_array_equal(mask, np.array([[True, True, False, False]]))

    data_u16, mask_u16 = rust._nexrad_cdm_moment_u16(
        np.array([[0, 1, 2, 65535]], dtype=np.uint16), -0.25, 2.0
    )
    np.testing.assert_array_equal(
        data_u16, np.array([[0.0, 1.0, 1.5, -16381.75]], dtype=np.float64)
    )
    np.testing.assert_array_equal(mask_u16, np.array([[True, True, False, False]]))

    data_i16, mask_i16 = rust._nexrad_cdm_moment_i16(
        np.array([[-2, 0, 1, 2]], dtype=np.int16), 0.5, -1.0
    )
    np.testing.assert_array_equal(
        data_i16, np.array([[-2.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    )
    np.testing.assert_array_equal(mask_i16, np.array([[True, True, True, False]]))

    data_f32, mask_f32 = rust._nexrad_cdm_moment_f32(
        np.array([[0.0, 1.0, 2.0, np.nan]], dtype=np.float32), 0.5, -1.0
    )
    np.testing.assert_array_equal(
        data_f32, np.array([[0.0, 1.0, 0.0, np.nan]], dtype=np.float64)
    )
    np.testing.assert_array_equal(mask_f32, np.array([[True, True, False, False]]))


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="installed extension required for direct Rust NEXRAD CDM checks",
)
@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda rust: rust._nexrad_cdm_moment_u8(
                np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2], 1.0, 0.0
            ),
            "C-contiguous",
        ),
        (
            lambda rust: rust._nexrad_cdm_moment_u16(
                np.array([[1]], dtype=np.uint16), np.inf, 0.0
            ),
            "finite",
        ),
    ],
)
def test_real_rust_nexrad_cdm_moment_direct_rejects_unsafe_inputs(call, match):
    rust = _rust_or_skip()

    with pytest.raises(ValueError, match=match):
        call(rust)
