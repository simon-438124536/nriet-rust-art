import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYART_QUIET", "1")

from pyart.map import gate_mapper  # noqa: E402


def _index_map(dtype=np.float64):
    return np.array(
        [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=dtype,
    )


def _source_data(dtype=np.float64):
    return np.ma.array(
        np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype=dtype),
        mask=[[False, True, False], [False, False, False]],
        fill_value=-12345.0,
    )


def _gate_excluded():
    return np.array(
        [[False, False, False], [False, True, False]],
        dtype=np.bool_,
    )


def _dest_data(dtype=np.float64):
    return (100.0 + np.arange(9, dtype=dtype).reshape(3, 3)).astype(dtype)


def _make_mapper(index_map=None, src=None, gate_excluded=None, dest=None):
    if index_map is None:
        index_map = _index_map()
    if src is None:
        src = _source_data()
    if gate_excluded is None:
        gate_excluded = _gate_excluded()
    if dest is None:
        dest = _dest_data()

    mapper = gate_mapper.GateMapper.__new__(gate_mapper.GateMapper)
    mapper.src_radar = SimpleNamespace(
        nrays=src.shape[0],
        ngates=src.shape[1],
        fields={
            "reflectivity": {
                "data": src,
                "units": "dBZ",
                "long_name": "source reflectivity",
            }
        },
    )
    mapper.dest_radar = SimpleNamespace(
        nrays=dest.shape[0],
        ngates=dest.shape[1],
        fields={
            "reflectivity": {
                "data": dest,
                "units": "dest_units",
                "long_name": "destination reflectivity",
            }
        },
    )
    mapper.gatefilter_src = SimpleNamespace(gate_excluded=gate_excluded)
    mapper._index_map = index_map
    return mapper


def _fallback_mapped_data(mapper, monkeypatch):
    monkeypatch.setattr(gate_mapper, "_rust_kernel", lambda _name: None)
    mapped = mapper.mapped_radar(["reflectivity"])
    return mapped.fields["reflectivity"]["data"]


def _assert_masked_array_exact(actual, expected):
    np.testing.assert_array_equal(np.ma.getdata(actual), np.ma.getdata(expected))
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    assert actual.dtype == expected.dtype
    assert actual.fill_value == expected.fill_value


def _python_apply_field(index_map, src_data, src_mask, out_data, out_mask):
    dest = np.ma.array(out_data.copy(), mask=out_mask.copy())
    src = np.ma.array(src_data, mask=src_mask)
    for ray in range(src_data.shape[0]):
        for gate in range(src_data.shape[1]):
            dest_ray = int(index_map[ray, gate, 0])
            dest_gate = int(index_map[ray, gate, 1])
            if dest_ray > 0:
                dest[dest_ray, dest_gate] = src[ray, gate]
    return dest.data, np.ma.getmaskarray(dest)


def test_gate_mapper_python_fallback_preserves_assignment_order_and_masks(monkeypatch):
    mapper = _make_mapper()

    actual = _fallback_mapped_data(mapper, monkeypatch)

    expected_data = _dest_data()
    expected_data[1, 1] = 12.0
    expected_data[2, 0] = 13.0
    expected_mask = np.ones((3, 3), dtype=np.bool_)
    expected_mask[1, 1] = False
    expected = np.ma.array(expected_data, mask=expected_mask)
    _assert_masked_array_exact(actual, expected)


def test_gate_mapper_dispatches_float64_fields_to_private_rust_kernel(monkeypatch):
    calls = []

    def rust_kernel(index_map, src_data, src_mask, out_data, out_mask):
        calls.append(
            (
                index_map.dtype,
                index_map.shape,
                src_data.dtype,
                src_mask.dtype,
                out_data.dtype,
                out_mask.dtype,
            )
        )
        out_data[:, :] = 42.0
        out_mask[:, :] = False
        return "ignored"

    monkeypatch.setattr(
        gate_mapper,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_gate_mapper_apply_field_f64" else None,
    )

    mapped = _make_mapper().mapped_radar(["reflectivity"])
    actual = mapped.fields["reflectivity"]["data"]

    assert calls == [((np.dtype("float64"), (2, 3, 2), np.dtype("float64"), np.bool_, np.dtype("float64"), np.bool_))]
    np.testing.assert_array_equal(actual.data, np.full((3, 3), 42.0))
    np.testing.assert_array_equal(actual.mask, np.zeros((3, 3), dtype=np.bool_))


def test_gate_mapper_fast_path_preserves_existing_field_metadata_and_fill_value(
    monkeypatch,
):
    def rust_kernel(_index_map, _src_data, _src_mask, out_data, out_mask):
        out_data[1, 1] = 55.0
        out_mask[1, 1] = False

    monkeypatch.setattr(
        gate_mapper,
        "_rust_kernel",
        lambda name: rust_kernel if name == "_gate_mapper_apply_field_f64" else None,
    )

    mapper = _make_mapper()
    mapped = mapper.mapped_radar(["reflectivity"])
    field = mapped.fields["reflectivity"]

    assert field["units"] == "dest_units"
    assert field["long_name"] == "destination reflectivity"
    assert field["data"].fill_value == np.ma.masked_where(
        True, _dest_data()
    ).fill_value
    assert field["data"].data[1, 1] == 55.0
    assert field["data"].mask[1, 1] == np.bool_(False)


@pytest.mark.parametrize(
    ("index_map", "src", "dest"),
    [
        (_index_map(np.float32), _source_data(), _dest_data()),
        (_index_map(), _source_data(np.float32), _dest_data()),
        (_index_map(), _source_data(), _dest_data(np.float32)),
    ],
)
def test_gate_mapper_unsupported_dense_inputs_keep_python_path(
    monkeypatch, index_map, src, dest
):
    expected = _fallback_mapped_data(
        _make_mapper(index_map=index_map, src=src, dest=dest), monkeypatch
    )

    def fail_if_called(name):
        if name != "_gate_mapper_apply_field_f64":
            return None

        def kernel(*_args):
            raise AssertionError("unsupported GateMapper input used Rust")

        return kernel

    monkeypatch.setattr(gate_mapper, "_rust_kernel", fail_if_called)
    actual = _make_mapper(index_map=index_map, src=src, dest=dest).mapped_radar(
        ["reflectivity"]
    )

    _assert_masked_array_exact(actual.fields["reflectivity"]["data"], expected)


def test_gate_mapper_negative_destination_column_stays_python_owned(monkeypatch):
    index_map = _index_map()
    index_map[0, 0] = [1.0, -1.0]
    expected = _fallback_mapped_data(_make_mapper(index_map=index_map), monkeypatch)

    def fail_if_called(name):
        if name != "_gate_mapper_apply_field_f64":
            return None

        def kernel(*_args):
            raise AssertionError("negative destination column used Rust")

        return kernel

    monkeypatch.setattr(gate_mapper, "_rust_kernel", fail_if_called)
    actual = _make_mapper(index_map=index_map).mapped_radar(["reflectivity"])

    _assert_masked_array_exact(actual.fields["reflectivity"]["data"], expected)


@pytest.mark.parametrize(
    ("bad_index", "exc_type"),
    [
        ((0, 0, 0, np.nan), ValueError),
        ((0, 0, 0, 99.0), IndexError),
    ],
)
def test_gate_mapper_malformed_index_map_keeps_oracle_exception(
    monkeypatch, bad_index, exc_type
):
    ray, gate, coord, value = bad_index
    index_map = _index_map()
    index_map[ray, gate, coord] = value

    def fail_if_called(name):
        if name != "_gate_mapper_apply_field_f64":
            return None

        def kernel(*_args):
            raise AssertionError("malformed index map used Rust")

        return kernel

    monkeypatch.setattr(gate_mapper, "_rust_kernel", fail_if_called)
    with pytest.raises(exc_type):
        _make_mapper(index_map=index_map).mapped_radar(["reflectivity"])


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="real pyart._rust parity is verified in installed-wheel mode",
)
def test_real_rust_gate_mapper_wrapper_matches_python_fallback(monkeypatch):
    expected = _fallback_mapped_data(_make_mapper(), monkeypatch)

    import pyart._rust as rust

    monkeypatch.setattr(gate_mapper, "_rust_kernel", lambda name: getattr(rust, name, None))
    mapped = _make_mapper().mapped_radar(["reflectivity"])

    _assert_masked_array_exact(mapped.fields["reflectivity"]["data"], expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust parity is verified in installed-wheel mode",
)
def test_real_rust_gate_mapper_apply_field_matches_python_oracle():
    import pyart._rust as rust

    index_map = _index_map()
    src = np.ma.masked_where(_gate_excluded(), _source_data())
    src_data = np.ascontiguousarray(np.ma.getdata(src), dtype=np.float64)
    src_mask = np.ascontiguousarray(np.ma.getmaskarray(src), dtype=np.bool_)
    out_data = _dest_data()
    out_mask = np.ones((3, 3), dtype=np.bool_)
    expected_data, expected_mask = _python_apply_field(
        index_map, src_data, src_mask, out_data, out_mask
    )

    rust._gate_mapper_apply_field_f64(index_map, src_data, src_mask, out_data, out_mask)

    np.testing.assert_array_equal(out_data, expected_data)
    np.testing.assert_array_equal(out_mask, expected_mask)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust parity is verified in installed-wheel mode",
)
def test_real_rust_gate_mapper_apply_field_f32_matches_python_fallback(monkeypatch):
    import pyart._rust as rust

    index_map = _index_map()
    src = _source_data(dtype=np.float32)
    mapper = _make_mapper(index_map=index_map, src=src, dest=_dest_data(dtype=np.float32))
    expected = _fallback_mapped_data(mapper, monkeypatch)

    actual = gate_mapper.GateMapper.mapped_radar(
        mapper, "reflectivity"
    ).fields["reflectivity"]["data"]

    _assert_masked_array_exact(actual, expected)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust parity is verified in installed-wheel mode",
)
def test_real_rust_gate_mapper_apply_field_truncates_float_indexes_like_python_int():
    import pyart._rust as rust

    index_map = np.array([[[1.9, 1.9]]], dtype=np.float64)
    src_data = np.array([[77.0]], dtype=np.float64)
    src_mask = np.array([[False]], dtype=np.bool_)
    out_data = _dest_data()
    out_mask = np.ones((3, 3), dtype=np.bool_)
    expected_data, expected_mask = _python_apply_field(
        index_map, src_data, src_mask, out_data, out_mask
    )

    rust._gate_mapper_apply_field_f64(index_map, src_data, src_mask, out_data, out_mask)

    np.testing.assert_array_equal(out_data, expected_data)
    np.testing.assert_array_equal(out_mask, expected_mask)


@pytest.mark.skipif(
    os.environ.get("PYART_TEST_INSTALLED") != "1",
    reason="direct Rust exception parity is verified in installed-wheel mode",
)
def test_real_rust_gate_mapper_apply_field_rejects_unsafe_direct_inputs():
    import pyart._rust as rust

    index_map = _index_map()
    src_data = np.ascontiguousarray(np.ma.getdata(_source_data()), dtype=np.float64)
    src_mask = np.zeros((2, 3), dtype=np.bool_)
    out_data = _dest_data()
    out_mask = np.ones((3, 3), dtype=np.bool_)

    with pytest.raises(ValueError, match="C-contiguous"):
        rust._gate_mapper_apply_field_f64(
            index_map[:, ::-1, :], src_data, src_mask, out_data, out_mask
        )

    bad_index_map = index_map.copy()
    bad_index_map[0, 0] = [99.0, 0.0]
    with pytest.raises(ValueError, match="in bounds"):
        rust._gate_mapper_apply_field_f64(
            bad_index_map, src_data, src_mask, out_data, out_mask
        )

    extreme_index_map = index_map.copy()
    extreme_index_map[0, 0] = [1.0e300, 0.0]
    with pytest.raises(ValueError, match="platform index range"):
        rust._gate_mapper_apply_field_f64(
            extreme_index_map, src_data, src_mask, out_data, out_mask
        )
