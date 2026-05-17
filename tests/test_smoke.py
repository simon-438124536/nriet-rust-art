import nriet_rust_art as nra


def test_package_imports():
    assert nra.version() == "0.1.0"


def test_sum_f64():
    assert nra.sum_f64([1.0, 2.5, 3.5]) == 7.0
