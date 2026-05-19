import os


def test_pyart_imports():
    os.environ.setdefault("PYART_QUIET", "1")
    import pyart

    assert pyart.__name__ == "pyart"
    assert hasattr(pyart, "load_config")
