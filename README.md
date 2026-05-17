# nriet-rust-art

Rust-first radar processing library with a thin Python package shell.

This repository is initialized for a staged rewrite of Py-ART-style workflows:

- keep Python as the ergonomic public API layer;
- move hot loops and data-heavy algorithms into Rust;
- expose Rust kernels through `PyO3` and `maturin`;
- keep local Py-ART source snapshots and comparison datasets outside git until import scope is explicit.

## Layout

```text
.
├── Cargo.toml              # Rust crate and Python extension configuration
├── pyproject.toml          # Python build metadata via maturin
├── python/nriet_rust_art/  # Python shell package
├── src/                    # Rust core
└── tests/                  # Python smoke tests
```

## Development

Install Rust and Python first, then build the extension in editable mode:

```powershell
python -m pip install -U pip
python -m pip install -e ".[dev]"
maturin develop --release
pytest
```

The first Rust module is intentionally small. It proves the packaging path before the rewrite starts moving real radar algorithms over.
