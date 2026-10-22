# Contributing

We welcome contributions! Here's how to get started.

## Development setup

```bash
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
pip install -e .
```

## Running tests

Per-file only (full suite hangs):

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/test_tensor.py -q -p no:cacheprovider
```

## Coding conventions

- `SNEPPX_` prefix for all public C functions, types, and macros
- `void` in empty parameter lists
- Return `int` (0 success, -1 error) for most API functions
- Python: `Tensor.data` is a COPY — rebuild perturbed tensors for FD checks
- Python: `Module.__setattr__` auto-registers Tensor attrs as params
- Additive-only: never rename/remove existing public items

## Pull request process

1. Fork the repo
2. Create a feature branch
3. Make your changes with per-file tests
4. Commit with a clear message
5. Push and open a PR

## Good first issues

Look for issues labeled `good first issue` or `help wanted`.

## Documentation

Docs use MkDocs Material. To preview:

```bash
pip install mkdocs-material mkdocs-material[imaging] mike
mkdocs serve
```