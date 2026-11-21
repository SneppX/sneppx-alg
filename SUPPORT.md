# Support

If you need help with SNEPPX-Alg, here are the available channels and resources:

## Official Documentation

- **Live Docs Site:** [https://sneppx.github.io/sneppx-alg/](https://sneppx.github.io/sneppx-alg/) — Full searchable reference, interactive API explorer, and guides
- [Quickstart Guide](QUICKSTART.md) — 5-minute setup and runnable verification
- [Architecture Overview](docs/ARCHITECTURE.md) — Deep dive into the 5-stage pipeline and mathematical formulations
- [Contributing Guide](CONTRIBUTING.md) — How to submit pull requests and test changes locally
- [FAQ](docs/FAQ.md) — Frequently asked questions and troubleshooting

## Community & Issue Tracker

- **GitHub Issues:** [https://github.com/SneppX/sneppx-alg/issues](https://github.com/SneppX/sneppx-alg/issues) — Bug reports and feature proposals
- **GitHub Discussions:** [https://github.com/SneppX/sneppx-alg/discussions](https://github.com/SneppX/sneppx-alg/discussions) — General questions and usage discussions
- **Security Inquiries:** Report privately via [Security Advisories](https://github.com/SneppX/sneppx-alg/security/advisories/new) or email `algoSNEPPX@gmail.com`
- **Updates & Announcements:** Follow [@SNEPPXdrv](https://x.com/SNEPPXdrv) on X/Twitter

## Common Troubleshooting Tips

1. **`ModuleNotFoundError: No module named 'SneppX_ALG'`**:
   Ensure `bindings/python` is added to your Python path:
   ```powershell
   $env:PYTHONPATH = "bindings/python"
   ```

2. **Tests hanging**:
   Only run unit tests **per-file** with pytest. Never execute GPU/LLM tests on CPU-only machines.

3. **C Compiler missing during pip install**:
   The pure-Python / NumPy engine runs out of the box without compiling C extensions:
   ```powershell
   pip install --no-build-isolation sneppx-alg
   ```
