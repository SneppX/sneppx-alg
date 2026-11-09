# Pre-commit / CI-equivalent gate (run before every commit)
#
# Usage (PowerShell, repo root):
#   .\scripts\precommit-gate.ps1
#
# Requires: py -3 (or python), numpy, pytest. Respects AGENTS machine constraints:
# no LLM/CUDA/inference-server test execution, no HF downloads, no GPU ops.

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "bindings/python"

$tests = @(
  "tests/python/test_readme_integrity.py",
  "tests/python/test_hf_safetensors.py",
  "tests/python/test_hf_integration.py",
  "tests/python/test_model_zoo.py",
  "tests/python/test_data_loader.py",
  "tests/python/test_tokenizer.py"
)

Write-Host "== SNEPPX pre-commit gate (safe CPU tests) =="
& py -3 -m pytest @tests -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) {
  Write-Error "Pre-commit gate FAILED. Fix failures before committing."
  exit $LASTEXITCODE
}
Write-Host "== Gate PASSED =="
