#!/usr/bin/env pwsh
# SNEPPX Developer Tool Chain - run all 7 tools against the ARIX_Algo codebase.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\dev-tools.ps1 [--all] [--no-test] [--build-dir build]
#
# Requires: sneppx-analyze, sneppx-format, sneppx-deps, sneppx-stats,
#           sneppx-test, sneppx-bench, sneppx-serve (all via `pip install sneppx-toolkit[all]`)

param(
    [switch]$All,
    [switch]$NoTest,
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $PSScriptRoot
$FAILED = 0

function Test-Tool {
    param([string]$name)
    if (Get-Command $name -ErrorAction SilentlyContinue) { return $true }
    Write-Host "  SKIP  $name (not installed - pip install sneppx-toolkit[all])" -ForegroundColor DarkYellow
    return $false
}

function Check {
    param([string]$name, [scriptblock]$block)
    try {
        & $block
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  PASS  $name" -ForegroundColor Green
        } else {
            Write-Host "  FAIL  $name (exit $LASTEXITCODE)" -ForegroundColor Red
            $script:FAILED = 1
        }
    } catch {
        Write-Host "  FAIL  $name ($_)" -ForegroundColor Red
        $script:FAILED = 1
    }
}

Write-Host "SNEPPX Developer Tool Chain" -ForegroundColor Cyan
Write-Host "Root: $ROOT" -ForegroundColor Cyan
Write-Host ""

# 1. Security scan
if (Test-Tool "sneppx-analyze") {
    Check "Security scan (kernel, algorithms, net, security)" {
        & sneppx-analyze --dirs kernel algorithms net security 2>$null
    }
}

# 2. Format / lint
if (Test-Tool "sneppx-format") {
    Check "Formatting / lint (C/C++ core)" {
        & sneppx-format --lint kernel > "$ROOT\.sneppx\lint-kernel.log" 2>&1
        & sneppx-format --lint algorithms > "$ROOT\.sneppx\lint-algorithms.log" 2>&1
        & sneppx-format --lint net > "$ROOT\.sneppx\lint-net.log" 2>&1
    }
}

# 2b. Documentation coverage (docs/COMMENTING.md Layers 1 & 4)
if (Test-Tool "sneppx-format") {
    Check "Documentation coverage (sneppx-format --docs)" {
        & sneppx-format --docs --no-color kernel > "$ROOT\.sneppx\docs-kernel.log" 2>&1
        & sneppx-format --docs --no-color algorithms > "$ROOT\.sneppx\docs-algorithms.log" 2>&1
        & sneppx-format --docs --no-color security > "$ROOT\.sneppx\docs-security.log" 2>&1
        & sneppx-format --docs --no-color net > "$ROOT\.sneppx\docs-net.log" 2>&1
    }
}

# 3. Circular dependencies
if (Test-Tool "sneppx-deps") {
    Check "Circular dependency check" {
        & sneppx-deps --circular . 2>$null
    }
}

# 4. Code stats
if (Test-Tool "sneppx-stats") {
    Check "Code statistics" {
        & sneppx-stats --save "$ROOT\.sneppx\stats.json" . 2>$null
    }
}

# 5. C test suite (ctest) - or sneppx-test if build dir exists
if (-not $NoTest) {
    if (Test-Path "$ROOT\$BuildDir") {
        if (Test-Tool "sneppx-test") {
            Check "Test suite (sneppx-test)" {
                & sneppx-test --build-dir "$ROOT\$BuildDir" --exclude cuda 2>$null
            }
        } else {
            Check "Test suite (ctest)" {
                & cmake --build "$ROOT\$BuildDir" --config Release 2>$null
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
                & ctest --test-dir "$ROOT\$BuildDir" -C Release --output-on-failure 2>$null
            }
        }
    } else {
        Write-Host "  SKIP  Test suite (no $BuildDir directory - run scripts\build.ps1 first)" -ForegroundColor DarkYellow
    }
}

# 6. Benchmarks
if ($All -and (Test-Tool "sneppx-bench")) {
    if (Test-Path "$ROOT\$BuildDir") {
        Check "Benchmarks" {
            & sneppx-bench --build-dir "$ROOT\$BuildDir" 2>$null
        }
    }
}

if ($FAILED -eq 1) {
    Write-Host "`nDev tool chain FAILED - fix issues before committing." -ForegroundColor Red
    exit 1
}
Write-Host "`nDev tool chain PASSED." -ForegroundColor Green
exit 0
