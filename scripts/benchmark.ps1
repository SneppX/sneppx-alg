param(
    [string]$BuildDir = "build_dir",
    [string]$Config = "Release",
    [int]$Warmup = 3,
    [int]$Iterations = 10
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$BuildPath = Join-Path $RootDir $BuildDir

Write-Host "=== SNEPPX Benchmark Runner ===" -ForegroundColor Cyan
Write-Host "Build: $BuildPath ($Config)"
Write-Host "Warmup: $Warmup, Iterations: $Iterations`n"

$testExes = @(
    "test_autodiff",
    "test_tensor_creation",
    "test_tokenizer",
    "test_checkpoint",
    "test_trainer"
)

foreach ($exe in $testExes) {
    $path = Join-Path $BuildPath "tests\$Config\$exe.exe"
    if (-not (Test-Path $path)) {
        Write-Host "SKIP: $exe (not found)" -ForegroundColor Yellow
        continue
    }
    Write-Host "Benchmarking $exe..." -ForegroundColor Green
    $times = @()
    for ($i = 0; $i -lt $Warmup + $Iterations; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process -FilePath $path -NoNewWindow -RedirectStandardOutput "NUL" -RedirectStandardError "NUL" -PassThru -Wait
        $sw.Stop()
        if ($p.ExitCode -ne 0) {
            Write-Host "  FAIL ($($p.ExitCode))" -ForegroundColor Red
            break
        }
        if ($i -ge $Warmup) { $times += $sw.Elapsed.TotalMilliseconds }
    }
    if ($times.Count -gt 0) {
        $avg = ($times | Measure-Object -Average).Average
        $min = ($times | Measure-Object -Minimum).Minimum
        $max = ($times | Measure-Object -Maximum).Maximum
        Write-Host "  Avg: $([math]::Round($avg, 1))ms  Min: $([math]::Round($min, 1))ms  Max: $([math]::Round($max, 1))ms (n=$($times.Count))" -ForegroundColor Cyan
    }
}

$pythonTests = @(
    "test_tensor",
    "test_nn",
    "test_optim",
    "test_quantization",
    "test_checkpoint"
)

$env:PYTHONPATH = Join-Path $RootDir "bindings\python"
foreach ($test in $pythonTests) {
    $path = Join-Path $RootDir "tests\python\$test.py"
    if (-not (Test-Path $path)) {
        Write-Host "SKIP: $test (not found)" -ForegroundColor Yellow
        continue
    }
    Write-Host "`nBenchmarking python $test..." -ForegroundColor Green
    $times = @()
    for ($i = 0; $i -lt $Warmup + $Iterations; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process -FilePath "python" -ArgumentList "-m pytest $path -x -q" -NoNewWindow -RedirectStandardOutput "NUL" -RedirectStandardError "NUL" -PassThru -Wait
        $sw.Stop()
        if ($p.ExitCode -ne 0) {
            Write-Host "  FAIL ($($p.ExitCode))" -ForegroundColor Red
            break
        }
        if ($i -ge $Warmup) { $times += $sw.Elapsed.TotalMilliseconds }
    }
    if ($times.Count -gt 0) {
        $avg = ($times | Measure-Object -Average).Average
        $min = ($times | Measure-Object -Minimum).Minimum
        $max = ($times | Measure-Object -Maximum).Maximum
        Write-Host "  Avg: $([math]::Round($avg, 1))ms  Min: $([math]::Round($min, 1))ms  Max: $([math]::Round($max, 1))ms (n=$($times.Count))" -ForegroundColor Cyan
    }
}

Write-Host "`n=== Benchmarking complete ===" -ForegroundColor Cyan
