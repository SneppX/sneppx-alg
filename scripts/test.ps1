param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "Release",
    [string]$Filter = "",
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RootDir "build"

if (-not (Test-Path $BuildDir)) {
    Write-Error "Build directory not found. Run build.ps1 first."
    exit 1
}

$ctestArgs = @(
    "--test-dir", $BuildDir,
    "-C", $Config
)

if ($Verbose) {
    $ctestArgs += "--output-on-failure"
}

if ($Filter) {
    $ctestArgs += "-R", $Filter
}

Write-Host "Running tests ($Config)..." -ForegroundColor Cyan
& ctest @ctestArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "All tests passed!" -ForegroundColor Green
} else {
    Write-Host "Some tests failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
