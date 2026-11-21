# SNEPPX-Algo v0.1.0 Installer (Windows)
Write-Host "=== SNEPPX-Algo v0.1.0 Installer ===" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Check dependencies
function Check-Command($cmd, $name) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "  FOUND: $name" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  MISSING: $name" -ForegroundColor Yellow
        return $false
    }
}

Write-Host ">> Checking dependencies..."
$hasCMake = Check-Command "cmake" "CMake"
$hasMSVC = (Get-Command "cl.exe" -ErrorAction SilentlyContinue) -ne $null
if (-not $hasMSVC) {
    Write-Host "  MISSING: Visual Studio Build Tools (cl.exe)" -ForegroundColor Yellow
}

if (-not $hasCMake) {
    Write-Host ""
    Write-Host "  Install with:"
    Write-Host "    choco install cmake"
    Write-Host "    winget install Kitware.CMake"
}

if (-not $hasMSVC) {
    Write-Host ""
    Write-Host "  Install Visual Studio 2022 Build Tools:"
    Write-Host "    choco install visualstudio2022buildtools"
    Write-Host "    winget install Microsoft.VisualStudio.2022.BuildTools"
}

# Build
Write-Host ""
Write-Host ">> Building..."
New-Item -ItemType Directory -Path "build" -Force | Out-Null
Set-Location "build"

if ($hasMSVC) {
    cmake .. -G Ninja -DSNEPPX_BUILD_TESTS=ON -DSNEPPX_BUILD_PYTHON=OFF
    cmake --build . --config Release
} else {
    Write-Host "  ERROR: No C++ compiler found. Install Visual Studio Build Tools and re-run." -ForegroundColor Red
    exit 1
}

# Test
Write-Host ""
Write-Host ">> Running tests..."
ctest --output-on-failure -C Release

Write-Host ""
Write-Host "=== SNEPPX-Algo v0.1.0 installed ===" -ForegroundColor Cyan
Write-Host "Run './scripts/test.sh' to verify."
Write-Host "See docs/ for documentation."
