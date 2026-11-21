# SNEPPX Compute Sanitizer CI Wrapper (Windows)
# Usage: .\scripts\run_sanitizers.ps1 [-Target <name>] [-ExtraArgs <string>]

param(
    [string]$Target = "",
    [string]$ExtraArgs = ""
)

$BuildDir = "build"
$Passed = 0
$Failed = 0

Write-Host "=== SNEPPX Sanitizer CI ===" -ForegroundColor Cyan

# 1. Build with ASan (MSVC)
Write-Host "`n--- Building with AddressSanitizer ---" -ForegroundColor Yellow
cmake -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_USE_ASAN=ON
if ($LASTEXITCODE -ne 0) { Write-Host "ASan build failed" -ForegroundColor Red; exit 1 }
cmake --build $BuildDir --config Debug
if ($LASTEXITCODE -ne 0) { Write-Host "ASan build failed" -ForegroundColor Red; exit 1 }
$Passed++

# 2. Run compute-sanitizer if available
$CS = Get-Command "compute-sanitizer.exe" -ErrorAction SilentlyContinue
if ($CS) {
    Write-Host "`n--- Running compute-sanitizer (memcheck) ---" -ForegroundColor Yellow
    Get-ChildItem -Path "$BuildDir/tests" -Recurse -Filter "test_*.exe" | ForEach-Object {
        Write-Host "Testing: $($_.FullName)"
        & compute-sanitizer --tool memcheck --report-api-errors all $_.FullName $ExtraArgs 2>&1 | Select-Object -Last 20
        if ($LASTEXITCODE -eq 0) { $Passed++ } else { $Failed++ }
    }

    Write-Host "`n--- Running compute-sanitizer (racecheck) ---" -ForegroundColor Yellow
    Get-ChildItem -Path "$BuildDir/tests" -Recurse -Filter "test_*.exe" | ForEach-Object {
        & compute-sanitizer --tool racecheck $_.FullName $ExtraArgs 2>&1 | Select-Object -Last 10
        if ($LASTEXITCODE -eq 0) { $Passed++ } else { $Failed++ }
    }

    Write-Host "`n--- Running compute-sanitizer (initcheck) ---" -ForegroundColor Yellow
    Get-ChildItem -Path "$BuildDir/tests" -Recurse -Filter "test_*.exe" | ForEach-Object {
        & compute-sanitizer --tool initcheck $_.FullName $ExtraArgs 2>&1 | Select-Object -Last 10
        if ($LASTEXITCODE -eq 0) { $Passed++ } else { $Failed++ }
    }
} else {
    Write-Host "WARNING: compute-sanitizer not found. Skipping CUDA memcheck." -ForegroundColor Magenta
}

# 3. Run ctest
Write-Host "`n--- Running ctest ---" -ForegroundColor Yellow
Push-Location $BuildDir
ctest -C Debug --output-on-failure $ExtraArgs
if ($LASTEXITCODE -eq 0) { $Passed++ } else { $Failed++ }
Pop-Location

Write-Host "`n=== Sanitizer CI Complete ($Passed passed, $Failed failed) ===" -ForegroundColor Green
