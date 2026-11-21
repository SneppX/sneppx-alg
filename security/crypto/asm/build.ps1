# Build script for SneppX-ALG ASM shared library
# Assembles all .asm files using MASM (ml64) and links them into a DLL
#
# Requirements:
#   - Visual Studio 2022 with x64 native tools prompt (vcvarsall.bat x64)
#   - MASM (ml64.exe) from the VS toolchain
#
# Usage:
#   .\build.ps1                  # Debug build
#   .\build.ps1 -Release         # Release build (optimized)
#   .\build.ps1 -Clean           # Clean artifacts

param(
    [switch]$Release,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Paths
$RepoRoot = Resolve-Path "$PSScriptRoot\..\..\.."
$AsmDir = Resolve-Path "$PSScriptRoot\x86_64"
$BuildDir = "$RepoRoot\build"
$OutDir = "$BuildDir\asm"
$OutputName = "neural_asm"

if ($Clean) {
    if (Test-Path $OutDir) {
        Remove-Item -Recurse -Force $OutDir
        Write-Host "Cleaned $OutDir"
    }
    exit 0
}
# Ensure output directory
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

# Check for ml64
$ml64 = Get-Command "ml64" -ErrorAction SilentlyContinue
if (-not $ml64) {
    Write-Warning "MASM (ml64.exe) not found. Skipping ASM build."
    Write-Warning "Open 'x64 Native Tools Command Prompt for VS 2022' and re-run."
    exit 0
}

# Collect all .asm files
$AsmFiles = Get-ChildItem -Path $AsmDir -Filter "*.asm" | Sort-Object Name
if ($AsmFiles.Count -eq 0) {
    Write-Warning "No .asm files found in $AsmDir"
    exit 0
}

Write-Host "Assembling $($AsmFiles.Count) ASM files..."

# Assemble each .asm to .obj
$ObjFiles = @()
foreach ($file in $AsmFiles) {
    $obj = "$OutDir\$($file.BaseName).obj"
    $cflags = "/c /nologo /Fo`"$obj`" /W3"
    if ($Release) {
        $cflags += " /O2"
    }
    $cmd = "ml64 $cflags `"$($file.FullName)`""
    Write-Host "  ml64 $($file.Name)..."
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ml64 failed for $($file.Name)"
        exit 1
    }
    $ObjFiles += $obj
}

# Link .obj files into DLL
$DllFile = "$OutDir\$OutputName.dll"
$LibFile = "$OutDir\$OutputName.lib"
$linkFlags = "/DLL /nologo /OUT:`"$DllFile`" /IMPLIB:`"$LibFile`""
if ($Release) {
    $linkFlags += " /OPT:REF /OPT:ICF"
}
$linkCmd = "link $linkFlags $($ObjFiles -join ' ')"
Write-Host "  Linking $OutputName.dll..."
Invoke-Expression $linkCmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "Link failed"
    exit 1
}

# Copy to build output for c_loader discovery
if (Test-Path "$BuildDir\Release") {
    Copy-Item $DllFile "$BuildDir\Release\$OutputName.dll" -Force
}
Copy-Item $DllFile "$BuildDir\$OutputName.dll" -Force

Write-Host "ASM build complete: $DllFile"
Write-Host "  Objects: $($ObjFiles.Count) files"
Write-Host "  Library: $LibFile"
