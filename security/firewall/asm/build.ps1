$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\..\.."

$AsmFiles = @(
    "ip_match.asm",
    "rate_counter.asm",
    "conn_track.asm",
    "port_knock.asm"
)

$Objs = @()
foreach ($file in $AsmFiles) {
    $src = Join-Path $ScriptDir $file
    $name = [System.IO.Path]::GetFileNameWithoutExtension($file)
    $obj = Join-Path $ScriptDir "$name.obj"
    Write-Host "Assembling $file..."
    & "ml64" /nologo /c /Fo"$obj" "$src" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Assembly failed: $file"
        exit 1
    }
    $Objs += $obj
}

$DllOut = Join-Path $ProjectRoot "build\firewall_core.dll"
Write-Host "Linking firewall_core.dll..."
& "link" /nologo /DLL /OUT:"$DllOut" /MACHINE:X64 $Objs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Link failed"
    exit 1
}

Write-Host "firewall_core.dll built at $DllOut"
