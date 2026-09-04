# Windows build-and-run for TetgenFEM (port of w_crFEM_build_and_run.sh).
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\w_crFEM_build_and_run.ps1

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $RootDir "out\build"

Write-Host "======================================"
Write-Host "Compile TetgenFEM (Windows)"
Write-Host "======================================"

function Test-CommandExists([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-CommandExists "cmake")) {
    Write-Host "Error: cmake not found. Install CMake and add it to PATH."
    exit 1
}

$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$VsInstall = $null
if (Test-Path $VsWhere) {
    $VsInstall = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}

if (-not $VsInstall) {
    Write-Host "Error: Visual Studio 2022 C++ workload not found."
    Write-Host "Install 'Desktop development with C++' from Visual Studio Installer."
    exit 1
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir
try {
    if (Test-Path "CMakeCache.txt") {
        Remove-Item "CMakeCache.txt" -Force
    }
    Get-ChildItem -Filter "TetgenFEM.exe" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force

    Write-Host "Running CMake..."
    cmake $RootDir -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed"
    }

    Write-Host "Building..."
    cmake --build . --config Release --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
}
finally {
    Pop-Location
}

Write-Host "======================================"
Write-Host "Build succeeded"
Write-Host "======================================"

$ExeCandidates = @(
    (Join-Path $BuildDir "Release\TetgenFEM.exe"),
    (Join-Path $BuildDir "TetgenFEM.exe")
)
$Exe = $ExeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Exe) {
    Write-Host "Error: TetgenFEM.exe not found under $BuildDir"
    exit 1
}

Write-Host ""
Write-Host "======================================"
Write-Host "Run TetgenFEM"
Write-Host "======================================"

$RunDir = Join-Path $RootDir "TetgenFEM"
Set-Location $RunDir
Write-Host "Running $Exe"
& $Exe
$Code = $LASTEXITCODE

Write-Host "======================================"
Write-Host "Program exited (code $Code)"
Write-Host "======================================"
exit $Code
