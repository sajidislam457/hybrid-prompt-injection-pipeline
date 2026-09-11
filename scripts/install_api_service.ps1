# Install SecureAI Python API as a Windows service (NSSM).
# Run from an elevated PowerShell in the repo root:
#   .\scripts\install_api_service.ps1
#   .\scripts\install_api_service.ps1 -Uninstall

param(
    [switch]$Uninstall,
    [string]$ServiceName = "SecureAI-API",
    [string]$NssmPath = "nssm"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Assert-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run this script as Administrator."
    }
}

Assert-Admin

if ($Uninstall) {
    & $NssmPath stop $ServiceName
    & $NssmPath remove $ServiceName confirm
    Write-Host "Removed service $ServiceName"
    exit 0
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = (Get-Command python -ErrorAction Stop).Source
}

$App = Join-Path $Root "run_api.py"
if (-not (Test-Path $App)) {
    throw "run_api.py not found at $App"
}

$nssmCmd = Get-Command $NssmPath -ErrorAction SilentlyContinue
if (-not $nssmCmd) {
    throw "nssm.exe not found. Install NSSM (https://nssm.cc) and add it to PATH, or pass -NssmPath."
}

& $NssmPath install $ServiceName $Py $App
& $NssmPath set $ServiceName AppDirectory $Root
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName AppStdout (Join-Path $Root "logs\api_service.out.log")
& $NssmPath set $ServiceName AppStderr (Join-Path $Root "logs\api_service.err.log")
& $NssmPath set $ServiceName AppRotateFiles 1
& $NssmPath set $ServiceName AppExit Default Restart
& $NssmPath set $ServiceName AppRestartDelay 3000
& $NssmPath start $ServiceName

Write-Host ""
Write-Host "Service $ServiceName installed and started."
Write-Host "Health: http://127.0.0.1:8000/health"
Write-Host "Uninstall: .\scripts\install_api_service.ps1 -Uninstall"
