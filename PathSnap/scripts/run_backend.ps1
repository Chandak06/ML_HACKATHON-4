param([string]$Mode="student_test",[string]$BindHost="0.0.0.0",[int]$Port=8000)
$ErrorActionPreference="Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PATHSNAP_MODE=$Mode
$env:PATHSNAP_HOST=$BindHost
$env:PATHSNAP_PORT="$Port"
& .\.venv\Scripts\python.exe -m app.api

