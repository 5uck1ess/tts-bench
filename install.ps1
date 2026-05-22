# Install 3 model venvs (pocket / neutts / luxtts). Idempotent — re-runs are no-ops.
#
# NOTE: We deliberately do NOT set $ErrorActionPreference = "Stop". Native commands
# like uv and pip write informational output to stderr, which PowerShell 5.1 treats
# as ErrorRecords. Instead we check $LASTEXITCODE after each native command.

Set-Location $PSScriptRoot

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Script)
    & $Script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Description (exit $LASTEXITCODE)" -ForegroundColor Red
        throw "$Description failed"
    }
}

Step "Pocket-TTS"
if (-not (Test-Path "venvs\pocket\Scripts\python.exe")) {
    Invoke-Checked "uv venv pocket" { uv venv venvs\pocket --python 3.11 }
    if (-not (Test-Path "venvs\pocket\src")) {
        Invoke-Checked "git clone pocket-tts" { git clone https://github.com/kyutai-labs/pocket-tts venvs\pocket\src }
    }
    Invoke-Checked "pip install pocket-tts" { venvs\pocket\Scripts\python.exe -m pip install -e venvs\pocket\src }
    Invoke-Checked "pip install pocket deps"  { venvs\pocket\Scripts\python.exe -m pip install soundfile numpy }
    Write-Host "pocket: ok" -ForegroundColor Green
} else {
    Write-Host "pocket: already installed" -ForegroundColor Gray
}

Step "NeuTTS (Air + Nano share this venv)"
if (-not (Test-Path "venvs\neutts\Scripts\python.exe")) {
    Invoke-Checked "uv venv neutts" { uv venv venvs\neutts --python 3.11 }
    # Try PyPI first; fall back to git source.
    & venvs\neutts\Scripts\python.exe -m pip install neutts
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyPI install failed, cloning from GitHub..." -ForegroundColor Yellow
        if (-not (Test-Path "venvs\neutts\src")) {
            Invoke-Checked "git clone neutts" { git clone https://github.com/neuphonic/neutts venvs\neutts\src }
        }
        Invoke-Checked "pip install neutts source" { venvs\neutts\Scripts\python.exe -m pip install -e venvs\neutts\src }
    }
    Invoke-Checked "pip install neutts deps" { venvs\neutts\Scripts\python.exe -m pip install torch soundfile numpy }
    Write-Host "neutts: ok" -ForegroundColor Green
} else {
    Write-Host "neutts: already installed" -ForegroundColor Gray
}

Step "LuxTTS"
if (-not (Test-Path "venvs\luxtts\Scripts\python.exe")) {
    Invoke-Checked "uv venv luxtts" { uv venv venvs\luxtts --python 3.11 }
    & venvs\luxtts\Scripts\python.exe -m pip install luxtts
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyPI install failed, cloning from GitHub..." -ForegroundColor Yellow
        if (-not (Test-Path "venvs\luxtts\src")) {
            Invoke-Checked "git clone LuxTTS" { git clone https://github.com/ysharma3501/LuxTTS venvs\luxtts\src }
        }
        Invoke-Checked "pip install luxtts source" { venvs\luxtts\Scripts\python.exe -m pip install -e venvs\luxtts\src }
    }
    Invoke-Checked "pip install luxtts deps" { venvs\luxtts\Scripts\python.exe -m pip install torch soundfile numpy }
    Write-Host "luxtts: ok" -ForegroundColor Green
} else {
    Write-Host "luxtts: already installed" -ForegroundColor Gray
}

Write-Host "`nDone. Run: python bench.py" -ForegroundColor Green
