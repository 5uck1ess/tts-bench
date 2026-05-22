# Install 3 model venvs (pocket / neutts / luxtts). Idempotent — re-runs are no-ops.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

Step "Pocket-TTS"
if (-not (Test-Path "venvs\pocket\Scripts\python.exe")) {
    uv venv venvs\pocket --python 3.11
    if (-not (Test-Path "venvs\pocket\src")) {
        git clone https://github.com/kyutai-labs/pocket-tts venvs\pocket\src
    }
    venvs\pocket\Scripts\python.exe -m pip install -e venvs\pocket\src
    venvs\pocket\Scripts\python.exe -m pip install soundfile numpy
} else {
    Write-Host "already installed"
}

Step "NeuTTS (Air + Nano share this venv)"
if (-not (Test-Path "venvs\neutts\Scripts\python.exe")) {
    uv venv venvs\neutts --python 3.11
    # Try PyPI first; if that fails, fall back to git source.
    try {
        venvs\neutts\Scripts\python.exe -m pip install neutts
    } catch {
        Write-Host "PyPI install failed, cloning from GitHub..." -ForegroundColor Yellow
        if (-not (Test-Path "venvs\neutts\src")) {
            git clone https://github.com/neuphonic/neutts venvs\neutts\src
        }
        venvs\neutts\Scripts\python.exe -m pip install -e venvs\neutts\src
    }
    venvs\neutts\Scripts\python.exe -m pip install torch soundfile numpy
} else {
    Write-Host "already installed"
}

Step "LuxTTS"
if (-not (Test-Path "venvs\luxtts\Scripts\python.exe")) {
    uv venv venvs\luxtts --python 3.11
    try {
        venvs\luxtts\Scripts\python.exe -m pip install luxtts
    } catch {
        Write-Host "PyPI install failed, cloning from GitHub..." -ForegroundColor Yellow
        if (-not (Test-Path "venvs\luxtts\src")) {
            git clone https://github.com/ysharma3501/LuxTTS venvs\luxtts\src
        }
        venvs\luxtts\Scripts\python.exe -m pip install -e venvs\luxtts\src
    }
    venvs\luxtts\Scripts\python.exe -m pip install torch soundfile numpy
} else {
    Write-Host "already installed"
}

Write-Host "`nDone. Run: python bench.py" -ForegroundColor Green
