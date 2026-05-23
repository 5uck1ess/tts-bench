# Install 3 model venvs (pocket / neutts / luxtts). Idempotent — re-runs are no-ops.
#
# Uses `uv pip install --python <venv-python>` rather than `python -m pip` because
# `uv venv` doesn't seed pip into the venv by default. uv's pip is also much faster.

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
    Invoke-Checked "uv pip install pocket-tts" { uv pip install --python venvs\pocket\Scripts\python.exe -e venvs\pocket\src }
    Invoke-Checked "uv pip install pocket deps" { uv pip install --python venvs\pocket\Scripts\python.exe soundfile numpy }
    Write-Host "pocket: ok" -ForegroundColor Green
} else {
    Write-Host "pocket: already installed" -ForegroundColor Gray
}

Step "NeuTTS (Air + Nano share this venv)"
if (-not (Test-Path "venvs\neutts\Scripts\python.exe")) {
    Invoke-Checked "uv venv neutts" { uv venv venvs\neutts --python 3.11 }
    # Try PyPI first; fall back to git source.
    & uv pip install --python venvs\neutts\Scripts\python.exe neutts
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyPI install failed, cloning from GitHub..." -ForegroundColor Yellow
        if (-not (Test-Path "venvs\neutts\src")) {
            Invoke-Checked "git clone neutts" { git clone https://github.com/neuphonic/neutts venvs\neutts\src }
        }
        Invoke-Checked "uv pip install neutts source" { uv pip install --python venvs\neutts\Scripts\python.exe -e venvs\neutts\src }
    }
    Invoke-Checked "uv pip install neutts deps" { uv pip install --python venvs\neutts\Scripts\python.exe torch soundfile numpy llama-cpp-python }
    Write-Host "neutts: ok" -ForegroundColor Green
} else {
    Write-Host "neutts: already installed" -ForegroundColor Gray
}

Step "LuxTTS"
if (-not (Test-Path "venvs\luxtts\Scripts\python.exe")) {
    Invoke-Checked "uv venv luxtts" { uv venv venvs\luxtts --python 3.11 }
    & uv pip install --python venvs\luxtts\Scripts\python.exe luxtts
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyPI install failed, cloning from GitHub..." -ForegroundColor Yellow
        if (-not (Test-Path "venvs\luxtts\src")) {
            Invoke-Checked "git clone LuxTTS" { git clone https://github.com/ysharma3501/LuxTTS venvs\luxtts\src }
        }
        Invoke-Checked "uv pip install luxtts source" { uv pip install --python venvs\luxtts\Scripts\python.exe -e venvs\luxtts\src }
    }
    Invoke-Checked "uv pip install luxtts deps" { uv pip install --python venvs\luxtts\Scripts\python.exe torch soundfile numpy }
    Write-Host "luxtts: ok" -ForegroundColor Green
} else {
    Write-Host "luxtts: already installed" -ForegroundColor Gray
}

Write-Host "`nDone. Run: python bench.py" -ForegroundColor Green
