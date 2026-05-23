# Install model venvs (one per model, isolated dependency trees). Idempotent.
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

Step "Kokoro-82M"
if (-not (Test-Path "venvs\kokoro\Scripts\python.exe")) {
    Invoke-Checked "uv venv kokoro" { uv venv venvs\kokoro --python 3.11 }
    Invoke-Checked "uv pip install kokoro" { uv pip install --python venvs\kokoro\Scripts\python.exe kokoro soundfile numpy }
    # misaki (Kokoro's tokenizer) auto-downloads spaCy en_core_web_sm via spacy.cli.download()
    # which shells out to pip. uv venvs have no pip, so pre-install the model wheel directly.
    Invoke-Checked "spacy en_core_web_sm" {
        uv pip install --python venvs\kokoro\Scripts\python.exe `
            https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
    }
    Write-Host "kokoro: ok" -ForegroundColor Green
} else {
    Write-Host "kokoro: already installed" -ForegroundColor Gray
}

Step "KittenTTS"
if (-not (Test-Path "venvs\kittentts\Scripts\python.exe")) {
    Invoke-Checked "uv venv kittentts" { uv venv venvs\kittentts --python 3.11 }
    # KittenTTS uses phonemizer + espeak-ng. espeakng-loader bundles the DLL.
    Invoke-Checked "uv pip install kittentts" { uv pip install --python venvs\kittentts\Scripts\python.exe kittentts espeakng-loader soundfile numpy }
    Write-Host "kittentts: ok" -ForegroundColor Green
} else {
    Write-Host "kittentts: already installed" -ForegroundColor Gray
}

Step "Piper"
if (-not (Test-Path "venvs\piper\Scripts\python.exe")) {
    Invoke-Checked "uv venv piper" { uv venv venvs\piper --python 3.11 }
    # piper-tts 1.4+ bundles espeak-ng, no piper-phonemize dependency, Windows-clean.
    Invoke-Checked "uv pip install piper" { uv pip install --python venvs\piper\Scripts\python.exe piper-tts soundfile numpy }
    Write-Host "piper: ok (voices auto-download on first use to ~/.cache/piper-voices)" -ForegroundColor Green
} else {
    Write-Host "piper: already installed" -ForegroundColor Gray
}

Step "ChatterBox-TTS"
if (-not (Test-Path "venvs\chatterbox\Scripts\python.exe")) {
    Invoke-Checked "uv venv chatterbox" { uv venv venvs\chatterbox --python 3.11 }
    Invoke-Checked "uv pip install chatterbox" { uv pip install --python venvs\chatterbox\Scripts\python.exe chatterbox-tts soundfile numpy }
    # perth (ChatterBox's audio watermarker) imports pkg_resources, which was removed
    # in setuptools 80+. Pin to <80 to keep the import working.
    Invoke-Checked "setuptools<80 (perth watermarker compat)" { uv pip install --python venvs\chatterbox\Scripts\python.exe "setuptools<80" }
    Write-Host "chatterbox: ok (GPU-targeted — expect <0.2x RTF on CPU)" -ForegroundColor Green
} else {
    Write-Host "chatterbox: already installed" -ForegroundColor Gray
}

Step "F5-TTS"
if (-not (Test-Path "venvs\f5tts\Scripts\python.exe")) {
    Invoke-Checked "uv venv f5tts" { uv venv venvs\f5tts --python 3.11 }
    Invoke-Checked "uv pip install f5-tts" { uv pip install --python venvs\f5tts\Scripts\python.exe f5-tts soundfile numpy }
    # torch 2.12+ routes torchaudio.load() through torchcodec, which needs
    # FFmpeg shared DLLs (not just ffmpeg.exe). HF datasets 3.0+ also pulls in
    # torchcodec via its audio feature. Pin datasets<3.0 and the runner
    # monkey-patches torchaudio.load to use soundfile directly.
    Invoke-Checked "datasets<3.0 (avoid torchcodec import)" { uv pip install --python venvs\f5tts\Scripts\python.exe "datasets<3.0" }
    Write-Host "f5tts: ok (GPU-targeted — expect <0.1x RTF on CPU)" -ForegroundColor Green
} else {
    Write-Host "f5tts: already installed" -ForegroundColor Gray
}

Step "VibeVoice-Realtime-0.5B (community fork)"
if (-not (Test-Path "venvs\vibevoice\Scripts\python.exe")) {
    Invoke-Checked "uv venv vibevoice" { uv venv venvs\vibevoice --python 3.11 }
    # The official microsoft/VibeVoice repo was taken down then partially restored
    # WITHOUT code. The community fork keeps the original code and added a
    # working streaming variant in 2025-12-04. The pypi `vibevoice==0.0.1` ships
    # the base architecture only (no streaming class), so install from the fork.
    Invoke-Checked "vibevoice (community fork)" {
        uv pip install --python venvs\vibevoice\Scripts\python.exe `
            "git+https://github.com/vibevoice-community/VibeVoice" torch soundfile numpy
    }
    Write-Host "vibevoice: ok (voice .pt presets auto-download on first use to ~/.cache/vibevoice-voices)" -ForegroundColor Green
} else {
    Write-Host "vibevoice: already installed" -ForegroundColor Gray
}

Write-Host "`nDone. Run: python bench.py" -ForegroundColor Green
