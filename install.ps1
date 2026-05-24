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
    Write-Host "chatterbox: ok (GPU-targeted - expect under 0.2x RTF on CPU)" -ForegroundColor Green
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
    Write-Host "f5tts: ok (GPU-targeted - expect under 0.1x RTF on CPU)" -ForegroundColor Green
} else {
    Write-Host "f5tts: already installed" -ForegroundColor Gray
}

Step "Coqui XTTS-v2 (idiap fork)"
if (-not (Test-Path "venvs\coqui\Scripts\python.exe")) {
    Invoke-Checked "uv venv coqui" { uv venv venvs\coqui --python 3.11 }
    # Original coqui-ai/TTS is archived; idiap/coqui-ai-TTS is the maintained fork.
    # PyPI package name is `coqui-tts` (not `TTS` - the old name is squatted).
    Invoke-Checked "uv pip install coqui-tts" { uv pip install --python venvs\coqui\Scripts\python.exe coqui-tts soundfile numpy }
    # Pin transformers<5.0 because XTTS imports `isin_mps_friendly` from
    # transformers.pytorch_utils which was removed in transformers 5.x.
    Invoke-Checked "pin transformers<5.0" { uv pip install --python venvs\coqui\Scripts\python.exe "transformers>=4.45,<5.0" }
    # Pin torch<2.9 because Coqui requires torchcodec for audio IO starting with
    # torch 2.9 / torchaudio 2.9, and torchcodec needs FFmpeg shared DLLs which
    # don't ship with the typical Windows FFmpeg static build. torch 2.8 still
    # uses the soundfile backend directly. CPU wheels by default - swap to CUDA
    # later via: uv pip install --python venvs\coqui\Scripts\python.exe --reinstall "torch<2.9" "torchaudio<2.9" --index-url https://download.pytorch.org/whl/cu128
    Invoke-Checked "pin torch<2.9" { uv pip install --python venvs\coqui\Scripts\python.exe "torch<2.9" "torchaudio<2.9" }
    Write-Host "coqui: ok (XTTS-v2 ~2GB downloads on first use; non-commercial CPML license)" -ForegroundColor Green
} else {
    Write-Host "coqui: already installed" -ForegroundColor Gray
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

Step "OmniVoice (k2-fsa, 600+ languages)"
if (-not (Test-Path "venvs\omnivoice\Scripts\python.exe")) {
    Invoke-Checked "uv venv omnivoice" { uv venv venvs\omnivoice --python 3.11 }
    Invoke-Checked "uv pip install omnivoice" { uv pip install --python venvs\omnivoice\Scripts\python.exe omnivoice soundfile numpy }
    # GPU-targeted (diffusion LM); cu128 wheels for Blackwell (RTX 5090, sm_120).
    Invoke-Checked "torch cu128 for omnivoice" { uv pip install --python venvs\omnivoice\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "omnivoice: ok (zero-shot cloning, 600+ langs, weights auto-download from HF on first use)" -ForegroundColor Green
} else {
    Write-Host "omnivoice: already installed" -ForegroundColor Gray
}

Step "VoxCPM-0.5B (OpenBMB, multilingual cloning)"
if (-not (Test-Path "venvs\voxcpm\Scripts\python.exe")) {
    # voxcpm requires Python >=3.10 <3.13 and torch >=2.5 with CUDA >=12.
    Invoke-Checked "uv venv voxcpm" { uv venv venvs\voxcpm --python 3.11 }
    Invoke-Checked "uv pip install voxcpm" { uv pip install --python venvs\voxcpm\Scripts\python.exe voxcpm soundfile numpy }
    # cu128 wheels for Blackwell (RTX 5090, sm_120).
    Invoke-Checked "torch cu128 for voxcpm" { uv pip install --python venvs\voxcpm\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "voxcpm: ok (zero-shot cloning, 0.5B variant; runner uses openbmb/VoxCPM-0.5B)" -ForegroundColor Green
} else {
    Write-Host "voxcpm: already installed" -ForegroundColor Gray
}

Step "Magpie-TTS Multilingual 357M (NVIDIA NeMo, predefined voices, 9 langs)"
if (-not (Test-Path "venvs\magpie\Scripts\python.exe")) {
    Invoke-Checked "uv venv magpie" { uv venv venvs\magpie --python 3.11 }
    # nemo_toolkit[tts] pulls nemo_text_processing -> pynini, which has no Windows
    # wheel and needs OpenFST + bazel + MSVC to build from source. Skip the [tts]
    # extra; install NeMo core + safe TTS deps + the [asr]-equivalent set
    # individually (NeMo's eager imports drag in TTS->audio_codec->ASR->lhotse
    # ->pyannote->IPython even if you only want MagpieTTSModel). Runner calls
    # do_tts(apply_TN=False) to avoid the pynini-using code path at inference.
    Invoke-Checked "uv pip install nemo_toolkit (core, no [tts])" { uv pip install --python venvs\magpie\Scripts\python.exe nemo_toolkit }
    # NeMo core deps (lightning pinned to <=2.4 per NeMo's requirements_lightning.txt).
    Invoke-Checked "uv pip install nemo core deps" { uv pip install --python venvs\magpie\Scripts\python.exe hydra-core omegaconf "lightning>2.2.1,<=2.4.0" "pytorch-lightning>2.2.1,<=2.4.0" fiddle cloudpickle wrapt "ruamel.yaml" wget braceexpand webdataset huggingface_hub editdistance "jiwer>=3.1.0,<4.0.0" "peft<=0.18.0" wandb sacremoses "sentencepiece<1.0.0" "datasets>=3.2.0" inflect }
    Invoke-Checked "uv pip install nemo TTS safe deps" { uv pip install --python venvs\magpie\Scripts\python.exe attrdict "cdifflib==1.2.6" einops kornia librosa matplotlib nltk pandas seaborn }
    # NeMo telemetry trio - required at import time even if not used.
    Invoke-Checked "uv pip install nemo nv_one_logger telemetry" { uv pip install --python venvs\magpie\Scripts\python.exe nv_one_logger_core nv_one_logger_training_telemetry nv_one_logger_pytorch_lightning_integration }
    # ASR-eager-import deps pulled by tts.models.audio_codec -> asr.parts.preprocessing.features.
    Invoke-Checked "uv pip install nemo ASR-eager deps" { uv pip install --python venvs\magpie\Scripts\python.exe lhotse pyannote.core pyannote.metrics kaldi-python-io marshmallow optuna pydub pyloudnorm resampy sacrebleu whisper_normalizer ipython }
    # Per-language G2P tokenizers - Magpie's multilingual config instantiates ALL
    # of these at load time even if you only generate English.
    Invoke-Checked "uv pip install nemo per-lang G2P (jieba/pyopenjtalk/etc)" { uv pip install --python venvs\magpie\Scripts\python.exe jieba pypinyin pypinyin-dict janome pyopenjtalk }
    Invoke-Checked "uv pip install magpie-specific deps" { uv pip install --python venvs\magpie\Scripts\python.exe kaldialign soundfile numpy }
    # cu128 wheels for Blackwell (RTX 5090, sm_120). Must come LAST because
    # nv_one_logger / lightning downgrades silently revert torch otherwise.
    Invoke-Checked "torch cu128 for magpie" { uv pip install --python venvs\magpie\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "magpie: ok (predefined voices, gated on HF - run 'uvx hf auth login' if not done; apply_TN forced False)" -ForegroundColor Green
} else {
    Write-Host "magpie: already installed" -ForegroundColor Gray
}

Write-Host "`nDone. Run: python bench.py" -ForegroundColor Green
