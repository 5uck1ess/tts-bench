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

Step "ChatterBox-TTS (base 1.2B + Turbo ~744M share this venv)"
if (-not (Test-Path "venvs\chatterbox\Scripts\python.exe")) {
    Invoke-Checked "uv venv chatterbox" { uv venv venvs\chatterbox --python 3.11 }
    Invoke-Checked "uv pip install chatterbox" { uv pip install --python venvs\chatterbox\Scripts\python.exe chatterbox-tts soundfile numpy }
    # perth (ChatterBox's audio watermarker) imports pkg_resources, which was removed
    # in setuptools 80+. Pin to <80 to keep the import working.
    Invoke-Checked "setuptools<80 (perth watermarker compat)" { uv pip install --python venvs\chatterbox\Scripts\python.exe "setuptools<80" }
    # Chatterbox Turbo (~744M GPT2-based AR model, Dec 2025). Same venv, same runner
    # dispatched via --variant turbo. Weights auto-download from ResembleAI/chatterbox-turbo
    # on first use; the turbo checkpoint uses the base tokenizer from ResembleAI/chatterbox.
    # cu128 wheels for Blackwell (RTX 5090, sm_120).
    Invoke-Checked "torch cu128 for chatterbox" { uv pip install --python venvs\chatterbox\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "chatterbox: ok (base 1.2B GPU-targeted + Turbo ~744M AR; both via --variant)" -ForegroundColor Green
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
    # f5-tts pins hydra-core 1.0.7 + omegaconf 2.0.6 which trip on Python 3.11's
    # stricter dataclass rules (mutable default ValueError). Bump to versions
    # that accept frozen field types.
    Invoke-Checked "hydra-core/omegaconf py3.11 fix" { uv pip install --python venvs\f5tts\Scripts\python.exe --upgrade "hydra-core>=1.3" "omegaconf>=2.3" }
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
    # bitsandbytes for the 7B Q8 weights (Linux Q8 path); harmless on Windows.
    Invoke-Checked "bitsandbytes for vibevoice 7b Q8" { uv pip install --python venvs\vibevoice\Scripts\python.exe "bitsandbytes>=0.48.1" }
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

Step "ZipVoice (k2-fsa, 123M, flow matching, zero-shot cloning)"
if (-not (Test-Path "venvs\zipvoice\Scripts\python.exe")) {
    # No pip wheel — source clone + editable install. Same lab as OmniVoice.
    # pyproject.toml in the repo has no [build-system]; we patch it during clone.
    Invoke-Checked "uv venv zipvoice" { uv venv venvs\zipvoice --python 3.11 }
    if (-not (Test-Path "venvs\zipvoice\src")) {
        Invoke-Checked "git clone zipvoice" { git clone --depth 1 https://github.com/k2-fsa/ZipVoice venvs\zipvoice\src }
    }
    # Patch pyproject.toml so editable install works (upstream has no [build-system]).
    $pj = "venvs\zipvoice\src\pyproject.toml"
    if (-not (Select-String -Path $pj -Pattern "\[build-system\]" -Quiet)) {
        $patch = "[build-system]`nbuild-backend = `"setuptools.build_meta`"`nrequires = [`"setuptools>=61`"]`n`n[project]`nname = `"zipvoice`"`nversion = `"0.1.0`"`n`n[tool.setuptools.packages.find]`ninclude = [`"zipvoice*`"]`n`n"
        $existing = Get-Content $pj -Raw
        Set-Content $pj ($patch + $existing)
    }
    Invoke-Checked "uv pip install zipvoice editable" { uv pip install --python venvs\zipvoice\Scripts\python.exe -e venvs\zipvoice\src }
    Invoke-Checked "uv pip install zipvoice requirements" { uv pip install --python venvs\zipvoice\Scripts\python.exe -r venvs\zipvoice\src\requirements.txt }
    # cu128 wheels for Blackwell (RTX 5090, sm_120). k2 is NOT required for inference.
    # Pin to 2.8.0: torchaudio>=2.9.0 requires torchcodec which has no Windows wheels.
    Invoke-Checked "torch cu128 for zipvoice" { uv pip install --python venvs\zipvoice\Scripts\python.exe "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "zipvoice: ok (zero-shot cloning zh+en, 123M, weights auto-download from HF on first use)" -ForegroundColor Green
} else {
    Write-Host "zipvoice: already installed" -ForegroundColor Gray
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
    # Pin hydra-core>=1.3 + omegaconf>=2.3 — nemo_toolkit's default resolver picks
    # hydra-core 1.0.7 / omegaconf 2.0.6 which trip on Python 3.11's stricter
    # dataclass rules ("mutable default ... Override"). Same fix as f5tts/indextts.
    Invoke-Checked "uv pip install nemo core deps" { uv pip install --python venvs\magpie\Scripts\python.exe "hydra-core>=1.3" "omegaconf>=2.3" "lightning>2.2.1,<=2.4.0" "pytorch-lightning>2.2.1,<=2.4.0" fiddle cloudpickle wrapt "ruamel.yaml" wget braceexpand webdataset huggingface_hub editdistance "jiwer>=3.1.0,<4.0.0" "peft<=0.18.0" wandb sacremoses "sentencepiece<1.0.0" "datasets>=3.2.0" inflect }
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

Step "Qwen3-TTS-Base 1.7B (Alibaba Qwen, zero-shot cloning, 10 langs)"
if (-not (Test-Path "venvs\qwentts\Scripts\python.exe")) {
    # qwen-tts requires Python 3.12 per upstream docs (works on 3.13 too but
    # 3.12 is what they test against).
    Invoke-Checked "uv venv qwentts" { uv venv venvs\qwentts --python 3.12 }
    Invoke-Checked "uv pip install qwen-tts" { uv pip install --python venvs\qwentts\Scripts\python.exe qwen-tts soundfile numpy }
    # cu128 wheels for Blackwell (RTX 5090, sm_120). qwen-tts pulls cu12 torch by
    # default; reinstall with the cu128 index AFTER qwen-tts so its deps don't
    # silently downgrade torch back to a non-cu128 build.
    Invoke-Checked "torch cu128 for qwentts" { uv pip install --python venvs\qwentts\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    # FlashAttention 2 is recommended upstream for best perf but flash-attn has
    # no first-party Windows wheels - skip it; runner falls back to default
    # SDPA implementation. ~20-40% slower inference but works.
    Write-Host "qwentts: ok (zero-shot cloning, wav+txt, 10 langs; flash-attn skipped on Windows)" -ForegroundColor Green
} else {
    Write-Host "qwentts: already installed" -ForegroundColor Gray
}

Step "faster-qwen3-tts (CUDA-graph fast path for Qwen3-TTS-Base 1.7B)"
if (-not (Test-Path "venvs\qwentts_fast\Scripts\python.exe")) {
    Invoke-Checked "uv venv qwentts_fast" { uv venv venvs\qwentts_fast --python 3.11 }
    Invoke-Checked "torch cu128 in qwentts_fast" {
        uv pip install --python venvs\qwentts_fast\Scripts\python.exe `
            --index-url https://download.pytorch.org/whl/cu128 `
            torch torchaudio
    }
    Invoke-Checked "faster-qwen3-tts" {
        uv pip install --python venvs\qwentts_fast\Scripts\python.exe `
            faster-qwen3-tts soundfile numpy
    }
    Write-Host "qwentts_fast: ok (CUDA-graph fast path, same Qwen3-TTS-Base 1.7B weights; CUDA-only)" -ForegroundColor Green
} else {
    Write-Host "qwentts_fast: already installed" -ForegroundColor Gray
}

Step "IndexTTS-2 (Bilibili Index, zero-shot cloning + emotion control)"
if (-not (Test-Path "venvs\indextts\Scripts\python.exe")) {
    # No pip wheel - upstream requires source clone. We mirror the neutts/luxtts
    # pattern: venv + clone into venvs\indextts\src + uv pip install -e ...
    # Model weights (IndexTeam/IndexTTS-2) are downloaded by huggingface_hub
    # on first runner call (~5GB), not by this install step.
    Invoke-Checked "uv venv indextts" { uv venv venvs\indextts --python 3.11 }
    if (-not (Test-Path "venvs\indextts\src")) {
        Invoke-Checked "git clone index-tts" { git clone --depth 1 https://github.com/index-tts/index-tts venvs\indextts\src }
    }
    Invoke-Checked "uv pip install indextts source" { uv pip install --python venvs\indextts\Scripts\python.exe -e venvs\indextts\src soundfile numpy huggingface_hub }
    # cu128 wheels for Blackwell (RTX 5090, sm_120).
    Invoke-Checked "torch cu128 for indextts" { uv pip install --python venvs\indextts\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    # IndexTTS-2 transitively pins omegaconf 2.0.6 (returns None for missing
    # keys instead of raising AttributeError, breaking RepCodec init) and a
    # protobuf >= 4 wheel that conflicts with its own legacy _pb2 stubs.
    Invoke-Checked "omegaconf 2.3 + protobuf 3.20 for indextts" { uv pip install --python venvs\indextts\Scripts\python.exe --upgrade "omegaconf>=2.3" "protobuf<3.21" }
    Write-Host "indextts: ok (zero-shot cloning, wav only, weights auto-download from HF on first use)" -ForegroundColor Green
} else {
    Write-Host "indextts: already installed" -ForegroundColor Gray
}

Step "Dia 1.6B (Nari Labs, Apache 2.0, dialogue + cloning)"
if (-not (Test-Path "venvs\dia\Scripts\python.exe")) {
    Invoke-Checked "uv venv dia" { uv venv venvs\dia --python 3.11 }
    Invoke-Checked "uv pip install dia from git" { uv pip install --python venvs\dia\Scripts\python.exe "git+https://github.com/nari-labs/dia.git" soundfile numpy }
    # Dia's pyproject pulls cu126 torch 2.6; reinstall cu128 for Blackwell (5090, sm_120).
    Invoke-Checked "torch cu128 for dia" { uv pip install --python venvs\dia\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "dia: ok (44.1kHz output, CUDA-only on Blackwell, default voice uses fixed seed=42)" -ForegroundColor Green
} else {
    Write-Host "dia: already installed" -ForegroundColor Gray
}

Step "Sesame CSM-1B (conversational speech model, in-context cloning)"
if (-not (Test-Path "venvs\sesame\Scripts\python.exe")) {
    # Native transformers support since 4.52.1 - no separate pip package needed.
    # MANUAL APPROVAL gating on HF: visit https://huggingface.co/sesame/csm-1b
    # and click "Ask for access" before first use. After approval lands in your
    # HF account, `hf auth login` will let the runner download weights.
    Invoke-Checked "uv venv sesame" { uv venv venvs\sesame --python 3.11 }
    Invoke-Checked "uv pip install sesame deps" { uv pip install --python venvs\sesame\Scripts\python.exe "transformers>=4.52.1" accelerate soundfile numpy librosa huggingface_hub }
    Invoke-Checked "torch cu128 for sesame" { uv pip install --python venvs\sesame\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "sesame: ok (CSM-1B; HF access REQUIRED - request at https://huggingface.co/sesame/csm-1b before running)" -ForegroundColor Green
} else {
    Write-Host "sesame: already installed" -ForegroundColor Gray
}

Step "MARS5-TTS (CAMB.AI, English zero-shot cloning, AGPL-3.0)"
if (-not (Test-Path "venvs\mars5\Scripts\python.exe")) {
    # Loaded via torch.hub - no source clone needed. ~1.2GB AR+NAR checkpoints
    # auto-download on first runner call. AGPL-3.0 license: not for commercial
    # use without a separate license from CAMB.AI.
    Invoke-Checked "uv venv mars5" { uv venv venvs\mars5 --python 3.10 }
    # numpy<2.0 pin: MARS5's torch.hub'd code uses np.array(obj, copy=False),
    # which was removed in NumPy 2.0 and raises ValueError at inference time.
    Invoke-Checked "uv pip install mars5 deps" { uv pip install --python venvs\mars5\Scripts\python.exe torch torchaudio librosa vocos encodec safetensors regex soundfile "numpy<2.0" }
    # cu128 wheels for Blackwell (RTX 5090, sm_120).
    Invoke-Checked "torch cu128 for mars5" { uv pip install --python venvs\mars5\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "mars5: ok (English only, AGPL-3.0; ref audio must be 1-12s, ~1.2GB weights auto-download on first run)" -ForegroundColor Green
} else {
    Write-Host "mars5: already installed" -ForegroundColor Gray
}

Step "Soprano 80M (ekwek1, Apache 2.0, predefined voice, 32kHz)"
if (-not (Test-Path "venvs\soprano\Scripts\python.exe")) {
    Invoke-Checked "uv venv soprano" { uv venv venvs\soprano --python 3.11 }
    if (-not (Test-Path "venvs\soprano\src")) {
        Invoke-Checked "git clone soprano" { git clone --depth 1 https://github.com/ekwek1/soprano venvs\soprano\src }
    }
    Invoke-Checked "uv pip install soprano source" { uv pip install --python venvs\soprano\Scripts\python.exe -e venvs\soprano\src }
    Invoke-Checked "uv pip install soprano deps" { uv pip install --python venvs\soprano\Scripts\python.exe soundfile numpy }
    # Windows CUDA gotcha: soprano pip install pulls CPU-only torch; reinstall cu128 for Blackwell (5090, sm_120).
    Invoke-Checked "torch cu128 for soprano" { uv pip install --python venvs\soprano\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "soprano: ok (~80M params, Apache 2.0, predefined voice, 32kHz, English only; weights auto-download from ekwek/Soprano-1.1-80M on first use)" -ForegroundColor Green
} else {
    Write-Host "soprano: already installed" -ForegroundColor Gray
}

Step "MOSS-TTS-Nano 100M (OpenMOSS/MOSI.AI, Apache 2.0, zero-shot cloning, 48kHz)"
if (-not (Test-Path "venvs\moss_tts_nano\Scripts\python.exe")) {
    # No PyPI wheel that ships the model code — install editable from source clone.
    # Upstream README recommends Python 3.12. The package's setup.py exposes
    # `moss-tts-nano` console-script + top-level py-modules (infer, app, etc.),
    # so an editable install also makes `from moss_tts_nano.defaults import ...`
    # and `import infer` available at the repo root.
    Invoke-Checked "uv venv moss_tts_nano" { uv venv venvs\moss_tts_nano --python 3.12 }
    if (-not (Test-Path "venvs\moss_tts_nano\src")) {
        Invoke-Checked "git clone MOSS-TTS-Nano" { git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano venvs\moss_tts_nano\src }
    }
    # Editable install pulls the pyproject deps (torch 2.7, transformers 4.57.1, etc).
    Invoke-Checked "uv pip install moss-tts-nano editable" { uv pip install --python venvs\moss_tts_nano\Scripts\python.exe -e venvs\moss_tts_nano\src }
    # The README notes WeTextProcessing + pynini are optional (the runner sets
    # enable_wetext_processing=0 to skip them). Pynini has no Windows wheel and
    # needs OpenFST/bazel to build, so we deliberately skip both — the upstream
    # text normalizer ("normalize_tts_text") is still applied.
    Invoke-Checked "uv pip install moss_tts_nano deps" { uv pip install --python venvs\moss_tts_nano\Scripts\python.exe soundfile numpy }
    # Upstream pins torch 2.7 (cu126). Reinstall cu128 wheels for Blackwell (5090, sm_120).
    # Pin torch == 2.8.0: 2.9+ routes torchaudio.load() through torchcodec, whose
    # libtorchcodec*.dll needs FFmpeg shared DLLs on PATH (not present on stock
    # Windows). 2.8.0 still uses the soundfile backend.
    Invoke-Checked "torch cu128 for moss_tts_nano" { uv pip install --python venvs\moss_tts_nano\Scripts\python.exe --reinstall "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "moss_tts_nano: ok (~100M params, 48kHz, voice cloning; weights auto-download from OpenMOSS-Team/MOSS-TTS-Nano on first use)" -ForegroundColor Green
} else {
    Write-Host "moss_tts_nano: already installed" -ForegroundColor Gray
}

Step "MOSS-TTS flagship (OpenMOSS, Apache 2.0, 8B Qwen3-backbone zero-shot cloning, 20 langs)"
if (-not (Test-Path "venvs\moss_tts\Scripts\python.exe")) {
    # Source-clone install. Upstream's [torch-runtime] extra pins torch==2.9.1
    # which routes torchaudio.load() through torchcodec — torchcodec's DLL fails
    # to load on Windows without FFmpeg shared libs on PATH. Install everything
    # else from the [torch-runtime] extra, then downgrade torch/torchaudio to
    # 2.8.0 so we get the soundfile backend.
    Invoke-Checked "uv venv moss_tts" { uv venv venvs\moss_tts --python 3.12 }
    if (-not (Test-Path "venvs\moss_tts\src")) {
        Invoke-Checked "git clone MOSS-TTS" { git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS venvs\moss_tts\src }
    }
    Invoke-Checked "uv pip install moss-tts torch-runtime" {
        uv pip install --python venvs\moss_tts\Scripts\python.exe `
            --extra-index-url https://download.pytorch.org/whl/cu128 `
            -e "venvs\moss_tts\src[torch-runtime]"
    }
    Invoke-Checked "uv pip install moss_tts deps" { uv pip install --python venvs\moss_tts\Scripts\python.exe soundfile }
    # Pin torch == 2.8.0 (avoid torchcodec — see comment above + moss_tts_nano block).
    Invoke-Checked "torch 2.8.0 for moss_tts" { uv pip install --python venvs\moss_tts\Scripts\python.exe --reinstall "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "moss_tts: ok (8B Qwen3 backbone, 20 langs, 24kHz, voice cloning; ~16GB weights auto-download from OpenMOSS-Team/MOSS-TTS on first use)" -ForegroundColor Green
} else {
    Write-Host "moss_tts: already installed" -ForegroundColor Gray
}

Step "Supertonic (Supertone Inc., ONNX, 99M, 31 langs, predefined voices)"
if (-not (Test-Path "venvs\supertonic\Scripts\python.exe")) {
    # Pure-ONNX runtime; no torch dependency. ~25MB weights auto-downloaded
    # from HF (Supertone/supertonic) on first run.
    # Open-weight release is fixed-voice only; cloning lives in the hosted
    # Voice Builder / Supertone Play API.
    Invoke-Checked "uv venv supertonic" { uv venv venvs\supertonic --python 3.11 }
    Invoke-Checked "uv pip install supertonic" { uv pip install --python venvs\supertonic\Scripts\python.exe supertonic soundfile psutil }
    Write-Host "supertonic: ok (~99M ONNX, 31 langs, MIT code + OpenRAIL-M weights, CPU-only)" -ForegroundColor Green
} else {
    Write-Host "supertonic: already installed" -ForegroundColor Gray
}

Step "Fish Speech 1.5 (fishaudio, zero-shot cloning, 44.1kHz)"
if (-not (Test-Path "venvs\fish\Scripts\python.exe")) {
    # Source clone of the v1.5.0 tag. [stable] extras are intentionally SKIPPED:
    # they pin torch<=2.4.1 which is incompatible with the RTX 5090 (Blackwell
    # sm_120), which needs cu128 / torch 2.7+. We install fish-speech with no
    # extras, hard-cap numpy<=1.26.4 (fish requirement), then reinstall torch cu128
    # LAST. The runner instantiates TTSInferenceEngine directly (NOT ModelManager,
    # which has an unconditional funasr import we want to avoid).
    Invoke-Checked "uv venv fish" { uv venv venvs\fish --python 3.10 }
    Invoke-Checked "clone fish-speech v1.5" { git clone --branch v1.5.0 https://github.com/fishaudio/fish-speech venvs\fish\src }
    Invoke-Checked "uv pip install fish-speech (no extras)" { uv pip install --python venvs\fish\Scripts\python.exe -e venvs\fish\src }
    Invoke-Checked "numpy<=1.26.4 (fish hard cap)" { uv pip install --python venvs\fish\Scripts\python.exe "numpy<=1.26.4" }
    Invoke-Checked "fish deps" { uv pip install --python venvs\fish\Scripts\python.exe soundfile }
    Invoke-Checked "torch cu128 for fish (LAST)" { uv pip install --python venvs\fish\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Invoke-Checked "download fish-speech-1.5 weights" { uv run --python venvs\fish\Scripts\python.exe -- hf download fishaudio/fish-speech-1.5 --local-dir venvs\fish\src\checkpoints\fish-speech-1.5 }
    Write-Host "fish: ok" -ForegroundColor Green
} else { Write-Host "fish: already installed" -ForegroundColor Gray }

Step "Maya1 (maya-research, Apache 2.0, voice-description default voice, 24kHz, SNAC codec)"
if (-not (Test-Path "venvs\maya1\Scripts\python.exe")) {
    # Default-voice model: no audio cloning. The voice is steered by a natural-
    # language description string; the runner uses a fixed DEFAULT_VOICE_DESC.
    # Llama-style causal LM emits flat SNAC codec tokens -> decoded by the
    # hubertsiuzdak/snac_24khz SNAC model (auto-downloads on first run alongside
    # the ~3B maya-research/maya1 weights). Windows/Linux use transformers+SNAC;
    # the Mac path uses MLX instead (see install.sh).
    Invoke-Checked "uv venv maya1" { uv venv venvs\maya1 --python 3.11 }
    Invoke-Checked "uv pip install maya1 deps" { uv pip install --python venvs\maya1\Scripts\python.exe "transformers>=4.50" snac soundfile numpy accelerate }
    Invoke-Checked "torch cu128 for maya1 (LAST)" { uv pip install --python venvs\maya1\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "maya1: ok" -ForegroundColor Green
} else { Write-Host "maya1: already installed" -ForegroundColor Gray }

Step "StyleTTS 2 (sidharthrajaram wrapper, MIT, LibriTTS, zero-shot cloning, 24kHz)"
if (-not (Test-Path "venvs\styletts2\Scripts\python.exe")) {
    # The `styletts2` PyPI wrapper (sidharthrajaram) uses gruut for phonemization
    # (no espeak-ng needed) and auto-downloads LibriTTS weights from HF on the
    # first StyleTTS2() call. Its dep `monotonic_align` is a Cython package that
    # compiles via setuptools' automatic MSVC discovery — VS Build Tools' VC++
    # compiler must be installed (confirmed on this box).
    Invoke-Checked "uv venv styletts2" { uv venv venvs\styletts2 --python 3.11 }
    Invoke-Checked "uv pip install styletts2" { uv pip install --python venvs\styletts2\Scripts\python.exe styletts2 soundfile numpy }
    # torch cu128 LAST (Blackwell sm_120); styletts2 pulls a non-cu128 torch otherwise.
    Invoke-Checked "torch cu128 for styletts2 (LAST)" { uv pip install --python venvs\styletts2\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "styletts2: ok" -ForegroundColor Green
} else { Write-Host "styletts2: already installed" -ForegroundColor Gray }

Step "Zonos-v0.1 transformer (Zyphra, zero-shot cloning, 44.1kHz, espeakng-loader)"
if (-not (Test-Path "venvs\zonos\Scripts\python.exe")) {
    # Transformer backbone only — do NOT install the `.[compile]` extras
    # (mamba-ssm/flash-attn are CUDA+Linux-only and unneeded for the transformer
    # variant). Zonos uses phonemizer -> espeak-ng; espeakng-loader bundles the
    # espeak-ng.dll + data so no system .msi install is required (the runner sets
    # PHONEMIZER_ESPEAK_LIBRARY/DATA from espeakng_loader before importing zonos).
    Invoke-Checked "uv venv zonos" { uv venv venvs\zonos --python 3.11 }
    Invoke-Checked "clone Zonos" { git clone https://github.com/Zyphra/Zonos venvs\zonos\src }
    Invoke-Checked "uv pip install zonos (transformer, no [compile])" { uv pip install --python venvs\zonos\Scripts\python.exe -e venvs\zonos\src soundfile numpy espeakng-loader }
    # cu128 wheels for Blackwell (RTX 5090, sm_120); zonos pulls a non-cu128 torch otherwise. LAST.
    Invoke-Checked "torch cu128 for zonos (LAST)" { uv pip install --python venvs\zonos\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "zonos: ok (espeak-ng bundled via espeakng-loader, no system install)" -ForegroundColor Green
} else { Write-Host "zonos: already installed" -ForegroundColor Gray }

Step "OpenVoice v2 (myshell-ai, MeloTTS base + tone-color converter, zero-shot cloning, 22.05kHz)"
if (-not (Test-Path "venvs\openvoice\Scripts\python.exe")) {
    # OpenVoice v2 = MeloTTS (base TTS) + a ToneColorConverter (cloning). 3 in-process
    # steps wrapped as one runner call. Python 3.11 (NOT 3.12 - fugashi, pulled by
    # MeloTTS's Japanese path, has no prebuilt wheel for 3.12 and the source build needs
    # MeCab; on 3.11 fugashi+mecab-python3 install cleanly from wheels).
    #
    # We deliberately do NOT `pip install -e` OpenVoice's setup.py: it pins
    # faster-whisper==0.9.0 -> av==10.0.0 (fails to Cython-build on Windows) plus
    # numpy==1.22 / librosa==0.9.1 / gradio that fight the MeloTTS+torch stack. The
    # runner only needs ToneColorConverter (api.py), whose runtime deps come from
    # the MeloTTS install; it adds venvs\openvoice\src to sys.path and instantiates
    # the converter with the default watermark ON (wavmark IS required — the
    # enable_watermark=False path is broken upstream, see the wavmark note below),
    # and calls extract_se([ref]) directly (no faster-whisper VAD). So: clone the
    # repo for its source + checkpoints, but skip the editable install.
    #
    # MeloTTS English uses g2p_en (CMUdict + NLTK), NOT espeak — no espeakng-loader
    # needed. The runner pre-downloads the NLTK tagger/tokenizer tables g2p_en
    # fetches at runtime.
    Invoke-Checked "uv venv openvoice" { uv venv venvs\openvoice --python 3.11 }
    Invoke-Checked "clone OpenVoice (source + checkpoints only; not pip-installed)" { git clone --depth 1 https://github.com/myshell-ai/OpenVoice venvs\openvoice\src }
    Invoke-Checked "uv pip install MeloTTS (provides the runner's torch/numpy/soundfile/librosa)" { uv pip install --python venvs\openvoice\Scripts\python.exe "git+https://github.com/myshell-ai/MeloTTS.git" }
    # MeloTTS may pull both mecab-python3 and python-mecab-ko; they conflict at import.
    # Uninstall python-mecab-ko if it landed (harmless if absent).
    & uv pip uninstall --python venvs\openvoice\Scripts\python.exe python-mecab-ko 2>&1 | Out-Null
    # wavmark: the ToneColorConverter's audio watermarker (loaded in its __init__;
    # enable_watermark defaults True and the False path is broken upstream — it
    # forwards the kwarg to a base __init__ that rejects it). Watermark is inaudible.
    Invoke-Checked "openvoice deps" { uv pip install --python venvs\openvoice\Scripts\python.exe soundfile numpy wavmark }
    # unidic uses its OWN downloader (not pip) — fine inside a uv venv. ~1GB.
    Invoke-Checked "unidic dict (~1GB)" { uv run --python venvs\openvoice\Scripts\python.exe -- python -m unidic download }
    # torch cu128 LAST (Blackwell sm_120); MeloTTS pulls a non-cu128 torch otherwise.
    Invoke-Checked "torch cu128 for openvoice (LAST)" { uv pip install --python venvs\openvoice\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    # OpenVoiceV2 checkpoints (converter + base-speaker SEs) into src\checkpoints_v2.
    Invoke-Checked "download OpenVoiceV2 ckpts" { uv run --python venvs\openvoice\Scripts\python.exe -- hf download myshell-ai/OpenVoiceV2 --local-dir venvs\openvoice\src\checkpoints_v2 }
    Write-Host "openvoice: ok (zero-shot tone-color cloning, 22.05kHz; MeloTTS base + ToneColorConverter)" -ForegroundColor Green
} else { Write-Host "openvoice: already installed" -ForegroundColor Gray }

Step "Echo-TTS (Jordan Darefsky, DiT + Fish S1-DAC, 44.1k, cloning, CUDA-only)"
if (-not (Test-Path "venvs\echo\Scripts\python.exe")) {
    # Source-clone install: inference.py/model.py/autoencoder.py are imported from
    # the cloned tree (no pip package). We DROP upstream's torchaudio + torchcodec —
    # the runner stubs both in sys.modules and reimplements load_audio via
    # soundfile + librosa; gradio (UI-only) is skipped too. librosa/scipy land via
    # the shared NAQ-deps step below.
    Invoke-Checked "uv venv echo" { uv venv venvs\echo --python 3.12 }
    if (-not (Test-Path "venvs\echo\src")) {
        Invoke-Checked "git clone echo-tts" { git clone --depth 1 https://github.com/jordandare/echo-tts venvs\echo\src }
    }
    Invoke-Checked "uv pip install echo-tts deps" { uv pip install --python venvs\echo\Scripts\python.exe huggingface-hub numpy safetensors einops soundfile }
    # torch cu128 LAST (Blackwell sm_120). Unlike Linux, PyPI's Windows torch is
    # CPU-only, so a normal resolution lands torch+cpu and the harness sees no CUDA.
    # Force the cu128 index as the sole source for torch.
    Invoke-Checked "torch cu128 for echo (LAST)" { uv pip install --python venvs\echo\Scripts\python.exe --reinstall-package torch "torch>=2.9.1" --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "echo: ok (echo-tts-base + fish-s1-dac-min weights auto-download from HF on first run; CUDA-only, bf16 DiT ~12GB)" -ForegroundColor Green
} else { Write-Host "echo: already installed" -ForegroundColor Gray }

Step "MiraTTS (Yatharth Sharma, MIT, 0.5B LLM-TTS + FastBiCodec, 48k, cloning, CUDA-only)"
if (-not (Test-Path "venvs\miratts\Scripts\python.exe")) {
    # pip package (project name FastNeuTTS). Pulls lmdeploy (TurboMind engine) + the
    # author's git deps ncodec (FastBiCodec) + fastaudiosr (FlashSR 48k upsampler) +
    # onnxruntime-gpu. soundfile for wav write. omegaconf is needed by the codec but
    # under-declared upstream, so add it explicitly.
    Invoke-Checked "uv venv miratts" { uv venv venvs\miratts --python 3.12 }
    Invoke-Checked "uv pip install MiraTTS" { uv pip install --python venvs\miratts\Scripts\python.exe "git+https://github.com/ysharma3501/MiraTTS.git" soundfile omegaconf }
    # torch cu128 LAST (Blackwell sm_120). lmdeploy pins torch 2.10.0, and PyPI's Windows
    # torch is CPU-only — so pin the SAME 2.10.0 from the cu128 index (keeps the TurboMind
    # ABI match) for torch + torchvision + torchaudio. Verified: lmdeploy 0.13 TurboMind
    # runs sm120 kernels on the RTX 5090, so the 40-series GPU-support ceiling in lmdeploy's
    # docs is stale. The Linux-3090 (Ampere) needs no cu128 swap (PyPI Linux torch is CUDA).
    Invoke-Checked "torch cu128 for miratts (LAST)" { uv pip install --python venvs\miratts\Scripts\python.exe --reinstall-package torch --reinstall-package torchvision torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "miratts: ok (YatharthS/MiraTTS + FastBiCodec + FlashSR weights auto-download from HF on first run; CUDA-only, lmdeploy TurboMind, ~1GB torch-allocator VRAM but TurboMind allocates more outside it)" -ForegroundColor Green
} else { Write-Host "miratts: already installed" -ForegroundColor Gray }

Step "OuteTTS 1.0 1B (edwko/OuteAI, CC-BY-NC-SA-4.0 + Llama-3.2, DAC, cloning + presets)"
if (-not (Test-Path "venvs\outetts\Scripts\python.exe")) {
    # pip package. We drive the HF/transformers backend (no llama.cpp compile), so we
    # also need accelerate. The runner monkeypatches torchaudio.load/save -> soundfile to
    # dodge torchcodec (FFmpeg-8 box; same reason echo avoids it), so torchaudio just
    # needs to import. torchvision MUST match torch or transformers' lazy Llama import
    # dies on torchvision::nms — so reinstall the matched cu128 trio together.
    Invoke-Checked "uv venv outetts" { uv venv venvs\outetts --python 3.12 }
    Invoke-Checked "uv pip install outetts" { uv pip install --python venvs\outetts\Scripts\python.exe outetts soundfile accelerate }
    # torch+torchvision+torchaudio cu128 LAST (Blackwell sm_120; PyPI Windows torch is CPU-only).
    Invoke-Checked "torch trio cu128 for outetts (LAST)" { uv pip install --python venvs\outetts\Scripts\python.exe --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "outetts: ok (Llama-OuteTTS-1.0-1B + DAC weights auto-download from HF on first run; HF backend, 44.1k, both preset voices and wav cloning)" -ForegroundColor Green
} else { Write-Host "outetts: already installed" -ForegroundColor Gray }

Step "Parler-TTS Mini v1 (parler-tts, Apache-2.0, description-controlled, DAC 44.1k)"
if (-not (Test-Path "venvs\parler\Scripts\python.exe")) {
    # git package (no PyPI release tracks the current model code). Pulls transformers +
    # descript-audio-codec + sentencepiece. accelerate for device_map. Then the cu128
    # torch trio LAST (Blackwell sm_120; PyPI Windows torch is CPU-only) — include
    # torchvision to keep transformers' lazy imports from tripping on a version skew.
    Invoke-Checked "uv venv parler" { uv venv venvs\parler --python 3.11 }
    Invoke-Checked "uv pip install parler-tts" { uv pip install --python venvs\parler\Scripts\python.exe "git+https://github.com/huggingface/parler-tts.git" soundfile accelerate }
    Invoke-Checked "torch trio cu128 for parler (LAST)" { uv pip install --python venvs\parler\Scripts\python.exe --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "parler: ok (parler-tts-mini-v1 weights auto-download from HF on first run; English, 44.1k, default-voice via text description)" -ForegroundColor Green
} else { Write-Host "parler: already installed" -ForegroundColor Gray }

Step "MeloTTS-English (myshell-ai, MIT, VITS predefined voice, 44.1k)"
if (-not (Test-Path "venvs\melotts\Scripts\python.exe")) {
    # git package + `unidic download` (dictionary the tokenizer loads at import). melo
    # imports ALL language modules at `from melo.api import TTS` (incl. pyopenjtalk/mecab),
    # even for English-only use. Upstream punts Windows to Docker, but the native build
    # DID succeed on Win-5090 / py3.11 (prebuilt wheels) — if it ever walls on another box,
    # MeloTTS still has the Linux/Mac rigs and the Windows cell just shows a fail row.
    Invoke-Checked "uv venv melotts" { uv venv venvs\melotts --python 3.11 }
    Invoke-Checked "uv pip install melotts" { uv pip install --python venvs\melotts\Scripts\python.exe "git+https://github.com/myshell-ai/MeloTTS.git" soundfile }
    Invoke-Checked "unidic download for melotts" { & venvs\melotts\Scripts\python.exe -m unidic download }
    Invoke-Checked "torch trio cu128 for melotts (LAST)" { uv pip install --python venvs\melotts\Scripts\python.exe --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 }
    Write-Host "melotts: ok (MeloTTS-English weights auto-download from HF on first run; EN-US speaker, 44.1k, predefined voice)" -ForegroundColor Green
} else { Write-Host "melotts: already installed" -ForegroundColor Gray }

# --- Higgs Audio v3 TTS — LINUX-ONLY, intentionally no Windows stanza ---
# Higgs v3 is server-backed: its only inference path is a Docker container running
# `sgl-omni serve` (an HTTP server), driven by a thin client venv. That path is set up on
# the Linux rig only (see install.sh's higgs_v3 stanza + Docker prerequisite). There is no
# Windows install here by design — leave higgs_v3 out of this script.

Step "DramaBox (Resemble AI, LTX-2 Community/NC, LTX-2.3 3.3B audio DiT, 48k, dialogue + cloning, CUDA-only)"
if (-not (Test-Path "venvs\dramabox\Scripts\python.exe")) {
    # Source repo (no PyPI package): the runner imports `src.inference_server.TTSServer`
    # from the cloned tree (venvs\dramabox\src). requirements pins torch==2.8.0; on Windows
    # we satisfy that from the cu128 index FIRST (Blackwell sm_120; PyPI Windows torch is
    # CPU-only), then install the rest with the torch lines stripped so it isn't pulled
    # back to a CPU wheel. bitsandbytes (>=0.45) powers the 4-bit Gemma-3-12B text encoder
    # and DOES ship a working Windows wheel (0.49.2 verified on the 5090). The optional
    # RE-USE ref-denoise deps (mamba-ssm/causal-conv1d) are intentionally skipped — no
    # Windows wheels; denoise_ref defaults off. Weights (~16 GB: DiT + audio-components +
    # Gemma) auto-download from HF on first run.
    Invoke-Checked "uv venv dramabox" { uv venv venvs\dramabox --python 3.11 }
    if (-not (Test-Path "venvs\dramabox\src")) {
        Invoke-Checked "git clone DramaBox" { git clone --depth 1 https://github.com/resemble-ai/DramaBox venvs\dramabox\src }
    }
    Invoke-Checked "torch 2.8.0 cu128 for dramabox (FIRST)" { uv pip install --python venvs\dramabox\Scripts\python.exe torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 }
    Get-Content venvs\dramabox\src\requirements.txt | Where-Object { $_ -notmatch '^\s*torch==' -and $_ -notmatch '^\s*torchaudio==' } | Set-Content venvs\dramabox\src\requirements.notorch.txt -Encoding utf8
    Invoke-Checked "uv pip install dramabox deps" { uv pip install --python venvs\dramabox\Scripts\python.exe -r venvs\dramabox\src\requirements.notorch.txt }
    Write-Host "dramabox: ok (LTX-2.3 audio DiT + audio-components + Gemma-3-12B-4bit auto-download from HF on first run; CUDA-only, 48k, ~18GB VRAM)" -ForegroundColor Green
} else { Write-Host "dramabox: already installed" -ForegroundColor Gray }

Step "dots.tts (rednote-hilab, Apache-2.0, 2B continuous AR TTS, 48k, cloning + default voice, CUDA-only)"
if (-not (Test-Path "venvs\dots_tts\Scripts\python.exe")) {
    # pip package `dots_tts`, installed editable from the cloned tree (venvs\dots_tts\src);
    # the runner does `from dots_tts.runtime import DotsTtsRuntime`. constraints/recommended.txt
    # pins torch==2.8.0 — on Windows we satisfy that from the cu128 index FIRST (Blackwell
    # sm_120; PyPI Windows torch is CPU-only), then apply the constraints with the torch lines
    # stripped so the editable install isn't dragged back to a CPU wheel. soundfile/numpy come
    # in via the constraints. Weights (rednote-hilab/dots.tts-soar, ~2B bf16) auto-download
    # from HF on first run. CUDA-only, 48 kHz.
    Invoke-Checked "uv venv dots_tts" { uv venv venvs\dots_tts --python 3.11 }
    if (-not (Test-Path "venvs\dots_tts\src")) {
        Invoke-Checked "git clone dots.tts" { git clone --depth 1 https://github.com/rednote-hilab/dots.tts venvs\dots_tts\src }
    }
    Invoke-Checked "torch 2.8.0 cu128 for dots_tts (FIRST)" { uv pip install --python venvs\dots_tts\Scripts\python.exe torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 }
    Get-Content venvs\dots_tts\src\constraints\recommended.txt | Where-Object { $_ -notmatch '^\s*torch==' -and $_ -notmatch '^\s*torchaudio==' } | Set-Content venvs\dots_tts\src\constraints\recommended.notorch.txt -Encoding utf8
    Invoke-Checked "uv pip install dots_tts" { uv pip install --python venvs\dots_tts\Scripts\python.exe -e venvs\dots_tts\src -c venvs\dots_tts\src\constraints\recommended.notorch.txt }
    Write-Host "dots_tts: ok (2B continuous AR TTS; dots.tts-soar weights auto-download from HF on first run; CUDA-only, 48k)" -ForegroundColor Green
} else { Write-Host "dots_tts: already installed" -ForegroundColor Gray }

Step "psutil in every venv (for bench memory tracking)"
# Bench reports include peak CPU RSS via psutil. The runner falls back to
# `None` if psutil is missing, so this is best-effort — but cheap to install.
Get-ChildItem venvs -Directory | ForEach-Object {
    $py = Join-Path $_.FullName "Scripts\python.exe"
    if (Test-Path $py) {
        & uv pip install --python $py psutil --quiet 2>&1 | Out-Null
    }
}
Write-Host "psutil: ensured in all venvs" -ForegroundColor Green

Step "NAQ deps in every venv (librosa + scipy for naq scoring)"
# NAQ scoring runs after every wav is written. Pure-acoustic;
# librosa + scipy are all that's needed. A learned-MOS predictor was considered
# and dropped — install portability across heterogeneous venvs wasn't workable.
Get-ChildItem venvs -Directory | ForEach-Object {
    $py = Join-Path $_.FullName "Scripts\python.exe"
    if (Test-Path $py) {
        & uv pip install --python $py librosa scipy --quiet 2>&1 | Out-Null
    }
}
Write-Host "naq deps: ensured in all venvs" -ForegroundColor Green

Write-Host "`nDone. Run: python bench.py" -ForegroundColor Green
