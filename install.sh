#!/usr/bin/env bash
# Install model venvs on macOS / Linux. Idempotent — re-runs skip already-installed venvs.

cd "$(dirname "${BASH_SOURCE[0]}")"

cyan()   { printf '\033[36m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
die()    { red "FAILED: $*"; exit 1; }

# --- Pocket-TTS ---
echo; cyan "=== Pocket-TTS ==="
if [ ! -x venvs/pocket/bin/python ]; then
    uv venv venvs/pocket --python 3.11 || die "uv venv pocket"
    if [ ! -d venvs/pocket/src ]; then
        git clone https://github.com/kyutai-labs/pocket-tts venvs/pocket/src \
            || die "git clone pocket-tts"
    fi
    uv pip install --python venvs/pocket/bin/python -e venvs/pocket/src \
        || die "uv pip install pocket-tts"
    uv pip install --python venvs/pocket/bin/python soundfile numpy \
        || die "uv pip install pocket deps"
    green "pocket: ok"
else
    echo "pocket: already installed"
fi

# --- NeuTTS (Air + Nano share this venv) ---
echo; cyan "=== NeuTTS (Air + Nano share this venv) ==="
if [ ! -x venvs/neutts/bin/python ]; then
    uv venv venvs/neutts --python 3.11 || die "uv venv neutts"
    if ! uv pip install --python venvs/neutts/bin/python neutts; then
        yellow "PyPI install failed, cloning from GitHub..."
        if [ ! -d venvs/neutts/src ]; then
            git clone https://github.com/neuphonic/neutts venvs/neutts/src \
                || die "git clone neutts"
        fi
        uv pip install --python venvs/neutts/bin/python -e venvs/neutts/src \
            || die "uv pip install neutts source"
    fi
    uv pip install --python venvs/neutts/bin/python torch soundfile numpy llama-cpp-python \
        || die "uv pip install neutts deps"
    green "neutts: ok"
else
    echo "neutts: already installed"
fi

# --- LuxTTS (should install cleanly on macOS — piper-phonemize has macOS wheels) ---
echo; cyan "=== LuxTTS ==="
if [ ! -x venvs/luxtts/bin/python ]; then
    uv venv venvs/luxtts --python 3.11 || die "uv venv luxtts"
    if ! uv pip install --python venvs/luxtts/bin/python luxtts; then
        yellow "PyPI install failed, cloning from GitHub..."
        if [ ! -d venvs/luxtts/src ]; then
            git clone https://github.com/ysharma3501/LuxTTS venvs/luxtts/src \
                || die "git clone LuxTTS"
        fi
        uv pip install --python venvs/luxtts/bin/python -e venvs/luxtts/src \
            || die "uv pip install luxtts source"
    fi
    uv pip install --python venvs/luxtts/bin/python torch soundfile numpy \
        || die "uv pip install luxtts deps"
    green "luxtts: ok"
else
    echo "luxtts: already installed"
fi

# --- Kokoro-82M ---
echo; cyan "=== Kokoro-82M ==="
if [ ! -x venvs/kokoro/bin/python ]; then
    uv venv venvs/kokoro --python 3.11 || die "uv venv kokoro"
    uv pip install --python venvs/kokoro/bin/python kokoro soundfile numpy \
        || die "uv pip install kokoro"
    # misaki (Kokoro's tokenizer) auto-downloads spaCy en_core_web_sm via spacy.cli.download()
    # which shells out to pip. uv venvs have no pip, so pre-install the model wheel directly.
    uv pip install --python venvs/kokoro/bin/python \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
        || die "spacy en_core_web_sm install"
    green "kokoro: ok"
else
    echo "kokoro: already installed"
fi

# --- KittenTTS ---
echo; cyan "=== KittenTTS ==="
if [ ! -x venvs/kittentts/bin/python ]; then
    uv venv venvs/kittentts --python 3.11 || die "uv venv kittentts"
    uv pip install --python venvs/kittentts/bin/python kittentts espeakng-loader soundfile numpy \
        || die "uv pip install kittentts"
    green "kittentts: ok"
else
    echo "kittentts: already installed"
fi

# --- Piper ---
echo; cyan "=== Piper ==="
if [ ! -x venvs/piper/bin/python ]; then
    uv venv venvs/piper --python 3.11 || die "uv venv piper"
    uv pip install --python venvs/piper/bin/python piper-tts soundfile numpy \
        || die "uv pip install piper-tts"
    green "piper: ok (voices auto-download on first use to ~/.cache/piper-voices)"
else
    echo "piper: already installed"
fi

# --- ChatterBox-TTS ---
echo; cyan "=== ChatterBox-TTS ==="
if [ ! -x venvs/chatterbox/bin/python ]; then
    uv venv venvs/chatterbox --python 3.11 || die "uv venv chatterbox"
    uv pip install --python venvs/chatterbox/bin/python chatterbox-tts soundfile numpy \
        || die "uv pip install chatterbox-tts"
    # perth (ChatterBox watermarker) imports pkg_resources (removed in setuptools 80+)
    uv pip install --python venvs/chatterbox/bin/python "setuptools<80" \
        || die "uv pip install setuptools"
    green "chatterbox: ok (GPU-targeted — expect <0.2x RTF on CPU)"
else
    echo "chatterbox: already installed"
fi

# --- F5-TTS ---
echo; cyan "=== F5-TTS ==="
if [ ! -x venvs/f5tts/bin/python ]; then
    uv venv venvs/f5tts --python 3.11 || die "uv venv f5tts"
    uv pip install --python venvs/f5tts/bin/python f5-tts soundfile numpy \
        || die "uv pip install f5-tts"
    # Pin datasets<3.0 to avoid pulling torchcodec into the import chain.
    # Mac/Linux generally have FFmpeg DLLs available, but consistency helps.
    uv pip install --python venvs/f5tts/bin/python "datasets<3.0" \
        || die "uv pip install datasets"
    green "f5tts: ok (GPU-targeted — expect <0.1x RTF on CPU)"
else
    echo "f5tts: already installed"
fi

# --- Coqui XTTS-v2 (idiap fork) ---
echo; cyan "=== Coqui XTTS-v2 (idiap fork) ==="
if [ ! -x venvs/coqui/bin/python ]; then
    uv venv venvs/coqui --python 3.11 || die "uv venv coqui"
    # Original coqui-ai/TTS is archived; idiap/coqui-ai-TTS is the maintained
    # fork. PyPI package is `coqui-tts` (the old `TTS` name is squatted).
    uv pip install --python venvs/coqui/bin/python coqui-tts soundfile numpy \
        || die "uv pip install coqui-tts"
    green "coqui: ok (XTTS-v2 ~2GB downloads on first use; non-commercial CPML license)"
else
    echo "coqui: already installed"
fi

# --- VibeVoice-Realtime-0.5B (community fork) ---
echo; cyan "=== VibeVoice-Realtime-0.5B (community fork) ==="
if [ ! -x venvs/vibevoice/bin/python ]; then
    uv venv venvs/vibevoice --python 3.11 || die "uv venv vibevoice"
    # The official microsoft/VibeVoice repo was taken down then partially restored
    # WITHOUT code. The community fork keeps the original code and added a
    # working streaming variant in 2025-12-04. The pypi `vibevoice==0.0.1` ships
    # the base architecture only (no streaming class), so install from the fork.
    uv pip install --python venvs/vibevoice/bin/python \
        "git+https://github.com/vibevoice-community/VibeVoice" torch soundfile numpy \
        || die "uv pip install vibevoice (community fork)"
    green "vibevoice: ok (voice .pt presets auto-download on first use to ~/.cache/vibevoice-voices)"
else
    echo "vibevoice: already installed"
fi

echo
green "Done. Run: python bench.py"
