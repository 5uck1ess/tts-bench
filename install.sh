#!/usr/bin/env bash
# Install 3 model venvs (pocket / neutts / luxtts) on macOS / Linux.
# Idempotent — re-runs skip already-installed venvs.

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

echo
green "Done. Run: python bench.py"
