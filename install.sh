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
    # Pin transformers<5.0 because XTTS imports `isin_mps_friendly` from
    # transformers.pytorch_utils which was removed in transformers 5.x.
    uv pip install --python venvs/coqui/bin/python "transformers>=4.45,<5.0" \
        || die "uv pip install transformers"
    # Pin torch<2.9 because Coqui requires torchcodec for audio IO starting
    # with torch 2.9 / torchaudio 2.9. torch 2.8 still uses the soundfile
    # backend directly so this avoids the torchcodec FFmpeg-DLL pain on Win.
    # For CUDA later: uv pip install --python venvs/coqui/bin/python --reinstall "torch<2.9" "torchaudio<2.9" --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python venvs/coqui/bin/python "torch<2.9" "torchaudio<2.9" \
        || die "uv pip install torch"
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

# --- OmniVoice (k2-fsa, 600+ languages) ---
echo; cyan "=== OmniVoice (k2-fsa, 600+ languages) ==="
if [ ! -x venvs/omnivoice/bin/python ]; then
    uv venv venvs/omnivoice --python 3.11 || die "uv venv omnivoice"
    uv pip install --python venvs/omnivoice/bin/python omnivoice soundfile numpy \
        || die "uv pip install omnivoice"
    # CPU wheels by default on Mac/Linux. CUDA users can swap to cu128 manually:
    # uv pip install --python venvs/omnivoice/bin/python --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    green "omnivoice: ok (zero-shot cloning, 600+ langs, weights auto-download from HF on first use)"
else
    echo "omnivoice: already installed"
fi

# --- VoxCPM-0.5B (OpenBMB, multilingual cloning) ---
echo; cyan "=== VoxCPM-0.5B (OpenBMB, multilingual cloning) ==="
if [ ! -x venvs/voxcpm/bin/python ]; then
    # voxcpm requires Python >=3.10 <3.13 and torch >=2.5.
    uv venv venvs/voxcpm --python 3.11 || die "uv venv voxcpm"
    uv pip install --python venvs/voxcpm/bin/python voxcpm soundfile numpy \
        || die "uv pip install voxcpm"
    green "voxcpm: ok (zero-shot cloning, 0.5B variant; runner uses openbmb/VoxCPM-0.5B)"
else
    echo "voxcpm: already installed"
fi

# --- Magpie-TTS Multilingual 357M (NVIDIA NeMo, predefined voices, 9 langs) ---
echo; cyan "=== Magpie-TTS Multilingual 357M (NVIDIA NeMo, predefined voices, 9 langs) ==="
if [ ! -x venvs/magpie/bin/python ]; then
    uv venv venvs/magpie --python 3.11 || die "uv venv magpie"
    # nemo_toolkit[tts] pulls nemo_text_processing -> pynini. On Mac/Linux pynini
    # has wheels and the [tts] extra works fine. We mirror the Windows recipe
    # below (skip [tts], install deps individually) for cross-platform parity
    # and so the runner's do_tts(apply_TN=False) call path is consistent.
    uv pip install --python venvs/magpie/bin/python nemo_toolkit \
        || die "uv pip install nemo_toolkit"
    uv pip install --python venvs/magpie/bin/python hydra-core omegaconf "lightning>2.2.1,<=2.4.0" "pytorch-lightning>2.2.1,<=2.4.0" fiddle cloudpickle wrapt "ruamel.yaml" wget braceexpand webdataset huggingface_hub editdistance "jiwer>=3.1.0,<4.0.0" "peft<=0.18.0" wandb sacremoses "sentencepiece<1.0.0" "datasets>=3.2.0" inflect \
        || die "uv pip install nemo core deps"
    uv pip install --python venvs/magpie/bin/python attrdict "cdifflib==1.2.6" einops kornia librosa matplotlib nltk pandas seaborn \
        || die "uv pip install nemo TTS safe deps"
    uv pip install --python venvs/magpie/bin/python nv_one_logger_core nv_one_logger_training_telemetry nv_one_logger_pytorch_lightning_integration \
        || die "uv pip install nemo telemetry"
    uv pip install --python venvs/magpie/bin/python lhotse pyannote.core pyannote.metrics kaldi-python-io marshmallow optuna pydub pyloudnorm resampy sacrebleu whisper_normalizer ipython \
        || die "uv pip install nemo ASR-eager deps"
    # Per-language G2P tokenizers - Magpie's multilingual config instantiates ALL
    # of these at load time even if you only generate English.
    uv pip install --python venvs/magpie/bin/python jieba pypinyin pypinyin-dict janome pyopenjtalk \
        || die "uv pip install nemo per-lang G2P"
    uv pip install --python venvs/magpie/bin/python kaldialign soundfile numpy \
        || die "uv pip install magpie-specific deps"
    green "magpie: ok (predefined voices, gated on HF - run 'uvx hf auth login' if not done; apply_TN forced False)"
else
    echo "magpie: already installed"
fi

# --- Qwen3-TTS-Base 1.7B (Alibaba Qwen, zero-shot cloning, 10 langs) ---
echo; cyan "=== Qwen3-TTS-Base 1.7B (Alibaba Qwen, zero-shot cloning, 10 langs) ==="
if [ ! -x venvs/qwentts/bin/python ]; then
    # qwen-tts targets Python 3.12 upstream.
    uv venv venvs/qwentts --python 3.12 || die "uv venv qwentts"
    uv pip install --python venvs/qwentts/bin/python qwen-tts soundfile numpy \
        || die "uv pip install qwen-tts"
    # FlashAttention 2 is recommended for best perf on Linux; on Mac the runner
    # falls back to default SDPA. We skip the flash-attn install here so the
    # script works on both - users on Linux with CUDA can `pip install flash-attn`
    # after for the perf bump.
    green "qwentts: ok (zero-shot cloning, wav+txt, 10 langs)"
else
    echo "qwentts: already installed"
fi

# --- IndexTTS-2 (Bilibili Index, zero-shot cloning + emotion control) ---
echo; cyan "=== IndexTTS-2 (Bilibili Index, zero-shot cloning + emotion control) ==="
if [ ! -x venvs/indextts/bin/python ]; then
    # Source clone (no pip wheel). Model weights (IndexTeam/IndexTTS-2) are
    # downloaded by huggingface_hub on first runner call (~5GB), not here.
    uv venv venvs/indextts --python 3.11 || die "uv venv indextts"
    if [ ! -d venvs/indextts/src ]; then
        git clone --depth 1 https://github.com/index-tts/index-tts venvs/indextts/src \
            || die "git clone index-tts"
    fi
    uv pip install --python venvs/indextts/bin/python -e venvs/indextts/src soundfile numpy huggingface_hub \
        || die "uv pip install indextts source"
    green "indextts: ok (zero-shot cloning, wav only, weights auto-download from HF on first use)"
else
    echo "indextts: already installed"
fi

# --- Sesame CSM-1B (conversational speech model, in-context cloning) ---
echo; cyan "=== Sesame CSM-1B (conversational speech model, in-context cloning) ==="
if [ ! -x venvs/sesame/bin/python ]; then
    # Native transformers support since 4.52.1.
    # MANUAL APPROVAL gating on HF - visit https://huggingface.co/sesame/csm-1b
    # and click "Ask for access". After approval, `hf auth login` enables download.
    uv venv venvs/sesame --python 3.11 || die "uv venv sesame"
    uv pip install --python venvs/sesame/bin/python "transformers>=4.52.1" accelerate soundfile numpy librosa huggingface_hub \
        || die "uv pip install sesame deps"
    green "sesame: ok (CSM-1B; HF access REQUIRED - request at https://huggingface.co/sesame/csm-1b before running)"
else
    echo "sesame: already installed"
fi

# --- MARS5-TTS (CAMB.AI, English zero-shot cloning, AGPL-3.0) ---
echo; cyan "=== MARS5-TTS (CAMB.AI, English zero-shot cloning, AGPL-3.0) ==="
if [ ! -x venvs/mars5/bin/python ]; then
    # torch.hub loads the model - no source clone needed. AGPL-3.0 license.
    uv venv venvs/mars5 --python 3.10 || die "uv venv mars5"
    # numpy<2.0 pin: MARS5's torch.hub'd code uses np.array(obj, copy=False),
    # which was removed in NumPy 2.0 and raises ValueError at inference time.
    uv pip install --python venvs/mars5/bin/python torch torchaudio librosa vocos encodec safetensors regex soundfile "numpy<2.0" \
        || die "uv pip install mars5 deps"
    green "mars5: ok (English only, AGPL-3.0; ref audio must be 1-12s, ~1.2GB weights auto-download on first run)"
else
    echo "mars5: already installed"
fi

echo
green "Done. Run: python bench.py"
