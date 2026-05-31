#!/usr/bin/env bash
# Install model venvs on macOS / Linux. Idempotent — re-runs skip already-installed venvs.

cd "$(dirname "${BASH_SOURCE[0]}")"

cyan()   { printf '\033[36m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
die()    { red "FAILED: $*"; exit 1; }

# --- Reference voices (default cloning voices; gitignored, fetched from upstream) ---
# Several cloning runners (neutts, f5tts, indextts, qwentts, mars5) fall back to a shared
# default reference voice in reference/ when --reference is omitted. These wavs are gitignored,
# so a fresh clone has none — fetch the canonical neutts-air samples here.
echo; cyan "=== Reference voices ==="
mkdir -p reference
NEUTTS_SAMPLES="https://raw.githubusercontent.com/neuphonic/neutts-air/main/samples"
fetch_ref() {  # fetch_ref <stem>: download reference/<stem>.{wav,txt} unless already present
    local stem="$1"
    if [ -f "reference/${stem}.wav" ] && [ -f "reference/${stem}.txt" ]; then
        echo "reference/${stem}: already present"; return 0
    fi
    if curl -fsSL -o "reference/${stem}.wav" "${NEUTTS_SAMPLES}/${stem}.wav" \
       && curl -fsSL -o "reference/${stem}.txt" "${NEUTTS_SAMPLES}/${stem}.txt"; then
        green "reference/${stem}: fetched"
    else
        rm -f "reference/${stem}.wav" "reference/${stem}.txt"
        yellow "reference/${stem}: fetch failed — cloning cells for this voice will be skipped"
    fi
}
fetch_ref jo        # English default voice (neutts, f5tts, indextts, qwentts, qwentts_fast)
fetch_ref juliette  # French default voice (neutts-fr, f5tts-fr, mars5)
# chris_hemsworth_15s.{wav,txt}: a ~15s English clip used as the default voice by vibevoice and
# zipvoice (and as the NAQ self-test's real-speech reference). Not redistributable here — drop a
# clean ~15s English wav + matching transcript at reference/chris_hemsworth_15s.{wav,txt} to bench
# those two models' default voice. Without it they are skipped, not failed.
if [ ! -f reference/chris_hemsworth_15s.wav ]; then
    yellow "reference/chris_hemsworth_15s.wav missing — vibevoice & zipvoice default-voice cells need it (+ .txt)"
fi

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

# --- ChatterBox-TTS (base 1.2B + Turbo ~744M share this venv) ---
echo; cyan "=== ChatterBox-TTS (base 1.2B + Turbo ~744M share this venv) ==="
if [ ! -x venvs/chatterbox/bin/python ]; then
    uv venv venvs/chatterbox --python 3.11 || die "uv venv chatterbox"
    uv pip install --python venvs/chatterbox/bin/python chatterbox-tts soundfile numpy \
        || die "uv pip install chatterbox-tts"
    # perth (ChatterBox watermarker) imports pkg_resources (removed in setuptools 80+)
    uv pip install --python venvs/chatterbox/bin/python "setuptools<80" \
        || die "uv pip install setuptools"
    # Chatterbox Turbo (~744M GPT2-based AR model, Dec 2025). Same venv, same runner
    # dispatched via --variant turbo. Weights auto-download from ResembleAI/chatterbox-turbo
    # on first use; the turbo checkpoint uses the base tokenizer from ResembleAI/chatterbox.
    # On CUDA Linux systems, swap in cu128 wheels for Blackwell; on Mac the default torch is fine.
    # uv pip install --python venvs/chatterbox/bin/python --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    green "chatterbox: ok (base 1.2B GPU-targeted + Turbo ~744M AR; both via --variant)"
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
    # f5-tts pins hydra-core 1.0.7 + omegaconf 2.0.6 which trip on Python 3.11's
    # stricter dataclass rules (mutable default ValueError). Bump to versions
    # that accept frozen field types.
    uv pip install --python venvs/f5tts/bin/python --upgrade "hydra-core>=1.3" "omegaconf>=2.3" \
        || die "uv pip install hydra-core/omegaconf upgrade"
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
    # bitsandbytes for the 7B Q8 weights (Linux Q8 path); harmless on Windows.
    uv pip install --python venvs/vibevoice/bin/python "bitsandbytes>=0.48.1" \
        || die "uv pip install bitsandbytes for vibevoice 7b Q8"
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

# --- ZipVoice (k2-fsa, 123M, flow matching, zero-shot cloning) ---
echo; cyan "=== ZipVoice (k2-fsa, 123M, flow matching, zero-shot cloning) ==="
if [ ! -x venvs/zipvoice/bin/python ]; then
    # No pip wheel — source clone + editable install. Same lab as OmniVoice.
    # pyproject.toml in the repo has no [build-system]; we patch it before installing.
    uv venv venvs/zipvoice --python 3.11 || die "uv venv zipvoice"
    if [ ! -d venvs/zipvoice/src ]; then
        git clone --depth 1 https://github.com/k2-fsa/ZipVoice venvs/zipvoice/src \
            || die "git clone zipvoice"
    fi
    # Patch pyproject.toml so editable install works (upstream has no [build-system]).
    if ! grep -q '\[build-system\]' venvs/zipvoice/src/pyproject.toml 2>/dev/null; then
        printf '[build-system]\nbuild-backend = "setuptools.build_meta"\nrequires = ["setuptools>=61"]\n\n[project]\nname = "zipvoice"\nversion = "0.1.0"\n\n[tool.setuptools.packages.find]\ninclude = ["zipvoice*"]\n\n' \
            | cat - venvs/zipvoice/src/pyproject.toml > /tmp/pj_tmp && mv /tmp/pj_tmp venvs/zipvoice/src/pyproject.toml
    fi
    uv pip install --python venvs/zipvoice/bin/python -e venvs/zipvoice/src \
        || die "uv pip install zipvoice editable"
    uv pip install --python venvs/zipvoice/bin/python -r venvs/zipvoice/src/requirements.txt \
        || die "uv pip install zipvoice requirements"
    # k2 is NOT required for inference — skip it.
    # requirements.txt pulls torch 2.11+, whose torchaudio routes audio loading through
    # torchcodec — install it so reference wavs load. torchcodec needs FFmpeg shared libs
    # (FFmpeg 4-8); zipvoice_runner.py adds the ffmpeg lib dir to LD_LIBRARY_PATH at runtime
    # so a non-standard FFmpeg (e.g. linuxbrew) is found. Windows needs the "full-shared"
    # FFmpeg build (ships DLLs). To skip torchcodec entirely, pin torch<2.9 instead, e.g.:
    #   uv pip install --python venvs/zipvoice/bin/python "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python venvs/zipvoice/bin/python torchcodec \
        || yellow "zipvoice: torchcodec install failed — reference loading will fail (pin torch<2.9 to avoid it)"
    green "zipvoice: ok (zero-shot cloning zh+en, 123M, weights auto-download from HF on first use)"
else
    echo "zipvoice: already installed"
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
    # Pin hydra-core>=1.3 + omegaconf>=2.3 — nemo_toolkit's default resolver picks
    # hydra-core 1.0.7 / omegaconf 2.0.6 which trip Python 3.11's stricter dataclass
    # rules ("mutable default ... Override"). Same fix as f5tts/indextts.
    uv pip install --python venvs/magpie/bin/python "hydra-core>=1.3" "omegaconf>=2.3" "lightning>2.2.1,<=2.4.0" "pytorch-lightning>2.2.1,<=2.4.0" fiddle cloudpickle wrapt "ruamel.yaml" wget braceexpand webdataset huggingface_hub editdistance "jiwer>=3.1.0,<4.0.0" "peft<=0.18.0" wandb sacremoses "sentencepiece<1.0.0" "datasets>=3.2.0" inflect \
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

# --- faster-qwen3-tts (CUDA-graph fast path for Qwen3-TTS-Base 1.7B) ---
echo; cyan "=== faster-qwen3-tts (CUDA-graph fast path for Qwen3-TTS-Base 1.7B) ==="
if [ ! -x venvs/qwentts_fast/bin/python ]; then
    uv venv venvs/qwentts_fast --python 3.11 || die "uv venv qwentts_fast"
    # CUDA-only on Linux/Windows; on Mac this installs CPU torch but the runner
    # will report CUDA unavailable and the bench cell will fail gracefully.
    uv pip install --python venvs/qwentts_fast/bin/python \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchaudio \
        || die "torch cu128 in qwentts_fast"
    uv pip install --python venvs/qwentts_fast/bin/python \
        faster-qwen3-tts soundfile numpy \
        || die "faster-qwen3-tts install"
    green "qwentts_fast: ok (CUDA-graph fast path, same Qwen3-TTS-Base 1.7B weights; CUDA-only)"
else
    echo "qwentts_fast: already installed"
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
    # IndexTTS-2 transitively pins omegaconf 2.0.6 (returns None for missing
    # keys instead of raising AttributeError, breaking RepCodec init) and a
    # protobuf >= 4 wheel that conflicts with its own legacy _pb2 stubs.
    uv pip install --python venvs/indextts/bin/python --upgrade "omegaconf>=2.3" "protobuf<3.21" \
        || die "uv pip install indextts omegaconf/protobuf pins"
    green "indextts: ok (zero-shot cloning, wav only, weights auto-download from HF on first use)"
else
    echo "indextts: already installed"
fi

# --- Dia 1.6B (Nari Labs, Apache 2.0, dialogue + cloning) ---
echo; cyan "=== Dia 1.6B (Nari Labs, Apache 2.0, dialogue + cloning) ==="
if [ ! -x venvs/dia/bin/python ]; then
    uv venv venvs/dia --python 3.11 || die "uv venv dia"
    uv pip install --python venvs/dia/bin/python "git+https://github.com/nari-labs/dia.git" soundfile numpy \
        || die "uv pip install dia from git"
    # Dia's pyproject pulls cu126 torch 2.6; on Blackwell (5090, sm_120) reinstall cu128.
    # On Mac/Linux non-CUDA, this is harmless (cu128 wheels are CPU+CUDA universal where applicable).
    uv pip install --python venvs/dia/bin/python --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 \
        || die "torch cu128 for dia"
    green "dia: ok (44.1kHz output, CUDA-only on Blackwell, default voice uses fixed seed=42)"
else
    echo "dia: already installed"
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

# --- Soprano 80M (ekwek1, Apache 2.0, predefined voice, 32kHz) ---
echo; cyan "=== Soprano 80M (ekwek1, Apache 2.0, predefined voice, 32kHz) ==="
if [ ! -x venvs/soprano/bin/python ]; then
    uv venv venvs/soprano --python 3.11 || die "uv venv soprano"
    if [ ! -d venvs/soprano/src ]; then
        git clone --depth 1 https://github.com/ekwek1/soprano venvs/soprano/src \
            || die "git clone soprano"
    fi
    uv pip install --python venvs/soprano/bin/python -e venvs/soprano/src \
        || die "uv pip install soprano source"
    uv pip install --python venvs/soprano/bin/python soundfile numpy \
        || die "uv pip install soprano deps"
    # On CUDA Linux/Windows, reinstall cu128 torch for Blackwell; on Mac, default torch is fine.
    # uv pip install --python venvs/soprano/bin/python --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    green "soprano: ok (~80M params, Apache 2.0, predefined voice, 32kHz, English only; weights auto-download from ekwek/Soprano-1.1-80M on first use)"
else
    echo "soprano: already installed"
fi

# --- MOSS-TTS-Nano 100M (OpenMOSS/MOSI.AI, Apache 2.0, zero-shot cloning, 48kHz) ---
echo; cyan "=== MOSS-TTS-Nano 100M (OpenMOSS/MOSI.AI, Apache 2.0, zero-shot cloning, 48kHz) ==="
if [ ! -x venvs/moss_tts_nano/bin/python ]; then
    # No PyPI wheel that ships the model code — install editable from source clone.
    # Upstream README recommends Python 3.12. The package's pyproject exposes
    # both the `moss_tts_nano` package + top-level py-modules (infer, app, ...),
    # so an editable install also makes `from moss_tts_nano.defaults import ...`
    # and `import infer` available at the repo root.
    uv venv venvs/moss_tts_nano --python 3.12 || die "uv venv moss_tts_nano"
    if [ ! -d venvs/moss_tts_nano/src ]; then
        git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano venvs/moss_tts_nano/src \
            || die "git clone MOSS-TTS-Nano"
    fi
    uv pip install --python venvs/moss_tts_nano/bin/python -e venvs/moss_tts_nano/src \
        || die "uv pip install moss-tts-nano editable"
    # The README notes WeTextProcessing + pynini are optional (the runner sets
    # enable_wetext_processing=0 to skip them). Pynini has wheels on Mac/Linux
    # but we deliberately skip the WeTextProcessing dep for cross-platform parity
    # with the Windows path — the upstream "normalize_tts_text" robust cleanup
    # is still applied at inference.
    uv pip install --python venvs/moss_tts_nano/bin/python soundfile numpy \
        || die "uv pip install moss_tts_nano deps"
    # On CUDA Linux, swap in cu128 wheels for Blackwell; on Mac/CPU-only the default torch is fine.
    # Pin torch == 2.8.0: torch 2.9+ routes torchaudio.load() through torchcodec, whose
    # libtorchcodec*.dll needs FFmpeg shared DLLs on PATH. Linux usually ships those
    # via apt but Windows doesn't; we pin globally so the venv layout matches across rigs.
    # uv pip install --python venvs/moss_tts_nano/bin/python --reinstall "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    green "moss_tts_nano: ok (~100M params, 48kHz, voice cloning; weights auto-download from OpenMOSS-Team/MOSS-TTS-Nano on first use)"
else
    echo "moss_tts_nano: already installed"
fi

# --- MOSS-TTS flagship (OpenMOSS, Apache 2.0, 8B Qwen3-backbone zero-shot cloning, 20 langs) ---
echo; cyan "=== MOSS-TTS flagship (OpenMOSS, Apache 2.0, 8B Qwen3-backbone zero-shot cloning, 20 langs) ==="
if [ ! -x venvs/moss_tts/bin/python ]; then
    # Source-clone install — upstream pins torch==2.9.1+cu128 + transformers==5.0.0
    # via the [torch-runtime] extra. Python 3.12 per upstream README.
    uv venv venvs/moss_tts --python 3.12 || die "uv venv moss_tts"
    if [ ! -d venvs/moss_tts/src ]; then
        git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS venvs/moss_tts/src \
            || die "git clone MOSS-TTS"
    fi
    # `.[torch-runtime]` pulls torch==2.9.1+cu128, torchaudio==2.9.1+cu128,
    # torchcodec, transformers==5.0.0 from the pytorch cu128 index.
    uv pip install --python venvs/moss_tts/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e "venvs/moss_tts/src[torch-runtime]" \
        || die "uv pip install moss-tts torch-runtime"
    uv pip install --python venvs/moss_tts/bin/python soundfile \
        || die "uv pip install moss_tts deps"
    # Pin torch == 2.8.0: 2.9+ routes torchaudio.load() through torchcodec, which
    # needs FFmpeg shared DLLs (no Windows wheels). 2.8.0 still uses soundfile.
    uv pip install --python venvs/moss_tts/bin/python --reinstall "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128 \
        || die "uv pip install moss_tts torch 2.8.0 pin"
    green "moss_tts: ok (8B Qwen3 backbone, 20 langs, 24kHz, voice cloning; ~16GB weights auto-download from OpenMOSS-Team/MOSS-TTS on first use)"
else
    echo "moss_tts: already installed"
fi

# --- Supertonic (Supertone Inc., ONNX, 99M, 31 langs, predefined voices) ---
echo; cyan "=== Supertonic (Supertone Inc., ONNX, 99M, 31 langs, predefined voices) ==="
if [ ! -x venvs/supertonic/bin/python ]; then
    # Pure-ONNX runtime; no torch dependency. ~25MB weights auto-downloaded
    # from HF (Supertone/supertonic) on first run.
    # Open-weight release is fixed-voice only; cloning lives in the hosted
    # Voice Builder / Supertone Play API.
    uv venv venvs/supertonic --python 3.11 || die "uv venv supertonic"
    uv pip install --python venvs/supertonic/bin/python supertonic soundfile psutil \
        || die "uv pip install supertonic"
    green "supertonic: ok (~99M ONNX, 31 langs, MIT code + OpenRAIL-M weights, CPU-only)"
else
    echo "supertonic: already installed"
fi

# --- Fish Speech 1.5 (fishaudio, zero-shot cloning, 44.1kHz) ---
echo; cyan "=== Fish Speech 1.5 (fishaudio, zero-shot cloning, 44.1kHz) ==="
if [ ! -x venvs/fish/bin/python ]; then
    # Source clone of the v1.5.0 tag. The [stable] extras are intentionally
    # SKIPPED — they pin torch<=2.4.1, incompatible with Blackwell (sm_120) which
    # needs cu128 / torch 2.7+. Install fish-speech with no extras, hard-cap
    # numpy<=1.26.4 (fish requirement), then reinstall torch cu128 LAST. On Mac/CPU
    # the cu128 reinstall is harmless. The runner instantiates TTSInferenceEngine
    # directly (NOT ModelManager — it has an unconditional funasr import to avoid).
    uv venv venvs/fish --python 3.10 || die "uv venv fish"
    if [ ! -d venvs/fish/src ]; then
        git clone --branch v1.5.0 https://github.com/fishaudio/fish-speech venvs/fish/src \
            || die "git clone fish-speech v1.5"
    fi
    uv pip install --python venvs/fish/bin/python -e venvs/fish/src \
        || die "uv pip install fish-speech (no extras)"
    uv pip install --python venvs/fish/bin/python "numpy<=1.26.4" \
        || die "uv pip install numpy<=1.26.4 (fish hard cap)"
    uv pip install --python venvs/fish/bin/python soundfile \
        || die "uv pip install fish deps"
    uv pip install --python venvs/fish/bin/python --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 \
        || die "torch cu128 for fish (LAST)"
    uv run --python venvs/fish/bin/python -- hf download fishaudio/fish-speech-1.5 --local-dir venvs/fish/src/checkpoints/fish-speech-1.5 \
        || die "download fish-speech-1.5 weights"
    green "fish: ok"
else
    echo "fish: already installed"
fi

# --- Maya1 (maya-research, Apache 2.0, voice-description default voice, 24kHz, SNAC codec) ---
echo; cyan "=== Maya1 (maya-research, Apache 2.0, voice-description default voice, 24kHz, SNAC codec) ==="
if [ ! -x venvs/maya1/bin/python ]; then
    # Default-voice model: no audio cloning. The voice is steered by a natural-
    # language description string; the runner uses a fixed DEFAULT_VOICE_DESC.
    # Llama-style causal LM emits flat SNAC codec tokens -> decoded by the
    # hubertsiuzdak/snac_24khz SNAC model (auto-downloads on first run alongside
    # the ~3B maya-research/maya1 weights).
    #
    # On Linux: transformers + snac, torch from the cu128 index (works on the
    # CUDA rigs; CPU fallback is fine for a non-CUDA host). On Apple Silicon the
    # transformers+SNAC stack is replaced by MLX — install `mlx-audio` instead of
    # the torch line below and the runner's device=="mps" branch loads
    # `mlx-community/maya1-4bit`. The Mac path is NOT tested here; uncomment the
    # mlx-audio install and skip the cu128 reinstall on Darwin.
    uv venv venvs/maya1 --python 3.11 || die "uv venv maya1"
    uv pip install --python venvs/maya1/bin/python "transformers>=4.50" snac soundfile numpy accelerate \
        || die "uv pip install maya1 deps"
    if [ "$(uname)" = "Darwin" ]; then
        # Apple-Silicon MLX path (untested): the runner loads mlx-community/maya1-4bit.
        uv pip install --python venvs/maya1/bin/python mlx-audio \
            || die "uv pip install mlx-audio for maya1 (Mac)"
    else
        # torch cu128 LAST (Blackwell sm_120). CPU-only hosts: harmless, falls back.
        uv pip install --python venvs/maya1/bin/python --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128 \
            || die "torch cu128 for maya1 (LAST)"
    fi
    green "maya1: ok"
else
    echo "maya1: already installed"
fi

# --- StyleTTS 2 (sidharthrajaram wrapper, MIT, LibriTTS, zero-shot cloning, 24kHz) ---
echo; cyan "=== StyleTTS 2 (sidharthrajaram wrapper, MIT, LibriTTS, zero-shot cloning, 24kHz) ==="
if [ ! -x venvs/styletts2/bin/python ]; then
    # The `styletts2` PyPI wrapper (sidharthrajaram) uses gruut for phonemization
    # (no espeak-ng needed) and auto-downloads LibriTTS weights from HF on the
    # first StyleTTS2() call. Its dep `monotonic_align` is a Cython package: Linux
    # builds with gcc (clean), Mac needs the CommandLineTools clang — run
    # `xcode-select --install` first. On Mac also run the runner with
    # PYTORCH_ENABLE_MPS_FALLBACK=1 (some ops lack MPS kernels).
    uv venv venvs/styletts2 --python 3.11 || die "uv venv styletts2"
    uv pip install --python venvs/styletts2/bin/python styletts2 soundfile numpy \
        || die "uv pip install styletts2"
    # torch LAST: on CUDA Linux reinstall cu128 wheels (Blackwell sm_120; harmless
    # CPU fallback on non-CUDA hosts); on Mac the default torch (MPS/CPU) is fine.
    if [ "$(uname)" = "Darwin" ]; then
        :  # Mac: keep the default torch wheel
    else
        uv pip install --python venvs/styletts2/bin/python --reinstall torch torchaudio \
            --index-url https://download.pytorch.org/whl/cu128 || die "torch cu128 for styletts2 (LAST)"
    fi
    green "styletts2: ok (LibriTTS weights auto-download from HF on first use)"
else
    echo "styletts2: already installed"
fi

# --- psutil in every venv (for bench memory tracking) ---
# Bench reports include peak CPU RSS via psutil. The runner falls back to
# `None` if psutil is missing, so this is best-effort — but cheap to install.
echo; cyan "=== psutil in every venv (for bench memory tracking) ==="
for v in venvs/*/bin/python; do
    if [ -x "$v" ]; then
        uv pip install --python "$v" psutil --quiet >/dev/null 2>&1 || true
    fi
done
green "psutil: ensured in all venvs"

# --- NAQ deps in every venv (librosa + scipy for naq scoring) ---
echo; cyan "=== NAQ deps in every venv (librosa + scipy for naq scoring) ==="
# Pure-acoustic NAQ; librosa + scipy are all that's needed. A learned
# MOS predictor was considered and dropped — install portability across
# heterogeneous venvs wasn't workable.
for v in venvs/*/bin/python; do
    if [ -x "$v" ]; then
        uv pip install --python "$v" librosa scipy --quiet >/dev/null 2>&1 || true
    fi
done
green "naq deps: ensured in all venvs"

echo
green "Done. Run: python bench.py"
