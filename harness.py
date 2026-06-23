"""Shared model registry + subprocess plumbing for bench.py and compare.py.

A "cell" is one (model, device) pair that's actually runnable on this machine
(venv installed AND device supported by the runner AND torch reports the
device as available). Each cell call spawns the runner subprocess once,
reads JSON-line results from stdout, and returns one dict per run.
"""

import json
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent


# (name, venv_dir, runner_relpath, multilingual?, devices, variant, can_clone)
# can_clone: True  = accepts user-supplied reference wav at inference (zero-shot)
#            False = predefined voice list only (Kokoro, KittenTTS, Piper)
#            "gated" = cloning works but requires HF accept-terms login
MODELS = [
    # Zero-shot voice cloning candidates
    ("pocket",      "pocket",     "runners/pocket_runner.py",     True,  ["cpu"],                None,   "gated"),
    ("neutts_air",  "neutts",     "runners/neutts_runner.py",     False, ["cpu", "cuda", "mps"], "air",  True),
    ("neutts_nano", "neutts",     "runners/neutts_runner.py",     True,  ["cpu", "cuda", "mps"], "nano", True),
    ("luxtts",      "luxtts",     "runners/luxtts_runner.py",     False, ["cpu", "cuda", "mps"], None,   True),
    ("chatterbox",       "chatterbox", "runners/chatterbox_runner.py", False, ["cpu", "cuda", "mps"], None,    True),
    ("chatterbox_turbo", "chatterbox", "runners/chatterbox_runner.py", False, ["cpu", "cuda", "mps"], "turbo", True),
    ("f5tts",       "f5tts",      "runners/f5tts_runner.py",      False, ["cpu", "cuda", "mps"], None,   True),
    ("coqui",       "coqui",      "runners/coqui_runner.py",      True,  ["cpu", "cuda", "mps"], None,   True),
    ("omnivoice",   "omnivoice",  "runners/omnivoice_runner.py",  True,  ["cpu", "cuda", "mps"], None,   True),
    ("zipvoice",    "zipvoice",   "runners/zipvoice_runner.py",   True,  ["cpu", "cuda", "mps"], None,   True),
    ("voxcpm",      "voxcpm",     "runners/voxcpm_runner.py",     True,  ["cpu", "cuda"],        None,   True),
    # base qwentts: cloning disabled — long prompts blow the 600s cell timeout on
    # this autoregressive sampler. qwentts_fast (CUDA-graph) handles cloning instead.
    ("qwentts",      "qwentts",      "runners/qwentts_runner.py",      True,  ["cpu", "cuda"],  "base", False),
    ("qwentts_fast", "qwentts_fast", "runners/qwentts_fast_runner.py", True,  ["cuda"],         "base", True),
    ("indextts",    "indextts",   "runners/indextts_runner.py",   False, ["cpu", "cuda"],        None,   True),
    ("fish_s2",     "fish_s2",    "runners/fish_s2_runner.py",    False, ["cuda"],               None,   True),
    ("metavoice",   "metavoice",  "runners/metavoice_runner.py",  False, ["cuda"],               None,   True),
    ("step_editx",  "step_editx", "runners/step_editx_runner.py", False, ["cuda"],               None,   True),
    ("sesame",      "sesame",     "runners/sesame_runner.py",     False, ["cpu", "cuda"],        None,   "gated"),
    ("mars5",       "mars5",      "runners/mars5_runner.py",      False, ["cpu", "cuda"],        None,   True),
    ("dia",         "dia",        "runners/dia_runner.py",        False, ["cuda"],               None,   True),
    ("fish_15",     "fish",       "runners/fish_runner.py",       True,  ["cpu", "cuda", "mps"], None,   True),
    # Predefined-voice-only (no cloning)
    ("kokoro",      "kokoro",     "runners/kokoro_runner.py",     True,  ["cpu", "cuda", "mps"], None,   False),
    ("kittentts",   "kittentts",  "runners/kittentts_runner.py",  False, ["cpu"],                None,   False),
    ("piper",       "piper",      "runners/piper_runner.py",      True,  ["cpu", "cuda"],        None,   False),
    ("vibevoice",      "vibevoice",  "runners/vibevoice_runner.py",  False, ["cpu", "cuda", "mps"], None,    False),
    ("vibevoice_15b",  "vibevoice",  "runners/vibevoice_runner.py",  False, ["cpu", "cuda", "mps"], "1.5b", False),
    ("vibevoice_7b",   "vibevoice",  "runners/vibevoice_runner.py",  False, ["cuda"],               "7b",   True),
    ("magpie",      "magpie",     "runners/magpie_runner.py",     True,  ["cpu", "cuda"],        None,   False),
    ("soprano",     "soprano",    "runners/soprano_runner.py",    False, ["cpu", "cuda", "mps"], None,   False),
    ("moss_tts_nano", "moss_tts_nano", "runners/moss_tts_nano_runner.py", True,  ["cpu", "cuda", "mps"], None, True),
    # MOSS-TTS: both checkpoints kept (1.0 still wins on some material by ear).
    # Same venv + runner; variant picks the HF checkpoint. v1.5 also gets a
    # per-prompt language tag (see the runner). Both are pure-cloning (no preset
    # voice) → both must be in publish.py / vote.py NO_PRESET_VOICE.
    ("moss_tts",      "moss_tts",      "runners/moss_tts_runner.py",      True,  ["cuda"],               "v1.0", True),
    ("moss_tts_v15",  "moss_tts",      "runners/moss_tts_runner.py",      True,  ["cuda"],               "v1.5", True),
    ("supertonic",  "supertonic", "runners/supertonic_runner.py", True,  ["cpu"],                None,   False),
    ("maya1",       "maya1",      "runners/maya1_runner.py",      False, ["cpu", "cuda", "mps"], None,   False),
    ("styletts2",   "styletts2",  "runners/styletts2_runner.py",  False, ["cpu", "cuda", "mps"], None,   True),
    ("zonos",       "zonos",      "runners/zonos_runner.py",      True,  ["cpu", "cuda"],        None,   True),
    ("openvoice",   "openvoice",  "runners/openvoice_runner.py",  True,  ["cpu", "cuda", "mps"], None,   True),
    # Voxtral: mps/cpu -> MLX (preset-voice only on Apple Silicon); cuda -> vllm-omni.
    # can_clone=True is for the cross-rig cuda path; the MLX runner fails a
    # --reference cell cleanly rather than mislabeling default-voice audio.
    ("voxtral",     "voxtral",    "runners/voxtral_runner.py",    True,  ["cpu", "cuda", "mps"], None,   True),
    # Echo-TTS (Jordan Darefsky): DiT flow-matching + Fish S1-DAC autoencoder, 44.1kHz,
    # zero-shot cloning. CUDA-only (40-step diffusion over a 2B-class DiT; torchcodec
    # decode path). English. Weights+outputs CC-BY-NC-SA-4.0 (same NC class as fish_*).
    ("echo",        "echo",       "runners/echo_runner.py",       False, ["cuda"],               None,   True),
    # MiraTTS (Yatharth Sharma, MIT): 0.5B LLM-TTS + FastBiCodec, 48kHz (FlashSR upsample),
    # ~6GB VRAM, zero-shot cloning from a reference wav. CUDA-only: the model hard-imports
    # lmdeploy and builds a TurboMind engine (no transformers fallback). lmdeploy's listed
    # GPU support stops at Ada (sm89) — Linux-3090 (Ampere) is the clean rig; Win-5090
    # (Blackwell sm120) is a stretch (prebuilt TurboMind wheel may lack sm120 kernels).
    ("miratts",     "miratts",    "runners/miratts_runner.py",    False, ["cuda"],               None,   True),
    # OuteTTS 1.0 (edwko/OuteAI, CC-BY-NC-SA-4.0 + Llama-3.2): ~1B Llama-3.2-1B LLM-TTS,
    # DAC codec. Does BOTH preset voices (default lens) and wav cloning (cloning lens).
    # HF/transformers backend (no llama.cpp compile); also has CPU/Metal backends, so
    # cross-rig incl. Mac. Multilingual (12 high-data langs incl. English).
    ("outetts",     "outetts",    "runners/outetts_runner.py",    True,  ["cpu", "cuda", "mps"], None,   True),
    # Parler-TTS (parler-tts/*, Apache-2.0): description-controlled TTS — the voice is
    # set by a natural-language prompt (gender/pitch/pace/quality), NOT a wav, so it's a
    # DEFAULT-VOICE model (can_clone=False; default lens only). T5 text encoder + decoder
    # LM -> DAC 44.1 kHz. English mini-v1 (multilingual=False). cpu/cuda (mps via DAC
    # untested). variant "large" = parler-tts-large-v1 (2.33B) — one-line add once the
    # mini's speed/disk cost is known.
    ("parler",      "parler",     "runners/parler_runner.py",     False, ["cpu", "cuda"],        None,   False),
    # MeloTTS (myshell-ai/MeloTTS, MIT): VITS multi-speaker predefined-voice TTS, EN-US
    # speaker, 44.1 kHz. The base speaker engine OpenVoice v2 wraps — here benched
    # standalone as a fast CPU baseline. No cloning. Multilingual=True: MeloTTS ships a
    # per-language checkpoint, so the runner loads the FR model (-> runs the FR prompt);
    # EN/ES/FR/ZH only — JP/KR need extra g2p deps (mecab/unidic, g2pkk) not installed.
    ("melotts",     "melotts",    "runners/melotts_runner.py",    True,  ["cpu", "cuda", "mps"], None,   False),
    # Higgs Audio v3 TTS (Boson AI, Research/Non-Commercial, ~4B) — SERVER-BACKED, Linux-only.
    # First server-backed model in the bench: v3 ships no modeling_*.py / auto_map and the
    # higgs_multimodal_qwen3 class isn't in stock transformers, so there's no single-process
    # Python path — the only supported inference is a Docker container running `sgl-omni
    # serve` (OpenAI-style HTTP /v1/audio/speech). higgs_v3_runner.py is a thin HTTP client
    # (no torch/model in its venv); stand the server up manually first (see install.sh
    # header + the runner docstring). 24 kHz, 100 langs, zero-shot in-context cloning.
    ("higgs_v3",    "higgs_v3",   "runners/higgs_v3_runner.py",   True,  ["cuda"],  None,   True),
    # Higgs Audio v2 (Boson AI, Apache-2.0) — still NOT registered: the installable
    # boson_multimodal (latest main) ships only the v1 HiggsAudioModel architecture, but
    # the v2 checkpoint (bosonai/higgs-audio-v2-generation-3B-base) is a different, larger
    # arch (DualFFN). Loading v2 weights into v1 code fails at the embedding layer even
    # after patching the unregistered model_type. runners/higgs_runner.py is kept on disk
    # for when Boson ships the v2 model class to the package; v3 supersedes it via the
    # server path above. Re-add a v2 line here if/when that class lands.
    # DramaBox (Resemble AI, LTX-2 Community License/NC) — expressive dialogue TTS, an
    # IC-LoRA fine-tune of the LTX-2.3 3.3B audio-only DiT (flow-matching). Prompt-driven
    # voice + emotion/laughs/sighs (runner wraps plain prompts in a neutral speaker
    # description) AND optional 10s+ wav cloning -> can_clone=True, populates both lenses.
    # 48 kHz out. Source-clone install (venvs/dramabox/src) + a bnb-4bit Gemma-3-12B text
    # encoder; ~18 GB VRAM peak (audio-only mode frees LTX's video stack) -> fits 5090 +
    # 3090. English (multilingual=False -> FR prompt skipped). CUDA-only.
    ("dramabox",    "dramabox",   "runners/dramabox_runner.py",   False, ["cuda"],  None,   True),
    # dots.tts (rednote-hilab, Apache-2.0, 2B) — fully-continuous AR TTS: semantic encoder
    # + LLM + flow-matching acoustic head over a 48 kHz AudioVAE (no codec tokens). Zero-shot
    # cloning from a reference wav (+ sibling .txt transcript for continuation cloning); pure
    # cloning model, so a no-reference run clones the bundled chris clip (bench default-voice
    # convention) -> can_clone=True populates both lenses.
    # Multilingual (24 langs incl. en/fr -> runs the FR prompt). pip package `dots_tts`,
    # source-clone editable install (venvs/dots_tts/src); weights snapshot-download from HF
    # on first run. CUDA-only here (bf16 backbone, ~2B; Ampere/3090 ok). Best on Seed-TTS-Eval.
    ("dots_tts",    "dots_tts",   "runners/dots_tts_runner.py",   True,  ["cuda"],  None,   True),
    # Miso TTS 8B (Miso Labs, modified-MIT) — Sesame-CSM architecture scaled to 8B:
    # Llama-8B backbone + Llama-300M audio decoder over 32 Mimi codebooks, 24 kHz.
    # Same conversational cloning paradigm as sesame (reference text+audio as a
    # prior turn from speaker 0); default mode = speaker 0, no context. Source-clone
    # import (venvs/miso/src) — upstream pins torch==2.4.0 but runs on cu128 torch
    # 2.7.1 + torchtune 0.6.1; the runner stubs the silentcipher watermark and swaps
    # the gated llama tokenizer for the unsloth mirror (see runner docstring).
    # English. CUDA-only: 8B bf16 ~16 GB VRAM (fits 5090 + 3090).
    ("miso",        "miso",       "runners/miso_runner.py",       False, ["cuda"],  None,   True),
    # LongCat-AudioDiT (Meituan, MIT) — non-autoregressive diffusion TTS that generates
    # directly in a Wav-VAE waveform latent space (no mel, no codec tokens): Wav-VAE + DiT
    # backbone, ODE-sampled over NFE steps, adaptive projection guidance (APG). SOTA-ish
    # zero-shot cloning on Seed (3.5B: EN SIM 0.786 / ZH SIM 0.818). Does BOTH zero-shot
    # default voice (no prompt) and reference-wav cloning (reference text prepended, like
    # sesame/miso -> sibling .txt required) -> can_clone=True, both lenses. ZH + EN only
    # (multilingual=False -> FR prompt skipped). Source-clone import (venvs/longcat/src;
    # `import audiodit` auto-registers the model with transformers>=5.3). Two sizes share
    # the venv+runner; --variant picks the HF checkpoint. CUDA-only (DiT + fp16 VAE), 24 kHz.
    ("longcat_1b",   "longcat",   "runners/longcat_runner.py",    False, ["cuda"],  "1b",   True),
    ("longcat_3p5b", "longcat",   "runners/longcat_runner.py",    False, ["cuda"],  "3.5b", True),
    # Orpheus-TTS (Canopy Labs, Apache-2.0): 3B Llama speech-LM -> SNAC codec, 24 kHz,
    # streaming (~200 ms TTFA). Served via vLLM (orpheus-speech pkg, AsyncLLMEngine) ->
    # CUDA-only. PRESET-VOICE only (named voices, no wav cloning) -> can_clone=False,
    # default lens only. English -> multilingual=False, FR prompt skipped. Gated HF
    # repo (accept the license once). Re-queued from the Windows-blocked list: the only
    # blocker was vLLM's no-Blackwell-wheel wall, native on the Ampere 3090.
    ("orpheus",     "orpheus",    "runners/orpheus_runner.py",    False, ["cuda"],  None,   False),
    # CosyVoice 3 (FunAudioLLM, Apache-2.0): 0.5B Qwen LLM-TTS + flow-matching, 24 kHz,
    # zero-shot multilingual cloning. Source-clone import (venvs/cosyvoice/src + its
    # third_party/Matcha-TTS). PURE CLONING (no preset voice -> NO_PRESET_VOICE in
    # publish.py/vote.py): default lens uses the house ref (chris_hemsworth_15s), cloning
    # lens uses the supplied wav; both need the reference's literal transcript (sibling
    # .txt). Multilingual=True (ZH/EN/JA/KO/yue+) -> FR prompt runs. CUDA-only (fp16).
    ("cosyvoice",   "cosyvoice",  "runners/cosyvoice_runner.py",  True,  ["cuda"],  None,   True),
    # LFM2.5-Audio-1.5B (Liquid AI, LFM Open License v1.0) — end-to-end omni speech<->text
    # model; we bench its TTS mode (sequential generation) only. PREDEFINED-VOICE (4 voices
    # via system prompt, no wav cloning -> can_clone=False, default lens). Single-process
    # `pip install liquid-audio` (py>=3.12), in-process torch model, 24 kHz, English-only.
    # cpu+cuda (cross-rig). NOT in NO_PRESET_VOICE.
    ("lfm2_audio",  "lfm2_audio", "runners/lfm2_audio_runner.py", False, ["cpu", "cuda"], None,  False),
    # MioTTS (Aratako) — LLM-codec TTS, SERVER-BACKED, runs on Linux-3090 (reuses its
    # llama.cpp). A llama.cpp/Ollama OpenAI server (emits MioCodec tokens) + the
    # MioTTS-Inference run_server.py REST orchestrator (decodes via MioCodec torch -> 44.1 kHz
    # wav); miotts_runner.py is a thin HTTP client (no torch in its venv -> SERVER_BACKED).
    # Pure zero-shot cloner, base64 reference in-band (no bind-mount, no transcript) ->
    # NO_PRESET_VOICE (cloning board only, like cosyvoice). The two sizes share runner+venv+
    # servers; --variant is labeling only (the LLM server hosts one GGUF at a time, so each
    # size benches in its own server session). 0.6B = Apache-2.0, 0.1B = Falcon-LLM. EN/JA.
    ("miotts_01b",  "miotts",     "runners/miotts_runner.py",     False, ["cuda"],  "0.1b", True),
    ("miotts_06b",  "miotts",     "runners/miotts_runner.py",     False, ["cuda"],  "0.6b", True),
]


# GPU-class models: measured sub-realtime (<0.5x RTF) on the best available
# CPU/MPS device, or they OOM / time out there. They only make sense on a CUDA
# rig. On a machine where CUDA isn't available, build_cells skips them by
# default (pass include_gpu_class=True / `bench.py --all` to force them in) so a
# non-CUDA pass doesn't burn time on models that can't be deploy candidates
# there. Tagged from the May 2026 Apple-M4 pass; they still run on Windows/Linux
# CUDA rigs unchanged. See docs/known-issues.md.
GPU_CLASS = {
    "f5tts", "indextts", "voxcpm", "qwentts", "sesame", "mars5",
    "chatterbox", "magpie",
    "fish_s2", "metavoice", "step_editx",
    # higgs_v3: server-backed CUDA model (the sgl-omni container runs on the GPU); the
    # thin HTTP-client venv would report cuda unavailable, so tag it gpu-class so non-CUDA
    # rigs skip it by default rather than trying to reach a server that isn't there.
    "higgs_v3",
    # dramabox: ~18 GB VRAM, 3.3B DiT + 12B 4-bit text encoder — CUDA-only (no CPU/MPS path).
    "dramabox",
    # dots_tts: 2B bf16 AR + flow-matching head — CUDA-only here (no CPU/MPS path benched).
    "dots_tts",
    # miso: 8B bf16 CSM — CUDA-only (fp32 CPU would need ~33 GB RAM, far sub-realtime).
    "miso",
    # longcat: DiT + fp16 Wav-VAE diffusion — CUDA-only (no CPU/MPS path benched).
    "longcat_1b", "longcat_3p5b",
    # orpheus: 3B Llama + vLLM — CUDA-only (vLLM has no CPU path here).
    "orpheus",
    # cosyvoice: 0.5B LLM + fp16 flow-matching — CUDA-only (no CPU/MPS path benched).
    "cosyvoice",
    # miotts: server-backed (the llama.cpp LLM + MioCodec servers run on the GPU); the
    # thin HTTP-client venv has no torch, so tag gpu-class to skip non-CUDA rigs by default.
    "miotts_01b", "miotts_06b",
}


# Server-backed models: inference happens in an external HTTP server (Docker), not in the
# runner process. Their client venv has NO torch, so detect_cuda() (which imports torch in
# the venv) can't see the GPU — build_cells uses detect_cuda_smi() for these instead so the
# cuda cell still builds on a CUDA rig. See harness comment on the MODELS entry + install.sh.
SERVER_BACKED = {
    "higgs_v3",
    # vibevoice_15b: ~0.03-0.07x RTF on M4 CPU (long-form times out at 600s) and
    # OOMs at load on mps (10.94 GiB > 16 GB). Runs on CUDA rigs; skip on non-CUDA.
    "vibevoice_15b",
    # miotts: inference is in the llama.cpp LLM server + MioTTS REST orchestrator; the
    # client venv has no torch, so build_cells uses detect_cuda_smi() for these.
    "miotts_01b", "miotts_06b",
}


def venv_python(venv_dir: str) -> Path:
    """Resolve the python.exe / bin/python path for a venv on this OS."""
    root = REPO / "venvs" / venv_dir
    if sys.platform.startswith("win"):
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _probe(py: Path, code: str) -> bool:
    try:
        out = subprocess.run(
            [str(py), "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        return "True" in out.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def detect_cuda(py: Path) -> bool:
    return _probe(py, "import torch; print(torch.cuda.is_available())")


def detect_cuda_smi() -> bool:
    """Torch-free CUDA probe via `nvidia-smi -L`. Used for SERVER_BACKED models whose thin
    client venv has no torch — detect_cuda would always report False there, so the cuda
    cell would never build even on a CUDA rig. nvidia-smi listing a GPU is enough signal
    that the sgl-omni server can run on this machine."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and "GPU 0" in out.stdout
    except (subprocess.SubprocessError, OSError):
        return False


def detect_mps(py: Path) -> bool:
    # Apple-Silicon GPU availability. Most venvs report it via torch's MPS
    # backend; MLX-only venvs (e.g. voxtral, maya1's Mac path) have no torch, so
    # fall back to MLX's own Metal probe. Either signal means the "mps" cell is
    # runnable on this rig.
    return _probe(
        py,
        "ok = False\n"
        "try:\n"
        "    import torch; b = getattr(torch.backends, 'mps', None); ok = bool(b and b.is_available())\n"
        "except Exception: pass\n"
        "if not ok:\n"
        "    try:\n"
        "        import mlx.core as mx; ok = mx.metal.is_available()\n"
        "    except Exception: pass\n"
        "print(ok)",
    )


def build_cells(reference=None, requested_models=None, requested_devices=None,
                verbose=True, include_gpu_class=False, skipped_out=None):
    """Return the list of runnable (model, device) cells on this machine.

    `requested_models` / `requested_devices`: optional sets to filter. If
    `reference` is truthy, predefined-voice-only models (can_clone=False) are
    skipped, matching bench.py's behavior. Cells with missing venvs or
    unavailable devices are silently dropped (with a printed note if verbose).

    GPU-class models (see GPU_CLASS) are skipped on a rig where CUDA isn't
    available unless `include_gpu_class` is True. When `skipped_out` is a list,
    one dict per skipped (model, device) is appended so callers can mark them in
    their output instead of dropping them silently.
    """
    cells = []
    for (model_name, venv_dir, runner_rel, multilingual,
         model_devices, variant, can_clone) in MODELS:
        if requested_models and model_name not in requested_models:
            continue
        if reference and can_clone is False:
            continue
        py = venv_python(venv_dir)
        if not py.exists():
            if verbose:
                print(f"skip {model_name}: venv not installed ({py})")
            continue
        if model_name in SERVER_BACKED:
            # Torch-free venv: probe the GPU via nvidia-smi, not torch in the venv.
            cuda_ok = ("cuda" in model_devices) and detect_cuda_smi()
        else:
            cuda_ok = ("cuda" in model_devices) and detect_cuda(py)
        mps_ok = ("mps" in model_devices) and detect_mps(py)
        if model_name in GPU_CLASS and not cuda_ok and not include_gpu_class:
            if verbose:
                print(f"skip {model_name}: gpu-class — sub-realtime without CUDA "
                      f"on this rig (pass --all to include)")
            if skipped_out is not None:
                for device in model_devices:
                    if device == "cuda":
                        continue
                    if requested_devices and device not in requested_devices:
                        continue
                    skipped_out.append({
                        "model": model_name, "device": device,
                        "variant": variant, "can_clone": can_clone,
                        "multilingual": multilingual,
                        "reason": "gpu-class: sub-realtime without CUDA (--all to include)",
                    })
            continue
        for device in model_devices:
            if device == "cuda" and not cuda_ok:
                continue
            if device == "mps" and not mps_ok:
                continue
            if requested_devices and device not in requested_devices:
                continue
            cells.append({
                "model": model_name, "device": device, "variant": variant,
                "multilingual": multilingual, "can_clone": can_clone,
                "venv_python": py, "runner": REPO / runner_rel,
            })
    return cells


def run_cell(cell, text, out_wav, language="en", runs=1, reference=None,
             timeout=600) -> list[dict]:
    """Run one (model, device) cell with N back-to-back generations.

    Returns one dict per run from the runner's JSON-line stdout. On failure
    returns a single dict with ok=False. The first row gets a wall_s key
    measuring total subprocess wall time (load + all runs).
    """
    cmd = [
        str(cell["venv_python"]), str(cell["runner"]),
        "--text", text, "--out", str(out_wav),
        "--device", cell["device"], "--runs", str(runs),
        "--language", language,
    ]
    if cell.get("variant"):
        cmd += ["--variant", cell["variant"]]
    if reference:
        cmd += ["--reference", str(reference)]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [{"ok": False, "error": f"timeout {timeout}s", "run_index": 0,
                 "wall_s": time.perf_counter() - t0}]
    wall = time.perf_counter() - t0

    parsed = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as e:
            parsed.append({"ok": False, "error": f"json parse failed: {e}", "run_index": -1})

    if not parsed:
        tail = " ".join(proc.stderr.strip().splitlines()[-3:])[:300] if proc.stderr else ""
        return [{"ok": False, "error": f"no json in stdout. stderr: {tail}",
                 "run_index": 0, "wall_s": wall}]

    parsed[0]["wall_s"] = wall
    return parsed


def play_wav(wav_path) -> None:
    """Play a wav file synchronously through the OS default player. Best-effort."""
    wav_path = str(wav_path)
    if sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            return
        except Exception as e:
            print(f"  (playback failed: {e})")
            return
    if sys.platform == "darwin":
        subprocess.run(["afplay", wav_path], check=False)
        return
    for tool in ("aplay", "paplay"):
        try:
            subprocess.run([tool, wav_path], check=False)
            return
        except FileNotFoundError:
            continue
