"""Voxtral-4B-TTS runner (mistralai/Voxtral-4B-TTS-2603, CC-BY-NC-4.0, 24 kHz).

Mac is Voxtral's primary rig and the cleanest path is MLX, so the MLX path is
the main branch here; the Linux/CUDA vllm-omni path is folded in under a device
branch so this one file serves both rigs.

Device dispatch (the harness tuple is ["cpu", "cuda", "mps"]):
    mps / cpu  -> MLX path (mlx_audio, Apple Silicon). Loads the 4-bit community
                  port mlx-community/Voxtral-4B-TTS-2603-mlx-4bit (~2 GB, fits
                  16 GB). This is the path actually exercised on this Mac.
    cuda       -> vllm-omni path (in-proc Omni engine, NOT a server). Uses the
                  Linux-verified accessors (see below). Not hardware-tested from
                  this Mac — written per the 2026-05-31 Linux handoff.

Voice / cloning, an important verified detail:
    The mlx-audio Voxtral port is PRESET-VOICE-ONLY. `Model.generate(text, voice=...)`
    resolves `voice` to a precomputed voice EMBEDDING loaded from files shipped in
    the model repo (Model._voice_embedding_files, populated at load time). There is
    no method to derive an embedding from an arbitrary reference wav, so zero-shot
    wav cloning is NOT available on the MLX path. The model FAMILY supports cloning
    (hence can_clone=True in the registry for the cross-rig cuda path), but on Apple
    Silicon a `--reference` request fails cleanly rather than silently falling back
    to a preset (which would mislabel default-voice audio as a clone in the bench's
    cloning column). The bench records that per-cell fail as expected data.

MLX API (verified 2026-05-30 against the installed mlx_audio):
    from mlx_audio.tts.utils import load
    model = load("mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
    for r in model.generate(text=TEXT, voice="casual_male"):   # generator
        r.audio          # mx.array waveform (float32)
        r.sample_rate    # 24000
    # voice presets: model._voice_embedding_files.keys()

vllm-omni API (Linux-verified 2026-05-31):
    import importlib.resources as ir
    from vllm_omni import Omni
    yaml = str(ir.files("vllm_omni") / "deploy" / "voxtral_tts.yaml")  # REQUIRED
    omni = Omni(model="mistralai/Voxtral-4B-TTS-2603", stage_configs_path=yaml)
    outs = omni.generate(prompts, sampling_params_list)
    audio = outs[0].multimodal_output["audio"]   # .audio_array (np.float32), .sampling_rate (24000)
"""

import argparse
import json
import sys
import time

import _meminfo
import _naq


MLX_REPO = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
VLLM_REPO = "mistralai/Voxtral-4B-TTS-2603"
SAMPLE_RATE = 24000  # native Voxtral rate (config.sample_rate / audio tokenizer)

# Preferred preset voice per language. Validated against the model's actual
# preset list at load time; falls back to "casual_male", then to any available
# preset if neither is present.
DEFAULT_VOICE = "casual_male"
VOICE_BY_LANG = {
    "en": "casual_male",
    "fr": "fr_female",
    "es": "es_female",
    "it": "it_female",
    "pt": "pt_female",
    "de": "de_female",
}


def _pick_voice(available, language):
    """Choose a preset voice name that exists in the model's embedding set."""
    if not available:
        return DEFAULT_VOICE
    requested = VOICE_BY_LANG.get(language, DEFAULT_VOICE)
    if requested in available:
        return requested
    if DEFAULT_VOICE in available:
        return DEFAULT_VOICE
    return sorted(available)[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Zero-shot reference wav. Supported only on the cuda/vllm-omni "
                        "path; the MLX path is preset-voice-only and fails cleanly here.")
    p.add_argument("--variant", default=None)        # unused
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
    # mps/cpu -> MLX (Apple Silicon primary); cuda -> vllm-omni.
    use_mlx = device != "cuda"

    try:
        import numpy as np
        import soundfile as sf

        if use_mlx:
            # The MLX port cannot clone from a wav. Fail the cell cleanly instead
            # of silently producing default-voice audio in a cloning column.
            if args.reference:
                print(json.dumps({
                    "ok": False, "run_index": 0,
                    "error": "voxtral MLX path is preset-voice-only; zero-shot wav "
                             "cloning is unsupported on Apple Silicon (use a preset "
                             "voice, or the cuda/vllm-omni path for cloning)",
                }))
                return 1
            from mlx_audio.tts.utils import load as mlx_load
            model = mlx_load(MLX_REPO)
            available = set(getattr(model, "_voice_embedding_files", {}).keys())
            voice = _pick_voice(available, args.language)
            # Stderr only — stdout is reserved for JSON-line results.
            print(f"voxtral MLX presets={sorted(available)} chosen={voice!r}",
                  file=sys.stderr, flush=True)
        else:
            import importlib.resources as ir
            from vllm_omni import Omni
            yaml = str(ir.files("vllm_omni") / "deploy" / "voxtral_tts.yaml")
            omni = Omni(model=VLLM_REPO, stage_configs_path=yaml)
            voice = _pick_voice(set(VOICE_BY_LANG.values()), args.language)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _gen_mlx(text):
        """Run the MLX generator, returning (audio float32 mono, ttfa_seconds)."""
        first = None
        chunks = []
        for r in model.generate(text=text, voice=voice, stream=False):
            if r.audio is None:
                continue
            if first is None:
                first = time.perf_counter()
            arr = np.asarray(r.audio, dtype="float32").reshape(-1)
            chunks.append(arr)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
        return audio, first

    def _gen_vllm(text):
        """Run the vllm-omni engine, returning (audio float32 mono, ttfa_seconds)."""
        from vllm import SamplingParams

        first = None
        sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
        prompt = text if voice is None else f"[voice={voice}] {text}"
        outs = omni.generate([prompt], [sampling])
        first = time.perf_counter()
        audio_obj = outs[0].multimodal_output["audio"]
        audio = np.asarray(audio_obj.audio_array, dtype="float32").reshape(-1)
        return audio, first

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio, first = _gen_mlx(text) if use_mlx else _gen_vllm(text)
            t_end = time.perf_counter()

            audio = np.asarray(audio, dtype="float32").reshape(-1)
            audio_s = float(len(audio) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, audio, SAMPLE_RATE)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000 if first else (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else {"naq": None, "naq_artifact": None, "naq_naturalness": None}),
            }), flush=True)
            return True
        except Exception as e:
            print(json.dumps({
                "ok": False, "run_index": run_index,
                "error": f"{type(e).__name__}: {e}",
            }), flush=True)
            return False

    if args.stdin:
        idx = 0
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                print(json.dumps({"ok": False, "run_index": idx,
                                  "error": f"json parse: {e}"}), flush=True)
                idx += 1
                continue
            _one(job["text"], job["out"], idx, write_wav=True)
            idx += 1
        return 0

    for i in range(args.runs):
        if not _one(args.text, args.out, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
