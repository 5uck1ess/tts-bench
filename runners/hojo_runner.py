"""Hojo-TTS-Light runner (HojoAI, Apache-2.0, ONNX Runtime, no torch). Two sizes:

  * ``--variant 40m``: 15 built-in voices, NO cloning. Runs the repo's default
    voice ``hojo_en_m_02``. ``--reference`` is accepted and ignored.
  * ``--variant 80m``: cloning ONLY (reference wav + sibling .txt transcript).
    Its ``voice.npz`` holds just token embeddings and ``generate()`` requires
    ``prompt_speech``, so there is no preset voice; the no-reference path clones
    the house chris_hemsworth_15s reference instead (NO_PRESET_VOICE).

Both are token-LMs over a 50 Hz codec, 24 kHz output. Source is the GitHub repo
pinned at 7b2c674 (2026-10-27; each size ships its own ``onnx_model.py``, so the
runner imports from that size's folder), weights pinned by HF revision below.

Sampling: upstream defaults (temperature 0.8, top_p 0.95, repetition penalty
1.1, max 2048 tokens = ~41 s, prompt 3 is ~18 s). ``generate()`` re-seeds
``np.random`` with ``seed`` on every call, so reruns are bit-identical.

Gotchas:
  * **The 80M continues the reference if it is long.** On the full 15 s house
    reference it often speaks the reference transcript's tail before or inside
    the target ("...otherwise because you, you're amazing"): CPU, 3 seeds x 4
    English prompts gave WER up to 2.1, and p2 ran to 30 s once. On a 5.27 s
    clean cut (Whisper-timed, cut in the 5.21-5.33 s pause) all 12 clips were
    sensible lengths, p1/p2 WER 0-0.14. Upstream pitches cloning from "a few
    seconds" of audio (its example ref is 5.5 s) and the speaker encoder reads
    only 6 s anyway. So, like zipvoice, the house reference is cut in-runner via
    ``_KNOWN_CLIPS``; other references are used as given.
  * **Text normalisation is skipped.** Upstream runs WeTextProcessing, which needs
    pynini, which does not build on Windows. The English bench prompts are already
    spelled out (no digits/symbols), so English TN would be a no-op; skipping it on
    every rig keeps the rigs comparable.
  * **CUDA on Windows needs the cuDNN wheel dirs on PATH.** ``ort.preload_dlls()``
    loads cudnn64_9.dll, but cuDNN lazy-loads its sub-libraries by name and fails
    with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED at the first Conv. ORT then logs
    "EP Error" and silently reruns that session on CPU, so the cell still "works".
    The nvidia/*/bin dirs are prepended to PATH before onnxruntime is imported, and
    every cuda run checks that no session fell back (a fallback fails the run).
  * onnxruntime-gpu pinned 1.21.1 (upstream: 1.25 breaks the 80M decoder on CUDA).
    Upstream promotes the BF16 weights to FP32 at load for every provider.

The API is non-streaming and returns the full waveform, so TTFA == gen_s.
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import _meminfo


REPO = Path(__file__).resolve().parent.parent
VENV = Path(sys.prefix)
SRC_DIRS = {"40m": VENV / "src" / "Hojo-TTS-Light-40M",
            "80m": VENV / "src" / "Hojo-TTS-Light-80M"}
MODEL_DIRS = {"40m": VENV / "models-40m", "80m": VENV / "models-80m"}
# HF revisions the install scripts download (recorded here; snapshot is local).
REVISIONS = {"40m": "c3cf21c77dc04b220aa4dabe20fc59e66a2a5485",   # HojoAI/Hojo-TTS-Light-40M
             "80m": "d9c02af95496b6363b44944326d71491ab8905b0"}   # HojoAI/Hojo-TTS-Light
VOICE_40M = "hojo_en_m_02"   # the repo's documented default voice
SEED = 42                    # upstream default
DEFAULT_REF = REPO / "reference" / "chris_hemsworth_15s.wav"
SR = 24000

# Clean cut points for house references: (seconds, exact transcript of that slice).
_KNOWN_CLIPS = {
    "chris_hemsworth_15s": (5.27, "Hey you. Yes, yes, yes, you. Today is your day. You got this."),
}


def _add_cuda_dll_dirs():
    nv = VENV / "Lib" / "site-packages" / "nvidia"
    bins = glob.glob(str(nv / "*" / "bin"))
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")


def _resolve_reference(ref_wav):
    """(wav_path, transcript) for 80M cloning; house refs are cut to a clean slice."""
    import soundfile as sf

    ref_wav = Path(ref_wav)
    txt = ref_wav.with_suffix(".txt")
    if not txt.exists():
        raise FileNotFoundError(f"reference {ref_wav} needs a sibling .txt transcript")
    known = _KNOWN_CLIPS.get(ref_wav.stem)
    if not known:
        return str(ref_wav), txt.read_text(encoding="utf-8").strip()
    cut_sec, text = known
    y, sr = sf.read(str(ref_wav), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    fd, clip = tempfile.mkstemp(suffix=".wav", prefix="hojo_ref_")
    os.close(fd)
    sf.write(clip, y[:int(cut_sec * sr)], sr)
    return clip, text


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="80m: wav + sibling .txt (defaults to the house reference). 40m: ignored.")
    p.add_argument("--variant", default="40m", choices=sorted(SRC_DIRS))
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1
    if args.language != "en":
        # Chinese is the model's only other language, and not a bench prompt language.
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"Hojo-TTS-Light runs English only here, got language={args.language}"}))
        return 1
    if args.device not in ("cpu", "cuda"):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unsupported device {args.device} (ONNX Runtime cpu/cuda only)"}))
        return 1

    try:
        if args.device == "cuda":
            _add_cuda_dll_dirs()
        sys.path.insert(0, str(SRC_DIRS[args.variant]))
        import soundfile as sf
        import onnx_model

        tts = onnx_model.HojoTTSLightOnnx(str(MODEL_DIRS[args.variant]), provider=args.device)
        sessions = ([tts.lm, tts.codec_encoder, tts.decoder, tts.speaker_encoder]
                    if args.variant == "80m" else [tts.lm, tts.fine_local, tts.codec_decode])
        if args.variant == "80m":
            ref_path, ref_text = _resolve_reference(args.reference or DEFAULT_REF)
            gen = lambda t: tts.generate(t, prompt_speech=ref_path, prompt_text=ref_text, seed=SEED)
        else:
            gen = lambda t: tts.generate(t, voice=VOICE_40M, seed=SEED)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            wav = gen(text)
            t_end = time.perf_counter()
            # ORT's fallback happens at run time (EP Error -> session rebuilt on CPU),
            # so a cuda cell must be checked after the call, not just at load.
            if args.device == "cuda" and any(
                    s.get_providers()[0] != "CUDAExecutionProvider" for s in sessions):
                raise RuntimeError("an ONNX session fell back to CPU on --device cuda")

            audio_s = float(len(wav) / SR)
            if write_wav:
                sf.write(out_path, wav, SR)

            # Non-streaming: TTFA = gen_s (no audio until the call returns).
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
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
