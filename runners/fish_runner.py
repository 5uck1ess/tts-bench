"""Fish Speech 1.5 runner (fishaudio, zero-shot voice cloning, 44.1 kHz).

Installed from the v1.5.0 source tag into venvs/fish/src. The inference API in
this tag lives under the `tools.*` package (NOT `fish_speech.inference_engine`,
which only exists in later releases):

    from tools.llama.generate import launch_thread_safe_queue
    from tools.vqgan.inference import load_model as load_decoder_model
    from tools.inference_engine import TTSInferenceEngine
    from tools.schema import ServeTTSRequest, ServeReferenceAudio

We instantiate TTSInferenceEngine DIRECTLY rather than going through
tools.server.model_manager.ModelManager, which has an unconditional `funasr`
import (ASR pipeline) we don't want to pull in.

The text->semantic LLAMA runs in a daemon worker thread launched by
launch_thread_safe_queue; the queue / engine are held at module scope so the
thread stays alive for the process lifetime. inference() is a generator that
yields InferenceResult(code=..., audio=(sample_rate, np_array)); we collect the
single `code == "final"` result (streaming=False).

Output is the decoder's native sample rate (44.1 kHz for fish-speech-1.5) — we
keep it native rather than resampling.

[stable] extras were intentionally skipped at install time (they pin
torch<=2.4.1, incompatible with Blackwell sm_120 / cu128 torch 2.7+).
"""

import argparse
import json
import sys
import time

import _meminfo
import _naq


REF_TEXT = "This is a reference voice sample used for zero-shot voice cloning."  # generic; accurate transcript would improve quality


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Reference wav path for zero-shot cloning (defaults to chris_hemsworth_15s.wav).")
    p.add_argument("--variant", default=None)        # unused
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from pathlib import Path

        # torchaudio 2.11 (cu128, required for Blackwell sm_120) removed the
        # legacy dispatcher helper list_audio_backends() that fish's
        # ReferenceLoader calls at init, AND routes torchaudio.load() through
        # TorchCodec (not installed — needs FFmpeg shared DLLs on Windows). Shim
        # both: advertise the soundfile backend, and replace load() with a
        # soundfile reader (same workaround the f5tts runner uses). The [stable]
        # extras that would have pinned torch<=2.4.1 / kept the old API are
        # skipped on purpose — incompatible with cu128.
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        def _sf_load(path, **kwargs):  # ignores backend= kwarg; handles path or BytesIO
            data, sr = sf.read(path, dtype="float32")
            if data.ndim == 1:
                data = data[None, :]          # (1, T) mono
            else:
                data = data.T                 # (C, T)
            return torch.from_numpy(data), sr

        torchaudio.load = _sf_load

        from tools.llama.generate import launch_thread_safe_queue
        from tools.vqgan.inference import load_model as load_decoder_model
        from tools.inference_engine import TTSInferenceEngine
        from tools.schema import ServeTTSRequest, ServeReferenceAudio

        CKPT = Path(__file__).resolve().parent.parent / "venvs" / "fish" / "src" / "checkpoints" / "fish-speech-1.5"
        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        precision = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(CKPT), device=device, precision=precision, compile=False)
        decoder_model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=str(CKPT / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
            device=device)
        engine = TTSInferenceEngine(
            llama_queue=llama_queue, decoder_model=decoder_model,
            precision=precision, compile=False)
        ref_wav = args.reference or str(
            Path(__file__).resolve().parent.parent / "reference" / "chris_hemsworth_15s.wav")
        ref_bytes = open(ref_wav, "rb").read()
        samplerate = 44100  # Fish 1.5 native; keep native SR
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(audio=ref_bytes, text=REF_TEXT)],
                streaming=False, format="wav")
            audio_np, sr = None, samplerate
            for result in engine.inference(req):
                if result.code == "error":
                    raise RuntimeError(f"Fish engine error: {result.error}")
                if result.code == "final":
                    sr, audio_np = result.audio
            if audio_np is None:
                raise RuntimeError("Fish engine returned no final audio")
            t_end = time.perf_counter()

            audio_np = np.asarray(audio_np, dtype="float32").squeeze()
            audio_s = float(len(audio_np) / sr)
            if write_wav:
                sf.write(out_path, audio_np, sr)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,  # non-streaming, so TTFA == gen_s
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
