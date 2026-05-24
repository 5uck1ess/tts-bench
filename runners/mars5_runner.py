"""MARS5-TTS runner (CAMB.AI, zero-shot voice cloning, English).

Loaded via `torch.hub.load('Camb-ai/mars5-tts', 'mars5_english')` - no
git clone, no pip wheel for the model itself. The package just needs
the runtime deps (torch, vocos, encodec, librosa, safetensors, regex).

API (from torch.hub):
    import torch, librosa
    mars5, config_class = torch.hub.load(
        'Camb-ai/mars5-tts', 'mars5_english', trust_repo=True
    )
    wav, sr = librosa.load(ref_wav, sr=mars5.sr, mono=True)
    wav = torch.from_numpy(wav)
    cfg = config_class(deep_clone=True)  # False for shallow clone (no transcript)
    ar_codes, gen_wav = mars5.tts(text, wav, ref_transcript, cfg=cfg)
    # gen_wav: torch tensor, sample_rate is mars5.sr (24kHz)

Cloning flavors:
- shallow (no transcript, fast) - matches ChatterBox / Coqui / VoxCPM
- deep    (wav + transcript, higher quality) - matches NeuTTS / F5 / etc.
This runner picks deep when a sibling .txt exists, shallow otherwise.

License: AGPL-3.0 (commercial use restricted). English-only.
Reference audio MUST be 1-12 seconds per upstream docs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo
import _naq


def _read_ref_transcript(ref_wav):
    if not ref_wav:
        return None
    txt_path = Path(ref_wav).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for cloning (1-12 sec). Sibling .txt enables deep clone.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # MARS5 is English-only.
    if args.language != "en":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"MARS5 is English-only; got language={args.language!r}"}))
        return 1

    # Cloning-only model: fall back to bundled juliette.wav for default-voice runs.
    # MARS5 requires reference audio 1-12 seconds; jo.wav is 13s (just over), so
    # juliette.wav (8.1s) is the bundled ref that fits the window.
    repo = Path(__file__).resolve().parent.parent
    if args.reference:
        ref_wav = Path(args.reference)
    else:
        ref_wav = repo / "reference" / "juliette.wav"

    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found: {ref_wav}"}))
        return 1

    # Deep clone if transcript exists, shallow otherwise.
    ref_text = _read_ref_transcript(str(ref_wav))
    deep_clone = ref_text is not None

    try:
        import librosa
        import numpy as np
        import soundfile as sf
        import torch

        mars5, config_class = torch.hub.load(
            "Camb-ai/mars5-tts", "mars5_english", trust_repo=True,
        )
        device = args.device if args.device in ("cuda", "cpu") else "cpu"
        if device == "cuda" and torch.cuda.is_available():
            mars5 = mars5.to(device)
        sample_rate = int(mars5.sr)

        # Load reference once - reused for every generation.
        ref_arr, _ = librosa.load(str(ref_wav), sr=sample_rate, mono=True)
        ref_tensor = torch.from_numpy(ref_arr)
        if device == "cuda":
            ref_tensor = ref_tensor.to(device)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            cfg = config_class(deep_clone=deep_clone)
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            result = mars5.tts(text, ref_tensor, ref_text or "", cfg=cfg)
            t_end = time.perf_counter()

            # mars5.tts returns (ar_codes, gen_wav) per docs.
            if isinstance(result, tuple) and len(result) == 2:
                _, gen_wav = result
            else:
                gen_wav = result
            if hasattr(gen_wav, "detach"):
                gen_wav = gen_wav.detach().cpu().float().numpy()
            arr = np.asarray(gen_wav).reshape(-1).astype(np.float32)
            audio_s = float(len(arr) / sample_rate)
            if write_wav:
                sf.write(out_path, arr, sample_rate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else {"naq": None, "naq_harm": None, "naq_buzz": None}),
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
