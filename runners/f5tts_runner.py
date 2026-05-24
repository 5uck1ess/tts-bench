"""F5-TTS runner (MIT, zero-shot voice cloning via flow matching).

API (f5-tts==0.x):
    from f5_tts.api import F5TTS
    m = F5TTS(model='F5TTS_v1_Base', device='cpu')
    wave, sr, spec = m.infer(ref_file=..., ref_text=..., gen_text=...,
                              file_wave=out_path, show_info=lambda *a, **k: None,
                              progress=None)

Voice cloning: requires a reference wav + its transcript text (similar to NeuTTS).
The runner reads <reference>.txt (same basename as the wav) automatically.

Heavy on CPU (flow matching + diffusion sampling, GPU-targeted).
Non-streaming: returns full waveform after all sampling steps. TTFA == gen_s.

Install gotchas:
- `torchaudio.load()` in torch 2.12+ routes through `torchcodec`, which
  requires FFmpeg shared DLLs (libtorchcodec_core4.dll etc.). On Windows
  with a static FFmpeg build, this fails. Workaround in this runner:
  monkey-patch torchaudio.load to use `soundfile` directly.
- Pin `datasets<3.0` to avoid pulling torchcodec into the import chain.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo


def _install_soundfile_loader():
    """Replace torchaudio.load with soundfile to avoid torchcodec DLL hell on Windows."""
    import soundfile as sf
    import numpy as np
    import torch
    import torchaudio

    def _sf_load(path, **kwargs):
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        return torch.from_numpy(data), sr

    torchaudio.load = _sf_load


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Reference wav for zero-shot cloning. Needs matching .txt with transcript next to it.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        _install_soundfile_loader()
        from f5_tts.api import F5TTS
        import numpy as np
        import soundfile as sf

        # Reference handling: default to jo (EN) / juliette (FR) shipped in reference/
        repo = Path(__file__).resolve().parent.parent
        if args.reference:
            ref_wav = Path(args.reference)
        else:
            default_ref = {"en": "jo.wav", "fr": "juliette.wav"}.get(args.language, "jo.wav")
            ref_wav = repo / "reference" / default_ref
        ref_txt = ref_wav.with_suffix(".txt")
        if not ref_wav.exists():
            raise FileNotFoundError(f"reference wav not found: {ref_wav}")
        if not ref_txt.exists():
            raise FileNotFoundError(f"reference transcript not found: {ref_txt} (F5-TTS needs .wav + .txt pair)")
        ref_text = ref_txt.read_text(encoding="utf-8").strip()

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        m = F5TTS(model="F5TTS_v1_Base", device=device)
        samplerate = 24000  # F5-TTS v1 fixed output rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            result = m.infer(
                ref_file=str(ref_wav), ref_text=ref_text, gen_text=text,
                file_wave=out_path if write_wav else None,
                show_info=lambda *a, **k: None, progress=None,
            )
            t_end = time.perf_counter()

            # F5TTS.infer returns (wave_ndarray, sample_rate, spectrogram)
            wave = result[0] if isinstance(result, tuple) else result
            arr = np.asarray(wave)
            audio_s = float(len(arr) / samplerate)

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
