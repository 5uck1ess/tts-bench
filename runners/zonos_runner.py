"""Zonos-v0.1 transformer runner (Zyphra, Apache 2.0, zero-shot cloning, 44.1kHz).

API (Zyphra/Zonos, transformer backbone) — verified against upstream sample.py:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    wav, sr = torchaudio.load(ref_wav)
    speaker = model.make_speaker_embedding(wav, sr)
    cond = make_cond_dict(text=text, speaker=speaker, language="en-us")
    codes = model.generate(model.prepare_conditioning(cond))
    wavs = model.autoencoder.decode(codes).cpu()   # native 44.1kHz
    sr = model.autoencoder.sampling_rate

Zero-shot voice cloning from a reference wav (no transcript needed — speaker
embedding is computed directly from the audio). Non-streaming: returns the
full waveform after all autoregressive steps, so ttfa_ms == gen_s.

espeak-ng: Zonos phonemizes via `phonemizer`, which needs espeak-ng. We do NOT
require a system install — the `espeakng-loader` pip package bundles the
espeak-ng shared lib + data. The PHONEMIZER_ESPEAK_LIBRARY/DATA env vars are
set from espeakng_loader at module top BEFORE importing zonos/phonemizer.

We install the TRANSFORMER variant only — the hybrid (mamba-ssm) backbone is
not installed (CUDA+Linux-only `.[compile]` extras). MPS is unsupported
upstream, so device is cpu/cuda only.

Install gotchas:
- torch 2.11 routes torchaudio.load() through torchcodec, which needs FFmpeg
  shared DLLs (not present on stock Windows). The runner monkey-patches
  torchaudio.load to use soundfile directly (same as f5tts_runner).
- torchaudio 2.11's MelSpectrogram constructor mixes a hardcoded CPU tensor
  with the active `torch.device("cuda")` context that SpeakerEmbeddingLDA uses,
  raising a device-mismatch RuntimeError. The runner builds the speaker
  embedding on CPU and moves the result to the model device.
- The bundled espeak-ng DLL has a hardcoded CI build path baked in, so
  ESPEAK_DATA_PATH must point at the real data dir (same as kittentts_runner).
- generate() falls back to torch.compile for the transformer backbone (CUDA
  Graphs are mamba-ssm only), which needs MSVC cl.exe on Windows. The runner
  passes disable_torch_compile=True to run eager.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo
import _naq


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
                   help="Reference wav for zero-shot cloning (no transcript needed).")
    p.add_argument("--variant", default=None)        # unused (transformer only)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # On a CUDA-equipped host, Zonos' generate path allocates a tensor on the
    # visible GPU even when we asked for device="cpu", colliding with the cpu
    # model ("mat1 is on cuda:0, ... other tensors on cpu"). Hide CUDA for the
    # cpu cell so the process behaves like a cpu-only box (how it runs on Mac).
    # Use "-1" (an invalid index), NOT "" — an empty string is treated as unset
    # and leaves all GPUs visible. Must run before torch is imported in the try
    # block below (_meminfo imports torch lazily, so torch is not yet loaded).
    if (args.device if args.device in ("cpu", "cuda") else "cpu") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        # Point phonemizer at the bundled espeak-ng (no system install) BEFORE
        # importing zonos/phonemizer. The bundled DLL has a hardcoded CI build
        # path ('D:/a/espeakng-loader/...') baked in, so ESPEAK_DATA_PATH must be
        # set to the real data dir or espeak errors on `phontab` lookup. Same
        # proven wiring as runners/kittentts_runner.py.
        import espeakng_loader
        os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
        espeakng_loader.make_library_available()
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())

        _install_soundfile_loader()
        import torch
        import torchaudio
        import numpy as np
        import soundfile as sf
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        from zonos.speaker_cloning import SpeakerEmbeddingLDA

        device = args.device if args.device in ("cpu", "cuda") else "cpu"  # upstream disables MPS
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

        repo = Path(__file__).resolve().parent.parent
        ref_wav = args.reference or str(repo / "reference" / "chris_hemsworth_15s.wav")
        if not Path(ref_wav).exists():
            raise FileNotFoundError(f"Voice reference wav not found: {ref_wav}")
        _w, _sr = torchaudio.load(ref_wav)
        # Build the speaker-embedding model on CPU regardless of target device.
        # torchaudio 2.11's MelSpectrogram constructor mixes a hardcoded CPU
        # tensor with the active `torch.device("cuda")` context inside
        # SpeakerEmbeddingLDA, raising a device-mismatch RuntimeError. Building on
        # CPU then moving the resulting embedding to the model device sidesteps it.
        model.spk_clone_model = SpeakerEmbeddingLDA(device="cpu")
        _, speaker = model.spk_clone_model(_w, _sr)
        speaker = speaker.to(model.device, dtype=next(model.parameters()).dtype)

        LANG = {"en": "en-us", "fr": "fr-fr", "de": "de", "ja": "ja", "zh": "cmn"}.get(args.language, "en-us")
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            cond = make_cond_dict(text=text, speaker=speaker, language=LANG)
            # disable_torch_compile: the transformer backbone can't use CUDA
            # Graphs (that path is mamba-ssm only), so generate() otherwise falls
            # back to torch.compile, which needs MSVC cl.exe on Windows. Disable
            # it to run with eager attention.
            codes = model.generate(model.prepare_conditioning(cond), disable_torch_compile=True)
            wavs = model.autoencoder.decode(codes).cpu()
            sr = model.autoencoder.sampling_rate
            t_end = time.perf_counter()

            arr = wavs[0].numpy().astype("float32").squeeze()
            audio_s = float(len(arr) / sr)
            if write_wav:
                sf.write(out_path, arr, sr)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,  # non-streaming: TTFA == gen
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
