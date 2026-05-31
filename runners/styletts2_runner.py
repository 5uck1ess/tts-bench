"""StyleTTS 2 runner (sidharthrajaram PyPI wrapper, MIT; LibriTTS weights, zero-shot cloning, 24kHz).

API discovered by inspection (2026-05-30, styletts2==0.1.6):
    from styletts2 import tts as styletts2_tts
    model = styletts2_tts.StyleTTS2()   # auto-downloads LibriTTS + ASR/F0/PLBERT
                                        # checkpoints to the cached_path cache on first init
    audio = model.inference(
        text,
        target_voice_path=None,         # None/missing => clones a bundled default LibriVox voice
        output_wav_file=None,           # if set, also writes a wav via scipy.io.wavfile.write
        output_sample_rate=24000,
        alpha=0.3, beta=0.7,            # timbre / prosody blend toward the target voice
        diffusion_steps=5,
        embedding_scale=1,
        ref_s=None,                     # pre-computed style vector (skips compute_style)
    )
    # returns the audio as a float32 numpy array (mono). Model output rate is 24kHz;
    # output_sample_rate only sets the wav-header rate, it does NOT resample, so we keep 24000.

Phonemizer is gruut (English) — NO espeak-ng needed. This wrapper is English-only.
`--language` is accepted for harness uniformity but not acted on.

Device: StyleTTS2.__init__ hard-codes self.device = 'cuda' if torch.cuda.is_available()
else 'cpu' — it ignores any caller device. We set torch best-effort and don't fail on it
(see note below); the wrapper auto-selects the device regardless of --device.
"""

import argparse
import json
import sys
import time

# StyleTTS2's gruut phonemizer surfaces IPA (e.g. the schwa U+0259) into strings
# that get tokenized/handled on the default Windows cp1252 stream, raising
# UnicodeEncodeError. Force UTF-8 on the std streams so IPA passes through.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import _meminfo
import _naq


DIFFUSION_STEPS = 5  # fixed for reproducibility


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Target voice wav for zero-shot cloning. Omit => wrapper's bundled default voice.")
    p.add_argument("--variant", default=None)        # unused
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")        # unused — wrapper is English-only (gruut)
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

        # torch 2.6+ flipped torch.load's `weights_only` default to True. The
        # styletts2 wrapper calls torch.load without weights_only=False on its
        # F0/ASR/.t7 checkpoints (which pickle `getattr` and other globals not on
        # the safe-list), so the default load raises UnpicklingError. These are
        # trusted yl4579 HF checkpoints — force weights_only=False for the loads
        # the wrapper does at init.
        _orig_torch_load = torch.load
        def _patched_torch_load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_torch_load(*a, **k)
        torch.load = _patched_torch_load

        from styletts2 import tts as styletts2_tts

        # The wrapper downloads the legacy `punkt` tokenizer, but NLTK>=3.9's
        # word_tokenize needs the newer `punkt_tab` table. Fetch it once (cached).
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        model = styletts2_tts.StyleTTS2()  # auto-downloads LibriTTS weights on first call
        ref_wav = args.reference  # None => wrapper default single-speaker; path => zero-shot clone
        samplerate = 24000  # StyleTTS2 LibriTTS fixed output rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            wav = model.inference(
                text=text,
                target_voice_path=ref_wav,
                output_sample_rate=samplerate,
                diffusion_steps=DIFFUSION_STEPS,
                alpha=0.3, beta=0.7, embedding_scale=1.0,
            )
            t_end = time.perf_counter()

            arr = np.asarray(wav, dtype="float32").squeeze()
            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,  # non-streaming: TTFA == full gen
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
