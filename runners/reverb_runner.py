"""Reverb-TTS runner (Software Mansion AI Labs, 297M, zero-shot cloning + presets).

Reverb is a pruned + distilled + DPO'd OmniVoice (our `omnivoice` row is its
parent): 24 of OmniVoice's 28 layers, smaller FFN/KV/vocab, retrained on seven
languages (en/de/fr/es/it/pl/pt). Masked-diffusion over codec tokens with a
FIXED number of passes (8 main steps + polish), so generation time is nearly
flat in output length — ~0.3 s warm per prompt on the 5090 from p1 to p3.
Short prompts therefore show low RTFx and the long prompt high RTFx (~50x warm
on p3); that's the model, not a measurement bug.

API (reverb-tts 1.0.0, PyPI):
    from reverb_tts import TTS
    tts = TTS.from_pretrained("software-mansion/Reverb-TTS", device="cuda")
    voice = tts.voice("ref.wav", "exact transcript", language="en")   # cloning
    voice = tts.preset_voice("en_male_narrator_1")                    # preset
    audio = tts.generate(text, voice, language="en", seed=42)          # float32 @ 24 kHz

Choices / gotchas (measured 2026-11-19, HF sha 9c0da9dd):
- Weights: package default variant `int8` (per-channel int8 safetensors). Every
  variant is upcast to the compute dtype at load (bf16 on CUDA, fp32 on CPU),
  so variant changes numerics only, not speed. The card's quality table uses
  full precision; this row is what `pip install reverb-tts` gives you.
- Preset voices are real presets (16 shipped, per language), not a fallback to
  a bundled reference — so the model sits in `_PRESET_AND_CLONE`. Pinned per
  bench language: en_male_narrator_1 / fr_male_standard / es_male_standard_1.
- Cloning needs the transcript (sibling .txt). The reference is encoded ONCE
  at load (`tts.voice(...)`, a cacheable Voice of codec codes), so encoding is
  OUTSIDE the timed region — same convention as pocket / zonos3_mini, and how
  the card times it ("from an already-encoded reference voice"). Runners that
  pass the ref into every call (omnivoice, audio8) include it in gen_s. The house ref is 15.1 s and the
  package warns "references over 15 s tend to make worse takes"; we keep the
  house ref for comparability with every other cloner.
- Text is normalized by Verbalis inside `generate` (numbers etc. to words);
  left on (package default) — the bench prompts are already spoken-form.
- Seed pinned at 42 for every call, so warm runs regenerate the same take.
- Non-streaming: ttfa_ms == gen_s (whole clip returned at once).
- Output has silent edges trimmed by default (`trim=True`); kept.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


SAMPLE_RATE = 24000
REPO = "software-mansion/Reverb-TTS"
SEED = 42
PRESETS = {"en": "en_male_narrator_1", "fr": "fr_male_standard", "es": "es_male_standard_1"}


def _read_ref_transcript(ref_wav):
    if not ref_wav:
        return None
    txt = Path(ref_wav).with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav for zero-shot cloning. Needs sibling .txt transcript.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    ref_text = _read_ref_transcript(args.reference)
    if args.reference and not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference {args.reference} provided but sibling .txt transcript missing"}))
        return 1

    try:
        import numpy as np
        import soundfile as sf
        from reverb_tts import TTS

        tts = TTS.from_pretrained(REPO, device=args.device)
        # Reference language: the house ref is English. Encode once at load.
        clone_voice = tts.voice(args.reference, ref_text, language="en") if args.reference else None
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _voice_for(lang):
        if clone_voice is not None:
            return clone_voice
        return tts.preset_voice(PRESETS.get(lang, PRESETS["en"]))

    def _one(text, out_path, lang, run_index, write_wav):
        try:
            voice = _voice_for(lang)
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio = tts.generate(text, voice, language=lang, seed=SEED)
            t_end = time.perf_counter()
            arr = np.asarray(audio, dtype=np.float32).reshape(-1)
            if write_wav:
                sf.write(out_path, arr, SAMPLE_RATE)
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": float(len(arr) / SAMPLE_RATE),
                **_meminfo.sample(args.device),
            }), flush=True)
            return True
        except Exception as e:
            print(json.dumps({"ok": False, "run_index": run_index,
                              "error": f"{type(e).__name__}: {e}"}), flush=True)
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
            _one(job["text"], job["out"], job.get("language", args.language), idx, write_wav=True)
            idx += 1
        return 0

    for i in range(args.runs):
        if not _one(args.text, args.out, args.language, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
