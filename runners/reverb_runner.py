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

Choices / gotchas (measured 2026-11-19; weights + presets pinned to the v1.0.0
tag, HF sha 72be799c — see REVISION):
- Presets are loaded by `_load_preset` (UTF-8, pinned revision, no Hub listing),
  not `tts.preset_voice()` — see its docstring for the cp1252 mojibake trap.
- Reference language for transcript normalization: sibling `<ref>.lang`, else en.
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
# The v1.0.0 tag the package defaults to, pinned by SHA so a moved tag can't
# silently change the row. Weights AND preset JSONs both come from here (the
# package itself fetches presets from `main`; the three we use are byte-identical
# at main 9c0da9dd, so rows benched before this pin are the same voices).
REVISION = "72be799c8fb6d56a206d260b425c48bf3bd5943d"
SEED = 42
PRESETS = {"en": "en_male_narrator_1", "fr": "fr_male_standard", "es": "es_male_standard_1"}
# sha256 of each preset JSON at REVISION. Lets an offline run accept a cached copy
# fetched under another revision (the package itself caches presets under `main`)
# only when it is byte-for-byte the pinned file.
PRESET_SHA256 = {
    "en_male_narrator_1": "e69282b398ae1d5adfbb81b462b0ab8d3d5c65c09a64eb72507fa7926e456f69",
    "fr_male_standard":   "6bdb764082b824729065037d9f61deaf6528f161f862045feb11fedf1e7770ff",
    "es_male_standard_1": "287882480b440cde3b77b6f8e9828776cbbbdf59e0f1a8103302ed0d7cf610bb",
}


def _read_ref_transcript(ref_wav):
    if not ref_wav:
        return None
    txt = Path(ref_wav).with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else None


def _read_ref_language(ref_wav):
    """Language of the REFERENCE recording (drives Verbalis normalization of its
    transcript) — not the text's language, which may differ (cross-lingual).
    The harness has no channel for it, so: a sibling `<ref>.lang` file holding a
    code like `fr`, else English (every house reference is English)."""
    lang = Path(ref_wav).with_suffix(".lang")
    if lang.exists():
        # utf-8-sig: Notepad-style BOMs would otherwise make check_language reject "fr".
        return lang.read_text(encoding="utf-8-sig").strip().lower()
    return "en"


def _load_preset(name):
    """Preset voice from the pinned revision, parsed as UTF-8.

    Not `tts.preset_voice()`: it lists the Hub repo on every call (breaks
    HF_HUB_OFFLINE with a full cache, and ran inside each warm iteration), and
    `Voice.load` reads the JSON with the locale codec — cp1252 on Windows, which
    turns the fr/es transcripts' é/ñ into mojibake that conditions generation.
    Mirrors Voice.load's JSON branch otherwise."""
    import json as _json
    import torch
    from huggingface_hub import hf_hub_download
    from reverb_tts.text import check_language
    from reverb_tts.tts import CODEBOOKS, Voice
    import hashlib
    try:
        raw = Path(hf_hub_download(REPO, f"voices/{name}.json", revision=REVISION)).read_bytes()
    except Exception as first_err:  # offline with the file cached under another revision
        from huggingface_hub.constants import HF_HUB_CACHE
        snaps = Path(HF_HUB_CACHE) / ("models--" + REPO.replace("/", "--")) / "snapshots"
        raw = next((b for b in (p.read_bytes() for p in snaps.glob(f"*/voices/{name}.json"))
                    if hashlib.sha256(b).hexdigest() == PRESET_SHA256[name]), None)
        if raw is None:
            raise first_err
    if hashlib.sha256(raw).hexdigest() != PRESET_SHA256[name]:
        raise ValueError(f"preset {name} does not match the pinned sha256")
    data = _json.loads(raw.decode("utf-8"))
    codes = torch.tensor(data["codes"], dtype=torch.long).view(CODEBOOKS, -1)
    return Voice(codes, data["transcript"], check_language(data["language"]), data.get("speech_frames"))


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

        tts = TTS.from_pretrained(REPO, device=args.device, revision=REVISION)
        # Encode the reference / load the presets once, at load (outside timing).
        if args.reference:
            ref_lang = _read_ref_language(args.reference)
            if ref_lang == "en" and args.language != "en" and not Path(args.reference).with_suffix(".lang").exists():
                print(f"reverb: no {Path(args.reference).with_suffix('.lang').name}; treating the reference "
                      f"recording as English (target text is {args.language}). Add the sidecar if it isn't.",
                      file=sys.stderr)
            clone_voice = tts.voice(args.reference, ref_text, language=ref_lang)
            presets = {}
        else:
            clone_voice = None
            presets = {lang: _load_preset(name) for lang, name in PRESETS.items()}
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _voice_for(lang):
        if clone_voice is not None:
            return clone_voice
        return presets.get(lang, presets["en"])

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
