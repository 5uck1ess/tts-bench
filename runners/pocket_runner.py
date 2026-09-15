"""Pocket-TTS runner.

Benched on **pocket-tts v3.1.0** (re-benched 2026-09-05 from v2.1.0; install.sh /
install.ps1 now clone `--branch v3.1.0`).

⚠ The pin only protects a **fresh** install. Both installers short-circuit on an
existing `venvs/pocket/` ("pocket: already installed") and never re-clone, and even
inside that branch the clone is guarded by `[ ! -d venvs/pocket/src ]`. A rig that
already has the venv therefore stays on whatever commit it installed — it must be
updated by hand. Verify with `git -C venvs/pocket/src describe --tags` before
trusting a row's version.

Resolved checkpoints — nothing here passes `revision=`, so these SHAs are the only
record of what a row was actually measured on:
    pocket (live)  english             @39592ff23c9ef80098bb74895d104c26275fe2c9
    --variant 24l  english_2026-04_24l @492522650173a0653b7575cdc25ae09810e5d741

`--variant 24l` has NO harness row: the 24-layer variant was benched 2026-09-05 and
skipped (336M for 1.62x RTFx against the base row's 4.90x — both are same-basis
medians from that scratch run, NOT the board's published 4.40x; see
docs/considered.md).
The mapping is kept so revisiting it is one harness row, not a re-implementation.

**The v2.1.0 -> v3.1.0 delta is sampling temperature, not weights.** `english`'s
weights are byte-identical across the two releases (same @39592ff2), but v3.1.0
added `default_temperature: 0.3` to the config; v2.1.0 had no such key and fell
back to the global DEFAULT_TEMPERATURE = 0.7. This runner deliberately does NOT
pass `temp`, so a row always reflects upstream's own default — which is why the
re-benched audio differs from the published v2.1.0 clips.

API discovered by inspection (2026-05-22, re-verified against v3.1.0 2026-09-05):
    from pocket_tts import TTSModel
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
    model = TTSModel.load_model(language="english")   # see LANGUAGE_CONFIG
    state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/...")
    for chunk in model.generate_audio_stream(state, text):
        ...

Default voices come from _ORIGINS_OF_PREDEFINED_VOICES — we use "anna" for
English and "estelle" for French (both clean VCTK/Unmute references). User
can pass --reference path/to/wav to override.
"""

import argparse
import json
import sys
import time

import _meminfo


LANGUAGE_CONFIG = {
    # "english", NOT "english_2026-04". They are different checkpoints despite
    # upstream's load_model docstring calling them "the same model":
    #   english          -> languages/english/model.safetensors@39592ff2
    #   english_2026-04  -> languages/english_2026-04/model.safetensors@19f95fe2
    # Every published `pocket` row was measured on "english" (the old code path
    # called load_model() with no argument, which resolves here), so this stays
    # "english" to keep the v2.1.0 -> v3.1.0 re-bench a clean version comparison
    # rather than a silent checkpoint swap.
    "en": "english",
    "fr": "french_24l",
    "de": "german_24l",
    "it": "italian_24l",
    "pt": "portuguese_24l",
    "es": "spanish_24l",
}

# Harness variant -> English model config. Only English ships a size variant;
# the non-English entries above are already upstream's 24-layer builds, so the
# variant is meaningful for English only.
VARIANT_EN_CONFIG = {
    None:  "english",
    "24l": "english_2026-04_24l",
}


DEFAULT_VOICE = {
    "en": "anna",       # VCTK p228
    "fr": "estelle",    # Unmute prod website
    "de": "juergen",
    "it": "giovanni",
    "pt": "anna",       # no pt-specific default in catalog; reuse en
    "es": "lola",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")        # pocket-tts is CPU-only
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)        # None = 16-layer; "24l" = 24-layer English
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true",
                   help="Interactive mode: read JSON jobs {text, out} from stdin, one per line.")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        from pocket_tts import TTSModel
        import numpy as np
        import soundfile as sf

        # Fail loudly rather than silently substituting English: a bench row
        # that runs fine under the wrong config is mislabeled data, which is
        # worse on a public board than a visibly failed cell.
        if args.language == "en":
            if args.variant not in VARIANT_EN_CONFIG:
                raise ValueError(
                    f"unknown --variant {args.variant!r}; expected "
                    f"{sorted(k for k in VARIANT_EN_CONFIG if k)} or omitted")
            lang_cfg = VARIANT_EN_CONFIG[args.variant]
        else:
            if args.language not in LANGUAGE_CONFIG:
                raise ValueError(
                    f"unsupported --language {args.language!r}; expected "
                    f"{sorted(LANGUAGE_CONFIG)}")
            lang_cfg = LANGUAGE_CONFIG[args.language]
        model = TTSModel.load_model(language=lang_cfg)
        samplerate = int(model.sample_rate)

        # Pocket-TTS accepts either a predefined voice name ("anna") or a path/url to
        # a wav for cloning. Cloning requires HF auth on the gated kyutai/pocket-tts
        # repo; named voices work without auth.
        voice = args.reference or DEFAULT_VOICE.get(args.language, "anna")
        state = model.get_state_for_audio_prompt(voice)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            first = None
            chunks = []
            for chunk in model.generate_audio_stream(state, text):
                if first is None:
                    first = time.perf_counter()
                arr = chunk.numpy() if hasattr(chunk, "numpy") else np.asarray(chunk)
                chunks.append(arr)
            t_end = time.perf_counter()

            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
            audio_s = float(len(audio) / samplerate)
            if write_wav:
                sf.write(out_path, audio, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000 if first else None,
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
        # Signal readiness so speak.py knows the model is loaded.
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
