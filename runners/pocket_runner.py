"""Pocket-TTS runner.

Benched on **pocket-tts v4.0.0** (re-benched 2026-10-20 from v3.1.0, which was
itself re-benched 2026-09-05 from v2.1.0; install.sh / install.ps1 clone
`--branch v4.0.0`).

⚠ The pin only protects a **fresh** install. Both installers short-circuit on an
existing `venvs/pocket/` ("pocket: already installed") and never re-clone, and even
inside that branch the clone is guarded by `[ ! -d venvs/pocket/src ]`. A rig that
already has the venv therefore stays on whatever commit it installed — it must be
updated by hand. Verify with `git -C venvs/pocket/src describe --tags` before
trusting a row's version.

Resolved checkpoints — nothing here passes `revision=`, so these SHAs are the only
record of what a row was actually measured on:
    pocket (live)  english_2026-09_2   @7f364c05d1be3faf2b234cdbe4cbb68a026d4218
                   french_24l          @3e82814a68665eec246ff649b14c71331f955c06
                   spanish_24l         @2dd944b099d06bb9edbd221fef138685711d9bf0
    --variant 24l  english_2026-04_24l @492522650173a0653b7575cdc25ae09810e5d741
    (v3.1.0 rows were english @39592ff2; fr/es _24l @39592ff2 — all superseded)

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

**The v3.1.0 -> v4.0.0 delta IS weights.** v4 turned `english` into an alias of
`english_2026-09_2` (new checkpoint), retrained French and Spanish (both _24l
configs point at new revisions), moved every config to a `tokenizers` tokenizer
with `replace_characters` rewrites, added a 5 ms fade-in at chunk starts, and
ends voice prompts on a short pause. We name `english_2026-09_2` explicitly so
a future alias move upstream cannot swap the checkpoint under a published row.

API discovered by inspection (2026-05-22, re-verified against v3.1.0 2026-09-05 and v4.0.0 2026-10-20):
    from pocket_tts import TTSModel
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
    model = TTSModel.load_model(language="english_2026-09_2")   # see LANGUAGE_CONFIG
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
    # v4.0.0's latest English, named explicitly rather than via the "english"
    # alias (which upstream re-pointed from @39592ff2 to english_2026-09_2 in
    # v4). Pinning the name means an alias move is a visible diff here, not a
    # silent checkpoint swap on a published row.
    # fr/es stay on the 24-layer configs (retrained in v4); v4's new 6-layer
    # "french"/"spanish" are a different model size and are not benched.
    "en": "english_2026-09_2",
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
    None:  LANGUAGE_CONFIG["en"],
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
        # v4.0.0 bug: utils/config.py reads the model YAML with a bare
        # open(path, "r"), i.e. the locale codec (cp1252 on Windows). v4's new
        # `replace_characters` keys ('’', '«', ...) then decode to multi-char
        # mojibake and load_model dies with "string keys in translate table
        # must be of length 1" -- for every language. Shadow `open` in that one
        # module (its only open() call) so the stock install runs as on Linux.
        import functools
        import pocket_tts.utils.config as _pocket_config
        _pocket_config.open = functools.partial(open, encoding="utf-8")
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
