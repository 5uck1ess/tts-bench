"""Scylla's Band runner (LiteRT on CPU, fixed voices, predefined-only).

Release v3-20261104 (scyllasband runtime 3.0, github.com/lowkeytea/scyllasband
at or after 26fa443, which reads "V3", "GPU" and the like correctly). Fourteen
voices; English plus es, it, fr, de and vi. Output is 24 kHz.

English dialect. Each voice speaks one English dialect: ink, orpheus and
tuesday are en_gb, the rest en_us. Harness "en" is passed to the runtime as
"en", which selects the voice's own dialect; v3 rejects the other one, so the
v1 runner's fixed "en" -> "en_us" mapping fails for the three en_gb voices.

Bundle. install.* downloads the LiteRT bundle (`download --flavor litert`) on
every rig, so all rigs bench the same graphs; LiteRT is the runtime's default
CPU path. `--variant onnx` runs the ONNX bundle instead (it needs
`python -m scyllasband download --flavor onnx` and `pip install -e .[onnx]`),
and SCYLLASBAND_BUNDLE points at any bundle directory.

Synthesis. synthesize_stream() speaks long text as passages and emits one
`audio_chunk` event per passage, the first one short, so ttfa_ms is a real time
to first audio. The seed is pinned (11) so reruns render the same audio.
Delivery (intensity, pace, pitch) is left unset: each voice's own delivery.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo


# Harness language code -> runtime language code. "en" stays "en": the runtime
# resolves it to the voice's own English dialect (see the docstring).
LANGUAGE_MAP = {
    "en": "en",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "de": "de",
    "vi": "vi",
}

# Runtime codes are also accepted verbatim for A/B work the harness never asks
# for; a voice only accepts its own English dialect.
NATIVE_LANGUAGES = {"en_us", "en_gb"}

FLAVORS = ("litert", "onnx")
SEED = 11


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Voice ID (e.g. 'gwen'); defaults to 'tuesday'. "
                        "Scylla's Band doesn't support wav cloning.")
    p.add_argument("--variant", default=None, help="Bundle flavor: litert (default) or onnx.")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    language = LANGUAGE_MAP.get(args.language)
    if language is None and args.language in NATIVE_LANGUAGES:
        language = args.language
    if not language:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unsupported Scylla's Band language={args.language}; "
                                   f"harness codes: {', '.join(sorted(LANGUAGE_MAP))}; native codes: "
                                   f"{', '.join(sorted(NATIVE_LANGUAGES))}"}))
        return 1

    flavor = (args.variant or "litert").lower()
    if flavor not in FLAVORS:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unknown variant {args.variant!r}; choose {', '.join(FLAVORS)}"}))
        return 1

    try:
        from scyllasband import ScyllasBandRuntime
        from scyllasband.planner import records_from_text
        import numpy as np
        import soundfile as sf

        repo_root = Path(__file__).resolve().parents[1]
        bundle_dir = Path(os.environ.get(
            "SCYLLASBAND_BUNDLE",
            repo_root / "venvs" / "scyllasband" / "src" / "scyllasband" / "models" / flavor,
        )).expanduser().resolve()
        # tuesday led the v3 voice sweep (UTMOS 4.27, WER 0.029 over prompts 1-4, two
        # seeds; ink 4.22 / 0.033 within the seed spread). See docs/known-issues.md.
        voice_id = args.reference or "tuesday"
        # With SCYLLASBAND_BUNDLE and no --variant, the bundle's own manifest picks the backend.
        explicit = args.variant or not os.environ.get("SCYLLASBAND_BUNDLE")
        rt = ScyllasBandRuntime.from_bundle(bundle_dir, backend=flavor if explicit else None)
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
            samplerate = None
            records = records_from_text(text, voice=voice_id, language=language)
            for event in rt.synthesize_stream(records, seed=SEED):
                if event.type != "audio_chunk" or event.audio is None:
                    continue
                if first is None:
                    first = time.perf_counter()
                chunks.append(np.asarray(event.audio, dtype=np.float32))
                samplerate = int(event.sample_rate)
            t_end = time.perf_counter()

            if not chunks:
                raise RuntimeError("no audio_chunk events emitted")
            audio = np.concatenate(chunks)
            audio_s = float(len(audio) / samplerate)
            if write_wav:
                sf.write(out_path, audio, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (first - t0) * 1000,
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
