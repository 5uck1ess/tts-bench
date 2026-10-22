"""KittenTTS 2 runner (KittenML / Stellon Labs; code Apache-2.0, weights Stellon Labs
Community License).

A 1.7B speech language model (Qwen-style LM body packed to 1.58-bit ternary, bf16
embedding) that writes S3 codec tokens, decoded to 24 kHz audio. 115 built-in voices
(the eight KittenTTS 0.8 speakers -- Bruno, Jasper, ... -- plus named per-language
voices) and in-context cloning from a 5-30 s reference.

Two rows share this runner, one per engine, in separate venvs:

    torch -> `kittenml` 0.9.6 (venvs/kittentts2). The reference Python package.
             cpu + cuda, presets and cloning. Weights packing "packed" (the package
             default); full / packed / emb4 measured within 5% of each other on
             CUDA p3, so the packing is not what sets its speed.
    cpp   -> `kitten_tts_cpp` 0.2.0 (venvs/kittentts2_cpp). KittenML's llama.cpp
             fork, the engine their "runs real time on a CPU" claim refers to.
             CPU only (the wheel is built GGML_CUDA=OFF), presets only: raw-audio
             cloning raises NotImplementedError and needs an offline
             prepare_reference.py step, so this row is default-lens only. Uses the
             library's default 8 threads and its own default decoder (config
             `cpp_decoder`), which is NOT the torch path's decoder -- the two rows
             are the same LM through two engines and two decoders, so compare their
             audio, not just their speed. Its sampler is not bit-identical to the
             torch one even at the same seed (upstream README says so).

Weights: HF KittenML/kitten-tts-2 (ungated) @ 1f00fee525c7b2e30bbe9ca9ba67529734fce9c9
(refs/main when benched 2026-10-22; not pinned by revision=, like every runner here).

API (both engines expose the same surface):
    m = KittenTTS("KittenML/kitten-tts-2", device=...)
    wav = m.generate(text, voice="Bruno" | reference=path, reference_text=...,
                     normalize=..., advanced={"seed": N})   # float32 numpy, m.sample_rate

Voice per language (default lens): en -> Bruno (the repo's own default_voice),
fr -> French, es -> Spanish. These are the model's own per-language voices; the
text normalizer is English-tuned, so non-English text runs with normalize=False as
upstream instructs.

Cloning: the reference's sibling .txt is passed as reference_text, which skips the
Whisper pass kittenml otherwise runs on every call (that would be timed).

Non-streaming: generate() returns the whole array (generate_stream is chunk-level
and every bench prompt fits one ~380-char chunk anyway), so TTFA == gen_s.

Generation cap: max_new_tokens=1000 per chunk at ~25 tokens/s = ~40 s per chunk;
prompt 3 measured 20.1 s (preset) / 16.9 s (cloned), well inside it.

Determinism: seed pinned per call via advanced={"seed": SEED}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ID = "KittenML/kitten-tts-2"
VARIANTS = ("torch", "cpp")
DEFAULT_VARIANT = "torch"
SEED = 1234
LANG_VOICE = {"en": "Bruno", "fr": "French", "es": "Spanish"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Reference wav to clone (torch engine only); sibling .txt is its transcript.")
    p.add_argument("--variant", default=None, choices=VARIANTS)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    variant = args.variant or DEFAULT_VARIANT
    if args.language not in LANG_VOICE:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"no KittenTTS 2 voice wired for language={args.language}"}))
        return 1
    if variant == "cpp" and args.reference:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "kitten_tts_cpp cannot clone raw audio (needs offline prepare_reference.py)"}))
        return 1

    try:
        if variant == "cpp":
            if args.device != "cpu":
                raise ValueError("kitten_tts_cpp is CPU-only")
            from kitten_tts_cpp import KittenTTS
        else:
            from kittenml import KittenTTS
        import soundfile as sf

        m = KittenTTS(REPO_ID, device=args.device)
        sample_rate = m.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    kw = {"normalize": args.language == "en", "advanced": {"seed": SEED}}
    if args.reference:
        kw["reference"] = args.reference
        txt = Path(args.reference).with_suffix(".txt")
        if txt.exists():
            kw["reference_text"] = txt.read_text(encoding="utf-8").strip()
    else:
        kw["voice"] = LANG_VOICE[args.language]

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio = m.generate(text, **kw)
            t_end = time.perf_counter()

            audio_s = float(len(audio) / sample_rate)
            if write_wav:
                sf.write(out_path, audio, sample_rate)

            # Non-streaming: TTFA = gen_s (no audio until the call returns).
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
