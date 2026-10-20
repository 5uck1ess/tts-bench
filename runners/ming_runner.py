"""Ming-3.0-tts runner (inclusionAI / Ant Group, 2026-10-19, MIT).

Variant `0.5b` = Ming-3.0-tts-0.5B: dense Qwen2.5-0.5B backbone + flow-matching
patch head, 44.1 kHz. Variant `tiny` = Ming-3.0-tts-tiny: 9.0B-total / ~1.3B-active
MoE (BailingMoeV3), same 44.1 kHz patch geometry; Linux-only (its own venv,
venvs/ming_tiny: Python 3.10 + torch 2.7.1+cu118 from requirements-tiny.txt, which
needs fla-core/triton), ~17.5 GB VRAM in bf16, ~0.4x realtime on a 3090. Source is a plain repo clone (no package) at
`venvs/ming/src`, pinned at `e999c39`; the runner puts it on sys.path and
chdirs into it because the repo imports its own top-level modules
(`tts_tasks`, `frontend`, ...).

API: `ming_tts.MingTTS(repo_id, device, text_normalization).synthesize(task, text, ...)`.

Voices:
- Default lens: task="IP", controls={"ip": "US_James"}, use_zero_spk_emb=True.
  A genuine built-in English voice (README-recommended), not a clone of a bundled
  wav -> `_PRESET_AND_CLONE`, both lenses.
- Cloning lens: task="TTS" with prompt_wav_path + prompt_text (sibling .txt,
  REQUIRED) + use_spk_emb=True (CAM++ speaker embedding), as the upstream CLI example.
  This is continuation-style cloning: a transcript that under-describes the
  reference makes the model swallow the start of the target text. That is how the
  truncated house transcript (fixed 2026-10-20, see docs/known-issues.md) was found.

Gotchas:
- text_normalization=False: the FST normalizer needs pynini, which has no usable
  Windows wheel. The bench prompts are already spelled out (no digits), so the
  only thing lost is TN on text the bench never sends. Same setting on Linux, so
  the two rigs run identical inputs.
- raw_text stays False (upstream default): text is split into ~50-char fragments
  and trailing silence trimmed per fragment. Upstream's published numbers used
  raw_text=True, but that is the benchmark harness, not the library default. One
  decode step = 3528 samples (882 hop x 4 patch) = 80 ms, i.e. 12.5 steps per
  second of audio, so max_decode_steps=1000 is an 80 s ceiling per fragment --
  no cap risk. The ~13 steps/s decode rate is why it runs at ~1x realtime.
- cache_prompts=False: encoding the reference samples the VAE posterior, so with
  the cache on, run 1 and runs 2-3 draw different noise under the same seed and
  are not the same computation. Re-encoding per call keeps runs reproducible and
  charges the encode to gen_s, like the other cloners.
- Seed 1895 (the upstream CLI default) before every call.
- Non-streaming: ttfa_ms == gen_s (whole clip, not a first-chunk latency).
- Languages: en/fr/es map to the upstream language_id; fr/es go through the
  "multilingual" path, which bypasses the zh/en text frontend.

Tiny (`ming_tiny.MingTinyTTS`, a different API in the same repo):
- Default lens: task="ip", ip="US_James" (same preset as 0.5B).
- Cloning lens: task="seed" (en) or "multilingual" (fr/es, target_language name),
  prompt_wav + prompt_text. `generate(seed=)` is passed as well as torch.manual_seed.
- Returns an output object; the audio is `.waveform`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo

VARIANTS = {"0.5b": "inclusionAI/Ming-3.0-tts-0.5B", "tiny": "inclusionAI/Ming-3.0-tts-tiny"}
TINY_LANG = {"en": "English", "fr": "French", "es": "Spanish", "de": "German",
             "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
DEFAULT_VOICE = "US_James"
SEED = 1895
SUPPORTED_LANGS = {"en", "fr", "es", "de", "ja", "ko", "zh"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for cloning. Sibling .txt transcript required.")
    p.add_argument("--variant", default="0.5b")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1
    lang = (args.language or "en").lower()
    if lang not in SUPPORTED_LANGS:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"language {lang!r} not supported by Ming-3.0-tts"}))
        return 1
    if args.variant not in VARIANTS:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unknown variant {args.variant!r} (have {sorted(VARIANTS)})"}))
        return 1

    ref_wav = ref_text = None
    if args.reference:
        ref_wav = Path(args.reference).resolve()
        txt = ref_wav.with_suffix(".txt")
        if not ref_wav.exists() or not txt.exists():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"reference wav + sibling .txt required: {ref_wav}"}))
            return 1
        try:
            ref_text = txt.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"reading transcript {txt}: {type(e).__name__}: {e}"}))
            return 1

    src = Path(__file__).resolve().parent.parent / "venvs" / "ming" / "src"
    cwd = Path.cwd()  # out paths are relative to the caller, not to src
    try:
        sys.path.insert(0, str(src))
        os.chdir(src)
        import soundfile as sf
        import torch

        device = "cuda:0" if args.device == "cuda" else args.device
        tiny = args.variant == "tiny"
        if tiny:
            from ming_tiny import MingTinyTTS
            tts = MingTinyTTS.from_pretrained(VARIANTS[args.variant], device=device)
        else:
            from ming_tts import MingTTS
            tts = MingTTS(VARIANTS[args.variant], device=device,
                          text_normalization=False, cache_prompts=False)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            torch.manual_seed(SEED)
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            if tiny and ref_wav is None:
                wav = tts.generate(task="ip", text=text, ip=DEFAULT_VOICE, seed=SEED).waveform
            elif tiny:
                wav = tts.generate(task="seed" if lang in ("en", "zh") else "multilingual",
                                   text=text, prompt_wav=str(ref_wav), prompt_text=ref_text,
                                   target_language=TINY_LANG[lang], seed=SEED).waveform
            elif ref_wav is None:
                wav = tts.synthesize(task="IP", text=text, language_id=lang,
                                     controls={"ip": DEFAULT_VOICE}, use_zero_spk_emb=True)
            else:
                wav = tts.synthesize(task="TTS", text=text, language_id=lang,
                                     prompt_wav_path=str(ref_wav), prompt_text=ref_text,
                                     use_spk_emb=True)
            if args.device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            arr = wav.squeeze().float().cpu().numpy()
            audio_s = float(len(arr) / tts.sample_rate)
            if write_wav:
                sf.write(str(cwd / out_path), arr, tts.sample_rate)

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
                text, out = job["text"], job["out"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(json.dumps({"ok": False, "run_index": idx,
                                  "error": f"bad job: {type(e).__name__}: {e}"}), flush=True)
                idx += 1
                continue
            _one(text, out, idx, write_wav=True)
            idx += 1
        return 0

    for i in range(args.runs):
        if not _one(args.text, args.out, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
