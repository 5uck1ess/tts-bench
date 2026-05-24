"""Kokoro-82M runner (Apache 2.0, predefined voices only, no zero-shot cloning).

API discovered by inspection (2026-05-23, kokoro==0.9.4):
    from kokoro import KPipeline
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    for result in pipe(text, voice='af_heart', speed=1):
        result.audio       # torch.Tensor (mono, 24kHz)
        result.graphemes   # str
        result.phonemes    # str

Language codes (single-letter): a=American EN, b=British EN, e=Spanish,
f=French, h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Mandarin.

Voice naming: <lang><gender>_<name>, e.g. af_heart (American Female heart).
Voice list comes from the HF repo at runtime — we pick a sensible default
per language. Caller can override via --reference (treated as voice NAME
string, not a wav path — Kokoro does not support zero-shot cloning).

Install gotcha: misaki (Kokoro's tokenizer) calls spacy.cli.download() at
init to fetch en_core_web_sm. That uses pip under the hood, which fails
in uv venvs (no pip seed). install.ps1 / install.sh pre-install the model
wheel to bypass this.
"""

import argparse
import json
import sys
import time

import _meminfo
import _naq


LANG_CODE = {
    "en": "a",   # American English
    "fr": "f",
    "es": "e",
    "it": "i",
    "pt": "p",
    "de": "a",   # no German in Kokoro — fall back to American EN
    "ja": "j",
    "zh": "z",
    "hi": "h",
}


DEFAULT_VOICE = {
    "en": "af_heart",
    "fr": "ff_siwis",
    "es": "ef_dora",
    "it": "if_sara",
    "pt": "pf_dora",
    "ja": "jf_alpha",
    "zh": "zf_xiaobei",
    "hi": "hf_alpha",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Voice NAME (e.g. 'af_heart') — Kokoro doesn't support zero-shot wav cloning.")
    p.add_argument("--variant", default=None)        # unused
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        from kokoro import KPipeline
        import numpy as np
        import soundfile as sf
        import torch

        lang_code = LANG_CODE.get(args.language, "a")
        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        pipe = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=device)
        voice = args.reference or DEFAULT_VOICE.get(args.language, "af_heart")
        samplerate = 24000  # Kokoro fixed output rate
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
            for r in pipe(text, voice=voice, speed=1):
                if r.audio is None:
                    continue
                if first is None:
                    first = time.perf_counter()
                arr = r.audio.cpu().numpy() if hasattr(r.audio, "cpu") else np.asarray(r.audio)
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
