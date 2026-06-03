"""Parler-TTS runner (parler-tts/*, Apache 2.0, description-controlled TTS).

DEFAULT-VOICE model: there is NO audio cloning. The voice is steered entirely
by a natural-language DESCRIPTION string (speaker gender, pitch, pace, recording
quality, reverb). So can_clone=False and Parler is benched in the default-voice
column only. --reference is accepted but ignored.

Variants (--variant):
    None / "mini"  -> parler-tts/parler-tts-mini-v1   (878M, English)
    "large"        -> parler-tts/parler-tts-large-v1  (2.33B, English)
Both share this runner and the same API; large is just a bigger checkpoint.

Architecture (transformers path, cuda/cpu):
    Parler is an encoder-decoder: a frozen T5 text encoder consumes the
    DESCRIPTION, a decoder LM consumes the PROMPT (the words to speak) and
    cross-attends to the description, emitting DAC (44.1 kHz) codec tokens that
    are decoded to a waveform. `model.generate(input_ids=<desc>,
    prompt_input_ids=<text>)` returns the waveform directly.

The voice that a description maps to is sampled, so we set a fixed torch seed
before each generate() to keep "the default voice" stable across prompts/runs
(otherwise the same description can drift speaker between cells).
"""

import argparse
import json
import sys
import time

import _meminfo


# A single fixed description = our "default voice" for the bench. Clear, close,
# neutral — the analogue of every other model's preset voice.
DEFAULT_DESCRIPTION = (
    "A clear adult male narrator with a neutral accent delivers his words at a "
    "moderate, even pace. The recording is of very high quality, with his voice "
    "sounding crisp and very close up, with no background noise."
)

# Fixed seed so the description maps to a stable speaker across prompts/runs.
VOICE_SEED = 0

VARIANT_REPO = {
    None:    "parler-tts/parler-tts-mini-v1",
    "mini":  "parler-tts/parler-tts-mini-v1",
    "large": "parler-tts/parler-tts-large-v1",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)       # unused (default-voice only)
    p.add_argument("--variant", default=None)         # None/"mini" or "large"
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    repo = VARIANT_REPO.get(args.variant)
    if repo is None:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unknown variant {args.variant!r}"}))
        return 1

    try:
        import numpy as np  # noqa: F401  (kept for parity / squeeze paths)
        import soundfile as sf
        import torch
        from transformers import AutoTokenizer
        from parler_tts import ParlerTTSForConditionalGeneration

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo, torch_dtype=dtype).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(repo)
        samplerate = int(model.config.sampling_rate)

        desc_ids = tokenizer(DEFAULT_DESCRIPTION, return_tensors="pt").input_ids.to(device)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            torch.manual_seed(VOICE_SEED)
            t0 = time.perf_counter()
            with torch.inference_mode():
                gen = model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids)
            arr = gen.to(torch.float32).cpu().numpy().squeeze()
            t_end = time.perf_counter()

            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                # Parler is non-streaming, so TTFA == gen_s.
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
