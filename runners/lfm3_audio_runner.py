"""LFM3-Audio-2B runner (Liquid AI, LFM Open License v1.0) — single-process.

LFM3-Audio-2B is an end-to-end audio foundation model (ASR + TTS + voice chat + tool
calls), NOT a dedicated TTS. We bench its TTS mode only: system prompt
`Perform TTS in <language> using a <gender> voice.`, assistant turn prefilled with
`<|audio_start|>`, then `model.generate()` returns `audio_codes` (12 codebooks per 80 ms
frame, 12.5 frames/s) that the bundled LiquidCodec decodes to 24 kHz.

A new row, not a replacement for `lfm2_audio`: different backbone (LFM3-1.7B), codec
(LiquidCodec vs Mimi) and decoding (parallel vs sequential audio-token prediction).

PREDEFINED-VOICE, not a cloner: the only voice control is female/male (+ American vs
British English) in the system prompt; the card has no reference-audio path. So
can_clone=False (default board); --reference accepts a voice key from VOICES, and a
--reference that points at a wav fails cleanly.

Gotchas:
- Generation cap. The card's TTS example uses max_new_tokens=256, which counts audio
  frames (prompt-3 seq grew 83 -> 317 tokens for 234 frames). 256 frames = 20.5 s, and
  canonical prompt 3 measured 234 frames (18.7 s) — 22 frames of margin. We use 2048;
  generation still stops at the model's own end-of-audio, so it costs nothing.
- Sampling. "female voice" is sampled, not a fixed speaker embedding: a different seed
  gives a different take (seed 0 vs 1 differ everywhere on prompt 2), the same seed is
  bit-identical. We reseed to SEED before every generate so runs 1-3 match and reruns
  reproduce. Sampling params are the card's recommended TTS settings.
- Non-streaming here: ttfa_ms == gen_s (model.generate returns all frames at once; the
  card's audio_streamer path is not used), so don't read ttfa as a streaming latency.
- Languages. Each bench language maps to a card-listed locale; an unknown code fails
  the cell instead of silently speaking English. fr/es measured WER 0.0 (Whisper-v3)
  on bench prompts 5/6 at revision below.
- trust_remote_code: transformers pinned ==5.20.2 (card floor) and the checkpoint
  pinned to REVISION — the remote modeling code is re-downloaded on any upstream push
  otherwise.

Resolved HF revision (2026-10-27): 9d49701c372d6011d900f6abb40c6b296fb706dd.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


HF_REPO = "LiquidAI/LFM3-Audio-2B"
REVISION = "9d49701c372d6011d900f6abb40c6b296fb706dd"
SAMPLE_RATE = 24_000
MAX_NEW_TOKENS = 2048   # audio frames at 12.5/s (~164 s); stops at end-of-audio
SEED = 0

LOCALES = {"en": "American English", "fr": "French", "es": "Spanish"}
VOICES = {"female": ("female", None), "male": ("male", None),
          "uk_female": ("female", "British English"), "uk_male": ("male", "British English")}
DEFAULT_VOICE_KEY = "female"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help=f"Voice key ({'/'.join(VOICES)}). LFM3-Audio has no wav cloning.")
    p.add_argument("--variant", default=None)        # unused (single checkpoint)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    voice_key = DEFAULT_VOICE_KEY
    if args.reference:
        if args.reference in VOICES:
            voice_key = args.reference
        elif Path(args.reference).exists():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": "LFM3-Audio has no voice cloning; female/male preset "
                                       "voices only (default board)."}))
            return 1
    if args.language not in LOCALES:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unsupported language {args.language!r}"}))
        return 1
    gender, locale_override = VOICES[voice_key]
    locale = locale_override if (locale_override and args.language == "en") else LOCALES[args.language]
    system_prompt = f"Perform TTS in {locale} using a {gender} voice."

    try:
        import torch
        import soundfile as sf
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForMultimodalLM.from_pretrained(
            HF_REPO, revision=REVISION, dtype=dtype, device_map=device,
            trust_remote_code=True).eval()
        processor = AutoProcessor.from_pretrained(HF_REPO, revision=REVISION,
                                                  trust_remote_code=True)
        codec = processor.audio_tokenizer.to(device=device).eval()
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
                {"role": "assistant", "content": "<|audio_start|>"},
            ]
            torch.manual_seed(SEED)
            t0 = time.perf_counter()
            inputs = processor.apply_chat_template(
                conversation, continue_final_message=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    audio_temperature=1.0, audio_top_k=128,
                    audio_repetition_penalty=1.1, audio_repetition_window=4096,
                    return_dict_in_generate=True)
                n = int(out.audio_lengths[0])
                if n < 1:
                    print(json.dumps({"ok": False, "run_index": run_index,
                                      "error": "no audio produced"}), flush=True)
                    return False
                waveform = codec.decode(out.audio_codes[:1, :, :n]).audio_values[0]
            audio = waveform.float().cpu().numpy().reshape(-1)
            gen_s = time.perf_counter() - t0
            if n >= MAX_NEW_TOKENS:
                print(json.dumps({"ok": False, "run_index": run_index,
                                  "error": f"hit generation cap ({n} frames)"}), flush=True)
                return False

            if write_wav:
                sf.write(out_path, audio, SAMPLE_RATE)
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": gen_s * 1000,   # non-streaming: ttfa == gen_s
                "gen_s": gen_s, "audio_s": float(len(audio) / SAMPLE_RATE),
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
            _one(job["text"], job["out"], idx, write_wav=True)
            idx += 1
        return 0

    for i in range(args.runs):
        if not _one(args.text, args.out, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
