"""LFM2.5-Audio-1.5B runner (Liquid AI, LFM Open License v1.0) — single-process.

LFM2.5-Audio is an end-to-end multimodal speech<->text foundation model (ASR + TTS +
speech-to-speech), NOT a dedicated TTS. We bench its TTS mode only: the *sequential*
generation routine (`generate_sequential`) emits text then audio tokens; the 8-entry
audio tokens (8 Mimi codebooks) are stacked and `processor.decode`d to a 24 kHz wav.

It is a PREDEFINED-VOICE model, not a cloner — TTS offers exactly four voices selected by
system prompt (US/UK x male/female); a custom voice needs finetuning, not a reference clip.
So can_clone=False (default board); --reference is a voice-key override (us_female/us_male/
uk_female/uk_male), and a --reference that points at a wav fails cleanly (no cloning).

Architecture: pretrained LFM2.5-1.2B backbone + FastConformer audio encoder (nvidia
canary-180m-flash) + RQ-transformer + LFM2-based Mimi-compatible detokenizer. bf16.
English only. Install: `pip install liquid-audio` (Python >=3.12; flash-attn optional —
torch-SDPA fallback). License: package+weights LFM Open License v1.0; bundled canary
encoder + Mimi weights CC-BY-4.0, Mimi code MIT (attribution).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


VOICES = {
    "us_female": "Perform TTS. Use the US female voice.",
    "us_male":   "Perform TTS. Use the US male voice.",
    "uk_female": "Perform TTS. Use the UK female voice.",
    "uk_male":   "Perform TTS. Use the UK male voice.",
}
DEFAULT_VOICE_KEY = "us_female"

HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"
SAMPLE_RATE = 24_000
MAX_NEW_TOKENS = 1024  # generous cap; the model stops on its own via modality tokens


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Voice key (us_female/us_male/uk_female/uk_male). LFM has no wav cloning.")
    p.add_argument("--variant", default=None)        # unused (single checkpoint)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")       # English-only model; kept for parity
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Resolve the voice. A --reference that is an existing file = a cloning request, which
    # LFM cannot satisfy -> fail cleanly so the cloning cell skips (this is a default model).
    voice_key = DEFAULT_VOICE_KEY
    if args.reference:
        if args.reference in VOICES:
            voice_key = args.reference
        elif Path(args.reference).exists():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": "LFM2.5-Audio has no voice cloning; it offers 4 preset "
                                       "voices only (default board)."}))
            return 1
        # else: unknown string -> fall back to default voice
    system_prompt = VOICES[voice_key]

    try:
        import torch
        import soundfile as sf
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
        model = LFM2AudioModel.from_pretrained(HF_REPO).eval()
        if device != "cpu":
            model = model.to(device)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            import torch

            _meminfo.reset_peak(args.device)
            chat = ChatState(processor)
            chat.new_turn("system")
            chat.add_text(system_prompt)
            chat.end_turn()
            chat.new_turn("user")
            chat.add_text(text)
            chat.end_turn()
            chat.new_turn("assistant")

            t0 = time.perf_counter()
            first = None
            audio_out = []
            for t in model.generate_sequential(**chat, max_new_tokens=MAX_NEW_TOKENS,
                                               audio_temperature=0.8, audio_top_k=64):
                if t.numel() > 1:                     # 8-entry = audio token (vs 1-entry text)
                    if first is None:
                        first = time.perf_counter()   # time-to-first-audio-token (TTFA proxy)
                    audio_out.append(t)

            if len(audio_out) < 2:
                print(json.dumps({"ok": False, "run_index": run_index,
                                  "error": "no audio produced (empty sequential generation)"}),
                      flush=True)
                return False

            audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
            waveform = processor.decode(audio_codes)
            audio = waveform.cpu()[0].float().numpy().reshape(-1)
            t_end = time.perf_counter()

            audio_s = float(len(audio) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, audio, SAMPLE_RATE)

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
