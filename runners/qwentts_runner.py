"""Qwen3-TTS runner (Alibaba Qwen team, 10 languages).

The 1.7B and 0.6B Base variants are *cloning* models (wav + transcript).
The 0.6B CustomVoice variant uses its checkpoint-provided preset speakers.

API (qwen-tts==latest):
    from qwen_tts import Qwen3TTSModel
    import torch
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",           # or "cpu"
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # skipped on Windows
    )
    wavs, sr = model.generate_voice_clone(
        text="...",
        language="English",            # or "Auto"
        ref_audio=ref_wav_path,
        ref_text=ref_text,
    )

Cloning flavor: wav + transcript (matches NeuTTS, F5-TTS, OmniVoice).
Base has no predefined-voice path, so it is REFERENCE-ONLY: the no-reference
run falls back to the house reference (chris_hemsworth_15s.wav) like every other
no-preset model, and it is listed in NO_PRESET_VOICE (cloning board only).

10 supported languages: zh, en, ja, ko, de, fr, ru, pt, es, it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


# Map our ISO-style language codes to Qwen3-TTS's full names.
LANG_MAP = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}

MODEL_IDS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "base_06b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "custom_06b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}


def _device_map_from(device: str) -> str:
    return {"cuda": "cuda:0", "cpu": "cpu"}.get(device, "cpu")


def _read_ref_transcript(ref_wav: str | None) -> str | None:
    if not ref_wav:
        return None
    txt_path = Path(ref_wav).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


# The 9 CustomVoice timbres carry no language metadata in the checkpoint — the
# names are just names (aiden, dylan, eric, ono_anna, ryan, serena, sohee,
# uncle_fu, vivian), so they can't be matched to a language programmatically.
# These are the bench's picks per language; anything else falls back to the
# checkpoint's own first entry rather than guessing.
DEFAULT_SPEAKERS = {"English": "vivian", "French": "serena"}


def _select_speaker(model, requested: str | None, language: str) -> str:
    if requested:
        return requested
    supported = list(model.get_supported_speakers())
    if not supported:
        raise RuntimeError("CustomVoice checkpoint reported no supported speakers")
    preferred = DEFAULT_SPEAKERS.get(language)
    if preferred in supported:
        return preferred
    return supported[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning. Needs sibling .txt transcript.")
    p.add_argument("--variant", default="base", choices=tuple(MODEL_IDS))
    p.add_argument("--speaker", default=None,
                   help="CustomVoice preset speaker. Defaults to runtime discovery.")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    ref_wav = None
    ref_text = None
    if args.variant != "custom_06b":
        # Default-voice path: borrow a bundled reference so Base (clone-only) can run.
        repo = Path(__file__).resolve().parent.parent
        if args.reference:
            ref_wav = Path(args.reference)
        else:
            default_ref = "chris_hemsworth_15s.wav"
            ref_wav = repo / "reference" / default_ref

        if not ref_wav.exists():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"reference wav not found: {ref_wav}"}))
            return 1

        ref_text = _read_ref_transcript(str(ref_wav))
        if not ref_text:
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"reference transcript missing: {ref_wav.with_suffix('.txt')} "
                                       f"(Qwen3-TTS Base needs wav + matching .txt)"}))
            return 1

    language = LANG_MAP.get(args.language, "Auto")

    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        device_map = _device_map_from(args.device)
        # bf16 on CUDA (Blackwell supports it); float32 on CPU.
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        model = Qwen3TTSModel.from_pretrained(
            MODEL_IDS[args.variant],
            device_map=device_map,
            dtype=dtype,
        )
        speaker = (_select_speaker(model, args.speaker, language)
                   if args.variant == "custom_06b" else None)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            if args.variant == "custom_06b":
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=str(ref_wav),
                    ref_text=ref_text,
                )
            t_end = time.perf_counter()

            # API returns (list_of_wavs, sample_rate); single text → single wav.
            arr = wavs[0] if isinstance(wavs, list) else wavs
            # Convert torch tensor → numpy if needed.
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            import numpy as np
            arr = np.asarray(arr).reshape(-1).astype(np.float32)
            audio_s = float(len(arr) / sr)
            if write_wav:
                sf.write(out_path, arr, sr)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **({"speaker": speaker} if speaker is not None else {}),
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
