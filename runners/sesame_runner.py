"""Sesame CSM-1B runner (conversational speech model, 1B Llama backbone + Mimi codec).

Cloning paradigm is different from wav+txt or wav-only:
CSM is conversation-aware and "sounds best when provided with context".
For cloning, we pass the reference (text + audio) as a prior turn from
speaker "0", then ask the model to continue with our target text as a
new turn from the SAME speaker. CSM picks up the voice from the audio
in the prior turn.

Default-voice mode uses speaker id "[0]" with no context — sounds OK but
the voice is generic; cloning mode with a real reference is much better.

HF-gated: requires `hf auth login` + accepting terms at
https://huggingface.co/sesame/csm-1b. Runner errors clearly if not done.

API (transformers >= 4.52.1):
    from transformers import CsmForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained("sesame/csm-1b")
    model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b",
                                                        device_map="cuda")

    # Default voice (no context):
    conversation = [{"role": "0",
                     "content": [{"type": "text", "text": "Hello."}]}]

    # Cloning (with reference as prior turn):
    conversation = [
        {"role": "0", "content": [
            {"type": "text",  "text": ref_text},
            {"type": "audio", "path": ref_audio_path},     # or numpy array
        ]},
        {"role": "0", "content": [{"type": "text", "text": target_text}]},
    ]

    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True,
    ).to(device)
    audio = model.generate(**inputs, output_audio=True)
    processor.save_audio(audio, out_path)

Output sample rate is 24 kHz (Mimi codec native).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo
import _naq


SAMPLE_RATE = 24000  # Mimi codec
MODEL_ID = "sesame/csm-1b"


def _read_ref_transcript(ref_wav: str | None) -> str | None:
    if not ref_wav:
        return None
    txt_path = Path(ref_wav).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for cloning. Sibling .txt transcript required (passed as prior-turn context to CSM).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Cloning requires the transcript of the reference wav. Default-voice mode
    # (no --reference) just uses speaker [0] with no context.
    ref_wav = Path(args.reference) if args.reference else None
    ref_text = _read_ref_transcript(args.reference) if args.reference else None
    if args.reference and not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference {args.reference} provided but sibling .txt transcript missing "
                                   f"(CSM cloning needs the literal words spoken in the wav)"}))
        return 1

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import CsmForConditionalGeneration, AutoProcessor

        device = args.device if args.device in ("cuda", "cpu") else "cpu"

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = CsmForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        # HF gating hint
        if "401" in err or "403" in err or "gated" in err.lower() or "access" in err.lower():
            err = (f"HF-gated model. Accept terms at https://huggingface.co/sesame/csm-1b "
                   f"and run `hf auth login`. Original: {err}")
        print(json.dumps({"ok": False, "run_index": 0, "error": f"load failed: {err}"}))
        return 1

    def _build_conversation(text):
        if ref_wav and ref_text:
            return [
                {"role": "0", "content": [
                    {"type": "text", "text": ref_text},
                    {"type": "audio", "path": str(ref_wav)},
                ]},
                {"role": "0", "content": [{"type": "text", "text": text}]},
            ]
        return [{"role": "0", "content": [{"type": "text", "text": text}]}]

    def _one(text, out_path, run_index, write_wav):
        try:
            conversation = _build_conversation(text)

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            inputs = processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True,
            ).to(device)
            # Audio reference comes back as float32; cast to model dtype so
            # the bfloat16 path on CUDA doesn't trip "Input type (float) and
            # bias type (struct c10::BFloat16) should be the same".
            for k in ("input_values", "audio_values"):
                if k in inputs and inputs[k] is not None and inputs[k].dtype != model.dtype:
                    inputs[k] = inputs[k].to(model.dtype)
            audio = model.generate(**inputs, output_audio=True)
            t_end = time.perf_counter()

            # model.generate(output_audio=True) returns a list of tensors;
            # take the first sample (we generated a batch of 1).
            wav = audio[0] if isinstance(audio, (list, tuple)) else audio
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu().float().numpy()
            arr = np.asarray(wav).reshape(-1).astype(np.float32)
            audio_s = float(len(arr) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, arr, SAMPLE_RATE)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else {"naq": None, "naq_harm": None, "naq_buzz": None}),
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
