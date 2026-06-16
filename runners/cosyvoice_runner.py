"""CosyVoice 3 runner (FunAudioLLM, Apache-2.0, 0.5B LLM-TTS + flow-matching, 24 kHz).

Fun-CosyVoice3-0.5B-2512: a Qwen-based speech LM that predicts supervised semantic
tokens, refined by a flow-matching decoder + HiFi vocoder to 24 kHz. Successor to
CosyVoice 2 (better content consistency, speaker similarity, prosody); zero-shot
multilingual synthesis in the wild. Source-clone import (venvs/cosyvoice/src is NOT
a pip package): the runner adds that tree + its third_party/Matcha-TTS to sys.path
and uses AutoModel(), which snapshot_downloads the checkpoint (modelscope) and
dispatches to the CosyVoice3 class. Weights cache on first run (~1-2 GB).

PURE CLONING (no preset voice): every synthesis needs a reference wav + its literal
transcript (read from a sibling .txt, like sesame/miso/longcat). can_clone=True ->
both lenses:
  * default (no --reference): the house voice, reference/chris_hemsworth_15s.wav.
  * cloning (--reference wav + sibling .txt): the supplied voice.
The reference transcript is prepended (after the v3 system prompt) so the model has
the literal words of the prompt audio. Multilingual (ZH/EN/JA/KO/yue + more) ->
multilingual=True, the FR prompt runs. CUDA-only (fp16 flow/LLM). License: Apache-2.0.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
COSY_SRC = REPO_ROOT / "venvs" / "cosyvoice" / "src"
MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"
# v3 expects the reference transcript behind its system prompt (see example.py).
SYS_PROMPT = "You are a helpful assistant.<|endofprompt|>"


def _read_ref_transcript(ref_wav: Path) -> str | None:
    txt_path = ref_wav.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for cloning. Sibling .txt transcript required "
                        "(prepended to the prompt, like sesame/miso/longcat).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    if args.device != "cuda":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"cosyvoice is CUDA-only here; device {args.device!r} unsupported"}))
        return 1

    # Resolve output to absolute BEFORE any chdir so a relative --out still lands
    # in the bench run dir (CosyVoice's frontend loads some resources relative to
    # the source tree, so we chdir there).
    out_abs = str(Path(args.out).resolve()) if args.out else None
    ref_wav = Path(args.reference).resolve() if args.reference else DEFAULT_REF
    ref_text = _read_ref_transcript(ref_wav)
    if not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference {ref_wav} needs a sibling .txt transcript "
                                   f"(cloning prepends the reference's literal words)"}))
        return 1
    prompt_text = SYS_PROMPT + ref_text

    try:
        sys.path.insert(0, str(COSY_SRC))
        sys.path.insert(0, str(COSY_SRC / "third_party" / "Matcha-TTS"))
        os.chdir(COSY_SRC)

        import soundfile as sf
        from cosyvoice.cli.cosyvoice import AutoModel

        model = AutoModel(model_dir=MODEL_ID, fp16=True)
        sr = model.sample_rate
        # The frontend calls load_wav(prompt_wav, ...) itself (at both 16 k and
        # 24 k), so prompt_wav must be a file PATH, not a pre-loaded tensor.
        prompt_wav = str(ref_wav)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            out = None
            for j in model.inference_zero_shot(text, prompt_text, prompt_wav, stream=False):
                out = j["tts_speech"]
                break
            t_end = time.perf_counter()
            if out is None:
                raise RuntimeError("inference_zero_shot yielded no audio")

            arr = out.squeeze().detach().cpu().float().numpy().reshape(-1)
            audio_s = float(len(arr) / sr)
            if write_wav:
                sf.write(out_path, arr, sr)

            # Non-streaming flow-matching: no partial audio, so TTFA == full gen.
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
        if not _one(args.text, out_abs, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
