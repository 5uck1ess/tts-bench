"""Orpheus-TTS runner (Canopy Labs, Apache-2.0, Llama-3B backbone + SNAC codec, 24 kHz).

A 3B Llama-architecture speech LM that emits SNAC audio codec tokens, decoded to
24 kHz PCM. Streaming by design (~200 ms to first audio), so TTFA is measured at
the first decoded chunk, not the full generation. Served through vLLM: the
`orpheus-speech` pip package wraps an `AsyncLLMEngine` around the checkpoint and
streams tokens into the SNAC decoder. vLLM is CUDA-only here (no CPU/MPS path) and
co-resolves a modern stack (vllm 0.23 / torch 2.11) — the v1 engine spawns its
EngineCore in a child process, which is why this runner must stay guarded behind
`if __name__ == "__main__"` (the spawned child re-imports the module).

PRESET-VOICE ONLY: the OrpheusModel wrapper exposes named voices (tara, leah,
...), not wav cloning, so can_clone=False in harness.py and a cloning run skips it
(harness build_cells drops can_clone=False cells when a reference is set). English
only -> multilingual=False, the FR prompt is skipped.

Weights auto-download from HF on first run (gated repo canopylabs/orpheus-3b-0.1-ft
~6.5 GB; accept the license once on the model page). License: Apache-2.0.
"""

import argparse
import json
import sys
import time

import _meminfo


MODEL_ID = "canopylabs/orpheus-3b-0.1-ft"
VOICE = "tara"          # canonical Orpheus voice (README default)
SAMPLE_RATE = 24000     # SNAC decoder output


def _gpu_used_mb():
    """Coarse whole-GPU memory.used via nvidia-smi. vLLM runs the model in a spawned
    EngineCore subprocess, so this process's torch.cuda peak is meaningless (~100 MB);
    report the whole-GPU figure for rough sizing instead (like higgs_v3), NOT a peak."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Unused: Orpheus is preset-voice only (can_clone=False).")
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
                          "error": f"orpheus is CUDA-only (vLLM); device {args.device!r} unsupported"}))
        return 1

    try:
        import asyncio
        import queue as _queue
        import threading

        import numpy as np
        import soundfile as sf
        import torch
        from orpheus_tts import OrpheusModel
        from orpheus_tts.decoder import tokens_decoder_sync
        from vllm import SamplingParams

        model = OrpheusModel(model_name=MODEL_ID, dtype=torch.bfloat16)
        engine = model.engine

        # vLLM's AsyncLLMEngine binds its output handler to the first event loop
        # it runs on. OrpheusModel.generate_speech calls asyncio.run() PER
        # generation (a fresh loop each time), so the 2nd call (our warm run)
        # hangs forever and leaks the engine subprocess. Drive the engine on ONE
        # persistent loop in a background thread, reused across cold+warm runs.
        # The SNAC token->audio decoder (tokens_decoder_sync) is vLLM-independent,
        # so we keep it as-is.
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    # Orpheus sampling defaults (engine_class.generate_tokens_sync).
    _SAMPLING = dict(temperature=0.6, top_p=0.8, max_tokens=1200,
                     stop_token_ids=[49158], repetition_penalty=1.3)

    def _token_gen(text, request_id):
        """Yield token strings from the engine via the persistent loop."""
        prompt_string = model._format_prompt(text, voice=VOICE)
        sp = SamplingParams(**_SAMPLING)
        q = _queue.Queue()

        async def _produce():
            try:
                async for result in engine.generate(prompt=prompt_string,
                                                     sampling_params=sp,
                                                     request_id=request_id):
                    q.put(result.outputs[0].text)
            finally:
                q.put(None)

        fut = asyncio.run_coroutine_threadsafe(_produce(), _loop)
        while True:
            tok = q.get()
            if tok is None:
                break
            yield tok
        fut.result()   # surface any exception raised inside the coroutine

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            ttfa_ms = None
            chunks = []
            # tokens_decoder_sync streams raw 16-bit PCM byte chunks from the SNAC
            # decoder; the first chunk is real time-to-first-audio.
            for chunk in tokens_decoder_sync(_token_gen(text, f"req-{run_index}")):
                if ttfa_ms is None:
                    ttfa_ms = (time.perf_counter() - t0) * 1000
                chunks.append(chunk)
            t_end = time.perf_counter()

            frames = b"".join(chunks)
            arr = np.frombuffer(frames, dtype=np.int16)
            audio_s = float(len(arr) / SAMPLE_RATE)
            if write_wav:
                sf.write(out_path, arr, SAMPLE_RATE)

            mem = _meminfo.sample(args.device)
            mem["peak_vram_mb"] = None          # real VRAM is in the EngineCore subprocess
            mem["gpu_used_mb"] = _gpu_used_mb()  # whole-GPU, coarse (not a per-call peak)
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": ttfa_ms if ttfa_ms is not None else (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **mem,
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
