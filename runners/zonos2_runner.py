"""Zonos2 runner (Zyphra, Apache-2.0, zero-shot cloning, 44.1kHz) — CUDA/Linux only.

DRAFT STATUS: the API surface below is confirmed against upstream at commit
194c0a3 (README quickstart + python/zonos2/tts/__init__.py + docs/tts_architecture.md),
but this runner has NOT yet been executed on-device. Zonos2 ships compiled CUDA
kernels (python/zonos2/kernel/csrc) and is Linux + x86_64 + NVIDIA-CUDA only, so
it cannot run on the Windows dev box — VALIDATE ON THE LINUX-3090 (install +
smoke with the canonical bench prompts, incl. the long prompt 3) before publish.

API (Zyphra/ZONOS2) — from upstream README + python/zonos2/tts/__init__.py:
    from zonos2.tts import TTSLLM               # tts/__init__.py -> from .llm import TTSLLM
    from zonos2.message import TTSSamplingParams
    tts = TTSLLM(model_path="Zyphra/ZONOS2")    # loads to CUDA (no CPU path)
    emb = tts.embed_speaker_file(ref_wav)       # zero-shot clone from ANY audio file
    result = tts.generate_one(
        text,
        TTSSamplingParams(seed=0),
        speaker_embedding=emb,                  # also: clean_speaker_background=, accurate_mode=
        language="en_us",
    )
    tts.save_audio(result["audio"], out_wav)    # DAC vocoder -> 44.1 kHz mono

Architecture: byte-level UTF-8 text -> AR transformer predicting 9 parallel audio
codebooks -> DAC vocoder @ 44.1 kHz. Text normalization is a vendored NeMo stack
(no espeak/phonemizer, unlike Zonos v0.1). Vendored normalizers cover en/de/zh
but NOT fr -> multilingual=False in harness.py (French prompt skipped).

Pure zero-shot cloner: with no --reference we clone the bundled bench default
(chris_hemsworth_15s.wav), same convention as zonos/echo/cosyvoice -> this model
belongs in the NO_PRESET_VOICE lists (cloning board only).

Non-streaming here: generate_one() blocks and returns the full result, so
ttfa_ms == gen_s (same convention as zonos_runner).

VERIFIED on Linux-3090 (2026-07-21, first on-device run):
  - TTSLLM(model_path="Zyphra/ZONOS2") auto-selects CUDA — no device kwarg. OK.
  - result["audio"] is raw PCM *bytes* (float32 @ 44.1 kHz), NOT a tensor/array
    (upstream llm.py comment + save_audio np.frombuffer). _audio_len() handles
    bytes (len//4); the earlier tensor/array assumption undercounted audio_s to
    ~1 sample (RTFx ~0). FIXED.
  - TTSLLM DOES spin up an internal sglang-style engine (Gloo rank-0, paged KV
    cache, CUDA-graph capture). It sizes the KV cache to fill *free* VRAM, so it
    grabs ~21 GB of the 24 GB card and reports whole-GPU peak_vram_mb (~21.8 GB,
    process-local VRAM is blind — same convention as orpheus/vLLM). Runs single
    process here; the --stdin persistent loop reuses the one warm engine. Free
    the GPU of other large allocations before benching or the KV cache shrinks.
  - Long prompts: TWO defaults truncate/OOM long-form on the 24 GB 3090, both
    fixed below. (a) generate_one defaults to max_tokens=1024 frames (~11.9 s at
    ~86 fps) and does NOT auto-raise to the model limit like the HTTP server —
    bench prompt 3 (~16 s) cut off at exactly 11.865 s mid-sentence. Fixed by
    passing the model's real cap via resolve_max_tokens(). (b) With the cap
    lifted, the default memory_ratio=0.9 leaves only ~2.1 GB free and the DAC
    vocoder then OOMs decoding the full clip. Fixed with memory_ratio=0.8
    (~4.5 GB headroom). With both, prompt 3 renders complete at ~14.4 s.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import _meminfo

SR = 44100  # DAC vocoder output; confirmed in docs/tts_architecture.md


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")   # CUDA-only upstream (compiled kernels)
    p.add_argument("--reference", default=None,
                   help="Reference audio for zero-shot cloning (no transcript needed).")
    p.add_argument("--variant", default=None)     # unused (single checkpoint)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        import numpy as np
        from zonos2.tts import TTSLLM
        from zonos2.message import TTSSamplingParams

        # CUDA-only: no CPU/MPS backend (compiled CUDA kernels + MoE). The harness
        # only schedules this model on cuda cells, so a non-cuda --device is a
        # config error rather than a fallback.
        if args.device != "cuda":
            raise RuntimeError(f"zonos2 is CUDA-only; got device={args.device!r}")

        # memory_ratio (vLLM-style gpu-util knob, default 0.9) caps total engine
        # VRAM at memory_ratio * free_before, leaving (1 - memory_ratio) * free_before
        # as headroom. At 0.9 on the 24 GB 3090 that headroom is only ~2.1 GB — the
        # DAC vocoder decoding a long (~16 s) clip then OOMs mid-generation (bench
        # prompt 3 with the max_tokens cap lifted). Drop to 0.8 (~4.5 GB free); the
        # KV pool (page_size=1) stays tens of thousands of pages, far above any
        # bench prompt's frame count, so max_seq_len / capacity is unaffected.
        tts = TTSLLM(model_path="Zyphra/ZONOS2", memory_ratio=0.8)

        repo = Path(__file__).resolve().parent.parent
        ref = args.reference or str(repo / "reference" / "chris_hemsworth_15s.wav")
        if not Path(ref).exists():
            raise FileNotFoundError(f"Voice reference not found: {ref}")
        emb = tts.embed_speaker_file(ref)

        # generate_one defaults to TTSSamplingParams.max_tokens=1024 audio frames
        # (~11.9 s at the 44.1 kHz DAC's ~86 fps) — the in-process path does NOT
        # auto-raise it to the model limit the way the HTTP server does. That
        # truncates any prompt whose speech runs past ~11.9 s: on-device, bench
        # prompt 3 (the ~16 s Parakeet paragraph) cut off at exactly 11.865 s,
        # mid-sentence, inflating its WER. Resolve the model's real ceiling
        # (config.max_seq_len) and pass it explicitly; resolve_max_tokens() clamps
        # a huge sentinel down to that ceiling, and the scheduler further clamps
        # per-request to max_seq_len - input_len (server parity). Short prompts
        # still stop early at EOS, so this only lifts the artificial 1024 cap.
        MAX_TOKENS = tts.resolve_max_tokens(1 << 30)

        # Vendored NeMo normalizer covers en/de/zh (no fr). The harness only feeds
        # this model English (multilingual=False), so en_us is the live path; the
        # rest are best-effort and unverified.
        LANG = {"en": "en_us", "de": "de_de", "zh": "zh_cn"}.get(args.language, "en_us")
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _audio_len(audio) -> int:
        """Sample count from result['audio'].

        Confirmed on-device (Linux-3090): generate_one() returns the waveform as
        raw PCM *bytes* (float32 @ 44.1 kHz), NOT a tensor/array — see upstream
        llm.py `# r["audio"] is PCM bytes` and save_audio()'s
        np.frombuffer(audio_bytes, dtype=np.float32). float32 = 4 bytes/sample.
        """
        if isinstance(audio, (bytes, bytearray, memoryview)):
            return len(bytes(audio)) // 4
        # Defensive fallback if a future upstream returns a tensor/array instead.
        try:
            import torch
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
        except Exception:
            pass
        arr = np.asarray(audio).squeeze()
        return int(arr.shape[-1]) if arr.ndim else int(arr.size)

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            result = tts.generate_one(
                text,
                TTSSamplingParams(seed=0),   # fixed seed: reproducible across warm runs
                speaker_embedding=emb,
                language=LANG,
                max_tokens=MAX_TOKENS,       # model's real cap, not the 1024 default (see above)
            )
            audio = result["audio"]
            t_end = time.perf_counter()

            audio_s = float(_audio_len(audio)) / SR
            if write_wav:
                tts.save_audio(audio, out_path)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,   # non-streaming: TTFA == gen
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
