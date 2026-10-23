"""AuroraTTS-100K runner (Apache-2.0, single LJSpeech voice, English-only, no cloning).

95,638 inference parameters, trained from scratch on LJSpeech-1.1. No neural
vocoder: the network predicts log-F0, voicing and harmonic/noise spectral
envelopes, and a 145-parameter differentiable harmonic+noise DSP synthesizer
renders 22.05 kHz audio.

The model ships as transformers ``trust_remote_code`` modeling code, so the
checkpoint is pinned by ``revision`` (commit ac60c05, 2026-10-22, the release
commit, reviewed before first run). A free resolve would silently pick up any
later code push. ``transformers`` is pinned to 4.57.1 in the install scripts.

FP32 weights (``variant=None``), matching the card's speed table. The fp16 and
int8 files are storage variants; int8 is dequantised to fp32 at load anyway.

The noise branch of the DSP draws ``torch.randn`` on every call, so the seed is
re-pinned before each run; with it, reruns are bit-identical. The first
synthesis initialises g2p_en (and calls ``nltk.download`` for its corpora,
which is a no-op once cached), so a throwaway warm-up call runs during load
to keep that cost out of run 0.

The API is non-streaming and returns the complete waveform, so TTFA == gen_s.
``--reference`` is accepted for harness compatibility and ignored.
"""

import argparse
import json
import sys
import time

import _meminfo


REPO = "AuroraAI-Research/AuroraTTS-100K"
REVISION = "ac60c05487c9853c1c8ffb7254aac2e12f4690c0"
SEED = 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — AuroraTTS-100K is a single-speaker model and cannot clone.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    if args.language != "en":
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"AuroraTTS-100K is English-only, got language={args.language}"}))
        return 1

    try:
        import soundfile as sf
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(REPO, revision=REVISION,
                                          trust_remote_code=True).eval().to(args.device)
        torch.manual_seed(SEED)
        model.synthesize("Warm up.")
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            torch.manual_seed(SEED)
            t0 = time.perf_counter()
            wav, sr = model.synthesize(text)
            t_end = time.perf_counter()

            audio_s = float(len(wav) / sr)
            if write_wav:
                sf.write(out_path, wav, sr)

            # Non-streaming: TTFA = gen_s (no audio until the call returns).
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
        if not _one(args.text, args.out, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
