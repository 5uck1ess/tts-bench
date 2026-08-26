"""Breeze TTS 2 runner (RESONIA, INC., BreezeBlue Research and Non-Commercial, 24kHz).

CSM-shaped autoregressive cloner: a Qwen3-flavour `llama-1B` backbone + a `llama-100M`
depth decoder emitting 16 codebooks over the kyutai/mimi codec at a 12.5 Hz frame rate,
decoded to 24 kHz. 3.466B params. en + zh only -> multilingual=False (the French prompt
is not claimed). Three modes upstream: voice clone (ref wav + exact transcript), voice
design (natural-language instruction, no ref), and voice direction (both).

This runner uses the CLONE path only, and the bench treats the model as NO_PRESET_VOICE.
That is a measurement, not a guess: with the shipped generic instruction the no-reference
path does not hold a speaker across prompts (median F0 ranged 114.6-215.1 Hz over the
canonical English prompts at a fixed seed). Pinning a specific maya1-style description
tightens that to 37.7 Hz and makes it seed-stable, but that is still looser than every
default-board row (maya1 8.8 Hz, kokoro 10.5, melotts 30.2), so there is no honest
preset voice to publish. Absent --reference we fall back to DEFAULT_REF, matching the
NO_PRESET_VOICE convention (see docs/known-issues.md) — but pointed at a consent-clean
reference rather than the house one, for the licence reason noted at DEFAULT_REF.

Streaming: genuine. `iter_audio_chunks` is a real generator, so ttfa_ms is a true
time-to-first-chunk, NOT a copy of gen_s. Measured through this runner on Win-5090
(eager, bf16, cloning prompt 2, 3 runs): cold TTFA 3567 ms / warm ~630 ms, warm gen
~15.1 s for 4.56 s of audio -> ~0.30x RTFx. The no-reference path is faster (~365 ms
warm TTFA, ~0.46x RTFx) because it skips reference prefill; the cloning numbers above
are the ones this row actually reports. The card claims <40 ms TTFA / 0.32 RTF on H100
via the `--fast-all` CUDA-graph path; that path is in-repo and would make a
`breeze_tts2_fast` sibling row measurable, but is not what this row runs.

Two install/runtime traps, both load-bearing:

1. INDEXED DEVICE. Anything that reaches `load_runtime` calls `torch.cuda.set_device`,
   which rejects a bare "cuda" with `ValueError: Expected a torch.device with a
   specified index`. harness.py passes "cuda", so we normalise to "cuda:0" for torch
   and keep args.device for _meminfo.

2. FLASH-ATTENTION IS PINNED IN THE CHECKPOINT. config.json sets
   `text_encoder_config.preferred_attn_implementation="flash_attention_2"`, and
   models/breeze.py applies that to the T5Gemma2 text encoder *regardless* of the
   top-level `attn_implementation=` passed to from_pretrained. flash_attn has no
   Windows wheel, so load dies with a hard ImportError. We load the BreezeConfig,
   rewrite that field to "sdpa", and pass config= into from_pretrained.
   models/t5gemma2_compat.py dispatches every non-flash impl through
   `ALL_ATTENTION_FUNCTIONS`, so sdpa is a mathematically equivalent substitute.
   Upstream's README says Linux-only; nothing in the eager path actually is.

Genuinely CUDA-only, though: models/fast_streaming.py:192 hard-gates on
`torch.cuda.is_available()`. Hence devices=["cuda"] in harness.py, no cpu row.

Generation cap: 1500 frames at 12.5 fps = 120 s of audio. The longest canonical prompt
(3, cloned) measured 221 frames / ~17.7 s, so the margin is ~7x. The upstream default of
750 (60 s) would also clear it, but the loop exits at EOS regardless, so the extra cap
costs memory rather than time.

State: the runtime prints "Residual tail decode detected; request must be reset or
closed after this call" and carries per-request state, so we always close the generator
in a finally: block, which triggers its own `close_request`. Verified clean through this
runner: three consecutive --runs produced 4.56 s of audio each, byte-identical duration,
with warm gen_s within 0.25 s. If run 2 or 3 ever drifts, suspect that close.

License note: weights are non-commercial, and §5(c)/(d) prohibit cloning a real person's
voice without their consent, with no research carve-out. Whatever wav is passed as
--reference must be one there is consent to clone.
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import _meminfo


@contextlib.contextmanager
def _stdin_shielded():
    """Point fd 0 at os.devnull for the duration of the block.

    Importing `qwen_tts` DRAINS STDIN — bisected against the other four imports, which
    are all clean. It pulls in `sox`, whose module-level probe spawns a subprocess that
    inherits fd 0 and swallows whatever is queued on it. In --stdin mode the harness has
    already begun feeding jobs by then, so without this the runner prints {"ready": true}
    and then reads EOF: every job silently vanishes and the cell reports no results at
    all rather than an error. Swapping fd 0 (not just sys.stdin) is what makes this hold,
    since the drain happens in a child process, below the Python level.
    """
    saved = os.dup(0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(devnull, 0)
        yield
    finally:
        os.dup2(saved, 0)
        os.close(devnull)
        os.close(saved)
        # sys.stdin still wraps the original fd 0, but its buffer may have been touched;
        # rebuild it so iteration below starts from a clean reader on the restored fd.
        sys.stdin = os.fdopen(0, "r", encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parent.parent
BREEZE_SRC = REPO_ROOT / "venvs" / "breeze_tts2" / "src" / "breeze-tts"
CKPT = REPO_ROOT / "venvs" / "breeze_tts2" / "src" / "Breeze-TTS-2"
# NOT the house chris_hemsworth_15s.wav that every other NO_PRESET_VOICE runner falls
# back to. License §5(c)/(d) forbid cloning a real person without their consent, with no
# research carve-out, so this model must never be pointed at the house reference — and a
# no-reference run is exactly where that would happen silently. tym.wav is a consent-clean
# reference (recorded by the repo owner, of himself). Keep this override if the house
# reference ever changes; it is a licence constraint, not a preference.
DEFAULT_REF = REPO_ROOT / "reference" / "tym.wav"

SAMPLERATE = 24_000
MAX_NEW_TOKENS = 1500   # frames; 12.5 fps -> 120 s. See docstring.
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
SEED = 42

# The ref_edit_tata template requires an `instruction` field even on the clone path
# (omitting it raises "missing template fields: ['instruction']"). Keep it neutral and
# about delivery only: identity must come from the reference, not from this string.
# This is the value the by-ear clone audition was judged on.
CLONE_INSTRUCTION = "Speak clearly and naturally."


def _ref_transcript(ref_wav: Path) -> str:
    """Cloning needs the reference's exact transcript. Bench convention is a sibling
    .txt (reference/foo.wav -> reference/foo.txt), same as sesame/miso."""
    txt = ref_wav.with_suffix(".txt")
    if not txt.exists():
        raise FileNotFoundError(f"reference transcript not found: {txt}")
    return txt.read_text(encoding="utf-8").strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot cloning; needs a sibling .txt transcript.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found: {ref_wav}"}))
        return 1

    try:
        import numpy as np
        import soundfile as sf
        import torch

        if args.device != "cuda":
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"breeze_tts2 is CUDA-only (fast_streaming.py "
                                       f"gates on torch.cuda.is_available); got "
                                       f"{args.device}"}))
            return 1
        if not torch.cuda.is_available():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": "CUDA requested but not available"}))
            return 1

        # Source repo, not a package.
        if str(BREEZE_SRC) not in sys.path:
            sys.path.insert(0, str(BREEZE_SRC))
        from transformers import AutoTokenizer
        from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
        from breeze_infer.templates import get_template, prepare_inputs
        from models.breeze import BreezeForConditionalGeneration
        from models.breeze_config import BreezeConfig
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
        with _stdin_shielded():   # qwen_tts drains stdin on import — see _stdin_shielded
            from qwen_tts import Qwen3TTSTokenizer

        ref_text = _ref_transcript(ref_wav)

        device = "cuda:0"          # MUST be indexed -- see docstring trap 1.
        torch.cuda.set_device(device)

        cfg = BreezeConfig.from_pretrained(CKPT)
        cfg.text_encoder_config.preferred_attn_implementation = "sdpa"  # trap 2.

        tokenizer = AutoTokenizer.from_pretrained(CKPT)
        model = BreezeForConditionalGeneration.from_pretrained(
            CKPT, config=cfg, dtype=torch.bfloat16, attn_implementation="eager")
        model.to(device).eval()
        update_generation_config_for_breeze(model)

        audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(CKPT / "audio_tokenizer"), device_map=device)

        runtime = FastBreezeStreamingRuntime(
            model, audio_tokenizer,
            FastStreamingConfig(
                max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN,
                fast_all=False, fast_text_encoder=False, fast_backbone_prefill=False,
                fast_backbone_decode=False, fast_depth_decoder=False, fast_codec=False,
                repetition_penalty=REPETITION_PENALTY),
            tokenizer=tokenizer)
        template = get_template("ref_edit_tata")
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            _meminfo.reset_peak(args.device)
            req_id = f"bench-{run_index}"
            req = {"id": req_id, "text": text, "speaker": "S0",
                   "instruction": CLONE_INSTRUCTION,
                   "ref_audio_path": str(ref_wav), "ref_text": ref_text}

            set_all_seeds(SEED)
            inputs = prepare_inputs(tokenizer, audio_tokenizer, model, [req], template,
                                    guidance_scale=1.0, guidance_scale_ref=None,
                                    guidance_scale_ins=None)

            chunks, ttfa = [], None
            t0 = time.perf_counter()
            stream = runtime.iter_audio_chunks(inputs, request_id=req_id)
            try:
                for ch in stream:
                    if ttfa is None:
                        ttfa = (time.perf_counter() - t0) * 1000.0
                    chunks.append(ch.audio)
            finally:
                # Carries per-request state; closing runs the generator's own
                # close_request and keeps runs 2/3 uncontaminated. See docstring.
                stream.close()
            t_end = time.perf_counter()

            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            if audio.size == 0:
                raise RuntimeError("model produced no audio chunks")
            if write_wav:
                sf.write(str(out_path), audio, SAMPLERATE, subtype="PCM_16")

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": ttfa if ttfa is not None else (t_end - t0) * 1000.0,
                "gen_s": t_end - t0,
                "audio_s": float(len(audio) / SAMPLERATE),
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
