"""Maya1 runner (maya-research/maya1, Apache 2.0, expressive voice generation).

DEFAULT-VOICE model: there is NO audio cloning. The voice is steered entirely
by a natural-language DESCRIPTION string (e.g. "A calm adult male narrator
with a neutral American accent"). So can_clone=False and Maya1 is benched in
the default-voice column only. --reference is accepted but ignored.

Architecture (transformers path, cuda/cpu):
    Maya1 is a Llama-style causal LM that emits a flat stream of SNAC codec
    tokens. The prompt format wraps a `<description="...">` tag + text in
    Maya1's special header tokens; generation stops at CODE_END_TOKEN_ID.
    Generated tokens are filtered to the SNAC code range, then regrouped from
    a flat 7-tokens-per-frame stream into SNAC's 3 hierarchical codebooks
    (1 / 2 / 4 codes per frame) before snac_model.decode(). The exact token
    offsets + regrouping come from the maya-research/maya1 model card.

Apple-Silicon path (mps): the transformers + SNAC stack is replaced by MLX via
    mlx-audio loading the 4-bit community port (mlx-community/maya1-4bit).
    Untested on this rig; install needs `mlx-audio` (see install.sh Mac note).

Token constants (from the maya-research/maya1 model card):
    CODE_START_TOKEN_ID = 128257   start-of-speech
    CODE_END_TOKEN_ID   = 128258   end-of-speech (eos for generate)
    CODE_TOKEN_OFFSET   = 128266   subtract to map token -> SNAC code
    SNAC_MIN_ID/MAX_ID  = 128266 / 156937   valid SNAC code-token range
    SNAC_TOKENS_PER_FRAME = 7
    SOH/EOH/SOA = 128259 / 128260 / 128261   header framing
    TEXT_EOT_ID = 128009
"""

import argparse
import json
import sys
import time

import _meminfo


DEFAULT_VOICE_DESC = "A calm, clear adult male narrator with a neutral American accent, professional tone."

# SNAC token constants (maya-research/maya1 model card).
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
TEXT_EOT_ID = 128009

# Generation params (from the maya-research/maya1 model card).
GEN_MAX_NEW_TOKENS = 2048          # from maya1 model card
GEN_MIN_NEW_TOKENS = 28            # from maya1 model card: 4 SNAC frames x 7 slots
GEN_TEMPERATURE = 0.4              # from maya1 model card
GEN_TOP_P = 0.9                    # from maya1 model card
GEN_REPETITION_PENALTY = 1.1       # from maya1 model card


def _build_prompt(tok, description, text):
    """Wrap description + text in Maya1's header/speech framing tokens."""
    soh = tok.decode([SOH_ID])
    eoh = tok.decode([EOH_ID])
    soa = tok.decode([SOA_ID])
    sos = tok.decode([CODE_START_TOKEN_ID])
    eot = tok.decode([TEXT_EOT_ID])
    bos = tok.bos_token
    formatted_text = f'<description="{description}"> {text}'
    return soh + bos + formatted_text + eot + eoh + soa + sos


def _unpack_snac_from_7(snac_tokens):
    """Regroup a flat 7-tokens-per-frame stream into SNAC's 3 hierarchical levels."""
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]
    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]
    l1, l2, l3 = [], [], []
    for i in range(frames):
        slots = snac_tokens[i * 7:(i + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])
    return [l1, l2, l3]


def _decode_snac(snac_model, token_tensor, device):
    """Maya1 generated-token stream -> float32 numpy audio at 24 kHz.

    token_tensor: 1-D tensor of generated token ids (prompt already stripped).
    """
    import torch

    token_ids = token_tensor.tolist()
    # Stop at end-of-speech if present.
    if CODE_END_TOKEN_ID in token_ids:
        token_ids = token_ids[:token_ids.index(CODE_END_TOKEN_ID)]
    # Keep only valid SNAC code tokens.
    snac_tokens = [t for t in token_ids if SNAC_MIN_ID <= t <= SNAC_MAX_ID]
    levels = _unpack_snac_from_7(snac_tokens)
    if not levels[0]:
        import numpy as np
        return np.zeros(0, dtype="float32")
    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
        for level in levels
    ]
    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()
    return audio


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)       # unused (default-voice only)
    p.add_argument("--variant", default=None)         # unused
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
        import soundfile as sf

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        samplerate = 24000
        USE_MLX = (device == "mps")
        if USE_MLX:
            from mlx_audio.tts.utils import load as mlx_load
            mlx_model = mlx_load("mlx-community/maya1-4bit")
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from snac import SNAC
            tok = AutoTokenizer.from_pretrained("maya-research/maya1")
            lm = AutoModelForCausalLM.from_pretrained(
                "maya-research/maya1", torch_dtype=torch.bfloat16,
                device_map=device, low_cpu_mem_usage=True)
            snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        voice_desc = DEFAULT_VOICE_DESC
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            if USE_MLX:
                # NOTE: this MLX prompt framing intentionally differs from the transformers
                # _build_prompt path (no header/speech tokens) and is untested on this rig.
                chunks = [np.asarray(r.audio, dtype="float32")
                          for r in mlx_model.generate(text=f"{voice_desc}\n\n{text}")]
                arr = np.concatenate(chunks).squeeze() if chunks else np.zeros(0, dtype="float32")
            else:
                prompt = _build_prompt(tok, voice_desc, text)
                inputs = tok(prompt, return_tensors="pt").to(device)
                out = lm.generate(
                    **inputs,
                    max_new_tokens=GEN_MAX_NEW_TOKENS,
                    min_new_tokens=GEN_MIN_NEW_TOKENS,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                    repetition_penalty=GEN_REPETITION_PENALTY,
                    do_sample=True,
                    eos_token_id=CODE_END_TOKEN_ID,
                    pad_token_id=tok.pad_token_id or CODE_END_TOKEN_ID,
                )
                gen = out[0, inputs["input_ids"].shape[1]:]
                arr = _decode_snac(snac_model, gen, device)
            t_end = time.perf_counter()

            arr = np.asarray(arr, dtype="float32").squeeze()
            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                # Maya1 is non-streaming, so TTFA == gen_s.
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
