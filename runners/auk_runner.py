"""AuK / AuK-Flash runner (Tencent, reference-only, 24 kHz).

Duration is CALLER-SUPPLIED. Upstream's get_gen_duration uses a byte-ratio
estimator: ref_seconds * len(gen_text_bytes) / len(ref_text_bytes). It inherits
the reference's speaking pace (~13.5 chars/s for the house clip). On canonical
prompt 3 it requested 18.21 s and AuK delivered 18.22 s; FireRedTTS3 spoke the
same sentence in 15.68 s. Over-asking inflates audio_s and therefore RTFx, and
can make the model hallucinate filler to fill the time: the measured French tail
became "votre code de joie de taille" instead of "votre code aujourd'hui".
We keep this estimator because it is upstream's own default, but AuK's RTFx is
not directly comparable to a model that chooses its own length.

Both variants are pure cloners with no model-native preset voice. The default
lens borrows reference/chris_hemsworth_15s.wav and its sibling .txt, like
IndexTTS-2. Missing reference text is a hard error: without it, the estimator
silently inherits the reference's LENGTH regardless of the target prompt.
English and French run; cross-lingual French from the English reference measured
WER 0.267, with the first 11 of 15 words verbatim.

One venv (venvs/auk) serves both --variant base (NFE 32, CFG 2.0) and flash
(NFE 4, CFG 0.0). CUDA-only; flash-attn is not needed because infer_auk.py:97
sets attn_backend="torch". AuK weights are MIT, but the mandatory frozen
Qwen2.5-Omni-3B text encoder uses the non-commercial qwen-research license;
the board therefore reports "MIT code / Qwen Research (NC) encoder" -- the
restrictive half is what governs use, so the cell must name it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
AUK_SRC = REPO_ROOT / "venvs" / "auk" / "src" / "src"
CK = REPO_ROOT / "venvs" / "auk" / "src" / "ckpts"

SEED = 0

VARIANT_MODELS = {
    "base": (CK / "AuK" / "auk_base.safetensors", CK / "AuK" / "config.yaml", 32, 2.0),
    "flash": (CK / "AuK-Flash" / "auk_flash.safetensors", CK / "AuK-Flash" / "config.yaml", 4, 0.0),
}


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
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for cloning. Sibling .txt transcript required; defaults to the house Chris reference.")
    p.add_argument("--variant", default="base", help="base | flash")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    variant = (args.variant or "base").lower()
    model_spec = VARIANT_MODELS.get(variant)
    if model_spec is None:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"unknown variant {args.variant!r} (expected one of {sorted(VARIANT_MODELS)})"}))
        return 1
    ckpt, config, nfe, cfg_strength = model_spec

    # Default-voice path: borrow a bundled reference (clone-only model).
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
                          "error": f"reference {ref_wav} provided but sibling .txt transcript missing "
                                   f"(without ref_text the duration estimator silently inherits the reference's LENGTH regardless of the prompt)"}))
        return 1

    try:
        if args.device != "cuda":
            raise ValueError("AuK requires CUDA")
        sys.path.insert(0, str(AUK_SRC))

        import torch

        from auk.infer.infer_auk import AukInfer, get_gen_duration, save_audio

        engine = AukInfer(str(config), str(ckpt), qwen_path=str(CK / "Qwen2.5-Omni-3B"))

        # Seed at load, like longcat_runner.py. `generate(seed=0)` alone is NOT enough:
        # it only pins the CFM's initial noise y0 (model/cfm_edit.py:220), while the
        # reference encode does a reparameterization draw *before* that seed is set --
        # `latents = mean + torch.randn_like(mean) * torch.exp(log_std)`
        # (model/vae/bigvgan_flow_vae.py:410). Measured consequence of leaving it unseeded:
        # two identical calls with seed=0 produced the same DURATION (caller-supplied, so it
        # is pinned) but a genuinely different take -- correlation 0.872, mean |diff| 0.036
        # against an RMS of 0.162. Not float noise. Seeding the global RNG here covers the
        # VAE draw too: verified by re-running the same prompt in two fresh processes and
        # comparing samples -- 0 of 437280 differ. Note `cmp` still reports 2 differing
        # bytes at offsets 61-62; that is the Unix timestamp libsndfile writes into the
        # PEAK chunk, not audio. Compare decoded samples, not bytes.
        torch.manual_seed(SEED)
        if args.device == "cuda":
            torch.cuda.manual_seed(SEED)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            est = get_gen_duration(audio=str(ref_wav), ref_text=ref_text, gen_text=text)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": f"Say the following with the same voice: '{text}'"},
                {"type": "audio", "audio": str(ref_wav)},
            ]}]

            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio, sr = engine.generate(
                messages, gen_seconds=est, nfe=nfe, cfg_strength=cfg_strength, seed=SEED,
            )
            t_end = time.perf_counter()

            audio_s = float(audio.shape[-1] / sr)
            if write_wav:
                save_audio(audio, sr, out_path)

            # Non-streaming CFM diffusion: no partial-audio concept, so TTFA == full gen.
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
