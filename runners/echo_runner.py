"""Echo-TTS runner (Jordan Darefsky, CC-BY-NC-SA-4.0, DiT flow-matching, 44.1kHz).

Echo is a multi-speaker, reference-conditioned TTS built on a diffusion-transformer
(EchoDiT) latent model + the Fish Speech S1-DAC autoencoder. Zero-shot voice cloning
from a reference wav; multi-speaker via [S1]/[S2] tags. The model generates up to ~30s
of audio per call (640 latents); longer text is squeezed into that window (faster
speaking rate) rather than truncated, so we leave the single-call sampler as-is and
let the bench prompts (all < 30s of speech) fit naturally.

Upstream is a source repo, NOT a pip package: inference.py / model.py / autoencoder.py
are imported from the cloned tree (venvs/echo/src). install.sh clones it.

torchcodec/torchaudio note: inference.py imports both at module level, but we install
neither (torchcodec only supports FFmpeg 4-7 and this rig has 8; the torch we resolve
ships an ABI-broken torchaudio wheel). The model + autoencoder are pure torch, so we
stub both modules in sys.modules to satisfy the import, and reimplement the one real
use — load_audio's torchcodec decode — here with soundfile + librosa. Output is also
written via soundfile (no torchaudio.save).

API (lifted from the upstream README Quick Start + inference.py signatures):
    from inference import (load_model_from_hf, load_fish_ae_from_hf,
                           load_pca_state_from_hf, load_audio, sample_pipeline,
                           sample_euler_cfg_independent_guidances)
    model     = load_model_from_hf(delete_blockwise_modules=True)   # jordand/echo-tts-base
    fish_ae   = load_fish_ae_from_hf()                              # jordand/fish-s1-dac-min
    pca_state = load_pca_state_from_hf()
    spk       = load_audio("ref.wav").to(device)                    # or None -> random voice
    sample_fn = partial(sample_euler_cfg_independent_guidances, num_steps=40,
                        cfg_scale_text=3.0, cfg_scale_speaker=8.0, cfg_min_t=0.5,
                        cfg_max_t=1.0, truncation_factor=0.8, sequence_length=640, ...)
    audio_out, _ = sample_pipeline(model, fish_ae, pca_state, sample_fn,
                                   text_prompt="[S1] ...", speaker_audio=spk, rng_seed=0)
    torchaudio.save(out, audio_out[0].cpu(), 44100)

Devices: CUDA-only in harness.py. The model loads with device default "cuda" and the
sampler is a 40-step diffusion pass over a 2B-class DiT — CPU/MPS are impractical and
load_audio's torchcodec decode path is built around the GPU pipeline.

License: weights AND audio outputs are CC-BY-NC-SA-4.0 (the latter forced by the Fish
S1-DAC autoencoder dependency) — same non-commercial constraint as fish_15 / fish_s2.
"""

import argparse
import json
import sys
import time
import types
from functools import partial
from pathlib import Path

import _meminfo
import _naq


REPO_ROOT = Path(__file__).resolve().parent.parent
ECHO_SRC = REPO_ROOT / "venvs" / "echo" / "src"
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"

SAMPLERATE = 44_100


def _tag(text: str) -> str:
    """Echo expects a speaker tag; bench prompts are plain. Prefix [S1] if absent."""
    return text if "[S1]" in text or "[S2]" in text else f"[S1] {text}"


def _stub_uninstalled_audio_libs() -> None:
    """inference.py imports torchcodec + torchaudio at module level; we install
    neither (see module docstring). Inject lightweight stubs so the import succeeds —
    the only code that would touch them (load_audio) is reimplemented below."""
    if "torchcodec" not in sys.modules:
        tc = types.ModuleType("torchcodec")
        dec = types.ModuleType("torchcodec.decoders")
        dec.AudioDecoder = object
        tc.decoders = dec
        sys.modules["torchcodec"] = tc
        sys.modules["torchcodec.decoders"] = dec
    if "torchaudio" not in sys.modules:
        ta = types.ModuleType("torchaudio")
        ta.functional = types.ModuleType("torchaudio.functional")
        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.functional"] = ta.functional


def _load_ref_audio(path, device):
    """Reimplements inference.load_audio without torchcodec: load → mono → resample
    to 44.1kHz → peak-normalize. Returns a (1, T) float tensor on `device`."""
    import librosa
    import numpy as np
    import torch
    # librosa returns mono float32 at the target SR directly (averages channels).
    y, _ = librosa.load(str(path), sr=SAMPLERATE, mono=True, duration=300)
    y = np.asarray(y, dtype=np.float32)
    audio = torch.from_numpy(y).unsqueeze(0)  # (1, T)
    audio = audio / torch.maximum(audio.abs().max(), torch.tensor(1.0))
    return audio.to(device)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning (wav only, no transcript).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--num-steps", type=int, default=40)
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
        import torch

        # Source repo, not a package: make its modules importable, and satisfy its
        # module-level torchcodec/torchaudio imports with stubs before importing it.
        _stub_uninstalled_audio_libs()
        if str(ECHO_SRC) not in sys.path:
            sys.path.insert(0, str(ECHO_SRC))
        from inference import (
            load_model_from_hf, load_fish_ae_from_hf, load_pca_state_from_hf,
            sample_pipeline, sample_euler_cfg_independent_guidances,
        )

        if args.device == "cuda" and not torch.cuda.is_available():
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": "CUDA requested but not available"}))
            return 1
        device = args.device if args.device in ("cuda", "cpu", "mps") else "cuda"

        # bf16 DiT (half VRAM, AR-sampling-stable); fp32 autoencoder for decode quality.
        model = load_model_from_hf(device=device, delete_blockwise_modules=True)
        fish_ae = load_fish_ae_from_hf(device=device)
        pca_state = load_pca_state_from_hf(device=device)
        speaker_audio = _load_ref_audio(ref_wav, device)

        sample_fn = partial(
            sample_euler_cfg_independent_guidances,
            num_steps=args.num_steps,
            cfg_scale_text=3.0,
            cfg_scale_speaker=8.0,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=0.8,
            rescale_k=None,
            rescale_sigma=None,
            speaker_kv_scale=None,
            speaker_kv_max_layers=None,
            speaker_kv_min_t=None,
            sequence_length=640,  # ~30s; shorter text auto-pads to a shorter clip
        )
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()

            audio_out, _ = sample_pipeline(
                model=model,
                fish_ae=fish_ae,
                pca_state=pca_state,
                sample_fn=sample_fn,
                text_prompt=_tag(text),
                speaker_audio=speaker_audio,
                rng_seed=0,
            )
            t_end = time.perf_counter()

            # audio_out[0] is (channels, T) at 44.1kHz.
            clip = audio_out[0].detach().cpu().float()
            if clip.dim() == 1:
                clip = clip.unsqueeze(0)
            audio_s = float(clip.shape[-1] / SAMPLERATE)

            if write_wav:
                import soundfile as sf
                # soundfile wants (T, channels); clip is (channels, T).
                sf.write(str(out_path), clip.numpy().T, SAMPLERATE)

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else
                   {"naq": None, "naq_artifact": None, "naq_naturalness": None}),
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
