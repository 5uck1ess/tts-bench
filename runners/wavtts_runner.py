"""WavTTS runner (cwx-worst-one / worstchan, MIT code / CC-BY-NC-4.0 weights).

WavTTS: "Towards High-Quality Zero-Shot TTS via Direct Raw Waveform Modeling"
(arXiv 2606.03455). A flow-matching DiT that generates speech DIRECTLY in the raw
waveform space — no mel-spectrogram, no VAE latent, no codec tokens — built on the
F5-TTS codebase (waveform patchification + multi-scale mel supervision + power
timestep mapping). 16 kHz output (the model's fixed target_sample_rate).

API (same shape as F5-TTS, from wavtts.infer.infer_cli):
    cfg   = OmegaConf.load(files("wavtts")/"configs/WavTTS.yaml")
    cls   = get_class(f"wavtts.model.{cfg.model.backbone}")        # DiT
    model = load_model(cls, cfg.model.arch, ckpt, vocab_file="", device=...,
                       cfm_kwargs=cfg.model.cfm, waveform_kwargs=cfg.model.waveform)
    ref_a, ref_t = preprocess_ref_audio_text(ref_wav, ref_text)   # clips ref <=12s
    wave, sr, _  = infer_process(ref_a, ref_t, gen_text, model, ...)  # sr == 16000

PURE CLONING (no preset voice): every synthesis needs a reference wav + its literal
transcript (read from a sibling .txt, like cosyvoice/sesame/longcat). can_clone=True
-> both lenses, but it's NO_PRESET_VOICE so it renders on the cloning board only:
  * default (no --reference): the house voice, reference/chris_hemsworth_15s.wav.
  * cloning (--reference wav + sibling .txt): the supplied voice.
ZH/EN only (Emilia ZH_EN, pinyin tokenizer) -> multilingual=False, FR prompt skipped.

Install / runtime gotchas (mirrors f5tts):
- torch routes torchaudio.load() through torchcodec, which needs FFmpeg shared DLLs
  (not just ffmpeg.exe) — broken with a static FFmpeg on Windows. This runner
  monkey-patches torchaudio.load to use soundfile directly, so torchcodec is never
  imported at inference time (it's a WavTTS dep but stays dormant).
- The reference-ASR path (whisper-large-v3-turbo) only fires when ref_text is empty;
  we always pass the sibling-.txt transcript, so no whisper download happens.
- Heavy diffusion DiT (NFE 50 over a ~0.5B-class backbone) -> CUDA-only here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"
DEFAULT_MODEL = "WavTTS"
DEFAULT_CKPT = "hf://worstchan/WavTTS/model_1200000.pt"


def _install_soundfile_loader():
    """Replace torchaudio.load with soundfile to avoid torchcodec DLL hell (same as f5tts)."""
    import soundfile as sf
    import numpy as np
    import torch
    import torchaudio

    def _sf_load(path, **kwargs):
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        return torch.from_numpy(data), sr

    torchaudio.load = _sf_load


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for cloning. Sibling .txt transcript required "
                        "(WavTTS prepends the reference's literal words, like cosyvoice/longcat).")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Resolve the reference + its transcript (pure cloner -> always needs a .txt).
    ref_wav = Path(args.reference).resolve() if args.reference else DEFAULT_REF
    ref_txt = ref_wav.with_suffix(".txt")
    if not ref_wav.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference wav not found: {ref_wav}"}))
        return 1
    if not ref_txt.exists():
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference {ref_wav} needs a sibling .txt transcript "
                                   f"(WavTTS prepends the reference's literal words)"}))
        return 1
    ref_text = ref_txt.read_text(encoding="utf-8").strip()

    try:
        _install_soundfile_loader()
        import numpy as np
        import soundfile as sf
        from importlib.resources import files
        from cached_path import cached_path
        from hydra.utils import get_class
        from omegaconf import OmegaConf
        from wavtts.infer.utils_infer import (
            load_model, infer_process, preprocess_ref_audio_text,
            target_rms, cross_fade_duration, nfe_step, cfg_strength,
            sway_sampling_coef, timestep_mapping, timestep_power, shift, speed,
            fix_duration,
        )

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"

        model_cfg = OmegaConf.load(str(files("wavtts").joinpath(f"configs/{DEFAULT_MODEL}.yaml")))
        model_cls = get_class(f"wavtts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch
        cfm_kwargs = getattr(model_cfg.model, "cfm", {}) or {}
        ckpt_file = str(cached_path(DEFAULT_CKPT))
        ema_model = load_model(
            model_cls, model_arc, ckpt_file, vocab_file="", device=device,
            cfm_kwargs=cfm_kwargs, waveform_kwargs=model_cfg.model.waveform,
        )
        samplerate = 16000  # WavTTS fixed target_sample_rate

        # Preprocess the reference once (clips to <=12s, normalizes, caches by hash).
        ref_audio_p, ref_text_p = preprocess_ref_audio_text(str(ref_wav), ref_text)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            wave, sr, _spec = infer_process(
                ref_audio_p, ref_text_p, text, ema_model,
                show_info=lambda *a, **k: None, progress=None,
                target_rms=target_rms, cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step, cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef, timestep_mapping=timestep_mapping,
                timestep_power=timestep_power, shift=shift, speed=speed,
                fix_duration=fix_duration, device=device,
            )
            t_end = time.perf_counter()
            if wave is None:
                raise RuntimeError("infer_process returned no audio")

            arr = np.asarray(wave).reshape(-1)
            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

            # Non-streaming flow-matching: full waveform after all NFE steps, TTFA == gen.
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
