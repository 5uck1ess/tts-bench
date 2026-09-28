"""SupraTTS-0.1-Beta runner (MIT, English-only, one fixed LJSpeech voice, no cloning).

Glow-TTS trained from scratch on LJSpeech in Coqui-TTS, with the deterministic
duration predictor swapped for VITS's stochastic one, plus a HiFi-GAN v1 vocoder
trained from scratch on ground-truth mels. Successor to SupraLabs' Flare-TTS v1.5.

    acoustic model  29,547,392 params   (the "29.6M" in the release post)
    HiFi-GAN v1     13,926,017 params
    full stack      43.47M              (what the board lists — Inflect/Vaniq count
                                          their built-in decoder the same way)

Output is 22.05 kHz. Weights pinned to HF SHA 49d39b7b53f75c47199fb5f1b7650b0ae7e5d874
(downloaded 2026-09-28); no runner here pins `revision=`, so that is the record.
The .pth files are full training checkpoints (~1.4 GB with optimizer state) even
though the models are small.

API (repo-local, not a PyPI package — the HF repo ships infer_v2.py defining
`FlareGlowTTS(GlowTTS)` on top of `coqui-tts`):
    config = GlowTTSConfig(); config.load_json("config.json"); config.model_args = None
    ap = AudioProcessor(**config.audio)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = FlareGlowTTS(config, ap, tokenizer=tokenizer, noise_scale_dp=0.8)
    model.load_checkpoint(config, "model.pth", eval=True)
    ... model.inference(ids) -> mel -> vocoder.inference(mel)

Upstream's infer_v2.py main() does NOT run on coqui-tts 0.27.5 (current); two fixes
here, neither changes the model's numerics:
  1. `AudioProcessor(config.audio)` passes the whole audio config as the first
     positional arg (`sample_rate`), so every other field defaults to None and the
     constructor dies dividing frame_length_ms by frame_shift_ms. Unpack it instead.
  2. config.json ships `"model_args": {}`. Coqui loads that as an empty, non-None
     Coqpit, and BaseTTS then reads `config.model_args.num_chars` instead of the
     top-level `num_chars` -> AttributeError. Nulling it restores the intended path.

Phonemizer gotcha: config uses Coqui's `espeak` phonemizer, which shells out to an
`espeak-ng` EXECUTABLE found via shutil.which at TTS import time (the
espeakng-loader wheel other runners use ships only the DLL, so it can't serve
here). Windows: install.ps1 admin-extracts the official espeak-ng 1.52.0 MSI to
venvs/supratts/espeak-ng/ (no system install); this runner prepends it to PATH
and sets ESPEAK_DATA_PATH before importing TTS — without ESPEAK_DATA_PATH that
portable exe segfaults on `--version`, which Coqui calls on import. Linux/Mac use
the system espeak-ng (apt/brew). If none is found Coqui would silently pick a
different phonemizer, so the runner asserts the backend is espeak.

Non-streaming: one inference call returns the full waveform, so TTFA == gen_s.
Reported that way to stay honest against streaming models.

Non-autoregressive (duration predictor), so there is no generation-token cap:
canonical prompt 3 renders in full (18.39 s, 1574 mel frames) on cpu and cuda.

Determinism: noise scales are upstream's (inference 0.333, SDP 0.8, length 1.0);
the torch seed is reset to 0 before every run so reruns of a prompt sample the
same durations and latent.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import _meminfo


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "venvs" / "supratts" / "src" / "SupraTTS-0.1-Beta"
ESPEAK_DIR = REPO / "venvs" / "supratts" / "espeak-ng"

# Upstream infer_v2.py defaults.
NOISE_SCALE = 0.333
NOISE_SCALE_DP = 0.8
LENGTH_SCALE = 1.0
SEED = 0


def _wire_espeak():
    # Must run before `import TTS`: Coqui resolves the espeak binary at import time.
    if (ESPEAK_DIR / "espeak-ng.exe").exists():
        os.environ["PATH"] = str(ESPEAK_DIR) + os.pathsep + os.environ.get("PATH", "")
        os.environ["ESPEAK_DATA_PATH"] = str(ESPEAK_DIR / "espeak-ng-data")
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        raise FileNotFoundError(
            "espeak-ng executable not found — run install.ps1/install.sh for `supratts`")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Ignored — SupraTTS ships one fixed LJSpeech voice and cannot clone.")
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
                          "error": f"SupraTTS is English-only, got language={args.language}"}))
        return 1

    try:
        if not (MODEL_DIR / "model.pth").exists():
            raise FileNotFoundError(
                f"{MODEL_DIR} is missing model.pth — run install.ps1/install.sh for `supratts`")
        _wire_espeak()

        import torch
        import soundfile as sf
        from TTS.config import load_config
        from TTS.tts.configs.glow_tts_config import GlowTTSConfig
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor
        from TTS.vocoder.models import setup_model as setup_vocoder
        sys.path.insert(0, str(MODEL_DIR))
        from infer_v2 import FlareGlowTTS

        config = GlowTTSConfig()
        config.load_json(str(MODEL_DIR / "config.json"))
        config.model_args = None  # fix 2 in the docstring
        ap = AudioProcessor(**config.audio)  # fix 1
        tokenizer, config = TTSTokenizer.init_from_config(config)
        if tokenizer.phonemizer is None or tokenizer.phonemizer.name() != "espeak":
            raise RuntimeError(f"expected espeak phonemizer, got {tokenizer.phonemizer!r}")

        model = FlareGlowTTS(config, ap, tokenizer=tokenizer, noise_scale_dp=NOISE_SCALE_DP)
        model.load_checkpoint(config, str(MODEL_DIR / "model.pth"), eval=True)
        model.inference_noise_scale, model.length_scale = NOISE_SCALE, LENGTH_SCALE
        model.to(args.device).eval()

        vconfig = load_config(str(MODEL_DIR / "vocoder_config.json"))
        vocoder = setup_vocoder(vconfig)
        vocoder.load_checkpoint(vconfig, str(MODEL_DIR / "vocoder.pth"), eval=True)
        vocoder.to(args.device).eval()
        sample_rate = ap.sample_rate
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            torch.manual_seed(SEED)
            t0 = time.perf_counter()
            x = torch.LongTensor(tokenizer.text_to_ids(text))[None].to(args.device)
            with torch.inference_mode():
                out = model.inference(
                    x, aux_input={"x_lengths": torch.LongTensor([x.shape[1]]).to(args.device)})
                wav = vocoder.inference(out["model_outputs"].transpose(1, 2))
            audio = wav.squeeze().float().cpu().numpy()  # .cpu() syncs cuda
            t_end = time.perf_counter()

            audio_s = float(len(audio) / sample_rate)
            if write_wav:
                sf.write(out_path, audio, sample_rate)

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
