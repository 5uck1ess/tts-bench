"""ZipVoice runner (k2-fsa, 123M, Apache 2.0, zero-shot voice cloning).

Flow-matching TTS. Zero-shot voice cloning requires a reference wav + sibling
.txt transcript. No built-in voices — falls back to the repo's default
reference if none is supplied.

API (source install, k2-fsa/ZipVoice):
  Weights auto-downloaded from HuggingFace `k2-fsa/ZipVoice` on first run.
  Inference path used here mirrors zipvoice.bin.infer_zipvoice.generate_sentence:

    from zipvoice.bin.infer_zipvoice import generate_sentence, get_vocoder
    from zipvoice.models.zipvoice import ZipVoice
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.feature import VocosFbank
    import safetensors.torch, json
    from huggingface_hub import hf_hub_download

    model_ckpt   = hf_hub_download("k2-fsa/ZipVoice", "zipvoice/model.pt")
    model_config = hf_hub_download("k2-fsa/ZipVoice", "zipvoice/model.json")
    token_file   = hf_hub_download("k2-fsa/ZipVoice", "zipvoice/tokens.txt")
    # ... build model, tokenizer, vocoder, feature_extractor ...
    generate_sentence(save_path, prompt_text, prompt_wav, text,
                      model, vocoder, tokenizer, feature_extractor, device,
                      num_step=16, guidance_scale=1.0, sampling_rate=24000)

  generate_sentence() saves the wav itself via torchaudio.save(); returns a
  metrics dict with t, wav_seconds, rtf etc.

Cloning: requires --reference AND sibling <reference>.txt transcript.
Default voice: falls back to reference/chris_hemsworth_15s.wav + .txt.
Sample rate: 24 kHz (vocos-mel-24khz vocoder).
Multilingual: zh + en (Emilia tokenizer).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _ensure_ffmpeg_libs():
    """ZipVoice loads audio via torchaudio 2.11+, which routes through torchcodec;
    torchcodec needs FFmpeg's shared libs on the loader path. When FFmpeg comes from
    a non-standard prefix (e.g. linuxbrew) its lib dir isn't searched, so the load
    fails with "Could not load libtorchcodec". Find the ffmpeg lib dir and re-exec
    once with it on LD_LIBRARY_PATH. No-op if already on the path or no ffmpeg found."""
    if os.environ.get("_ZIPVOICE_FFMPEG_LDPATH"):
        return
    import shutil
    cands = []
    ff = shutil.which("ffmpeg")
    if ff:
        cands.append(Path(ff).resolve().parent.parent / "lib")
    cands += [Path("/home/linuxbrew/.linuxbrew/lib"),
              Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/local/lib")]
    libdirs = [str(d) for d in cands if d.is_dir() and any(d.glob("libavcodec.so*"))]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    missing = [d for d in libdirs if d not in cur.split(":")]
    if not missing:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + ([cur] if cur else []))
    os.environ["_ZIPVOICE_FFMPEG_LDPATH"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_ffmpeg_libs()

import _meminfo

SAMPLE_RATE = 24000
HF_REPO = "k2-fsa/ZipVoice"
MODEL_DIR_HF = "zipvoice"   # subfolder inside the HF repo

# Relative to the runner file — used as default reference when none supplied.
_RUNNER_DIR = Path(__file__).resolve().parent
_DEFAULT_REFERENCE = _RUNNER_DIR.parent / "reference" / "chris_hemsworth_15s.wav"


def _read_ref_transcript(ref_wav: str | None) -> str | None:
    """Return contents of <ref_wav>.txt, or None if missing/unset."""
    if not ref_wav:
        return None
    txt_path = Path(ref_wav).with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def _maybe_clip_reference(wav_path: str, ref_text: str, max_sec: float = 3.0):
    """Clip the reference wav + text proportionally if wav > max_sec.

    ZipVoice's docs recommend 1-3s reference clips for inference. A longer
    reference (e.g. our standard 15s chris_hemsworth) causes generation to
    balloon to ~5 min per prompt. We clip in-runner so the global reference
    file stays the same as other models, but ZipVoice sees a 3s slice.

    Word-boundary alignment can't be inferred without ASR, so the transcript
    is trimmed by word count proportional to the audio fraction kept. This
    is approximate but adequate for flow-matching's acoustic-conditioning
    use of the prompt (the prompt is used for voice characteristics, not for
    strict text alignment).

    Returns (clipped_wav_path, clipped_text). The clipped wav is written to
    a temp file; if no clipping was needed the original path/text are
    returned unchanged.
    """
    import tempfile
    import soundfile as sf

    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    duration = len(y) / sr
    if duration <= max_sec:
        return wav_path, ref_text

    clip_samples = int(max_sec * sr)
    y_clip = y[:clip_samples]

    words = ref_text.split()
    n_words = max(1, int(len(words) * max_sec / duration))
    text_clip = " ".join(words[:n_words])

    fd, clip_path = tempfile.mkstemp(suffix=".wav", prefix="zipvoice_ref_")
    os.close(fd)
    sf.write(clip_path, y_clip, sr)
    return clip_path, text_clip


def _load_model(device_str: str):
    """Download weights (cached after first run) and return model components."""
    import json as _json

    import safetensors.torch
    import torch
    from huggingface_hub import hf_hub_download
    from vocos import Vocos

    from zipvoice.bin.infer_zipvoice import get_vocoder
    from zipvoice.models.zipvoice import ZipVoice
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.checkpoint import load_checkpoint
    from zipvoice.utils.feature import VocosFbank

    model_ckpt_path   = hf_hub_download(HF_REPO, filename=f"{MODEL_DIR_HF}/model.pt")
    model_config_path = hf_hub_download(HF_REPO, filename=f"{MODEL_DIR_HF}/model.json")
    token_file_path   = hf_hub_download(HF_REPO, filename=f"{MODEL_DIR_HF}/tokens.txt")

    with open(model_config_path, "r") as f:
        model_config = _json.load(f)

    tokenizer = EmiliaTokenizer(token_file=token_file_path)
    tokenizer_config = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id":     tokenizer.pad_id,
    }

    model = ZipVoice(**model_config["model"], **tokenizer_config)

    if str(model_ckpt_path).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt_path)
    else:
        load_checkpoint(filename=model_ckpt_path, model=model, strict=True)

    device = torch.device(
        {"cuda": "cuda:0", "mps": "mps", "cpu": "cpu"}.get(device_str, "cpu")
    )
    model = model.to(device)
    model.eval()

    vocoder = get_vocoder()   # downloads charactr/vocos-mel-24khz
    vocoder = vocoder.to(device)
    vocoder.eval()

    feature_extractor = VocosFbank()
    sampling_rate = model_config["feature"]["sampling_rate"]   # should be 24000

    return model, vocoder, tokenizer, feature_extractor, device, sampling_rate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot voice cloning. "
                        "Requires sibling .txt transcript. "
                        "Defaults to reference/chris_hemsworth_15s.wav.")
    p.add_argument("--variant", default=None)   # ignored for base model; reserved
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")  # informational; ZipVoice is auto-multilingual
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    # Resolve reference: user-supplied > repo default.
    ref_wav = args.reference or str(_DEFAULT_REFERENCE)
    ref_text = _read_ref_transcript(ref_wav)
    if not ref_text:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"reference transcript missing: expected {Path(ref_wav).with_suffix('.txt')}"}))
        return 1

    # ZipVoice's inference time balloons on references >5s (upstream
    # recommends 1-3s). Clip to first 3s if needed; transcript is trimmed
    # proportionally by word count.
    ref_wav, ref_text = _maybe_clip_reference(ref_wav, ref_text, max_sec=3.0)

    # ------------------------------------------------------------------ load
    try:
        model, vocoder, tokenizer, feature_extractor, device, sampling_rate = \
            _load_model(args.device)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    from zipvoice.bin.infer_zipvoice import generate_sentence

    # ------------------------------------------------------------------ infer
    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()

            # generate_sentence() saves the wav itself; we pass a temp path for
            # warm runs (write_wav=False) and the real path for the first run.
            save_path = out_path if write_wav else str(
                Path(out_path).with_suffix("") ) + f"._tmp_run{run_index}.wav"

            metrics = generate_sentence(
                save_path=save_path,
                prompt_text=ref_text,
                prompt_wav=ref_wav,
                text=text,
                model=model,
                vocoder=vocoder,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                device=device,
                num_step=16,
                guidance_scale=1.0,
                sampling_rate=sampling_rate,
            )

            t_end = time.perf_counter()

            # Clean up warm-run temp file
            if not write_wav:
                try:
                    Path(save_path).unlink(missing_ok=True)
                except OSError:
                    pass

            audio_s = float(metrics.get("wav_seconds", 0.0))
            gen_s   = t_end - t0

            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": gen_s * 1000,
                "gen_s": gen_s, "audio_s": audio_s,
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
