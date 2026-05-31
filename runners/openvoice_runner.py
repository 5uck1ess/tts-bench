"""OpenVoice v2 runner (myshell-ai, MIT, zero-shot tone-color cloning, 22.05kHz).

OpenVoice v2 = MeloTTS (base TTS) + a ToneColorConverter (cloning). Three
in-process steps wrapped as one runner call:
    1. MeloTTS generates base audio from text (a neutral base-speaker voice).
    2. The reference wav is run through ToneColorConverter.extract_se() to get
       the target speaker's tone-color embedding (tgt_se).
    3. ToneColorConverter.convert() re-tones the base audio toward tgt_se,
       using the base speaker's own SE (loaded from checkpoints) as src_se.

API discovered by inspection (2026-05-30, MeloTTS 0.1.2 + OpenVoice src @ main):
    from melo.api import TTS as MeloTTS
    from openvoice.api import ToneColorConverter
    melo = MeloTTS(language="EN", device=device)
    melo.tts_to_file(text, melo.hps.data.spk2id["EN-Default"], tmp_wav, speed=1.0)
    conv = ToneColorConverter("converter/config.json", device=device)
    conv.load_ckpt("converter/checkpoint.pth")
    src_se = torch.load("base_speakers/ses/en-default.pth")    # base speaker SE
    tgt_se = conv.extract_se([ref_wav])                        # target speaker SE
    conv.convert(audio_src_path=tmp_wav, src_se=src_se, tgt_se=tgt_se,
                 output_path=out_path)                         # writes 22.05kHz wav

Notes / install gotchas:
- We do NOT install OpenVoice's setup.py (it pins faster-whisper==0.9.0 ->
  av==10.0.0, which fails to Cython-build on Windows, plus numpy==1.22 /
  librosa==0.9.1 / gradio that fight the MeloTTS+torch stack). Instead we add
  the cloned src dir to sys.path and use only ToneColorConverter, whose runtime
  deps (torch/numpy/soundfile/librosa) come from the MeloTTS install, plus
  wavmark (installed at setup) for the converter's audio watermarker. The
  watermark is inaudible and does not affect the bench comparison.
- se_extractor.get_se (the usual cloning entry point) wraps extract_se with
  faster-whisper VAD segmentation to clean long/noisy reference audio. The bench
  reference is a clean ~15s clip, so we call extract_se([ref_wav]) directly and
  avoid faster-whisper entirely.
- MeloTTS English uses g2p_en (CMUdict + NLTK averaged_perceptron_tagger), NOT
  espeak/phonemizer. We pre-download the NLTK tables g2p_en fetches at runtime
  (guarded). No espeak wiring is needed for the English path.
- torch 2.11's torchaudio routes load() through torchcodec (FFmpeg DLLs absent
  on Windows). librosa is used for audio IO inside the converter, but we still
  monkey-patch torchaudio.load -> soundfile defensively (same as f5tts/zonos).
- ToneColorConverter.load_ckpt calls torch.load without weights_only=False; on
  torch 2.6+ the default flipped to True and rejects these checkpoints, so we
  patch torch.load to default weights_only=False (trusted myshell-ai weights).
- Base TTS speaker is EN-Default; base SE is base_speakers/ses/en-default.pth.
  Output is the ToneColorConverter's native 22.05kHz.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# OpenVoice's text path (via the converter's pulled-in deps) can surface IPA /
# non-ASCII into the default cp1252 Windows stream. Force UTF-8 like styletts2.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import _meminfo
import _naq


# Map harness language -> (MeloTTS language code, MeloTTS speaker key, base SE file).
# OpenVoice v2 ships base-speaker SEs under base_speakers/ses/<name>.pth; the
# MeloTTS speaker key lowercased matches the ses filename for English.
LANG_CFG = {
    "en": ("EN", "EN-Default", "en-default.pth"),
    "es": ("ES", "ES",         "es.pth"),
    "fr": ("FR", "FR",         "fr.pth"),
    "zh": ("ZH", "ZH",         "zh.pth"),
    "ja": ("JP", "JP",         "jp.pth"),
    "ko": ("KR", "KR",         "kr.pth"),
}


def _install_soundfile_loader():
    """Replace torchaudio.load with soundfile to avoid torchcodec DLL hell on Windows."""
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
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Target voice wav for zero-shot tone-color cloning. Omit => reference/chris_hemsworth_15s.wav.")
    p.add_argument("--variant", default=None)        # unused
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

    try:
        import os
        import tempfile
        import numpy as np
        import soundfile as sf
        import torch

        repo = Path(__file__).resolve().parent.parent
        src = repo / "venvs" / "openvoice" / "src"
        ckpt_dir = src / "checkpoints_v2"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"OpenVoiceV2 checkpoints not found: {ckpt_dir} (run install.ps1)")
        # Use the cloned OpenVoice source directly (we don't pip-install its
        # setup.py — see module docstring).
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        # torch 2.6+ flipped torch.load weights_only default to True; OpenVoice's
        # load_ckpt / SE loads pass no weights_only and need the full pickle.
        _orig_torch_load = torch.load
        def _patched_torch_load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_torch_load(*a, **k)
        torch.load = _patched_torch_load

        _install_soundfile_loader()

        from melo.api import TTS as MeloTTS
        from openvoice.api import ToneColorConverter

        # MeloTTS English uses g2p_en, which lazily fetches these NLTK tables at
        # runtime via NLTK's own downloader. Pre-fetch (cached, guarded).
        import nltk
        for _res, _path in [
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("averaged_perceptron_tagger",     "taggers/averaged_perceptron_tagger"),
            ("cmudict",                         "corpora/cmudict"),
            ("punkt",                           "tokenizers/punkt"),
            ("punkt_tab",                       "tokenizers/punkt_tab"),
        ]:
            try:
                nltk.data.find(_path)
            except LookupError:
                try:
                    nltk.download(_res, quiet=True)
                except Exception:
                    pass

        lang = args.language if args.language in LANG_CFG else "en"
        melo_lang, melo_spk_key, base_se_file = LANG_CFG[lang]

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        # ToneColorConverter asserts torch.cuda.is_available() for any 'cuda' device.
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        melo = MeloTTS(language=melo_lang, device=device)
        spk2id = melo.hps.data.spk2id
        if melo_spk_key not in spk2id:
            # Fall back to whatever single speaker MeloTTS exposes for this lang.
            melo_spk_key = next(iter(spk2id))
        speaker_id = spk2id[melo_spk_key]

        # NOTE: ToneColorConverter forwards **kwargs (incl. enable_watermark) to its
        # base __init__, which rejects them — so enable_watermark=False actually
        # raises. We leave the default (watermark on; wavmark installed at setup) and
        # the converter loads the wavmark model. The watermark is inaudible and does
        # not affect the bench's quality/latency comparison.
        converter = ToneColorConverter(str(ckpt_dir / "converter" / "config.json"),
                                       device=device)
        converter.load_ckpt(str(ckpt_dir / "converter" / "checkpoint.pth"))

        base_se_path = ckpt_dir / "base_speakers" / "ses" / base_se_file
        if not base_se_path.exists():
            raise FileNotFoundError(f"base-speaker SE not found: {base_se_path}")
        src_se = torch.load(str(base_se_path), map_location=device)

        ref_wav = Path(args.reference) if args.reference else (repo / "reference" / "chris_hemsworth_15s.wav")
        if not ref_wav.exists():
            raise FileNotFoundError(f"reference wav not found: {ref_wav}")
        # Target tone-color embedding from the (clean) reference clip. extract_se
        # accepts a list of wav paths and averages their SEs.
        tgt_se = converter.extract_se([str(ref_wav)])
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        tmp_wav = None
        warm_wav = None
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            # Step 1: MeloTTS base synthesis to a temp wav.
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            melo.tts_to_file(text, speaker_id, tmp_wav, speed=1.0, quiet=True)
            # Step 3: re-tone toward the target speaker. convert() writes a wav as
            # its output, so we always run it (honest timing) but only let the cold
            # run write the real out_path. Warm runs convert to a throwaway temp —
            # every run of a cell shares out_path, so writing warm output there
            # would overwrite and then delete the cold sample wav.
            if write_wav:
                conv_target = out_path
            else:
                fd2, warm_wav = tempfile.mkstemp(suffix=".wav")
                os.close(fd2)
                conv_target = warm_wav
            converter.convert(audio_src_path=tmp_wav, src_se=src_se, tgt_se=tgt_se,
                              output_path=conv_target)
            t_end = time.perf_counter()

            data, sr = sf.read(conv_target, dtype="float32")
            if len(data) == 0:
                raise RuntimeError("convert produced empty audio")
            audio_s = float(len(data) / sr)

            # Non-streaming: TTFA == full gen (audio not available until both
            # the base synthesis and the conversion finish).
            print(json.dumps({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                **_meminfo.sample(args.device),
                **(_naq.score(out_path) if write_wav else {"naq": None, "naq_artifact": None, "naq_naturalness": None}),
            }), flush=True)
            return True
        except Exception as e:
            print(json.dumps({
                "ok": False, "run_index": run_index,
                "error": f"{type(e).__name__}: {e}",
            }), flush=True)
            return False
        finally:
            for pth in (tmp_wav, warm_wav):
                if pth is not None:
                    try:
                        os.remove(pth)
                    except OSError:
                        pass

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
