"""MeloTTS runner (myshell-ai/MeloTTS, MIT, predefined-voice TTS, multilingual).

PREDEFINED-VOICE model: VITS-style multi-speaker TTS with baked-in speaker IDs.
MeloTTS ships a separate checkpoint per language; --language picks which one to
load and its default speaker (see LANG_CFG). No zero-shot cloning, so
can_clone=False and --reference is accepted but ignored. This is the base
speaker engine that OpenVoice v2 wraps with a tone-color converter; here we
bench it standalone as a fast CPU baseline.

API (melo.api.TTS):
    model = TTS(language="EN", device=...)
    spk = model.hps.data.spk2id            # {"EN-US": 0, ...}
    arr = model.tts_to_file(text, spk["EN-US"], output_path=None, speed=1.0)
    # output_path=None returns the float waveform instead of writing it.

Non-streaming, so TTFA == gen_s.
"""

import argparse
import json
import sys
import time

import _meminfo


# (melo language code, default speaker key) per ISO-639-1 --language.
# EN keeps EN-US so the existing English sample is unchanged; FR loads the
# French checkpoint (myshell-ai/MeloTTS-French), same mapping OpenVoice uses.
# JP/KR are omitted: they need extra g2p deps (mecab/unidic, g2pkk) not installed.
LANG_CFG = {
    "en": ("EN", "EN-US"),
    "es": ("ES", "ES"),
    "fr": ("FR", "FR"),
    "zh": ("ZH", "ZH"),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)       # unused (predefined voice)
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
        from melo.api import TTS

        lang = (args.language or "en").lower()
        if lang not in LANG_CFG:
            print(json.dumps({"ok": False, "run_index": 0,
                              "error": f"MeloTTS runner not wired for language={args.language!r} "
                                       f"(supported: {sorted(LANG_CFG)})"}))
            return 1
        melo_lang, speaker_key = LANG_CFG[lang]

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        model = TTS(language=melo_lang, device=device)
        spk2id = model.hps.data.spk2id
        if speaker_key not in spk2id:
            # Fall back to whatever single speaker this language exposes.
            speaker_key = next(iter(spk2id))
        speaker_id = spk2id[speaker_key]
        samplerate = int(model.hps.data.sampling_rate)
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            arr = model.tts_to_file(text, speaker_id, output_path=None,
                                    speed=1.0, quiet=True)
            arr = np.asarray(arr, dtype="float32").squeeze()
            t_end = time.perf_counter()

            audio_s = float(len(arr) / samplerate)
            if write_wav:
                sf.write(out_path, arr, samplerate)

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
