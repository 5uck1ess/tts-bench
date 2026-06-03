"""MeloTTS runner (myshell-ai/MeloTTS-English, MIT, predefined-voice TTS).

PREDEFINED-VOICE model: VITS-style multi-speaker TTS with baked-in speaker IDs
(EN-US / EN-BR / EN-AU / EN-Default / EN_INDIA for English). No zero-shot
cloning, so can_clone=False and --reference is accepted but ignored. This is the
base speaker engine that OpenVoice v2 wraps with a tone-color converter; here we
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


SPEAKER = "EN-US"


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

        device = args.device if args.device in ("cpu", "cuda", "mps") else "cpu"
        model = TTS(language="EN", device=device)
        spk2id = model.hps.data.spk2id
        if SPEAKER not in spk2id:
            raise RuntimeError(f"speaker {SPEAKER} not in {list(spk2id)}")
        speaker_id = spk2id[SPEAKER]
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
