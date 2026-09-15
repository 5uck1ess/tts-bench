"""FireRedTTS3-Base runner (FireRedTeam, Apache 2.0, 24 kHz).

Reference-only: there is no no-reference generation path (prompt_audio=None
raises TypeError). The default lens borrows reference/chris_hemsworth_15s.wav
and its sibling .txt, like IndexTTS-2; a missing transcript is a hard error.

The bench runs English only: French cloning from a French reference is excellent
(WER 0.000), but French from the English house reference fails (WER 0.867 and
1.000). This is a harness-condition limit, not a model language limit; see the
harness.MODELS comment. Always pass language="English" explicitly because
use_fasttext=False makes automatic language detection mislabel non-English text.

CUDA-only: flash-attn is hard-coded in llm/fireredtts3_base.py:49 and
redae/redae.py:37,122. Source and checkpoints live in venvs/firered3/src.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _meminfo


REPO_ROOT = Path(__file__).resolve().parent.parent
FIRERED3_SRC = REPO_ROOT / "venvs" / "firered3" / "src"


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
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()
    if not args.stdin and (args.text is None or args.out is None):
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": "either --stdin or both --text and --out are required"}))
        return 1

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
                                   f"(FireRedTTS3 requires the reference's literal words)"}))
        return 1

    try:
        if args.device != "cuda":
            raise ValueError("FireRedTTS3 requires CUDA (hard-coded flash-attn)")
        if args.language != "en":
            raise ValueError("FireRedTTS3 runner supports only the English harness condition")
        sys.path.insert(0, str(FIRERED3_SRC))

        import torchaudio
        from fireredtts3.core import FireRedTTS3

        tts = FireRedTTS3(
            str(FIRERED3_SRC / "checkpoints"), use_wetext=True, use_llm_tn=False,
            use_fasttext=False,
        )
        prompt_audio, prompt_audio_sr = torchaudio.load(str(ref_wav))
        # Measured: seeds 0 and 7 and a fresh process produced byte-identical audio; seed is inert.
    except Exception as e:
        print(json.dumps({"ok": False, "run_index": 0,
                          "error": f"load failed: {type(e).__name__}: {e}"}))
        return 1

    def _one(text, out_path, run_index, write_wav):
        try:
            _meminfo.reset_peak(args.device)
            t0 = time.perf_counter()
            audio, sr = tts.generate(
                language="English", prompt_text=ref_text,
                prompt_audio=prompt_audio, prompt_audio_sr=int(prompt_audio_sr),
                text=text, do_tn=True,
            )
            t_end = time.perf_counter()

            audio_s = float(audio.shape[-1] / sr)
            if write_wav:
                torchaudio.save(out_path, audio.cpu(), sr)

            # Non-streaming AR LLM + RedAE decode: no partial-audio concept, so TTFA == full gen.
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
