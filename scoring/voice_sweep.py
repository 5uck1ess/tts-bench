"""Rank a model's selectable voices objectively, to pick its bench default voice.

Multi-voice models ship no "best" voice — Scylla's Band's manifest lists all ten
as peers, Piper has one per language, Kokoro ships a pack per speaker. Picking the
board's default by ear is taste; this picks it by measurement.

Generates every voice over the canonical bench prompts through the model's OWN
runner (`--reference <voice>`, the same flag the harness uses for predefined-voice
models), then scores each clip with UTMOS (naturalness, reference-free) and WER
(intelligibility). SIM is deliberately absent: this is a default-voice question,
not a cloning one.

Needs only `venvs/scoring` (UTMOS + WER). It does NOT need `venvs/scoring_sim` —
SIM is the cloning metric and is out of scope here, and scoring_sim's py3.10
fairseq stack is the part that's Windows-hostile. UTMOS (torch.hub SpeechMOS) and
WER (transformers Whisper-large-v3) are portable; run this wherever the scoring
venv and a GPU live, which by convention is the Linux box.

Run under the SCORING venv; it subprocesses the MODEL venv to generate:

    venvs/scoring/bin/python -m scoring.voice_sweep \
        --venv scyllasband \
        --runner runners/scyllasband_runner.py \
        --voices ariadne,felix,gwen,ink,max,orpheus,rex,scylla,stone,tuesday \
        --out /tmp/voice_sweep

Writes <out>/voice_sweep.csv and prints a ranking. Per-clip failures blank that
cell and keep going, matching score_all's never-abort behavior.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from scoring.prompts import PROMPTS


def _fmt(v, nd=3):
    return "" if v is None else f"{v:.{nd}f}"


def generate(venv_python, runner, voice, prompts, out_dir, device, language):
    """Drive one voice through the runner's --stdin mode; return {pid: wav_path}.

    One process per voice so the model loads once and amortizes across prompts.
    """
    jobs, wavs = [], {}
    for pid, _lang, text in prompts:
        wav = out_dir / f"{voice}_p{pid}.wav"
        jobs.append(json.dumps({"text": text, "out": str(wav)}))
        wavs[pid] = wav

    proc = subprocess.run(
        [str(venv_python), str(runner), "--stdin",
         "--device", device, "--language", language, "--reference", voice],
        input="\n".join(jobs) + "\n",
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        print(f"  ! {voice}: runner exited {proc.returncode}: "
              f"{proc.stderr.strip()[:300]}", file=sys.stderr)

    ok = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ready"):
            continue
        idx = rec.get("run_index")
        if idx is None or idx >= len(prompts):
            continue
        pid = prompts[idx][0]
        if rec.get("ok"):
            ok[pid] = wavs[pid]
        else:
            print(f"  ! {voice} p{pid}: {rec.get('error')}", file=sys.stderr)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--venv", required=True, help="venv dir name under venvs/")
    p.add_argument("--runner", required=True, help="runner path, e.g. runners/scyllasband_runner.py")
    p.add_argument("--voices", required=True, help="comma-separated voice ids")
    p.add_argument("--out", required=True, help="output dir for wavs + csv")
    p.add_argument("--device", default="cpu")
    p.add_argument("--language", default="en", help="harness language code passed to the runner")
    p.add_argument("--langs", default="en",
                   help="comma-separated prompt languages to include (default: en only)")
    p.add_argument("--repo", default=".", help="repo root")
    args = p.parse_args()

    repo = Path(args.repo).resolve()
    # Linux/Mac layout first, Windows second — the scoring half is Linux-only, but
    # the generation half is testable anywhere, so don't hardcode posix here.
    venv_root = repo / "venvs" / args.venv
    venv_python = next(
        (c for c in (venv_root / "bin" / "python",
                     venv_root / "Scripts" / "python.exe") if c.exists()),
        None,
    )
    if venv_python is None:
        sys.exit(f"model venv python not found under {venv_root} "
                 f"(run from the repo root, with the model installed)")

    runner = repo / args.runner
    if not runner.exists():
        sys.exit(f"runner not found: {runner}")

    want_langs = {s.strip() for s in args.langs.split(",") if s.strip()}
    prompts = [t for t in PROMPTS if t[1] in want_langs]
    if not prompts:
        sys.exit(f"no prompts match --langs {args.langs}")

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from scoring.utmos import UtmosScorer
    from scoring.wer import WerScorer
    utmos, wer = UtmosScorer(), WerScorer()

    text_by_pid = {pid: text for pid, _l, text in prompts}
    lang_by_pid = {pid: lang for pid, lang, _t in prompts}

    rows = []
    for voice in voices:
        print(f"[{voice}] generating {len(prompts)} prompts…", flush=True)
        wavs = generate(venv_python, runner, voice, prompts, out_dir,
                        args.device, args.language)
        for pid, wav in sorted(wavs.items()):
            u = w = None
            try:
                u = utmos.score(str(wav))
            except Exception as e:
                print(f"  ! utmos {wav.name}: {type(e).__name__}: {e}", file=sys.stderr)
            try:
                w = wer.score(str(wav), text_by_pid[pid], lang_by_pid[pid])
            except Exception as e:
                print(f"  ! wer {wav.name}: {type(e).__name__}: {e}", file=sys.stderr)
            rows.append({"voice": voice, "prompt_id": pid,
                         "utmos": u, "wer": w, "wav": wav.name})
            print(f"  p{pid}: utmos={_fmt(u)} wer={_fmt(w)}", flush=True)

    csv_path = out_dir / "voice_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["voice", "prompt_id", "utmos", "wer", "wav"])
        wtr.writeheader()
        for r in rows:
            wtr.writerow({**r, "utmos": _fmt(r["utmos"]), "wer": _fmt(r["wer"])})

    # Aggregate: mean over prompts, ranked by UTMOS desc (higher = more natural).
    print(f"\n{'voice':12s} {'UTMOS':>7s} {'WER':>7s} {'n':>3s}")
    agg = []
    for voice in voices:
        vr = [r for r in rows if r["voice"] == voice]
        us = [r["utmos"] for r in vr if r["utmos"] is not None]
        ws = [r["wer"] for r in vr if r["wer"] is not None]
        agg.append((voice,
                    sum(us) / len(us) if us else None,
                    sum(ws) / len(ws) if ws else None,
                    len(vr)))
    for voice, u, w, n in sorted(agg, key=lambda t: (t[1] is None, -(t[1] or 0))):
        print(f"{voice:12s} {_fmt(u, 3):>7s} {_fmt(w, 4):>7s} {n:3d}")

    print(f"\nwrote {csv_path}")
    print("Higher UTMOS = more natural. Lower WER = more intelligible. "
          "Prefer a voice that wins UTMOS without a WER outlier.")


if __name__ == "__main__":
    main()
